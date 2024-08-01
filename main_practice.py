import yaml
import torch
import cv2
import argparse
from scipy.spatial.transform import Rotation as R
import math
import time
import matplotlib.pyplot as plt
import rospy
from feeding_scripts.pc_utils import *
from feeding_scripts.pixel_selector import PixelSelector
from feeding_scripts.utils import Trajectory, Robot
from feeding_scripts.feeding import FrankaFeedingBot
from food_detector.spanet_detector import SPANetDetector
import food_detector.ada_feeding_demo_config as conf
import robots
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor, Compose
import supervision as sv

TEST_KEYPOINT = False

def crop_food_item(mask, image):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        box = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.int0)
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image
    else:
        return None

if __name__ == '__main__':


    rospy.init_node("main_practice")
    print("Set up Configuration")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/vt-collab/Kiri_Spoon/2024_Kiri_Testing/franka_feeding-main/configs/feeding.yaml")
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    print("Finish setting up config")
    
    print("Connecting to Robot...")
    robot = FrankaFeedingBot(config)
    print("Connected to Robot")
    spanet = SPANetDetector(use_cuda=conf.use_cuda)

    color_image, depth_image = robot.take_rgbd()
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    if TEST_KEYPOINT:
        robot.skewering_skill(image, depth_image)
    else:
        annotated_image, detections, labels = robot.detect_items(image, detection_classes = ['green peas'])

        mask = np.array(detections.mask[0]).astype(np.uint8)*255
        mask = robot.cleanup_mask(mask)


        cropped_image = crop_food_item(mask, image)     

        
        positions, angles, actions, scores, rotations, features  = spanet.publish_spanet(cropped_image, "test", True, torch.tensor([[1., 0., 0.]]))
        # Get skewer action
        center, major_axis, skewer_image = robot.get_skewer_action(mask, image)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        viz_image = np.hstack([annotated_image, mask, skewer_image])
        plt.imshow( viz_image)
        plt.show()

        print("Major Axis: ", np.degrees(major_axis))
        
        # Call skewering skill
        robot.skewering_skill(image, depth_image, keypoint=center, major_axis=major_axis, actions = actions)
    robot.robot.go2position(robot.conn)
