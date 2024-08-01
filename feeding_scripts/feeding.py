import yaml
import torch
import cv2
import argparse
from scipy.spatial.transform import Rotation as R
import math
import time

from feeding_scripts.pc_utils import *
from feeding_scripts.pixel_selector import PixelSelector
from feeding_scripts.utils import Trajectory, Robot
import robots
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor, Compose
import supervision as sv

PATH_TO_GROUNDED_SAM = '/home/vt-collab/Kiri_Spoon/2024_Kiri_Testing/Grounded-Segment-Anything'
TEST_KEYPOINT = False

class FrankaFeedingBot:
    def __init__(self, config):

        # Constants for acquisition
        # self.HOME_POS = np.array([0.30496958, -0.00216635, 0.67])
        self.EE_POS_AT_PLATE_HEIGHT = 0.1
        self.transforms = None

        # Constants for GroundingDINO
        
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.GROUNDING_DINO_CONFIG_PATH = PATH_TO_GROUNDED_SAM + "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.GROUNDING_DINO_CHECKPOINT_PATH = PATH_TO_GROUNDED_SAM + "/groundingdino_swint_ogc.pth"
        self.BOX_THRESHOLD = 0.3
        self.TEXT_THRESHOLD = 0.2
        self.NMS_THRESHOLD = 0.4
        
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=self.GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT_PATH)
        # Segment-Anything checkpoint
        SAM_ENCODER_VERSION = "vit_h"
        SAM_CHECKPOINT_PATH = PATH_TO_GROUNDED_SAM + "/sam_vit_h_4b8939.pth"

        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=self.DEVICE)
        
        self.sam_predictor = SamPredictor(sam)

        self.pixel_selector = PixelSelector()
        
        self.env = robots.RobotEnv(**config)
        self.robot = Robot()
        self.conn = self.robot.connect2robot(port=8080)
        self.robot.go2position(self.conn)

    def fork_to_pixel(self):
        
        world_x = 0.033
        world_y = 0.015
        world_z = 0.1

        # convert to pixel using intrinsic matrix
        wrist_intrinsics = self.env.cameras['wrist'].get_intrinsics()['matrix']

        fx = wrist_intrinsics[0, 0]
        fy = wrist_intrinsics[1, 1]
        cx = wrist_intrinsics[0, 2]
        cy = wrist_intrinsics[1, 2]

        print(f"Camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        print("World coordinates", world_x, world_y, world_z)

        image_x = world_x * (fx / world_z) + cx
        image_y = world_y * (fy / world_z) + cy

        return int(image_x), int(image_y)

    def pixel_to_world(self, pixel_x, pixel_y, depth_image):
        obs = self.robot.find_pos(self.conn)
        ee_pos = obs[:3]
        ee_eul = obs[3:]
        r = R.from_euler('xyz', ee_eul)
        ee_quat = R.as_quat(r)

        T_Base2EE = np.eye(4)  
        T_Base2EE[:3, :3] = R.from_quat(ee_quat).as_matrix() 
        T_Base2EE[:3, 3] = ee_pos 

        T_Cam2EE = np.eye(4)
        T_Cam2EE[:3, :3] = R.from_quat([-0.004, 0.002, -0.380, 0.925]).as_matrix()
        # T_Cam2EE[:3, 3] = np.array([-0.04, -0.04, -0.1])
        T_Cam2EE[:3, 3] = np.array([-0.04, 0.04, -0.1])
        T_EE2Cam = np.linalg.inv(T_Cam2EE)

        wrist_intrinsics = self.env.cameras['wrist'].get_intrinsics()['matrix']
        cx, cy, cz = deproject_pixels(np.array([[pixel_x,pixel_y]]), depth_image.squeeze(), wrist_intrinsics)[0]
        print('cz', cz)
        P_Cam = np.eye(4)
        P_Cam[:3, 3] = np.array([cx, cy, cz])

        T_Base2ForkTarget = T_Base2EE @ T_EE2Cam @ P_Cam

        T_Fork2EE = np.eye(4)
        T_Fork2EE[:3, :3] = R.from_quat([-0.000, -0.000, 0.383, 0.924]).as_matrix()
        # T_Fork2EE[:3, 3] = np.array([-0.02, 0.000, -0.18])
        T_Fork2EE[:3, 3] = np.array([-0.02, 0.000, -0.18])

        T_Base2EETarget = T_Base2ForkTarget @ T_Fork2EE

        return T_Base2EETarget[:3, 3]


    def skewering_skill(self, color_image, depth_image, actions, keypoint=None, major_axis=None):
        """
        Args
            color_image: np.array, depth_image: np.array, camera_info: sensor_msgs.msg.CameraInfo
            keypoint: (int, int) or None, major_axis: float or None
        """
        if keypoint is not None:
            (center_x, center_y) = keypoint
        else:
            clicks = self.pixel_selector.run(color_image)
            (center_x, center_y) = clicks[0]
            major_axis = 0

        if major_axis < -np.pi/2:
            major_axis += np.pi
            major_axis = np.pi/2 - major_axis
        else:
            major_axis = (-np.pi/2) - major_axis
        
        print(f"Center x {center_x}, Center y {center_y} Major axis {major_axis}")

        ee_pos = self.pixel_to_world(center_x, center_y, depth_image)

        print("DEBUG: EE POS", ee_pos)
        if ee_pos[2] > self.EE_POS_AT_PLATE_HEIGHT:
            ee_pos[2] = self.EE_POS_AT_PLATE_HEIGHT

        obs = self.robot.find_pos(self.conn)
        # major_axis -= np.pi/4

        if obs[3] < 0:
            target_euler = np.array([-np.pi, 0, major_axis])
        else:
            target_euler = np.array([np.pi, 0, major_axis])


        self.robot.rotate_robot(self.conn,target_euler)
        self.robot.translate_robot(self.conn,ee_pos + [0, 0, 0.03],  max_delta=0.025)
        time.sleep(0.5)
        self.robot.translate_robot(self.conn,ee_pos - [0, 0, 0.03], max_delta=0.025)
        time.sleep(0.2)
        self.robot.translate_robot(self.conn,ee_pos + [0, 0, 0.03],  max_delta=0.025)

    def scooping_skill(self, color_image, depth_image, keypoint=None, major_axis=None):
        """
        Args
            color_image: np.array, depth_image: np.array, camera_info: sensor_msgs.msg.CameraInfo
            keypoint: (int, int) or None, major_axis: float or None
        """

        if keypoint is not None:
            (center_x, center_y) = keypoint
        else:
            clicks = self.pixel_selector.run(color_image)
            (center_x, center_y) = clicks[0]
            major_axis = 0

        # if major_axis < -np.pi/2:
        #     major_axis += np.pi
        #     major_axis = np.pi/2 - major_axis
        # else:
        #     major_axis = (-np.pi/2) - major_axis
        
        print(f"Center x {center_x}, Center y {center_y} Major axis {major_axis}")

        ee_pos = self.pixel_to_world(center_x, center_y, depth_image)

        print("DEBUG: EE POS", ee_pos)
        if ee_pos[2] < self.EE_POS_AT_PLATE_HEIGHT:
            ee_pos[2] = self.EE_POS_AT_PLATE_HEIGHT

        obs = self.robot.find_pos(self.conn)
        target_euler = np.array([obs[3], obs[4] + np.pi/4, obs[5]])
        self.robot.rotate_robot(self.conn,target_euler)

        self.robot.translate_robot(self.conn,ee_pos + [0, 0, 0.03],  max_delta=0.025)
        time.sleep(0.5)
        target_euler = np.array([obs[3], obs[4] - np.pi/2, obs[5]])
        self.robot.move_robot(self.conn,ee_pos,target_euler)
        time.sleep(0.2)
        self.robot.translate_robot(self.conn,ee_pos + [0, 0, 0.03],  max_delta=0.025)

    def take_rgbd(self):
        frames = self.env._get_frames()
        color_image = frames['wrist_image']
        depth_image = frames['wrist_depth']
        return color_image, depth_image

    def rgbd2pointCloud(self, color_image, depth_image):
        depth_image = depth_image.squeeze()
        agent_intrinsics = self.env.cameras['agent'].get_intrinsics()['matrix']
        denoised_idxs = denoise(depth_image)
        if self.transforms is not None:
            tf = self.transforms['agent']['tcr']
        else:
            tf = np.eye(4)
        points_3d = deproject(depth_image, agent_intrinsics, tf)
        colors = color_image.reshape(points_3d.shape)/255.

        points_3d = points_3d[denoised_idxs]
        colors = colors[denoised_idxs]

        idxs = crop(points_3d)
        points_3d = points_3d[idxs]
        colors = colors[idxs]

        pcd_merged = merge_pcls([points_3d], [colors])
        pcd_merged.remove_duplicated_points()
        return pcd_merged

    def detect_items(self, image, detection_classes = ['food item']):
        
        '''
        Detects items in the image and returns the annotated image with bounding boxes and masks.
        Args:
            image: np.array, detection_classes: list of str
        '''

        cropped_image = image.copy()

        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=cropped_image,
            classes=detection_classes,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD
        )
        # print(f"Detected {len(detections)} items")
        # print(detections)

        # annotate image with detections
        box_annotator = sv.BoundingBoxAnnotator()
        labels = [
            f"{detection_classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        annotated_frame = box_annotator.annotate(scene=cropped_image.copy(), detections=detections)
        
        # NMS post process
        #print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy).float().detach(), 
            torch.from_numpy(detections.confidence).float().detach(), 
            self.NMS_THRESHOLD
        ).tolist()

        # remove boxes which are union of two boxes
        
        detections.xyxy = detections.xyxy[nms_idx]

        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        #print(f"After NMS: {len(detections.xyxy)} boxes")

        # Prompting SAM with detected boxes
        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
            return np.array(result_masks)

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # print(f"Detected {len(detections)} items")
        # print(detections)
        
        # annotate image with detections
        box_annotator = sv.BoundingBoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()
        labels = [
            f"{detection_classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]

        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        
        return annotated_image, detections, labels

    def detect_angular_bbox(self, mask):
        contours,hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            return box
        else:
            return None

    def get_skewer_action(self, mask, image):
        '''
        Detects the center and major axis of the skewer action. Major axis is detected in the opencv frame (camera_color_optical_frame). 
        Args:
            mask: np.array
            image: np.array
        '''
        bbox = self.detect_angular_bbox(mask)

        center = np.array([bbox[0][0], bbox[0][1]]) + np.array([bbox[1][0], bbox[1][1]]) + np.array([bbox[2][0], bbox[2][1]]) + np.array([bbox[3][0], bbox[3][1]])
        center = center / 4
        center = center.astype(int)

        if np.linalg.norm(np.array(bbox[0]) - np.array(bbox[1])) > np.linalg.norm(np.array(bbox[1]) - np.array(bbox[2])):
            if bbox[0][1] == bbox[1][1]:
                major_axis = 0
            elif bbox[0][1] > bbox[1][1]:
                major_axis = math.atan2(bbox[1][1] - bbox[0][1], bbox[1][0] - bbox[0][0])
            else:
                major_axis = math.atan2(bbox[0][1] - bbox[1][1], bbox[0][0] - bbox[1][0])
        else:
            if bbox[1][1] == bbox[2][1]:
                major_axis = 0
            elif bbox[1][1] > bbox[2][1]:
                major_axis = math.atan2(bbox[2][1] - bbox[1][1], bbox[2][0] - bbox[1][0])
            else:
                major_axis = math.atan2(bbox[1][1] - bbox[2][1], bbox[1][0] - bbox[2][0])
        print("Major Axis (with opencv frame): (pi - 2*pi)", major_axis)

        viz = image.copy()
        for i in range(4):
            cv2.putText(viz, str(i), (bbox[i][0], bbox[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.line(viz, tuple(bbox[i]), tuple(bbox[(i+1)%4]), (0, 0, 255), 2)
            
        cv2.arrowedLine(viz, center, (int(center[0] + 100 * math.cos(major_axis)), int(center[1] + 100 * math.sin(major_axis))), (0, 0, 255), 2)
        return center, major_axis, viz
    
    def cleanup_mask(self, mask, blur_kernel_size=(5, 5), threshold=127, erosion_size=3):
        """
        Applies low-pass filter, thresholds, and erodes an image mask.

        :param image: Input image mask in grayscale.
        :param blur_kernel_size: Size of the Gaussian blur kernel.
        :param threshold: Threshold value for binary thresholding.
        :param erosion_size: Size of the kernel for erosion.
        :return: Processed image.
        """
        # Apply Gaussian Blur for low-pass filtering
        blurred = cv2.GaussianBlur(mask, blur_kernel_size, 0)
        # Apply thresholding
        _, thresholded = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        # Create erosion kernel
        erosion_kernel = np.ones((erosion_size, erosion_size), np.uint8)
        # Apply erosion
        eroded = cv2.erode(thresholded, erosion_kernel, iterations=1)
        return eroded

