import argparse
import matplotlib.pyplot as plt
import numpy as np
import serial
import utils

from tkinter import *
from tkinter.ttk import *

from PIL import Image, ImageTk
import pyrealsense2 as rs
from queue import Queue
import SPAnet as SPAN
import yaml
import torch
import cv2
import argparse
from scipy.spatial.transform import Rotation as R
import math
import time
import matplotlib.pyplot as plt

from feeding_scripts.pc_utils import *
from feeding_scripts.pixel_selector import PixelSelector
from feeding_scripts.utils import Trajectory, Robot
from feeding_scripts.feeding import FrankaFeedingBot
from feeding_scripts.spanet_detector import SPANetDetector
from franky import Affine, CartesianMotion, Robot, ReferenceType


import robots
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor, Compose
import supervision as sv


# CONNECT TO PRESSURE PORT
comm_arduino = serial.Serial('/dev/ttyACM0', baudrate=9600)
print("Connected to kiri-spoon")

# Initialize Variables
det_food = False
acq_food = False
feed_pos = False
det_user = False
feed_user = False
acq_pos = True
data = []

print("Set up Configuration")
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="/home/vt-collab/Kiri_Spoon/2024KiriTesting/franka_feeding-main/configs/feeding.yaml")
parser.add_argument("--utensil", type=str, default="fork")
args = parser.parse_args()
config_path = args.config
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)
print("Finish setting up config")

print('[*] Connecting to robot/SPANET...')
robot = FrankaFeedingBot(config)
spanet = SPANetDetector()
print("Connection Established")

# Initialize GUI
input('Press Enter to Start Trial')
root = Tk()
GUI = utils.GUI_Display(root)

while True:

    try:
################ACQUISITION PERIOD###############

    ##############Identify dishware location####################
        if acq_pos:  
            root.title("Acquisition Period")
            robot.robot.go2position(robot.conn)
            plate_pos, bowl_pos, plate_img, bowl_img, color_img = SPAN.find_dishes()

            des_pos = plate_pos
            img1_tk, img2_tk, des_pos, des_dish = GUI.choose_dish_gui(plate_img, bowl_img, plate_pos, bowl_pos)

            cur_pos = robot.robot.find_pos(robot.conn)
            cur_pos[3] = np.pi if cur_pos[3] > 0 else -np.pi
            dish_pos = [des_pos[0], des_pos[1], 0.25, cur_pos[3], 0, 0]

            traj = robot.robot.make_traj(cur_pos, dish_pos, 10)
            robot.robot.run_traj(robot.conn, traj, 10)
            
            # GUI.continue_acquiring_gui()
            acq_pos = False
            det_food = True

        ##############Identify food location in relation to utensil####################

        if det_food:
            image, depth_image, centers, major_axes, actions = SPAN.find_food(robot, spanet)

            images, center, major_axis, action = GUI.choose_food_gui(image, centers, major_axes, actions)
            det_food = False
            acq_food = True

        ##############Acquire food using specific acquisition method####################
        if acq_food:
            if args.utensil == 'fork':
                robot.skewering_skill(image, depth_image, keypoint=center, major_axis=major_axis, actions = action[0])
            if args.utensil == 'spoon':
                robot.scooping_skill(image, depth_image, keypoint=center, major_axis=major_axis)
            if args.utensil == 'kiri':
                robot.kiri_skill(image, depth_image, des_dish, comm_arduino, keypoint=center, major_axis=major_axis)

            food_trig = GUI.food_present()

            if food_trig.get() == 'c': # If food on utensil continue to feeding
                acq_food = False
                feed_pos = True
            else: # If not, repeat acquisition
                cur_pos = robot.robot.find_pos(robot.conn)
                dish_pos = [des_pos[0], des_pos[1], 0.25, cur_pos[3], 0, 0]

                traj = robot.robot.make_traj(cur_pos, dish_pos, 5) # Move to desired dishware
                robot.robot.run_traj(robot.conn, traj, 5)

                if args.utensil == 'kiri': utils.send_arduino(comm_arduino, 3)

                acq_food = False
                det_food = True


    ################FEEDING PERIOD###############

        ##############Go to feeding position, in front of user####################
        if feed_pos:
            root.title("Feeding Period")
            cur_pos = robot.robot.find_pos(robot.conn)
            des_pos = np.array([0.5,0,0.2])

            if args.utensil == 'fork':
                cur_pos[3] = np.pi if cur_pos[3] > 0 else -np.pi
                des_pos = [des_pos[0], des_pos[1], des_pos[2], cur_pos[3], -np.pi/3, 0]
            if args.utensil == 'spoon':
                des_pos = [des_pos[0], des_pos[1], des_pos[2], cur_pos[3], cur_pos[4], 0]
            if args.utensil == 'kiri':
                if des_dish == 'plate':
                    des_pos = [des_pos[0], des_pos[1], des_pos[2], cur_pos[3], -np.pi/3, 0]
                if des_dish == 'bowl':
                    des_pos = [des_pos[0], des_pos[1], des_pos[2], cur_pos[3], -np.pi/3, 0]

            traj = robot.robot.make_traj(cur_pos, des_pos, 6)
            data = robot.robot.run_traj(robot.conn, traj, 6)

            GUI.continue_feeding_gui()
            feed_pos = False
            det_user = True

        ##############Identify mouth location in relation to utensil####################
        if det_user:
            cur_pos = robot.robot.find_pos(robot.conn)

            image, depth_image, center = SPAN.find_face(robot) # Find location of mouth
            des_pos = [0.7,0,0.2]

            des_pos = [des_pos[0], des_pos[1], des_pos[2], cur_pos[3], cur_pos[4], cur_pos[5]]
            
            traj = robot.robot.make_traj(cur_pos, des_pos, 3)
            robot.robot.run_traj(robot.conn, traj, 3)

            if np.linalg.norm(cur_pos - des_pos) < 0.5:
                det_user = False
                feed_user = True

        ##############Feed user using specific feeding method####################
        if feed_user:
            if args.utensil == 'kiri': utils.send_arduino(comm_arduino, 2)

            GUI.more_food_gui()

            if args.utensil == 'kiri': utils.send_arduino(comm_arduino, 3)

            feed_user = False
            acq_pos = True

    except KeyboardInterrupt:
        exit()        