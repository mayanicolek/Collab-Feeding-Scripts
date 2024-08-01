import argparse
import matplotlib.pyplot as plt
import numpy as np
import serial
import utils

from tkinter import *
from tkinter.ttk import *

import cv2
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

import robots
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor, Compose
import supervision as sv

# import serial

# CONNECT TO PRESSURE PORT
# comm_arduino = serial.Serial('ttyACM0', baudrate=9600)
# print("Connected to spoon")

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
parser.add_argument("--config", type=str, default="/home/vt-collab/Kiri_Spoon/2024_Kiri_Testing/franka_feeding-main/configs/feeding.yaml")
parser.add_argument("--utensil", type=str, default="fork")
args = parser.parse_args()
config_path = args.config
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)
print("Finish setting up config")

print('[*] Connecting to robot...')
robot = FrankaFeedingBot(config)
print("Connection Established")

# Initialize GUI
input('Press Enter to Start Trial')
root = Tk()

while True:

    # ACQUISITION PERIOD
    if acq_pos:  # Identify dishware location
        root.title("Acquisition Period")
        robot.robot.go2position(robot.conn)
        dish1_pos, dish2_pos, dish1_img, dish2_img = SPAN.find_dishes()
        root, img1_tk, img2_tk, des_pos = utils.choose_dish_gui(root,dish1_img,dish2_img,dish1_pos,dish2_pos)

        cur_pos = robot.robot.find_pos(robot.conn)

        if cur_pos[3] < 0:
            dish_pos = [des_pos[0], des_pos[1], 0.25, -np.pi, 0, 0]
        else:
            dish_pos = [des_pos[0], des_pos[1], 0.25, np.pi, 0, 0]
        
        traj = robot.robot.make_traj(cur_pos, dish_pos, 5) # Move to desired dishware
        robot.robot.run_traj(robot.conn, traj, 5)
        
        root = utils.continue_acquiring_gui(root)
        acq_pos = False
        det_food = True


    if det_food: # Identify food location in relation to utensil
        image, depth_image, centers, major_axes = SPAN.find_food(robot)

        root, images, center, major_axis = utils.choose_food_gui(root, image, centers, major_axes)
        det_food = False
        acq_food = True


    if acq_food: # Acquire food using specific acquisition method
        if args.utensil == 'fork':
            robot.skewering_skill(image, depth_image, keypoint=center, major_axis=major_axis)
        if args.utensil == 'spoon':
            robot.scooping_skill(image, depth_image, keypoint=center, major_axis=major_axis)

        root, food_trig = utils.food_present(root)

        if food_trig.get() == 'c': # If food on utensil continue to feeding
            acq_food = False
            feed_pos = True
        else: # If not, repeat acquisition
            cur_pos = robot.robot.find_pos(robot.conn)
            traj = robot.robot.make_traj(cur_pos, dish_pos, 3) # Move to desired dishware
            robot.robot.run_traj(robot.conn, traj, 3)
            acq_food = False
            det_food = True


    # FEEDING PERIOD
    if feed_pos: # Go to feeding position, in front of user
        root.title("Feeding Period")
        cur_pos = robot.robot.find_pos(robot.conn)
        des_pos = np.array([0.6,0,0.2])

        if cur_pos[3] < 0:
            des_pos = [des_pos[0], des_pos[1], des_pos[2], -np.pi, cur_pos[4], 0]
        else:
            des_pos = [des_pos[0], des_pos[1], des_pos[2], np.pi, cur_pos[4], 0]

        traj = robot.robot.make_traj(cur_pos, des_pos, 5)
        data = robot.robot.run_traj(robot.conn, traj, 5)

        if args.utensil == 'fork':
            for idx in range(300):
                qdot = np.array([0,0,0,0,0,0.5,0])
                robot.robot.send2robot(robot.conn, qdot, 'v')

        root = utils.continue_feeding_gui(root)
        feed_pos = False
        det_user = True

    if det_user: # Identify mouth location in relation to utensil
        cur_pos = robot.robot.find_pos(robot.conn)

        image, depth_image, center = SPAN.find_face(robot) # Find location of mouth
        des_pos = [0.9,0,0.2]

        if cur_pos[3] < 0:
            des_pos = [des_pos[0], des_pos[1], des_pos[2], -np.pi, cur_pos[4], 0]
        else:
            des_pos = [des_pos[0], des_pos[1], des_pos[2], np.pi, cur_pos[4], 0]
        
        traj = robot.robot.make_traj(cur_pos, des_pos, 3)
        robot.robot.run_traj(robot.conn, traj, 3)

        if np.linalg.norm(cur_pos - des_pos) < 0.3:
            det_user = False
            feed_user = True

    if feed_user: # Feed user using specific feeding method
        root = utils.more_food_gui(root)

        cur_pos = robot.robot.find_pos(robot.conn)
        dish_pos = [cur_pos[0], cur_pos[1], cur_pos[2], cur_pos[3], 0, 0]
        traj = robot.robot.make_traj(cur_pos, des_pos, 3)
        robot.robot.run_traj(robot.conn, traj, 3)

        feed_user = False
        acq_pos = True

        