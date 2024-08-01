from operator import truediv
import numpy as np
import cv2
import time
import pickle
import socket
import matplotlib.pyplot as plt
import pickle as pkl
import sys
from scipy.interpolate import interp1d
import pygame
import pyrealsense2 as rs
from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk,ImageEnhance
import torch
from queue import Queue
import threading
import serial
import yaml
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

def find_face(robot):
    try:
        frames = robot.env.cameras['wrist'].get_frames()
    finally:
        print("*Image Found*")

    image = frames['image']
    depth_image = frames['depth']
    annotated_image, detections, labels = robot.detect_items(image, detection_classes = ['face'])
    confidence = detections.confidence
    for idx, mask in enumerate(detections.mask):
        if confidence[idx] > 0.5:
            mask = np.array(mask).astype(np.uint8)*255
            mask = robot.cleanup_mask(mask)
            
    try:    
        bbox = robot.detect_angular_bbox(mask)
        center = np.array([bbox[0][0], bbox[0][1]]) + np.array([bbox[1][0], bbox[1][1]]) + np.array([bbox[2][0], bbox[2][1]]) + np.array([bbox[3][0], bbox[3][1]])
        center = center / 4
        center = center.astype(int)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        viz_image = np.hstack([annotated_image, mask])
        plt.imshow(viz_image)
        plt.show()
    except:
        print("No Face Detected")
        center = [0,0]
        
    return image, depth_image, center

def find_food(robot):
    try:
        frames = robot.env.cameras['wrist'].get_frames()
    finally:
        print("*Image Found*")

    image = frames['image']
    depth_image = frames['depth']
    annotated_image, detections, labels = robot.detect_items(image, detection_classes = ['food'])
    height, width = image.shape[:2]
    combined_mask = np.zeros((height,width), dtype=np.uint8)
    confidence = detections.confidence
    centers = []
    major_axes = []
    for idx, mask in enumerate(detections.mask):
        if confidence[idx] > 0.5:
            mask = np.array(mask).astype(np.uint8)*255
            mask = robot.cleanup_mask(mask)
            center, major_axis, skewer_image = robot.get_skewer_action(mask, image)
            centers.append(center)
            major_axes.append(major_axis)
            combined_mask |= mask
    # mask = np.array(detections.mask[0]).astype(np.uint8)*255
    # mask = robot.cleanup_mask(mask)
    mask = robot.cleanup_mask(combined_mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    viz_image = np.hstack([annotated_image, mask, skewer_image])
    plt.imshow(viz_image)
    plt.show()
    return image, depth_image, centers, major_axes

def find_dishes():  
    cap0 = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    _, frame0 = cap0.read()
    img = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB) 
    img = img[200:400, 100:450]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 50, 150, cv2.THRESH_BINARY)
    dish1_img, dish2_img, dish1_pos, dish2_pos, object_img = identify_dishes(img, gray_img, binary_img)
    print(dish1_pos, dish2_pos)
    cap0.release()
    return dish1_pos, dish2_pos, dish1_img, dish2_img

def identify_dishes(img, gray_img, binary_img):
    (contoursred1, hierarchy) = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    counter = 0
    for pic, contourred in enumerate(contoursred1):
        area = cv2.contourArea(contourred)
        if (area > 1000 and area < 20000):
            x, y, w, h = cv2.boundingRect(contourred)
            gray_img = cv2.rectangle(gray_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if counter == 0: 
                dish1_img = img[y:y+h,x:x+w]
                dish1_center = [200 - ((y+h) + y)/2, 350 - ((x+w) + x)/2]
            elif counter == 1: 
                dish2_img = img[y:y+h,x:x+w]
                dish2_center = [200 - ((y+h) + y)/2, 350 - ((x+w) + x)/2]
            counter += 1

    dish1_x = (dish1_center[0]*0.4)/200 + 0.35
    dish2_x = (dish2_center[0]*0.4)/200 + 0.35

    dish1_y = (dish1_center[1]*0.79)/350 - 0.36
    dish2_y = (dish2_center[1]*0.79)/350 - 0.36

    dish1_pos = np.array([dish1_x,dish1_y])
    dish2_pos = np.array([dish2_x,dish2_y])

    return dish1_img, dish2_img, dish1_pos, dish2_pos, gray_img