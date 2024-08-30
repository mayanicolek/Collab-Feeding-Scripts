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
# import pygame
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

def find_food(robot, spanet):
    try:
        frames = robot.env.cameras['wrist'].get_frames()
    finally:
        print("*Image Found*")

    
    image = frames['image']
    depth_image = frames['depth']
    annotated_image, detections, labels = robot.detect_items(image, detection_classes = ['white food'])
    height, width = image.shape[:2]
    combined_mask = np.zeros((height,width), dtype=np.uint8)
    confidence = detections.confidence
    centers = []
    major_axes = []
    actions = []
    for idx, mask in enumerate(detections.mask):
        if confidence[idx] > 0.5:
            mask = np.array(mask).astype(np.uint8)*255
            mask = robot.cleanup_mask(mask)

            cropped_image = crop_food_item(mask, image)
            positions, angles, action, scores, rotations, features  = spanet.publish_spanet(cropped_image, "test", True, torch.tensor([[1., 0., 0.]]))
            center, major_axis, skewer_image = robot.get_skewer_action(mask, image)
            centers.append(center)
            major_axes.append(major_axis)
            actions.append(action)
            combined_mask |= mask
             
    mask = robot.cleanup_mask(combined_mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    viz_image = np.hstack([annotated_image, mask, skewer_image])
    plt.imshow(viz_image)
    plt.show()
    print(centers)
    return image, depth_image, centers, major_axes, actions

def find_dishes():  
    cap0 = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    _, frame0 = cap0.read()
    img = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB) 
    img = img[200:325, 150:450]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 50, 150, cv2.THRESH_BINARY)
    plate_img, bowl_img, plate_pos, bowl_pos = identify_dishes(img, gray_img, binary_img)
    print("Plate: ", plate_pos, "Bowl: ", bowl_pos)
    cap0.release()
    return plate_pos, bowl_pos, plate_img, bowl_img, img

def identify_dishes(img, gray_img, binary_img):
    (contoursred1, hierarchy) = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dish = {"Image": [], "Center": [], "Area": [], "Dish": ['dish1','dish2']}
    for pic, contourred in enumerate(contoursred1):
        area = cv2.contourArea(contourred)
        if (area > 2000 and area < 20000):
            print(area)
            x, y, w, h = cv2.boundingRect(contourred)
            gray_img = cv2.rectangle(gray_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            dish['Image'].append(img[y:y+h,x:x+w])
            dish['Center'].append([200 - ((y+h) + y)/2, 350 - ((x+w) + x)/2])
            dish['Area'].append(area)

    if dish['Area'][1] > dish['Area'][0]:
        dish['Dish'][1] = 'plate'
        dish['Dish'][0] = 'bowl'
        plate_img = dish["Image"][1]
        bowl_img = dish["Image"][0]
        plate_center = dish['Center'][1]
        bowl_center = dish['Center'][0]
    if dish['Area'][0] > dish['Area'][1]:
        dish['Dish'][0] = 'plate'
        dish['Dish'][1] = 'bowl'
        plate_img = dish["Image"][0]
        bowl_img = dish["Image"][1]        
        plate_center = dish['Center'][0]
        bowl_center = dish['Center'][1]

    plate_x = (plate_center[0]*0.4)/200 + 0.12
    bowl_x = (bowl_center[0]*0.4)/200 + 0.12

    plate_y = (plate_center[1]*0.82)/350 - 0.5
    bowl_y = (bowl_center[1]*0.82)/350 - 0.5

    plate_pos = np.array([plate_x,plate_y])
    bowl_pos = np.array([bowl_x,bowl_y])

    return plate_img, bowl_img, plate_pos, bowl_pos

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