import argparse
import os
import pickle
import socket
import sys
import time
from fileinput import filename
from multiprocessing import Process
from tracemalloc import start
import cv2
import matplotlib.pyplot as plt
import numpy as np
import serial
from utils import *

HOME = [3.600000e-05, -7.852010e-01,  1.350000e-04, -2.356419e+00, -3.280000e-04,  1.571073e+00,  7.856910e-01]

# CONNECT TO ROBOT
robot = FrankaResearch3()
PORT_robot = 8080
print('[*] Connecting to low-level controller...')
conn = robot.connect(PORT_robot)
print("Connection Established")

# # CONNECT TO GRIPPER
PORT_gripper = 8081
print("[*] Connecting to gripper")
conn_gripper = robot.connect(PORT_gripper)
print("Connected to gripper")

# CONNECT TO SPOON
# comm_arduino = serial.Serial('/dev/ttyACM0', baudrate=9600)
# print("Connected to spoon")

# GO TO HOME POSITION
print("Returning Home")
robot.go2position(conn)

# Initialize Variables
interface = Joystick()
data = {"Time": [], "Position": [], "Force": [], "Inputs": []}
motor = False
input('Press Enter to Start Trial (X Mode Light Off)')
timestep = time.time()

while True:

    z, A_pressed, B_pressed, X_pressed, Y_pressed, START_pressed, STOP_pressed, RT, LT = interface.input()
    Joystick_inputs = [z, A_pressed, B_pressed, X_pressed, Y_pressed, START_pressed, STOP_pressed, RT, LT]
    x_dot = [0.] * 6
    cur_pos = robot.find_pos(conn)

    # Regular Mode
    scale = 0.6

    if A_pressed:
        robot.send2gripper(conn_gripper, 'o')
    if B_pressed:
        robot.send2gripper(conn_gripper, 'c')


    x_dot[0:3] = scale * np.array([z[1], z[0], -z[2]])
    # robot.run_xdot(x_dot, conn)
    states = robot.readState(conn)
    q_dot = robot.xdot2qdot(x_dot, states)
    robot.send2robot(conn, q_dot, 'v')
    # data = append_data(data, time.time(), cur_pos, wrench, Joystick_inputs)
