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
import SPAnet as SPAN
import torch
from queue import Queue
import serial  


##########################################
class GUI_Display:
    def __init__(self, root):
        self.root = root

    def food_present(self):

        continue_trig = StringVar(self.root, "1")
        Radiobutton(self.root, text="Food picked successfully?",style = 'W.TButton', variable=continue_trig, width=25, value = 'c').grid(row = 6, column = 0,columnspan=3)
        Radiobutton(self.root, text="Food NOT picked successfully?",style = 'W.TButton', variable=continue_trig, width=25, value = 'd').grid(row = 6, column = 4,columnspan=3)

        self.root.wait_variable(continue_trig)
        return continue_trig

    def more_food_gui(self):

        repeat_trig = StringVar(self.root, "1")
        Radiobutton(self.root, text="Ready for more food?",style = 'W.TButton', variable=repeat_trig, width=54, value = {"CONTINUE": 'c'}).grid(row = 8,column=0, columnspan = 6)

        self.root.wait_variable(repeat_trig)

        
    def continue_feeding_gui(self):

        feed_trig = StringVar(self.root, "1")
        Radiobutton(self.root, text="Ready to Eat?", style = 'W.TButton',variable=feed_trig, width=54, value = {"CONTINUE": 'c'}).grid(row=7,column=0, columnspan=6)

        self.root.wait_variable(feed_trig)

    def continue_acquiring_gui(self):

        food_trig = StringVar(self.root, "1")
        Radiobutton(self.root, text="Ready to Pick Food?",style = 'W.TButton', variable=food_trig, width=75, value = {"CONTINUE": 'c'}).grid(row = 2,column=0, columnspan = 6)

        self.root.wait_variable(food_trig)

    def choose_food_gui(self, image, centers, major_axes, actions):

        Label(self.root,text='What food would you like to pick?').grid(row = 3,column=0, columnspan = 6)
        food_trig = StringVar(self.root,"a")
        img_tk_dict = {}
        for idx in range(3):
            try:
                center = centers[idx]
                x = center[0]
                y = center[1]
                food_img = image[y-25:y+25,x-25:x+25]
                img_tk_dict[idx] = ImageTk.PhotoImage(Image.fromarray(food_img).resize((133,133)))
            except:
                img_tk_dict[idx] = ImageTk.PhotoImage(Image.fromarray(food_img).resize((133,133)))
                print("not enough objects")

        names = list(range(3))
        food_options = {name: value for name, value in zip(names, names)}

        for idx,(text, value) in enumerate(food_options.items()):
            if idx < 3:
                Label(self.root,image=img_tk_dict[idx]).grid(row = 4,column = idx*2,columnspan=2)
                Radiobutton(self.root,text = text,style = 'W.TButton',variable = food_trig,value = value).grid(row = 5, column = idx*2,columnspan=2) # ASK USER WHAT FOOD THEY WANT

        self.root.wait_variable(food_trig)

        if food_trig.get() == '0':
            center = centers[0]
            major_axis = major_axes[0]
            action = actions[0]
        elif food_trig.get() == '1':
            center = centers[1]
            major_axis = major_axes[1]
            action = actions[1]
        elif food_trig.get() == '2':
            center = centers[2]
            major_axis = major_axes[2]
            action = actions[2]

        return img_tk_dict, center, major_axis, action

    def choose_dish_gui(self,dish1_img,dish2_img, dish1_pos, dish2_pos):
        # root=Toplevel(root)
        self.root.geometry('646x650')
        dish1_img = Image.fromarray(dish1_img).resize((321,321))
        dish2_img = Image.fromarray(dish2_img).resize((321,321))
        img1_tk = ImageTk.PhotoImage(dish1_img)
        img2_tk = ImageTk.PhotoImage(dish2_img)
        dish_trig = StringVar(self.root,"0")
        dish_options = {"Plate" : '1',"Bowl" : '2'}
        for(text, value) in dish_options.items():
            if value == '1':
                Label(self.root,image=img1_tk).grid(row = 0, column = 0,columnspan=3)
                Radiobutton(self.root,text = text,style = 'W.TButton',variable = dish_trig,value = value, width = 25).grid(row = 1, column = 0,columnspan=3)
            if value == '2':
                Label(self.root,image=img2_tk).grid(row = 0, column = 3,columnspan=3)
                Radiobutton(self.root,text = text,style = 'W.TButton',variable = dish_trig,value = value, width = 25).grid(row = 1, column = 3,columnspan=3)
        
        self.root.wait_variable(dish_trig)

        if dish_trig.get() == '1':
            des_pos = dish1_pos
            des_dish = "plate"
        elif dish_trig.get() == '2':
            des_pos = dish2_pos
            des_dish = "bowl"

        return img1_tk, img2_tk, des_pos, des_dish
        
    def update(self):
        self.root.update()
    
    def close(self):
        self.root.withdraw()

    def open(self):
        self.root.deiconify()

#######################################################

# ##########################################

# def food_present(root):
#     continue_trig = StringVar(root, "1")
#     Radiobutton(root, text="Food picked successfully?",style = 'W.TButton', variable=continue_trig, width=25, value = 'c').grid(row = 6, column = 0,columnspan=3)
#     Radiobutton(root, text="Food NOT picked successfully?",style = 'W.TButton', variable=continue_trig, width=25, value = 'd').grid(row = 6, column = 4,columnspan=3)
#     while continue_trig.get() == "1":
#         root.update()
#     return root, continue_trig

# def more_food_gui(root):
#     repeat_trig = StringVar(root, "1")
#     Radiobutton(root, text="Ready for more food?",style = 'W.TButton', variable=repeat_trig, width=54, value = {"CONTINUE": 'c'}).grid(row = 8,column=0, columnspan = 6)
#     while repeat_trig.get() == "1":
#         root.update()
#     return root

# def continue_feeding_gui(root):
#     feed_trig = StringVar(root, "1")
#     Radiobutton(root, text="Ready to Eat?", style = 'W.TButton',variable=feed_trig, width=54, value = {"CONTINUE": 'c'}).grid(row=7,column=0, columnspan=6)
#     while feed_trig.get() == "1":
#         root.update()
#     return root

# def continue_acquiring_gui(root):
#     food_trig = StringVar(root, "1")
#     Radiobutton(root, text="Ready to Pick Food?",style = 'W.TButton', variable=food_trig, width=75, value = {"CONTINUE": 'c'}).grid(row = 2,column=0, columnspan = 6)
#     while food_trig.get() == "1":
#         root.update()
#     return root

# def choose_food_gui(root, image, centers, major_axes, actions):
#     Label(root,text='What food would you like to pick?').grid(row = 3,column=0, columnspan = 6)
#     food_trig = StringVar(root,"a")
#     img_tk_dict = {}
#     for idx in range(3):
#         try:
#             center = centers[idx]
#             x = center[0]
#             y = center[1]
#             food_img = image[y-25:y+25,x-25:x+25]
#             img_tk_dict[idx] = ImageTk.PhotoImage(Image.fromarray(food_img).resize((133,133)))
#         except:
#             img_tk_dict[idx] = ImageTk.PhotoImage(Image.fromarray(food_img).resize((133,133)))
#             print("not enough objects")

#     names = list(range(3))
#     food_options = {name: value for name, value in zip(names, names)}

#     for idx,(text, value) in enumerate(food_options.items()):
#         if idx < 3:
#             Label(root,image=img_tk_dict[idx]).grid(row = 4,column = idx*2,columnspan=2)
#             Radiobutton(root,text = text,style = 'W.TButton',variable = food_trig,value = value).grid(row = 5, column = idx*2,columnspan=2) # ASK USER WHAT FOOD THEY WANT

#     while food_trig.get() == "a":
#         root.update()

#     if food_trig.get() == '0':
#         center = centers[0]
#         major_axis = major_axes[0]
#         action = actions[0]
#     elif food_trig.get() == '1':
#         center = centers[1]
#         major_axis = major_axes[1]
#         action = actions[1]
#     elif food_trig.get() == '2':
#         center = centers[2]
#         major_axis = major_axes[2]
#         action = actions[2]

#     return root, img_tk_dict, center, major_axis, action

# def choose_dish_gui(root,dish1_img,dish2_img, dish1_pos, dish2_pos):
#     root.withdraw()
#     root=Toplevel(root)
#     # root.geometry('400x465')
#     # root.geometry('646x550')
#     root.geometry('646x650')
#     #dish1_img = ImageEnhance.Color(dish1_img).enhance(2.5)
#     dish1_img = Image.fromarray(dish1_img).resize((321,321))
#     dish2_img = Image.fromarray(dish2_img).resize((321,321))
#     img1_tk = ImageTk.PhotoImage(dish1_img)
#     img2_tk = ImageTk.PhotoImage(dish2_img)
#     dish_trig = StringVar(root,"0")
#     dish_options = {"Plate" : '1',"Bowl" : '2'}
#     #Label(root,width=20).grid(column=2,columnspan=3)
#     for(text, value) in dish_options.items():
#         if value == '1':
#             Label(root,image=img1_tk).grid(row = 0, column = 0,columnspan=3)
#             Radiobutton(root,text = text,style = 'W.TButton',variable = dish_trig,value = value, width = 25).grid(row = 1, column = 0,columnspan=3)
#         if value == '2':
#             Label(root,image=img2_tk).grid(row = 0, column = 3,columnspan=3)
#             Radiobutton(root,text = text,style = 'W.TButton',variable = dish_trig,value = value, width = 25).grid(row = 1, column = 3,columnspan=3)

#     while dish_trig.get() == "0":
#         root.update()

#     if dish_trig.get() == '1':
#         des_pos = dish1_pos
#         des_dish = "plate"
#     elif dish_trig.get() == '2':
#         des_pos = dish2_pos
#         des_dish = "bowl"

#     return root, img1_tk, img2_tk, des_pos, des_dish

# #######################################################


def append_data(data, timestamp, cur_pos, wrench, voltage, Joystick_inputs):
    data["Time"].append(timestamp)
    data["Position"].append(cur_pos)
    data["Force"].append(wrench)
    data["Inputs"].append(Joystick_inputs)
    data["Voltage"].append(voltage)
    return data

def send_arduino(comm_arduino, user_input):
    string = '<' + str(user_input) + '>'
    comm_arduino.write(string.encode())


# class Joystick(object):

#     def __init__(self):
#         pygame.init()
#         self.gamepad = pygame.joystick.Joystick(0)
#         self.gamepad.init()
#         self.deadband = 0.1
#         self.timeband = 0.5
#         self.lastpress = time.time()

#     def input(self):
#         pygame.event.get()
#         curr_time = time.time()
#         z1 = self.gamepad.get_axis(0)
#         z2 = self.gamepad.get_axis(1)
#         z3 = self.gamepad.get_axis(4)
#         if abs(z1) < self.deadband:
#             z1 = 0.0
#         if abs(z2) < self.deadband:
#             z2 = 0.0
#         if abs(z3) < self.deadband:
#             z3 = 0.0
#         A_pressed = self.gamepad.get_button(0) and (curr_time - self.lastpress > self.timeband)
#         B_pressed = self.gamepad.get_button(1) and (curr_time - self.lastpress > self.timeband)
#         X_pressed = self.gamepad.get_button(2) and (curr_time - self.lastpress > self.timeband)
#         Y_pressed = self.gamepad.get_button(3) and (curr_time - self.lastpress > self.timeband)
#         START_pressed = self.gamepad.get_button(7) and (curr_time - self.lastpress > self.timeband)
#         STOP_pressed = self.gamepad.get_button(6) and (curr_time - self.lastpress > self.timeband)
#         Right_trigger = self.gamepad.get_button(5)
#         Left_Trigger = self.gamepad.get_button(4)
#         if A_pressed or START_pressed or B_pressed:
#             self.lastpress = curr_time
#         return [z1, z2, z3], A_pressed, B_pressed, X_pressed, Y_pressed, START_pressed, STOP_pressed, Right_trigger, Left_Trigger


class Trajectory(object):

    def __init__(self, xi, T):
        """ create cublic interpolators between waypoints """
        self.xi = np.asarray(xi)
        self.T = T
        self.n_waypoints = xi.shape[0]
        timesteps = np.linspace(0, self.T, self.n_waypoints)
        self.f1 = interp1d(timesteps, self.xi[:,0], kind='cubic')
        self.f2 = interp1d(timesteps, self.xi[:,1], kind='cubic')
        self.f3 = interp1d(timesteps, self.xi[:,2], kind='cubic')
        self.f4 = interp1d(timesteps, self.xi[:,3], kind='cubic')
        self.f5 = interp1d(timesteps, self.xi[:,4], kind='cubic')
        self.f6 = interp1d(timesteps, self.xi[:,5], kind='cubic')
        # self.f7 = interp1d(timesteps, self.xi[:,6], kind='cubic')

    def get(self, t):
        """ get interpolated position """
        if t < 0:
            q = [self.f1(0), self.f2(0), self.f3(0), self.f4(0), self.f5(0), self.f6(0)]
        elif t < self.T:
            q = [self.f1(t), self.f2(t), self.f3(t), self.f4(t), self.f5(t), self.f6(t)]
        else:
            q = [self.f1(self.T), self.f2(self.T), self.f3(self.T), self.f4(self.T), self.f5(self.T), self.f6(self.T)]
        return np.asarray(q)


class FrankaResearch3(object):

    def __init__(self):
        # self.home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
        self.home = [0.00720989, -1.45442,  0.00506945, -2.21578, -3.280000e-04,  0.832898,  7.856910e-01]

    def connect(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('172.16.0.3', port))
        s.listen()
        conn, addr = s.accept()
        return conn

    def send2gripper(self, conn, command):
        send_msg = "s," + command + ","
        conn.send(send_msg.encode())

    def send2robot(self, conn, qdot, control_mode, limit=1):
        qdot = np.asarray(qdot)
        scale = np.linalg.norm(qdot)
        if scale > limit:
            qdot *= limit/scale
        send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
        if send_msg == '0.,0.,0.,0.,0.,0.,0.':
            send_msg = '0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000'
        send_msg = "s," + send_msg + "," + control_mode + ","
        conn.send(send_msg.encode())

    def listen2robot(self, conn):
        state_length = 7 + 42
        message = str(conn.recv(2048))[2:-2]
        state_str = list(message.split(","))
        for idx in range(len(state_str)):
            if state_str[idx] == "s":
                state_str = state_str[idx+1:idx+1+state_length]
                break
        try:
            state_vector = [float(item) for item in state_str]
        except ValueError:
            return None
        if len(state_vector) is not state_length:
            return None
        state_vector = np.asarray(state_vector)
        state = {}
        state["q"] = state_vector[0:7]
        state["J"] = state_vector[7:49].reshape((7,6)).T
        xyz_lin, R = self.joint2pose(state_vector[0:7])
        beta = -np.arcsin(R[2, 0])
        alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
        gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
        xyz_ang = [alpha, beta, gamma]
        xyz = np.asarray(xyz_lin).tolist() + np.asarray(xyz_ang).tolist()
        state["x"] = np.array(xyz)
        return state

    def readState(self, conn):
        while True:
            states = self.listen2robot(conn)
            if states is not None:
                break
        return states

    def xdot2qdot(self, xdot, states):
        J_inv = np.linalg.pinv(states["J"])
        return J_inv @ np.asarray(xdot)

    def joint2pose(self, q):
        def RotX(q):
            return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
        def RotZ(q):
            return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        def TransX(q, x, y, z):
            return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
        def TransZ(q, x, y, z):
            return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
        H1 = TransZ(q[0], 0, 0, 0.333)
        H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
        H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
        H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
        H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
        H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
        H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
        H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.2105)
        T = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
        R = T[:,:3][:3]
        xyz = T[:,3][:3]
        return xyz, R

    def go2position(self, conn, goal=False):
        if not goal:
            goal = self.home
        total_time = 20.0
        start_time = time.time()
        states = self.readState(conn)
        dist = np.linalg.norm(states["q"] - goal)
        elapsed_time = time.time() - start_time
        while dist > 0.05 and elapsed_time < total_time:
            qdot = np.clip(goal - states["q"], -0.2, 0.2)
            self.send2robot(conn, qdot, "v")
            states = self.readState(conn)
            dist = np.linalg.norm(states["q"] - goal)
            elapsed_time = time.time() - start_time
    
    def find_pos(self,conn):
        state = self.readState(conn)
        return state['x']
    
    def play_traj(self,conn, data, traj_name, total_time):

        traj = np.array(pickle.load(open(traj_name, "rb")))
        traj = Trajectory(traj[:, :6], total_time)

        start_t = None
        play_traj = False
        state = self.readState(conn)
        corrections = []
        steptime = 0.1
        scale = 1.0
        mode = "v"
        time_elapsed = 0

        while True:
            state = self.readState(conn)
            if not play_traj:
                # go2home(conn)
                play_traj = True
                start_t = start_time = time.time()

            if time_elapsed >= total_time:
                print('reached', state['x'])
                return data

            if play_traj:
                curr_t = time.time() - start_t
                x_des = traj.get(curr_t)
                x_curr = state['x']

                x_des[3] = wrap_angles(x_des[3])
                x_des[4] = wrap_angles(x_des[4])
                x_des[5] = wrap_angles(x_des[5])
                xdot = 1 * scale * (x_des - x_curr)
                xdot[3] = wrap_angles(xdot[3])
                xdot[4] = wrap_angles(xdot[4])
                xdot[5] = wrap_angles(xdot[5])

                qdot = self.xdot2qdot(xdot, state)

                self.send2robot(conn, qdot, mode)
                time_elapsed = time.time() - start_t

            curr_time = time.time()

            if curr_time - start_time >= steptime:
                corrections.append(state["x"].tolist())
                start_time = curr_time


def wrap_angles(theta):
    if theta < -np.pi:
        theta += 2*np.pi
    elif theta > np.pi:
        theta -= 2*np.pi
    else:
        theta = theta
    return theta

def make_traj(start_pos, des_pos, val):
    traj = []
    for idx in range(val):
        traj.append(start_pos + (1/val) * idx * (des_pos - start_pos))
    traj.append(des_pos)
    pickle.dump(traj, open('traj.pkl', 'wb'))