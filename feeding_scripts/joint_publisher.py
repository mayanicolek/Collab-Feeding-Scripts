#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from polymetis import RobotInterface
import torch
import time

from scipy.spatial.transform import Rotation as R

def publish_joint_states():
    rospy.init_node('polymetis_joint_state_publisher', anonymous=True)
    joint_state_publisher = rospy.Publisher('/joint_states', JointState, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    # Initialize robot interface
    robot = RobotInterface(
        ip_address="192.168.2.121",
    )

    joint_positions = robot.get_joint_positions()
    ee_pose = robot.get_ee_pose()

    print("Current joint positions: ", joint_positions)
    print("Current end effector pose: ", ee_pose)

    # desired_ee_position = torch.Tensor([0.3748, -0.1293,  0.4373])
    # desired_rotation = R.from_quat([[1, 0, 0, 0]]).as_matrix() # desired panda hand rotation
    # desired_rotation = desired_rotation @ R.from_quat([[0.000, 0.000, 0.383, 0.924]]).as_matrix() # desired panda_link8 rotation
    # desired_rotation = R.from_matrix(desired_rotation).as_quat()
    # desired_rotation = desired_rotation[0].tolist()
    
    # print("Desired rotation: ", desired_rotation)
    
    # state_log = robot.move_to_ee_pose(
    #     position=desired_ee_position, orientation=desired_rotation, time_to_go=2.0
    # )

    joint_state_msg = JointState()
    joint_state_msg.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']

    while not rospy.is_shutdown():
        # Get current joint positions
        joint_positions = robot.get_joint_positions()
        joint_positions = joint_positions.tolist()
        joint_positions.append(0.0) # gripper joint position
        joint_positions.append(0.0) # gripper joint position

        # Populate joint state message
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.position = joint_positions

        # Publish joint state message
        joint_state_publisher.publish(joint_state_msg)

        rate.sleep()

if __name__ == "__main__":
    try:
        publish_joint_states()
    except rospy.ROSInterruptException:
        pass
