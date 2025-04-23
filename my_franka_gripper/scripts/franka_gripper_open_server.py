#!/usr/bin/env python

import rospy
import numpy as np
import os
import time
import numpy
import geometry_msgs.msg
from franka_gripper.msg import HomingAction, HomingActionGoal, GraspAction, GraspGoal
import actionlib
import time
from std_srvs.srv import Empty, EmptyResponse

# ON ROBOT COMPUTER: python franka_gripper_open_server.py

def handle_franka_gripper_open(req):
    print("Received request to open the Franka gripper.")
    
    # Implement gripper opening logic here
    gripper_client = actionlib.SimpleActionClient('/franka_gripper/homing', HomingAction)
    gripper_client.wait_for_server()
    gripper_goal = HomingActionGoal()
    gripper_client.send_goal(gripper_goal)
    gripper_client.wait_for_result()

    print("Franka gripper is now open.")
    return EmptyResponse()

def franka_gripper_open_server():
    rospy.init_node('franka_gripper_open_server')
    service = rospy.Service('franka_gripper_open', Empty, handle_franka_gripper_open)
    print("Ready to open the Franka gripper.")
    rospy.spin()

if __name__ == "__main__":
    franka_gripper_open_server()
