# IL-DVS

This is the official implementation of the paper "Imitation Learning-based Direct Visual Servoing using the Large Projection Formulation" ([paper link](https://www.sciencedirect.com/science/article/pii/S0921889025000570)). This paper introduces a dynamical system-based imitation learning for direct visual servoing. It leverages off-the-shelf deep learning-based perception modules to extract robust features from the raw input images, and an imitation learning strategy to execute sophisticated robot motions. The learning blocks are integrated using the large projection task priority formulation. As demonstrated through extensive experimental analysis, the proposed method realizes complex tasks with a robotic manipulator.

<p align="center">
  <img src="assets/video_v4_gif_opt.gif" width="620" alt="animated_overview" />
</p>

## Video

A video demonstrating our experiments on [YouTube](https://youtu.be/b0lviYlXarI):

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/b0lviYlXarI/0.jpg)](https://www.youtube.com/watch?v=b0lviYlXarI)


## Experiments
### Dependencies

We recommend working with Python virtual environments. Python dependencies can be installed using:
```bash
# Clone this repository
git clone git@github.com:sayantanauddy/il-dvs.git  ## Using SSH
cd il-dvs

# Create and activate a virtual env
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python3 -m pip install -r requirements.txt 
## Check https://pytorch.org/get-started/previous-versions/ if there is an issue with PyTorch
```

### ROS Setup

In our setup, we use 2 separate computers connected to same network. We use a Franka Emika Panda robot with a wrist-mounted RealSense camera. The ROS nodes for the robot run on the computer connected to the robot. Due to the [requirement of having a real-time kernel](https://frankaemika.github.io/docs/installation_linux.html) on the robot's computer, we run the ROS nodes that may require a GPU (e.g. YOLO) on a separate computer. We assume that `roscore` runs on the robot's computer. `ROS_MASTER_URI` should be correctly configured on the other computers ([see here](https://wiki.ros.org/ROS/Tutorials/MultipleMachines)).   

This code has been created and tested with [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu) and it is assumed that it has already been installed on both computers. We also require the following ROS packages:
 - [franka_example_controllers](https://wiki.ros.org/franka_example_controllers): This should be installed on the computer connected to the robot.
 - [realsense2_camera](https://wiki.ros.org/realsense2_camera): It's easier to have this on the robot's computer.
 - [darknet_ros](http://wiki.ros.org/darknet_ros): This needs a GPU and needs to be on a separate computer (i.e. not the one controlling the robot).
 - [aruco_ros](https://wiki.ros.org/aruco_ros): Can be on either computer (only needed for camera calibration).

```bash
##### Install ROS #####

# Install ROS Noetic on all machines and set set ROS_MASTER_URI

# Remedy for CA certificate issue while running rosdep init
# https://gitlab.com/voxl-public/system-image-build/poky/-/issues/3
sudo apt update
sudo apt install ca-certificates
update-ca-certificates
cp /etc/ssl/certs/ca-certificates.crt /usr/lib/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/usr/lib/ssl/certs/ca-certificates.crt

# Update rosdep
rosdep update

# Install ROS dependencies
rosdep install --from-paths /path/to/catkin_ws/src --ignore-src -r -y

# Source ROS environment
source /opt/ros/noetic/setup.bash

# Install the ROS packages needed
python3 -m pip install rospy rospkg


##### Setup for franka_example_controllers #####
# Follow the steps here: https://frankaemika.github.io/docs/installation_linux.html


##### Setup for realsense2_camera #####
# Instructions can be found here: https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy


##### Setup for darknet_ros: #####
# Check https://github.com/leggedrobotics/darknet_ros
# Install Boost and OpenCV
sudo apt update
sudo apt install libboost-all-dev
sudo apt install libopencv-dev python3-opencv

cd /path/to/catkin_ws/src
git clone --recursive git@github.com:leggedrobotics/darknet_ros.git
source /opt/ros/noetic/setup.bash
cd ..
catkin_make -DCMAKE_BUILD_TYPE=Release
source /opt/ros/noetic/setup.bash

# Download YOLO weights
cd src/darknet_ros/darknet_ros/yolo_network_config/weights/
wget http://pjreddie.com/media/files/yolov3.weights --no-check-certificate
wget http://pjreddie.com/media/files/yolov2-tiny.weights --no-check-certificate

##### Setup for aruco_ros #####
# Check https://wiki.ros.org/aruco_ros
```

### Data Collection
To collect a demonstration, follow the approach described in our paper.

```bash
# On the robot's computer (each command in a separate terminal):
# Safety should be pressed down (robot LED=white)
roscore
roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=INSERT_ROBOT_IP load_gripper:=true

# On camera computer (each command in a separate terminal): 
roslaunch realsense2_camera rs_camera.launch color_width:=640 color_height:=480 color_fps:=30
roslaunch darknet_ros darknet_ros.launch
python ros_scripts/bbox_tracker.py
python ros_scripts/bbox_image_pub.py
rosrun image_view image_view image:=/img_bbox_tracker

python ros_scripts/record_demo_images_bbox.py --freq 40 --n_pts 500 --savedir_root /PATH/TO/SAVE/DEMO --img_width 640 --img_height 480

```

### Preprocessing the Collected Demonstration

```bash
cd il-dvs
source .venv/bin/activate
python create_training_dataset.py --data_root /PATH/TO/SAVE/DEMO  
```

### Processed Datasets
The previous command creates a file `network_data.txt` for a single demonstration. We stack multiple demonstrations to create training and validation datasets than can be used for training a NODE. 

These datasets are available in the `datasets`directory. The `.npy` files contain the data and the `.md`files describe the ordering of the columns.

```bash
cd datasets
tree

# Below is the output
# .
# ├── data_node_cup
# │   ├── train_data.md
# │   ├── train_data.npy
# │   ├── val_data.md
# │   └── val_data.npy
# └── data_node_mouse
#     ├── train_data.md
#     ├── train_data.npy
#     ├── val_data.md
#     └── val_data.npy
```

The training dataset contains 4 demonstrations, each of 500 steps and with 11 recorded items in each step (see the `.md` files for the columns).
```python
import numpy as np
mouse_train = np.load("datasets/data_node_mouse/train_data.npy")
mouse_val = np.load("datasets/data_node_mouse/val_data.npy")
print(f"train shape: {mouse_train.shape}, val shape: {mouse_val.shape}")
# Output
# train shape: (4, 500, 11), val shape: (1, 500, 11)
```


### Training Neural ODEs (NODEs)

To train a NODE on the recorded demonstrations, use the below command in the root of this repository:
```bash
source .venv/bin/activate

# Example for training on the cup dataset
python tr_node.py \
--log_dir logs \
--description node_cup \
--data_root datasets/data_node_cup \
--train_file train_data.npy \
--val_file val_data.npy \
--tsub 30 \
--start_deadzone 0 \
--seed 200 \
--num_iter 20000 \
--lr 0.0005 \
--hidden_dim 256 \
--num_hidden 3 \
--explicit_time 0 \
--rot_scale 100.0
```

### Robot Setup and Callibration
To calibrate the extrinsic camera parameters, run
```bash
# Go to root of repo
# On robot computer (each command in a different terminal)
# launch Franka controller: 
roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=INSERT_ROBOT_IP load_gripper:=true

# On camera computer (each command in a different terminal)
# launch Realsense camera: 
roslaunch realsense2_camera rs_camera.launch

# launch aruco_ros
roslaunch aruco_ros single.launch 

# view aruco: 
rosrun image_view image_view image:=/aruco_single/result (check z does not flip)

# Repeat n times:
## 1. Place arm in different poses (high above table)
## 2. View aruko marker image (make sure z-axis does not flip between observations)
## 3. run this script: 
python ros_scripts/calibration_record.py # (each obs is saved in calibration_data) 

# Run the following to process all observations
python ros_scripts/calibration_process.py 
```


### Robot Experiments

To control the robot using only the NODE:
```bash
# On robot computer (each command in a different terminal, and in order):
roscore
roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=INSERT_ROBOT_IP load_gripper:=true

python franka_gripper_open_server.py # (in ros_ws/src/my_franka_gripper)

# On camera computer (each command in a different terminal, and in order): 
roslaunch realsense2_camera rs_camera.launch color_width:=640 color_height:=480 color_fps:=30
roslaunch darknet_ros darknet_ros.launch
python ros_scripts/bbox_tracker.py --tracked_obj_class cup --prob_thresh 0.4  # CHANGE OBJECT AND THRESHOLD AS NEEDED!
python ros_scripts/bbox_image_pub.py --test_center_goal
rosrun image_view image_view image:=/img_bbox_tracker

# MOUSE: 
python ros_scripts/robot_NODE.py --log_dir results_mouse --max_steps 900 --train_steps 500 --deadzone 0 --train_path datasets/data_node_mouse/train_data.npy --node_path PATH/TO/TRAINED/MOUSE/node.pt --desc pos0

# CUP: 
python ros_scripts/robot_NODE.py --log_dir results_cup_solo_20240118 --max_steps 700 --train_steps 500 --deadzone 0 --train_path datasets/data_node_cup/train_data.npy --node_path PATH/TO/TRAINED/CUP/node.pt --desc pos1 --open_gripper

```

To control the robot using only YOLO:

```bash
# On robot computer (each command in a different terminal, and in order):
roscore
roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=INSERT_ROBOT_IP load_gripper:=true

python franka_gripper_open_server.py

# On camera computer (each command in a different terminal, and in order): 
roslaunch realsense2_camera rs_camera.launch color_width:=640 color_height:=480 color_fps:=30
roslaunch darknet_ros darknet_ros.launch
python ros_scripts/bbox_tracker.py --tracked_obj_class mouse --prob_thresh 0.4 # <<< CHANGE OBJECT AND THRESHOLD AS NEEDED!
python ros_scripts/bbox_image_pub.py --test_center_goal
rosrun image_view image_view image:=/img_bbox_tracker

# MOUSE: 
python ros_scripts/robot_YOLO.py --max_steps 900 --gain_lambda 0.5 --log_dir results_mouse --test_center_goal --deadzone 0 --train_path datasets/data_node_mouse/train_data.npy --rob_init --desc pos0

# CUP: 
python ros_scripts/robot_YOLO.py --max_steps 700 --gain_lambda 0.5 --log_dir results_cup_solo_20240118 --deadzone 0 --train_path datasets/data_node_cup/train_data.npy --rob_init --desc pos1 --open_gripper
```

To control the robot using our proposed IL-DVS approach (i.e. with NODE and YOLO together):

```bash
# On robot computer (each command in a different terminal, and in order):
roscore
roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=INSERT_ROBOT_IP load_gripper:=true

python franka_gripper_open_server.py


# On camera computer (each command in a different terminal, and in order): 
roslaunch realsense2_camera rs_camera.launch color_width:=640 color_height:=480 color_fps:=30
roslaunch darknet_ros darknet_ros.launch
python ros_scripts/bbox_tracker.py --tracked_obj_class mouse --prob_thresh 0.4 # <<< CHANGE OBJECT AND THRESHOLD AS NEEDED! (for cup 1 dataset, thresh=0.6, hist_len=50, without test_goal_center)
python ros_scripts/bbox_image_pub.py --test_center_goal
rosrun image_view image_view image:=/img_bbox_tracker

# MOUSE: 
python ros_scripts/robot_NODE_YOLO.py --log_dir results_mouse_20231219 --description NODE_YOLO --gain 0.5 --max_steps 900 --rob_init --train_steps 500 --deadzone 0 --train_path datasets/data_node_mouse/train_data.npy --node_path PATH/TO/TRAINED/MOUSE/node.pt --desc pos0

# CUP: 
python ros_scripts/robot_NODE_YOLO.py --log_dir results_cup_solo_20240118 --description NODE_YOLO --gain 0.5 --max_steps 700 --rob_init --train_steps 500 --deadzone 0 --train_path datasets/data_node_cup/train_data.npy --node_path PATH/TO/TRAINED/CUP/node.pt --desc pos1 --open_gripper
```

## Citation

If you find our paper or code useful, please cite us using

```bash
@article{auddy_paolillo2025ildvs,
  title={{Imitation learning-based Direct Visual Servoing using the Large Projection Formulation}},
  author={Auddy, Sayantan and Paolillo, Antonio and Piater, Justus and Saveriano, Matteo},
  journal={Robotics and Autonomous Systems},
  volume={190},
  pages={104971},
  year={2025},
  doi = {https://doi.org/10.1016/j.robot.2025.104971},
  url = {https://www.sciencedirect.com/science/article/pii/S0921889025000570},
  publisher={Elsevier}
}
```