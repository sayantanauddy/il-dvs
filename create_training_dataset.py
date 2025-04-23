import numpy as np
import os
from panda_vision_imitation.logging import read_dict
import cv2
from pyquaternion import Quaternion
from scipy import signal
from argparse import ArgumentParser

from panda_vision_imitation.logging import read_dict
from panda_vision_imitation.tracker import Tracker
from panda_vision_imitation.geometry import Point, Box

def parse_args():

    parser = ArgumentParser('Create training dataset')
    parser.add_argument('--data_root', type=str, required=True, help='Path to robot data recording folder')
    parser.add_argument('--img_width', type=int, default=640, help='Image width')
    parser.add_argument('--img_height', type=int, default=480, help='Image height')
    # Some times even with quaternion correction, different demos have different quaternion
    # trajectories even though the movement is similar. Here we decide whether to flip all
    # quaternions for some demos, so that all demos have similar quaternion trajectories
    parser.add_argument('--flipquat', type=int, default=1, help='Flip quaternions')
    parser.add_argument('--data_filename', type=str, default='network_data.txt', help='Name of the training dataset file')
    args = parser.parse_args()
    return args

def rectify_quaternions_single_demo(orientation_quat):

    orientation_quat_cp = orientation_quat.copy()

    # Apply the quaternion correction
    num_steps = orientation_quat_cp.shape[0]
    
    for t in range(num_steps-1):
        qt = orientation_quat_cp[t]
        qtplus1 = orientation_quat_cp[t+1]
        checkval = np.dot(qt, qtplus1.T)
        if checkval <= 0.0:
            qtplus1 *= -1.0

        orientation_quat_cp[t] = qt
        orientation_quat_cp[t+1] = qtplus1

    return orientation_quat_cp

def butter_lowpass(cutoff, nyq_freq, order=4):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform 
    
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform 
    
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def process_input_data(args):

    # Set the paths to the required files/dirs
    raw_image_dir = os.path.join(args.data_root, 'images')
    tracked_image_dir = os.path.join(args.data_root, 'tracked_images')

    # Fetch the recorded YOLO data
    yolo_results = read_dict(os.path.join(args.data_root, 'yolo_results.json'))

    # Create a tracker
    tracker = Tracker(store_hist=True)

    # Run the tracker over the YOLO results
    # to order the bbox points and fill in missing frames
    for yolo in yolo_results:
        object = yolo['object']
        bbox = None
        # We assume that only one object can be detected
        if object == 'None':
            # If no object is detected
            bbox = None
        else:
            # If an object is detected
            bbox = Box.init_from_minmax(pmin=Point(x=yolo['xmin'], y=yolo['ymin']),
                                        pmax=Point(x=yolo['xmax'], y=yolo['ymax']))
        # Track the bounding box
        prev_bbox, curr_bbox = tracker.track(bbox=bbox)

    # Create a dir for images with tracked bboxes
    if not os.path.exists(tracked_image_dir):
        os.makedirs(tracked_image_dir)

    # Numpy array to hold the training data that will be computed
    num_data_rows = len(yolo_results)
    # bbox coords: 8
    num_data_cols = 8
    network_input = np.zeros((num_data_rows, num_data_cols))

    # Iterate over the YOLO frames again
    # If needed, we can make use of tracker history
    for idx, yolo in enumerate(yolo_results):

        image_name = yolo['image_name']

        tracked_info = tracker.hist[idx]
        assert tracked_info['frame_id'] == idx
        tracked_bbox = tracked_info['curr_bbox']

        # Create an image with the tracked bounding box
        img = cv2.imread(os.path.join(raw_image_dir, image_name))
        tracked_bbox.drawbox(img, lw=5)
        cv2.imwrite(os.path.join(tracked_image_dir, image_name), img)

        # Tracked bounding box
        tracked_bbox_coords = tracked_bbox.get_all_xy()
        for coord in tracked_bbox_coords:
            # Check if all coordinates are valid
            assert coord is not None and coord >= 0, f'frame: {idx}, coord: {tracked_bbox_coords}'
                    
        x0,y0,x1,y1,x2,y2,x3,y3 = tracked_bbox_coords

        # Covert pixels to the range 0.0 to 1.0
        x0 /= args.img_width
        x1 /= args.img_width
        x2 /= args.img_width
        x3 /= args.img_width

        y0 /= args.img_height
        y1 /= args.img_height
        y2 /= args.img_height
        y3 /= args.img_height

        # Record the training data for the current frame
        network_input[idx] = [x0, y0, x1, y1, x2, y2, x3, y3]     # Bounding box coordinates

    return network_input


def process_output_data(args):

    # Fetch the recorded robot data
    joint_ee_info_path = os.path.join(args.data_root, 'joint_ee_data.txt')

    # Load ground truth output data
    op_data = np.loadtxt(joint_ee_info_path)

    # Fetch position data
    pos_noisy = op_data[:, 0:3]
    pos = pos_noisy.copy()

    # Apply low-pass filter to position data
    pos[:,0] = butter_lowpass_filter(pos[:,0], 0.4, 20.0/2, order=4)
    pos[:,1] = butter_lowpass_filter(pos[:,1], 0.4, 20.0/2, order=4)
    pos[:,2] = butter_lowpass_filter(pos[:,2], 0.4, 20.0/2, order=4)

    # Compute velocities from positions
    # Convert vels from m/s to cm/s (*100.0)
    vel = np.diff(pos, n=1, axis=0) * 100.0
    vel_noisy = np.diff(pos_noisy, n=1, axis=0) * 100.0

    # Fetch quaternions
    quat_raw = op_data[:, 3:7]
    quat = quat_raw.copy()

    # Apply quaternion correction
    quat = rectify_quaternions_single_demo(quat)
    
    if args.flipquat==1:
        print('flipping all quaternions after correction')
        quat *= -1.0

    # Verify that quat and quat_raw represent the same rotation
    for qq, qq_raw in zip(quat, quat_raw):
        qq = Quaternion(qq)
        qq_raw = Quaternion(qq_raw)
        assert np.allclose(qq.rotation_matrix, qq_raw.rotation_matrix)

    # If pos and quat have N steps, vel has N-1 steps
    # Discard the first element of pos and quat
    pos = pos[1:,:]
    pos_noisy = pos_noisy[1:,:]

    quat = quat[1:,:]
    quat_raw = quat_raw[1:,:]

    # Noisy output
    output_noisy = np.hstack([vel_noisy, quat_raw])
    # Processed output
    output = np.hstack([vel, quat])

    return output, output_noisy

if __name__ == '__main__':
    args = parse_args()

    # Process inputs (bounding boxes)
    network_input = process_input_data(args)

    # Remove the first element of input (to make len same as vel traj)
    network_input = network_input[1:,:]

    # Process outputs (EE velocities and quaternions)
    output, output_noisy = process_output_data(args)

    assert network_input.shape[0] == output.shape[0] == output_noisy.shape[0]

    # Combine inputs and outputs into a single file on disk
    save_path = os.path.join(args.data_root, args.data_filename)
    np.savetxt(save_path, 
               np.hstack([network_input, output]))

    save_path = os.path.join(args.data_root, f'noisy_{args.data_filename}')
    np.savetxt(save_path, 
               np.hstack([network_input, output_noisy]))
