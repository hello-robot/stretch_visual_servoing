import zmq
import numpy as np
import cv2
import sys
import argparse
import time
from ultralytics import YOLO
import aruco_detector as ad
import aruco_to_fingertips as af
import yaml
from yaml.loader import SafeLoader
import d405_helpers_without_pyrealsense as dh
from copy import deepcopy
import loop_timer as lt
import yolo_servo_perception as yp
import yolo_networking as yn
import argparse



def main(use_remote_computer):
    print('cv2.__version__ =', cv2.__version__)
    print('cv2.__path__ =', cv2.__path__)
    print('sys.version =', sys.version)
    
    yolo_context = zmq.Context()
    yolo_socket = yolo_context.socket(zmq.PUB)
    if use_remote_computer:
        yolo_address = 'tcp://*:' + str(yn.yolo_port)
    else:
        yolo_address = 'tcp://' + '127.0.0.1' + ':' + str(yn.yolo_port)
    yolo_socket.setsockopt(zmq.SNDHWM, 1)
    yolo_socket.setsockopt(zmq.RCVHWM, 1)
    yolo_socket.bind(yolo_address)
    
    d405_context = zmq.Context()
    d405_socket = d405_context.socket(zmq.SUB)
    d405_socket.setsockopt(zmq.SUBSCRIBE, b'')
    d405_socket.setsockopt(zmq.SNDHWM, 1)
    d405_socket.setsockopt(zmq.RCVHWM, 1)
    d405_socket.setsockopt(zmq.CONFLATE, 1)
    if use_remote_computer:
        d405_address = 'tcp://' + yn.robot_ip + ':' + str(yn.d405_port)
        model_name = yn.yolo_model_on_remote_computer
    else:
        d405_address = 'tcp://' + '127.0.0.1' + ':' + str(yn.d405_port)
        model_name = yn.yolo_model_on_robot
        
    d405_socket.connect(d405_address)

    yolo_servo_perception = yp.YoloServoPerception(model_name=model_name)

    loop_timer = lt.LoopTimer()
    
    try:
        start_time = time.time()
        iterations = 0
        first_frame = True
        
        while True:

            loop_timer.start_of_iteration()

            d405_output = d405_socket.recv_pyobj()
            color_image = d405_output['color_image']
            depth_image = d405_output['depth_image']
            depth_camera_info = d405_output['depth_camera_info']
            color_camera_info = d405_output['color_camera_info']
            depth_scale = d405_output['depth_scale']

            # It looks like the depth_camera_info is more
            # accurate. Online documentation indicates that the D405
            # can be treated as a single depth camera looking out of
            # the left camera and that the RGB and depth images are
            # already aligned.
            
            #print('depth_camera_info =', depth_camera_info)
            #print('color_camera_info =', color_camera_info)

            if first_frame: 
                yolo_servo_perception.set_camera_parameters(depth_camera_info, depth_scale)
                first_frame = False
            
            send_dict = yolo_servo_perception.apply(color_image, depth_image)
            yolo_socket.send_pyobj(send_dict)
            
            cv2.waitKey(1)

            loop_timer.end_of_iteration()
            loop_timer.pretty_print()

    
    finally:
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Receive and Process D405 Images with YOLOv8',
        description='Receives and processes D405 images for visual servoing with the Stretch mobile manipulator from Hello Robot.'
    )
    parser.add_argument('-r', '--remote', action='store_true', help = 'Use this argument when running the code on a remote computer. By default, the code assumes that it is running on a Stretch robot. Prior to using this option, configure the network with the file yolo_networking.py on the robot and the remote computer.') 

    
    args = parser.parse_args()
    use_remote_computer =args.remote
    main(use_remote_computer)
    

    
