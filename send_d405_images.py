import zmq
import cv2
import sys
import argparse
import time
import numpy as np
import d405_helpers as dh
import loop_timer as lt
import yolo_networking as yn
import argparse


###########################
# Initial code copied from
# https://answers.opencv.org/question/75510/how-to-make-auto-adjustmentsbrightness-and-contrast-for-image-android-opencv-image-correction/
def autoAdjustments_with_convertScaleAbs(img):
    alow = img.min()
    #ahigh = img.max()
    ahigh = np.percentile(img, 90)
    amax = 255
    amin = 0
    
    # calculate alpha, beta
    alpha = ((amax - amin) / (ahigh - alow))
    beta = amin - alow * alpha
    # perform the operation g(x,y)= α * f(x,y)+ β
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    #return [new_img, alpha, beta]
    return new_img
###########################

    
def main(use_remote_computer, exposure):
    try: 
        print('cv2.__version__ =', cv2.__version__)
        print('cv2.__path__ =', cv2.__path__)
        print('sys.version =', sys.version)
        
        pipeline, profile = dh.start_d405(exposure)
        first_frame = True
        
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.RCVHWM, 1)
        #pub.setsockopt(zmq.SNDBUF, 2*1024)
        #sub.setsockopt(zmq.RCVBUF, 2*1024)

        if use_remote_computer:
            address = 'tcp://*:' + str(yn.d405_port)
        else:
            address = 'tcp://' + '127.0.0.1' + ':' + str(yn.d405_port)
            
        socket.bind(address)

        loop_timer = lt.LoopTimer()

        #adaptive_equalization = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #adaptive_equalization.apply(gray_image)
        
        start_time = time.time()
        iterations = 0
        while True:
            loop_timer.start_of_iteration()

            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if (not depth_frame) or (not color_frame):
                continue

            if first_frame:
                depth_scale = dh.get_depth_scale(profile)
                print('depth_scale = ', depth_scale)
                print()

                depth_camera_info = dh.get_camera_info(depth_frame)
                color_camera_info = dh.get_camera_info(color_frame)
                print_camera_info = True
                if print_camera_info: 
                    for camera_info, name in [(depth_camera_info, 'depth'), (color_camera_info, 'color')]:
                        print(name + ' camera_info:')
                        print(camera_info)
                        print()
                first_frame = False
                del depth_camera_info['distortion_model']
                del color_camera_info['distortion_model']

                d405_output = {'depth_camera_info' : depth_camera_info,
                               'color_camera_info' : color_camera_info,
                               'depth_scale': depth_scale}

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            brighten_image = False
            if brighten_image: 
                color_image = autoAdjustments_with_convertScaleAbs(color_image)

            image = color_image
            camera_info = color_camera_info
            
            d405_output['color_image'] = color_image
            d405_output['depth_image'] = depth_image
                
            socket.send_pyobj(d405_output)

            loop_timer.end_of_iteration()
            loop_timer.pretty_print()
        
    finally:
        pipeline.stop()

        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Send D405 Images',
        description='Send D405 images to local and remote processes.'
    )
    parser.add_argument('-r', '--remote', action='store_true', help = 'Use this argument when allowing a remote computer to receive D405 images. Prior to using this option, configure the network with the file yolo_networking.py on the robot and the remote computer.')
    
    parser.add_argument('-a', '--auto', action='store_true', help = 'Use auto exposure for the D405. This can result in greater motion blur due to longer integration time. The default is for auto exposure to be disabled.')
    parser.add_argument('-l', '--low', action='store_true', help = 'Use a low exposure setting for the D405. This can reduce motion blur.')
    parser.add_argument('-m', '--medium', action='store_true', help = 'Use a medium exposure setting for the D405. This can reduce motion blur compared to auto exposure.')
    parser.add_argument('-e', '--exposure', type=dh.check_exposure,  default=None, help = 'Provide a custom exposure setting for the D405. 330000 is the default setting.')
        
    args = parser.parse_args()

    use_remote_computer = args.remote
    
    exposure_value = args.exposure
    low_exposure = args.low
    medium_exposure = args.medium
    auto_exposure = args.auto

    exposure_command_count = 0

    exposure = 'medium'
    
    if exposure_value is not None:
        exposure_command_count = exposure_command_count + 1
        exposure = exposure_value
    if low_exposure:
        exposure_command_count = exposure_command_count + 1
        exposure = 'low'
    if medium_exposure:
        exposure_command_count = exposure_command_count + 1
        exposure = 'medium'
    if auto_exposure:
        exposure_command_count = exposure_command_count + 1
        exposure = 'auto'
    
    if exposure_command_count > 1: 
        raise argparse.ArgumentTypeError('Only a single exposure argument can be provided, but you provided ' + str(exposure_command_count) + ' exposure arguments.')

    
    main(use_remote_computer, exposure)
    
