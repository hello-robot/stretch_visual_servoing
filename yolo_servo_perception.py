import numpy as np
import cv2
from ultralytics import YOLO
import aruco_detector as ad
import aruco_to_fingertips as af
import yaml
from yaml.loader import SafeLoader
import d405_helpers_without_pyrealsense as dh
from copy import deepcopy

def draw_text(image, origin, text_lines):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    location = origin + np.array([0, -55])
    location = location.astype(np.int32)
        
    for i, line in enumerate(text_lines):
        text_size = cv2.getTextSize(line, font, font_size, 4)
        (text_width, text_height), text_baseline = text_size
        center = int(text_width / 2)
        offset = np.array([-center, i * (1.7*text_height)]).astype(np.int32)
        cv2.putText(image, line, location + offset, font, font_size, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, location + offset, font, font_size, (255, 255, 255), 1, cv2.LINE_AA)


class YoloServoPerception():
    def __init__(self, model_name=None):
        self.camera_info = None
        self.depth_scale = None
        if model_name is not None:
            self.model_name = model_name
        else: 
            self.model_name = 'yolov8n-seg.pt'

        print('YoloServoPerception: self.model_name =', self.model_name)
            
        self.masks_available = False
        if 'seg' in self.model_name:
            self.masks_available = True

        self.model = YOLO(self.model_name)
        self.example_tennis_ball_diameter = 0.0658
        
        self.marker_info = {}
        with open('aruco_marker_info.yaml') as f:
            self.marker_info = yaml.load(f, Loader=SafeLoader)
        self.aruco_detector = ad.ArucoDetector(marker_info=self.marker_info, show_debug_images=False, use_apriltag_refinement=False, brighten_images=False)
        self.fingertip_part = 'cup_top' #'cup_bottom'
        self.aruco_to_fingertips = af.ArucoToFingertips(default_height_above_mounting_surface=af.suctioncup_height[self.fingertip_part])


    def set_camera_parameters(self, camera_info, depth_scale):
        self.camera_info = camera_info
        self.depth_scale = depth_scale
        
    def apply(self, color_image, depth_image): 
       
        assert ((self.camera_info is not None) and (self.depth_scale is not None)), 'ERROR: YoloServoPerception: set_camera_parameters must be called prior to apply. self.camera_info or self.depth_scale is None'
        
        self.aruco_detector.update(color_image, self.camera_info)
        markers = self.aruco_detector.get_detected_marker_dict()
        fingertips = self.aruco_to_fingertips.get_fingertips(markers)

        display_received_images = False
        if display_received_images: 
            cv2.imshow('Received RGB Image', color_image)
            cv2.imshow('Received Depth Image', depth_image)

        conf = 0.1 #0.25 is the default threshold
        yolo_results = self.model.predict(color_image, conf=conf)[0]
        
        names = yolo_results.names

        yolo_output = []
        best_ball = None
        boxes = yolo_results.boxes.cpu().numpy()
        if len(boxes) > 0:
            masks = None
            if self.masks_available: 
                masks = yolo_results.masks.xy
            for i, box in enumerate(boxes):
                class_name = names[box.cls[0]]
                if class_name in ['apple', 'sports ball']:

                    box_min_x, box_min_y, box_max_x, box_max_y = box.xyxy[0]
                    box_width_x = box_max_x - box_min_x
                    box_width_y = box_max_y - box_min_y
                    max_box_side_pix = max(box_width_x, box_width_y)

                    if masks is not None: 
                        # Find the boundaries of the mask
                        int_mask = masks[i].astype(np.int32)
                        min_mask = np.min(int_mask, axis=0)
                        max_mask = np.max(int_mask, axis=0)
                        mask_min_x, mask_min_y = min_mask
                        mask_max_x, mask_max_y = max_mask
                        mask_width_x = mask_max_x - mask_min_x
                        mask_width_y = mask_max_y - mask_min_y

                        display_crop = False
                        crop_rgb = False
                        if crop_rgb:
                            ball_crop = color_image[mask_min_y:mask_max_y, mask_min_x:mask_max_x, :]
                            if display_crop:
                                cv2.imshow('Ball RGB Crop', ball_crop)

                        # Crop the depth image around the ball
                        ball_depth_crop = depth_image[mask_min_y:mask_max_y, mask_min_x:mask_max_x]
                        if display_crop:
                            cv2.imshow('Ball Depth Crop', ball_depth_crop)

                        # Create segmentation mask for cropped region
                        crop_polygon = masks[i] - [mask_min_x, mask_min_y]
                        mask_crop = np.zeros_like(ball_depth_crop, np.uint8)
                        cv2.fillPoly(mask_crop, [crop_polygon.astype(np.int32)], 255)
                        if display_crop:
                            cv2.imshow('Ball Mask', mask_crop)


                        # Find the estimated depth across the mask
                        estimated_depth = np.percentile(ball_depth_crop[mask_crop > 0], 50)
                        estimated_z_m = estimated_depth * self.depth_scale
                    else: 
                        # Use the bounding box to estimate the
                        # range instead of a segmentation
                        # mask.
                        box_min_x_int, box_min_y_int, box_max_x_int, box_max_y_int = box.xyxy[0].astype(np.int32)

                        display_crop = False
                        crop_rgb = False
                        if crop_rgb:
                            ball_crop = color_image[box_min_y_int:box_max_y_int, box_min_x_int:box_max_x_int, :]
                            if display_crop:
                                cv2.imshow('Ball RGB Crop', ball_crop)

                        # Crop the depth image around the ball
                        ball_depth_crop = depth_image[box_min_y_int:box_max_y_int, box_min_x_int:box_max_x_int]
                        if display_crop:
                            cv2.imshow('Ball Depth Crop', ball_depth_crop)

                        # Find the estimated depth across the mask
                        estimated_depth = np.percentile(ball_depth_crop, 50)
                        estimated_z_m = estimated_depth * self.depth_scale

                    # Compute the 3D grasp point
                    center_pix = np.array([(box_max_x + box_min_x)/2.0, (box_max_y + box_min_y)/2.0])
                    left_side_pix = np.array([box_min_x, (box_max_y + box_min_y)/2.0])
                    right_side_pix = np.array([box_max_x, (box_max_y + box_min_y)/2.0])
                    width_pix = box_max_x - box_min_x

                    center_xyz = dh.pixel_to_3d(center_pix, estimated_z_m , self.camera_info)
                    center_ray = center_xyz / np.linalg.norm(center_xyz)
                    left_side_xyz = dh.pixel_to_3d(left_side_pix, estimated_z_m, self.camera_info)
                    right_side_xyz = dh.pixel_to_3d(right_side_pix, estimated_z_m, self.camera_info)
                    width_m = np.linalg.norm(right_side_xyz - left_side_xyz)

                    grasp_depth = width_m / 2.0
                    grasp_center_xyz = center_xyz + (grasp_depth * center_ray)

                    if (best_ball is None) or (best_ball['max_box_side_pix'] < max_box_side_pix): 
                        best_ball = {
                            'name': class_name,
                            'max_box_side_pix' : max_box_side_pix,
                            'confidence': box.conf[0],
                            'width_m': width_m,
                            'width_pix': width_pix,
                            'estimated_z_m': estimated_z_m,
                            'grasp_center_xyz': grasp_center_xyz,
                            'left_side_xyz': left_side_xyz,
                            'left_side_pix': left_side_pix,
                            'right_side_xyz': right_side_xyz,
                            'right_side_pix': right_side_pix,
                            'box': {'min_x': box_min_x,
                                    'min_y': box_min_y,
                                    'max_x': box_max_x,
                                    'max_y': box_max_y}
                        }
                        if masks is not None:
                            best_ball['mask'] = masks[i]


        display_yolo_results_image = False
        if display_yolo_results_image:
            results_image = yolo_results.plot()
            cv2.imshow('YOLOv8 result', results_image)

        if best_ball is not None:
            # minimize the object before sending
            minimal_ball = deepcopy(best_ball)
            # Pixel information is not currently useful for
            # visual servoing using remote visual processing,
            # since the local visual servoing process does not
            # have access to the D405.
            del minimal_ball['max_box_side_pix']
            if self.masks_available: 
                del minimal_ball['mask']
            del minimal_ball['width_pix']
            del minimal_ball['left_side_pix']
            del minimal_ball['right_side_pix']
            del minimal_ball['box']

            yolo_output.append(minimal_ball)

        send_dict = {
            'fingertips': fingertips,
            'yolo': yolo_output
        } 

        display_task_relevant_image = True
        if display_task_relevant_image:
            task_relevant_image = np.copy(color_image)
            self.aruco_to_fingertips.draw_fingertip_frames(send_dict['fingertips'],
                                                           task_relevant_image,
                                                           self.camera_info,
                                                           axis_length_in_m=0.02,
                                                           draw_origins=True,
                                                           write_coordinates=True)
            ball = best_ball
            if ball is not None:
                if self.masks_available: 
                    mask = ball['mask']
                    brighten_mask = np.ones(task_relevant_image.shape[:2], np.float32)
                    #cv2.polylines(task_relevant_image, [mask.astype(np.int32)], True, (0,0,255), 3, lineType=cv2.LINE_AA)
                    cv2.polylines(brighten_mask, [mask.astype(np.int32)], True, 4.0, 3, lineType=cv2.LINE_AA)
                    cv2.fillPoly(brighten_mask, [mask.astype(np.int32)], 2.0, lineType=cv2.LINE_AA)
                    task_relevant_image[:,:,0] = np.minimum(brighten_mask * task_relevant_image[:,:,0], 255).astype(np.uint8)
                    task_relevant_image[:,:,1] = np.minimum(brighten_mask * task_relevant_image[:,:,1], 255).astype(np.uint8)
                    task_relevant_image[:,:,2] = np.minimum(brighten_mask * task_relevant_image[:,:,2], 255).astype(np.uint8)
                    
                    

                display_bounding_box = False
                if display_bounding_box: 
                    box = ball['box']
                    x0 = box['min_x']
                    x1 = box['max_x']
                    y0 = box['min_y']
                    y1 = box['max_y']
                    lines = np.array([[x0,y0], [x1,y0], [x1,y1], [x0,y1]]).astype(np.int32)
                    cv2.polylines(task_relevant_image, [lines], True, (255,0,0), 2)
                    middle = np.round(np.array([(x1+x0)/2.0, (y1+y0)/2.0])).astype(np.int32)

                display_sides = False
                if display_sides:
                    left_side_color = (0,255,0)
                    right_side_color = (255, 0, 0)
                    side_thickness = 2
                    side_length = ball['width_pix']

                    left_side_pix = ball['left_side_pix']
                    left_side_start = np.copy(left_side_pix)
                    left_side_start[1] = left_side_start[1] + side_length/2.0
                    left_side_end = np.copy(left_side_pix)
                    left_side_end[1] = left_side_end[1] - side_length/2.0
                    cv2.line(task_relevant_image,
                             left_side_start.astype(np.int32),
                             left_side_end.astype(np.int32),
                             left_side_color, side_thickness)

                    right_side_pix = ball['right_side_pix']
                    right_side_start = np.copy(right_side_pix)
                    right_side_start[1] = right_side_start[1] + side_length/2.0
                    right_side_end = np.copy(right_side_pix)
                    right_side_end[1] = right_side_end[1] - side_length/2.0
                    cv2.line(task_relevant_image,
                             right_side_start.astype(np.int32),
                             right_side_end.astype(np.int32),
                             right_side_color, side_thickness)

                grasp_center_xy = dh.pixel_from_3d(ball['grasp_center_xyz'], self.camera_info)
                grasp_point = grasp_center_xy.astype(np.int32)
                radius = 6
                cv2.circle(task_relevant_image, grasp_point, radius, (255, 0, 0), -1, lineType=cv2.LINE_AA)

                x,y,z = ball['grasp_center_xyz'] * 100.0
                text_lines = [
                    "{:.1f} cm wide".format(ball['width_m']*100.0),
                    "{:.1f}, {:.1f}, {:.1f} cm".format(x,y,z)
                    ]
                draw_text(task_relevant_image, grasp_point, text_lines)

            cv2.imshow('Task Relevant Results', task_relevant_image)
        
        return send_dict
        

        
