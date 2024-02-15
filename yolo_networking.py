
# Set these values for your network
robot_ip = 'stretch-se3-####'
remote_computer_ip = 'MACHINE_NAME_HERE'

# Set these to your preferred port numbers
# hello d405 => 4ello d405 => 4405
d405_port = 4405
# hello YOLO => 4ello Y010 => 4010
yolo_port = 4010

# Specify the models to run. Larger models run more slowly. You should
# target a rate of 10 Hz or higher. Examples of rates can be found in
# the comments below.
yolo_model_on_robot = 'yolov8n-seg.pt'
yolo_model_on_remote_computer = 'yolov8x-seg.pt'


# -----------------------------------------
# Detection Model Names and Performance
# 15Hz is the maximum achievable fps due to the D405’s frame rate

# 'yolov8n.pt'
# 15 Hz with 640x480 images on Stretch 3's NUC 12

# 'yolov8s.pt'
# 10 Hz with 640x480 images on Stretch 3's NUC 12

# 'yolov8m.pt'
# 4.5 Hz with 640x480 images on Stretch 3's NUC 12

# 'yolov8l.pt'

# 'yolov8x.pt'
# 15 Hz with 640x480 images on desktop with NVIDIA 4090 RTX GPU

# -----------------------------------------
# Segmentation Model Names and Performance
# 15Hz is the maximum achievable fps due to the D405’s frame rate

# 'yolov8n-seg.pt'
# 15 Hz with 640x480 images on Stretch 3's NUC 12

# 'yolov8s-seg.pt'
# 6.5 Hz with 640x480 images on Stretch 3's NUC 12

# 'yolov8m-seg.pt'

# 'yolov8l-seg.pt'

# 'yolov8x-seg.pt'
# 15 Hz with 640x480 images on desktop with NVIDIA 4090 RTX GPU
# -----------------------------------------
