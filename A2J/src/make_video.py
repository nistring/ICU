''' A simple visualizing tool for making videos from frames '''
import os
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
import cv2 as cv

# Configuration of the video
width = 640
height = 480
FPS = 15
linewidth = 3
upperBody = np.arange(9)
data_name = 'ICU'
data_type = 'test'
max_depth = 3.0
min_depth = 0.3

# Your video is stored at
fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter(os.path.join('/media/nistring/0f737a3f-358d-45e0-827c-473d6a7d555b/@ICU/datasets/train/res', 'GT_pose_estimation.avi'), fourcc, float(FPS), (width, height))



for i in range(1, 1721):
    frame = cv.imread(os.path.join('/media/nistring/0f737a3f-358d-45e0-827c-473d6a7d555b/@ICU/datasets/train/res/vis', f'{i}.jpg'), cv.IMREAD_COLOR)
    video.write(frame)
video.release()