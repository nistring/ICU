''' A simple visualizing tool for making videos from frames '''
import os
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
import cv2 as cv
from tqdm import tqdm

# Configuration of the video
width = 640
height = 480
FPS = 15

# Your video is stored at
fourcc = VideoWriter_fourcc('D', 'I', 'V', 'X')
video = VideoWriter('runs/exp59/depth_pose_estimation.mp4', fourcc, float(FPS), (width, height))
img_dir = 'runs/exp59/images'

total_frames = len(os.listdir(img_dir))//20
for i in tqdm(range(total_frames)):
    frame = cv.imread(img_dir + f'/{i}.jpg')
    video.write(frame)
video.release()