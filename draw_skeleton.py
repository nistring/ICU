''' A simple visualizing tool for making videos from frames '''
import os
root = os.path.abspath(os.path.dirname(__file__))

import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
import cv2 as cv

# Configuration of the video
width = 320
height = 240
FPS = 15
linewidth = 3
upperBody = np.arange(9)
data_name = 'ICU'
data_type = 'test'
max_depth = 3.0
min_depth = 0.3
idx = 0

# Your video is stored at
fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter(os.path.join(root, f'res/{data_name}_pose_estimation.avi'), fourcc, float(FPS), (width, height))

# We do not use z coordinate when ploting
pred = np.load(os.path.join(root, f'res/{data_name}_result.npy'))
# pred = world2pixel(pred).astype(np.int16)[:,:,:2]

print('pred: ', pred.shape)

# Draw skeleton
for idx in range(pred.shape[0]):

    # Depthmap at ith frame
    frame = np.load(os.path.join(os.path.dirname(root), 'datasets', data_type, 'valid_depth_data', f'{idx}.npy'))
    frame = (frame * 255 / max_depth).astype(np.uint8)
    frame = cv.applyColorMap(frame, cv.COLORMAP_OCEAN)
    p = pred[idx].astype(np.int16)
    
    cv.line(frame, p[0], (p[1]+p[2])//2, (102,102,255), linewidth)
    cv.line(frame, p[1], p[2], (102,178,255), linewidth)
    cv.line(frame, p[1], p[3], (102,255,255), linewidth)
    # cv.line(frame, p[1], p[7], (102,255,178), linewidth)
    cv.line(frame, p[2], p[4], (178,255,102), linewidth)
    # cv.line(frame, p[2], p[8], (255,178,102), linewidth)
    cv.line(frame, p[3], p[5], (255,102,102), linewidth)
    cv.line(frame, p[4], p[6], (255,102,178), linewidth)
    # cv.line(frame, p[7], p[8], (255,102,255), linewidth)        

    idx += 1
        
    video.write(frame)
video.release()