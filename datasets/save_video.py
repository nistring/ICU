import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import os
import timeit

import sys
import os
import shutil
root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root)
max_distance = 3.0
min_distance = 0.3
state_list = ['Normal', 'Seizure', 'Touching face', 'Sedation']
frame_id_list = []
curr_state = 1

# Post processing filters
spatial_filter = rs.spatial_filter()
# temporal_filter = rs.temporal_filter()

depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

# Filter options
spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.6)
spatial_filter.set_option(rs.option.filter_smooth_delta, 8)
# temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.5)

def create_folder():

    while True:
        name = input('Choose train or test : ')
        if name in ['train', 'test']:
            break
    try:
        directory = os.path.join(root, name)
        if os.path.exists(directory):
            while True:
                overwrite = input('You already have files in the datasets folder\nContinue? [y/n]')
                if overwrite in ('y', 'n'):
                    break
            if overwrite == 'y':
                shutil.rmtree(directory)
            else:
                sys.exit()
        os.makedirs(directory)
        for n in ('depth_data', 'images', 'color_img', 'res', 'labels', 'valid_depth_data'):
            os.makedirs(os.path.join(directory, n))
    except OSError:
        print('Error: Creating directory. ' + directory)
    
    return directory
            

if __name__ == '__main__':

    directory = create_folder()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    try:
        save = False
        idx = 0
        cv.namedWindow('depth_frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('depth_frame', width=640, height=480)
        cv.namedWindow('color_frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('color_frame', width=640, height=480)
        
        while True:

            start_t = timeit.default_timer()

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not depth_frame or not color_frame:
                continue
            
            # Post-processing
            depth_frame = depth_to_disparity.process(depth_frame)
            depth_frame = spatial_filter.process(depth_frame)
            # depth_frame = temporal_filter.process(depth_frame)
            depth_frame = disparity_to_depth.process(depth_frame)
            
            depth_data = np.asanyarray(depth_frame.get_data()) * depth_scale
            depth_data = np.clip(depth_data, min_distance, max_distance).astype(np.float32)
            depth_map = (depth_data * (255 / max_distance)).astype(np.uint8)
            depth_map = np.stack([depth_map, depth_map, depth_map], axis=-1)
            color_map = np.asanyarray(color_frame.get_data())

            if save is True:
                np.save(os.path.join(directory, 'depth_data', str(idx) + '.npy'), depth_data)
                cv.imwrite(os.path.join(directory, 'images', str(idx) + '.jpg'), depth_map)
                cv.imwrite(os.path.join(directory, 'color_img', str(idx) + '.jpg'), color_map)
                cv.circle(color_map, (17, 44), 6, (0,0,255), -1)
                cv.putText(color_map, 'REC', (30, 50), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0,0,255), thickness=2)
                frame_id_list.append(curr_state)
                idx += 1

            k = cv.waitKey(1)
            # Press q to quit
            if k == ord('q'):
                cv.destroyAllWindows()
                break
            # Press r to save the frames
            if k == ord('r'):
                if save is False:
                    save = True
                else:
                    save = False
            
            for i in range(1, len(state_list)+1):
                if k == ord(str(i)):
                    curr_state = i
                if curr_state == i:
                    color = (0,255,0)
                else:
                    color = (0,0,255)
                cv.putText(color_map, str(i) + '. ' + str(state_list[i-1]), (480, i*20), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=2)

            terminate_t = timeit.default_timer()
            FPS = int(1./(terminate_t - start_t))
            cv.putText(color_map, 'FPS : ' + str(FPS), (15, 25), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 0, 0), thickness=2)
            cv.putText(color_map, f'Index : {idx}', (15, 68), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 0, 0), thickness=2)
            cv.imshow('depth_frame', depth_map)
            cv.imshow('color_frame', color_map)
            

    finally:
        np.save(os.path.join(directory, 'frame_id_list.npy'), np.array(frame_id_list))
        pipeline.stop()