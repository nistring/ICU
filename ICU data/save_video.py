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
angle_list = ['15', '30', '45']
action_list = ['normal', 'tube', 'escape', 'tossturn', 'seizure']
state = 0
#tossturn900-1300
# Post processing filters
spatial_filter = rs.spatial_filter()
temporal_filter = rs.temporal_filter()

depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

# Filter options
spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.6)
spatial_filter.set_option(rs.option.filter_smooth_delta, 8)
temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.5)

def create_folder():

    name = input('Name of the participant : ')
    directory = os.path.join(root, name)
    try:
        os.makedirs(directory)
    except:
        pass
    
    print(angle_list)
    while True:
        angle = input('Choose one of the angles : ')
        if angle in angle_list:
            break
    directory = os.path.join(directory, angle)
    try:
        os.makedirs(directory)
    except:
        pass

    for action in action_list:
        action_dir = os.path.join(directory, action)
        try:
            if os.path.exists(action_dir):
                while True:
                    overwrite = input(f'You already have files in folder\nContinue? [y/n]')
                    if overwrite in ('y', 'n'):
                        break
                if overwrite == 'y':
                    shutil.rmtree(directory)
                else:
                    sys.exit()
            for n in ('depth_data', 'depth_img', 'color_img', 'res', 'labels'):
                os.makedirs(os.path.join(action_dir, n))
        except OSError:
            print('Error: Creating directory. ' + directory)
    
    return name, action, directory
            

if __name__ == '__main__':

    name, action, angle_dir = create_folder()
    directory = os.path.join(angle_dir, action_list[state])

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
            depth_frame = temporal_filter.process(depth_frame)
            depth_frame = disparity_to_depth.process(depth_frame)
            
            depth_data = np.asanyarray(depth_frame.get_data()) * depth_scale
            depth_data = np.clip(depth_data, min_distance, max_distance).astype(np.float32)
            depth_map = ((depth_data-min_distance) * (255 / (max_distance-min_distance)))
            depth_map = np.clip(depth_map, 0, 255).astype(np.uint8)
            depth_map = np.stack([depth_map, depth_map, depth_map], axis=-1)
            color_map = np.asanyarray(color_frame.get_data())

            if save is True:
                np.save(os.path.join(directory, 'depth_data', str(idx) + '.npy'), depth_data)
                cv.imwrite(os.path.join(directory, 'depth_img', str(idx) + '.jpg'), depth_map)
                cv.imwrite(os.path.join(directory, 'color_img', str(idx) + '.jpg'), color_map)
                cv.circle(color_map, (17, 69), 6, (0,0,255), -1)
                cv.putText(color_map, 'REC', (30, 75), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0,0,255), thickness=2)
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
            if k in list(map(ord, ('1','2','3','4','5'))):
                state = k - ord('1')
                directory = os.path.join(angle_dir, action_list[state])
            
            cv.putText(color_map, f'Name : {name}', (480, 25), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,128,255), thickness=2)
            cv.putText(color_map, f'Action : {action_list[state]}', (480, 50), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,128,255), thickness=2)

            terminate_t = timeit.default_timer()
            FPS = int(1./(terminate_t - start_t))
            cv.putText(color_map, 'FPS : ' + str(FPS), (15, 25), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0,255,128), thickness=2)
            cv.putText(color_map, f'Index : {idx}', (15, 50), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0,255,128), thickness=2)
            cv.imshow('depth_frame', depth_map)
            cv.imshow('color_frame', color_map)
            

    finally:
        pipeline.stop()