import sys
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(root)

import cv2 as cv
from cv2 import VideoWriter, VideoWriter_fourcc
import h5py
import torch
import torch.optim.lr_scheduler as lr_scheduler
import logging
import torch.utils.data
import numpy as np
import model
import anchor
from tqdm import tqdm
from random_erasing import RandomErasing
import time
import random
import pickle
from lib.accuracy import compute_mean_err

# DataHyperParms 
keypointsNumber = 9 # I will only use upper body
inputSize = 256 # 288 * 288
batch_size = 128
learning_rate = 0.00035
weight_decay = 1e-4
width = 640
height = 480
is_3D = False

nepoch = 30
step_size = 10
gamma = 0.1
data_name = 'ICU'

shift_range = 10

randomseed = 12345
random.seed(randomseed)
np.random.seed(randomseed)
torch.manual_seed(randomseed)

data_dir = os.path.join(root, 'datasets')
check_dir = os.path.join(root, 'A2J', 'checkpoint/exp2')
res_dir = os.path.join(root, 'A2J', 'res')

class my_dataloader(torch.utils.data.Dataset):
    
    def __init__(self, data_name, data_type, augmentation=False):
        self.data_name = data_name
        self.data_type = data_type
        self.augmentation = augmentation
        self.random_erasing = RandomErasing()
        with h5py.File(os.path.join(data_dir, self.data_type, f'{self.data_name}_{self.data_type}_labels.h5'), 'r') as f:
            valid_frames = np.where(f.get('is_valid')[:])[0]
            self.bounding_box = f.get('bounding_box')[valid_frames].astype(np.int16).copy()
            if data_type == 'train':
                self.skeleton = f.get('image_coordinates')[valid_frames].copy()


    def __getitem__(self, index):
        self.depth_map = np.load(os.path.join(data_dir, self.data_type, 'valid_depth_data', str(index) + '.npy'))
        self.box = self.bounding_box[index].copy()
        if self.data_type == 'train':
            self.label = self.skeleton[index].copy()
        self.crop()
        if self.augmentation == True:
            self.depth_map = self.random_erasing(self.depth_map)

        if self.data_type == 'train':
            return self.depth_map, self.label, self.depth_mean, self.box
        else:
            return self.depth_map, self.depth_mean, self.box

        # depth_save = self.depth_map + self.depth_mean
        # depth_save[depth_save < 0] = 0
        # depth_save =  cv.applyColorMap((depth_save * 255 / depth_save.max()).astype(np.uint8), colormap=cv.COLORMAP_OCEAN)
        # cv.imwrite('/media/nistring/0f737a3f-358d-45e0-827c-473d6a7d555b/@ICU/A2J/res/test.jpg', depth_save)
        # return self.depth_map, self.label, self.depth_mean, self.box
        ########### data augmentation visualization ##########
        # depth_save = self.depth_map + depth_map_mean
        # depth_save[depth_save < 0] = 0
        # depth_save =  cv.applyColorMap((depth_save * 255 / depth_save.max()).astype(np.uint8), colormap=cv.COLORMAP_OCEAN)
        # p = self.label[:,:2].astype(np.int16)
        # linewidth = 2
        # cv.line(depth_save, p[0], p[1], (178,102,255), linewidth)
        # cv.line(depth_save, p[1], p[2], (153,153,255), linewidth)
        # cv.line(depth_save, p[2], p[4], (102,102,255), linewidth)
        # cv.line(depth_save, p[4], p[6], (51,51,255), linewidth)
        # cv.line(depth_save, p[1], p[3], (255,153,153), linewidth)
        # cv.line(depth_save, p[3], p[5], (255,102,102), linewidth)
        # cv.line(depth_save, p[5], p[7], (255,51,51), linewidth)
        # cv.line(depth_save, p[1], p[8], (230,0,230), linewidth)
        
    
    def __len__(self):
        return self.bounding_box.shape[0]
    
   
    # def rotation(self):
    #     if self.augmentation == True:
    #         rand_angle = np.random.randint(-angle_range, angle_range)
    #         matrix = cv.getRotationMatrix2D((width/2, height/2), rand_angle)
    #         self.depth_map = cv.warpAffine(self.depth_map, matrix, (width, height), borderMode=cv.BORDER_REPLICATE)
    #         self.label[:,:2] = np.matmul(matrix, np.hstack((self.label[:,:2], np.ones((keypointsNumber, 1))))[:,:,np.newaxis]).squeeze(2)

    def crop(self):
        [x_min, y_min, x_max, y_max] = self.box

        if self.augmentation == True:
            x_min = max(x_min + np.random.randint(-shift_range, shift_range), 0)
            y_min = max(y_min + np.random.randint(-shift_range, shift_range), 0)
            x_max = min(x_max + np.random.randint(-shift_range, shift_range), width)
            y_max = min(y_max + np.random.randint(-shift_range, shift_range), height)
            self.box = np.array([x_min, y_min, x_max, y_min])

        self.depth_map = self.depth_map[y_min:y_max, x_min:x_max]
        self.depth_map = cv.resize(self.depth_map, (inputSize, inputSize), interpolation=cv.INTER_CUBIC)
        self.depth_mean = self.depth_map.mean()

        self.depth_map -= self.depth_mean

        if self.data_type == 'train':
            self.label[:,0] = (self.label[:,0] - x_min) * inputSize / (x_max - x_min)
            self.label[:,1] = (self.label[:,1] - y_min) * inputSize / (y_max - y_min)
            if is_3D == True:
                self.label[:,2] -= self.depth_mean


train_image_datasets = my_dataloader(data_name, 'train', True)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size = batch_size, shuffle = True, num_workers = 8)
val_image_datasets = my_dataloader(data_name, 'train', False)
val_dataloaders = torch.utils.data.DataLoader(val_image_datasets, batch_size = batch_size, shuffle = False, num_workers = 8)
test_image_datasets = my_dataloader(data_name, 'test', False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size, shuffle = False, num_workers = 8)

if data_name == 'ITOP':
    with h5py.File(os.path.join(root, 'data/ITOP_side_test_labels.h5'), 'r') as f:
        valid_frames = np.where(f['is_valid'][:])[0]
        gt = f['real_world_coordinates'][valid_frames]
        gt = gt[:,np.arange(keypointsNumber),:]

def train():
   
    net = model.A2J_model(num_classes = keypointsNumber, is_3D=is_3D)
    net = net.cuda()
    
    post_process = anchor.post_process(is_3D=is_3D)
    criterion = anchor.A2J_loss(is_3D=is_3D)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(check_dir, 'train.log'), level=logging.INFO)
    logging.info('======================================================')

    for epoch in range(nepoch):
        net = net.train()
        train_loss_sum = 0.0
        timer = time.time()
    
        # Training loop
        for i, (img, label, mean, box) in enumerate(train_dataloaders):
            
            torch.cuda.synchronize()

            img, label, mean, box = img.cuda(), label.cuda(), mean.cuda(), box.cuda()
            heads  = net(img)
            optimizer.zero_grad()  
            
            loss = criterion(heads, label, box)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            
            train_loss_sum += loss.item()
            # printing loss info
            if i%10 == 0:
                print('epoch: ',epoch, ' step: ', i, ' total loss ',loss.item())

        scheduler.step()

        # time taken
        torch.cuda.synchronize()
        timer = time.time() - timer
        timer = timer / len(train_image_datasets)
        print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

        train_loss_sum /= len(train_image_datasets)
        print('mean train_loss_sum of 1 sample: %f, #train_indexes = %d' %(train_loss_sum, len(train_image_datasets)))

        # Validation
        net = net.eval()
        output = torch.FloatTensor()
        
        Val_loss_sum = 0.0

        for i, (img, label, mean, box) in tqdm(enumerate(val_dataloaders)):
            with torch.no_grad():
                img, label, mean, box = img.cuda(), label.cuda(), mean.cuda(), box.cuda()
                heads = net(img)  
                
                pred_keypoints = post_process(heads, mean, box).data.cpu()
                output = torch.cat([output, pred_keypoints], 0)
                
                loss = criterion(heads, label, box)

                Val_loss_sum += loss.item()
        
        Val_loss_sum /= len(val_image_datasets)

        result = output.numpy()
        
        if is_3D == True:
            result = pixel2world(result)

            mean_err, mean_avg_precision = compute_mean_err(gt.reshape((-1,1,3)), result.reshape((-1,1,3)))
            mean_err = mean_err.item()
            mean_avg_precision = mean_avg_precision.item()
        
            log = 'Epoch#%d: train loss=%.4f, validation loss=%.4f, mean error(m)=%.4f, mean_avg_precision=%.4f, lr = %.6f' \
            %(epoch, train_loss_sum, Val_loss_sum, mean_err, mean_avg_precision, scheduler.get_last_lr()[0])
            print(log)
            logging.info(log)
        else:
            log = 'Epoch#%d: train loss=%.4f, validation loss=%.4f, lr = %.6f' \
            %(epoch, train_loss_sum, Val_loss_sum, scheduler.get_last_lr()[0])
            print(log)
            logging.info(log)

        
        saveNamePrefix = '%s/epoch#%d_lr_%.5f_wetD_%.5f_stepSize_%d_gamma_%d' % (check_dir, epoch, learning_rate, weight_decay, step_size, gamma)
        torch.save(net.state_dict(), saveNamePrefix + '.pth')

def test():

    net = model.A2J_model(num_classes = keypointsNumber, is_3D=is_3D)
    net.load_state_dict(torch.load(os.path.join(check_dir, 'epoch#25_lr_0.00035_wetD_0.00010_stepSize_10_gamma_0.pth')))
    net = net.cuda()
    net.eval()
    
    post_process = anchor.post_process(is_3D=is_3D)

    output = torch.FloatTensor()

    while True:
        for img, mean, box in tqdm(test_dataloaders):
            with torch.no_grad():
                img, mean, box = img.cuda(), mean.cuda(), box.cuda()
                heads = net(img)  
                pred_keypoints = post_process(heads, mean, box).data.cpu()
                output = torch.cat([output, pred_keypoints], 0)

        result = output.numpy()

        for i in range(result.shape[0]-1):
            result[i+1] = (result[i] + result[i+1]) / 2

        np.save(os.path.join(res_dir, f'{data_name}_result.npy'), result)

def pixel2world(pixel_coord, scale=1, width=160.0, height=120.0, fx=0.0035, fy=0.0035):
    world_coord = pixel_coord.copy()
    
    x = world_coord[:,:,0]
    y = world_coord[:,:,1]
    z = world_coord[:,:,2]
 
    world_coord[:,:,0] = (x / scale - width) * fx * z
    world_coord[:,:,1] = (height - y / scale) * fy * z
    world_coord[:,:,2] = z
    
    return world_coord

if __name__ == '__main__':
    test()