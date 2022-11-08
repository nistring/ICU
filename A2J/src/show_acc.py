import sys
sys.path.append('/csehome/baneling100/nistring/A2J')

import numpy as np
import matplotlib.pyplot as plt
from lib.accuracy import *
import h5py


with h5py.File('/csehome/baneling100/nistring/A2J/data/ITOP_side_test_labels.h5', 'r') as f:
    validFrames = np.where(f['is_valid'][:])[0]
    gt = f['real_world_coordinates'][validFrames]
    gt = gt[:,np.arange(9),:] # upper body

pred = np.load('/csehome/baneling100/nistring/A2J/res/itop_side_result.npy')

print('gt: ', gt.shape)
print('pred: ', pred.shape)


keypoints_num = 9
names = ['joint'+str(i+1) for i in range(keypoints_num)]


dist, acc = compute_dist_acc_wrapper(pred, gt, max_dist=1, num=100)

fig, ax = plt.subplots()
plot_acc(ax, dist, acc, names)
fig.savefig('/csehome/baneling100/nistring/A2J/res/itop_side_joint_acc.png')
plt.show()


mean_err, avg_precision = compute_mean_err(pred, gt)
fig, ax = plt.subplots()
plot_mean_err(ax, mean_err, names)
fig.savefig('/csehome/baneling100/nistring/A2J/res/itop_side_joint_mean_err.png')
plt.show()


print('mean_err: {}'.format(mean_err))
print('avg_precision: {}'.format(avg_precision))
mean_err_all, avg_precision_all = compute_mean_err(pred.reshape((-1, 1, 3)), gt.reshape((-1, 1,3)))
print('mean_err_all: ', mean_err_all)
print('average_precision_all: ', avg_precision_all)