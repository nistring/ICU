#! /usr/bin/env python
#! coding:utf-8:w

from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
import pickle
from pathlib import Path
import h5py
import sys
sys.path.insert(0, '..')
from utils import *  # noqa
current_file_dirpath = Path(__file__).parent.parent.absolute()


def load_icu_data(
        train_path=current_file_dirpath / Path("../datasets/train/ICU_train_labels.h5"),
        test_path=current_file_dirpath / Path("../datasets/test/ICU_test_labels.h5")):
    with h5py.File(train_path, 'r') as f:
        Train = f
    with h5py.File(test_path, 'w') as f:
        Test = f
    le = preprocessing.LabelEncoder()
    le.fit(Train['frame_id_list'])
    print("Loading ICU Dataset")
    return Train, Test, le


class JConfig():
    def __init__(self):
        self.frame_l = 32  # the length of frames
        self.joint_n = 9  # the number of joints
        self.joint_d = 2  # the dimension of joints
        self.clc_num = 4  # the number of class
        self.feat_d = 21
        self.filters = 64

# Genrate dataset
# T: Dataset  C:config le:labelEncoder


def ICU_generator(T, C, le):
    X_0 = []
    X_1 = []
    Y = []
    labels = le.transform(T['frame_id_list'])
    pose = T['image_coordinates']
    for i in range(0, pose.shape[0]-1, 3):
        pose[i+1] = 0.5 * (pose[i] + pose[i+1]) # low pass filter / smooth_alpha = 0.5
    for i in tqdm(range(pose.shape[0] - C.frame_l)):
        p = np.copy(pose[i:i+C.frame_l])
        print(p.shape)
        # p.shape (frame,joint_num,joint_coords_dims)
        p = zoom(p, target_l=C.frame_l,
                 joints_num=C.joint_n, joints_dim=C.joint_d)
        # p.shape (target_frame,joint_num,joint_coords_dims)
        # label = np.zeros(C.clc_num)
        # label[labels[i]] = 1
        label = labels[i:i+C.frame_l]
        label = max(label, key=label.count)
        # M.shape (target_frame,(joint_num - 1) * joint_num / 2)
        M = get_CG(p, C)

        X_0.append(M)
        X_1.append(p)
        Y.append(label)

    X_0 = np.stack(X_0)
    X_1 = np.stack(X_1)
    Y = np.stack(Y)
    return X_0, X_1, Y
