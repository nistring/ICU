from torch.utils.data import Dataset, DataLoader
import os
import h5py
import cv2 as cv
import numpy as np
from random_erasing import RandomErasing
import albumentations as A
from sklearn.model_selection import train_test_split

shift_range = 0.05
rotation_range = [-45, 45]
shear_range = [-45, 45]

randomseed = 12345
np.random.seed(randomseed)

root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
data_dir = os.path.join(root, "datasets")


class train_dataset(Dataset):
    def __init__(self, valid_frames, bounding_box, keypoints, opt):
        self.valid_frames = valid_frames
        self.bounding_box = bounding_box
        self.keypoints = keypoints
        self.width = opt.width
        self.height = opt.height
        self.input_size = opt.input_size
        self.transform = self._get_transform()
        self.random_erasing = RandomErasing()

    def __getitem__(self, index):
        depth_map = np.load(
            os.path.join(
                data_dir, "train", "depth_data", str(self.valid_frames[index]) + ".npy"
            )
        ).astype(np.float32)
        box = self.bounding_box[index]
        keypoints = self.keypoints[index].astype(np.float32)

        transformed = self.transform(
            image=depth_map, bboxes=[list(box) + ["person"]], keypoints=keypoints
        )
        depth_map = transformed["image"]
        box = transformed["bboxes"][0]
        keypoints = transformed["keypoints"]

        depth_map, keypoints, box = self.crop(depth_map, keypoints, box)
        depth_map = (depth_map - depth_map.mean()) / depth_map.std()
        depth_map = self.random_erasing(depth_map)

        # visualize_depth_map(depth_map, keypoints)
        return depth_map, keypoints, box

    def __len__(self):
        return self.valid_frames.size

    def _get_transform(self):
        transform = A.Compose(
            [
                A.Affine(scale=1.0, rotate=rotation_range, shear=shear_range),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            bbox_params=A.BboxParams(format="yolo"),
        )
        return transform

    def crop(self, depth_map, keypoints, box):
        box = np.asarray(box[:4], dtype=float)
        box[:2] -= 0.5 * box[2:]  # x,y,w,h -> x_min,y_min,x_max,y_max
        box[2:] += box[:2]
        box += (np.random.rand(box.shape[0]) - 0.5) * shift_range  # add random crop
        box = np.clip(box, 0, 1)

        box[[0, 2]] *= self.width
        box[[1, 3]] *= self.height
        box = box.astype(np.int16)

        depth_map = depth_map[box[1] : box[3], box[0] : box[2]]
        depth_map = cv.resize(
            depth_map, (self.input_size, self.input_size), interpolation=cv.INTER_CUBIC
        )

        keypoints = np.array(keypoints)
        keypoints = (
            (keypoints - box[np.newaxis, :2])
            * self.input_size
            / (box[np.newaxis, 2:] - box[np.newaxis, :2])
        )

        return depth_map, keypoints, box


class test_dataset(Dataset):
    def __init__(self, valid_frames, bounding_box, keypoints, opt):
        self.valid_frames = valid_frames
        self.bounding_box = bounding_box
        self.keypoints = keypoints
        self.width = opt.width
        self.height = opt.height
        self.input_size = opt.input_size
        self.is_test = True if opt.weight else False

    def __getitem__(self, index):
        depth_map = np.load(
            os.path.join(
                data_dir,
                "test" if self.is_test else "train",
                "depth_data",
                str(self.valid_frames[index]) + ".npy",
            )
        ).astype(np.float32)
        box = self.bounding_box[index]
        keypoints = self.keypoints[index].astype(np.float32)

        depth_map, keypoints, box = self.crop(depth_map, keypoints, box)
        depth_map = (depth_map - depth_map.mean()) / depth_map.std()

        return depth_map, keypoints, box

    def __len__(self):
        return self.valid_frames.size

    def crop(self, depth_map, keypoints, box):
        box = np.asarray(box[:4], dtype=float)
        box[:2] -= 0.5 * box[2:]  # x,y,w,h -> x_min,y_min,x_max,y_max
        box[2:] += box[:2]
        box = np.clip(box, 0, 1)

        box[[0, 2]] *= self.width
        box[[1, 3]] *= self.height
        box = box.astype(np.int16)

        depth_map = depth_map[box[1] : box[3], box[0] : box[2]]
        depth_map = cv.resize(
            depth_map, (self.input_size, self.input_size), interpolation=cv.INTER_CUBIC
        )

        keypoints = np.array(keypoints)
        keypoints = (
            (keypoints - box[np.newaxis, :2])
            * self.input_size
            / (box[np.newaxis, 2:] - box[np.newaxis, :2])
        )

        return depth_map, keypoints, box


def build_train_dataloader(opt):

    with h5py.File(os.path.join(data_dir, "train/res/ICU_train_labels.h5"), "r") as f:
        bounding_box = f["bounding_box"][:].copy()
        keypoints = f["image_coordinates"][:].copy()
        valid_frames = np.where(np.logical_and(bounding_box[:, 0], keypoints[:, 0, 0]))[
            0
        ]

        train_frames, val_frames = train_test_split(
            valid_frames, test_size=0.25, shuffle=False
        )
        train_bounding_box = bounding_box[train_frames]
        val_bounding_box = bounding_box[val_frames]
        train_keypoints = keypoints[train_frames]
        val_keypoints = keypoints[val_frames]

    train_dataloader = DataLoader(
        train_dataset(train_frames, train_bounding_box, train_keypoints, opt),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=16,
    )
    val_dataloader = DataLoader(
        test_dataset(val_frames, val_bounding_box, val_keypoints, opt),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=16,
    )

    return train_dataloader, val_dataloader


def build_test_dataloader(opt):

    with h5py.File(os.path.join(data_dir, "test/res/ICU_test_labels.h5"), "r") as f:
        bounding_box = f["bounding_box"][valid_frames].copy()
        keypoints = f["keypoints"][valid_frames].copy()
        valid_frames = np.where(np.logical_and(bounding_box[:, 0], keypoints[:, 0, 0]))[0]

    test_dataloader = DataLoader(
        test_dataset(valid_frames, bounding_box, keypoints, opt),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=8,
    )

    return test_dataloader


def visualize_depth_map(depth_map, keypoints):

    depth_save = depth_map + np.min(depth_map)
    depth_save = cv.applyColorMap(
        (depth_save * 255 / depth_save.max()).astype(np.uint8),
        colormap=cv.COLORMAP_OCEAN,
    )

    p = keypoints[:, :2].astype(np.int16)
    linewidth = 2
    cv.line(
        depth_save,
        p[0],
        ((p[1] + p[2]) / 2).astype(np.int16),
        (178, 102, 255),
        linewidth,
    )
    cv.line(depth_save, p[1], p[2], (153, 153, 255), linewidth)
    cv.line(depth_save, p[1], p[3], (102, 102, 255), linewidth)
    cv.line(depth_save, p[3], p[5], (51, 51, 255), linewidth)
    cv.line(depth_save, p[2], p[4], (255, 153, 153), linewidth)
    cv.line(depth_save, p[4], p[6], (255, 102, 102), linewidth)
    cv.line(depth_save, p[1], p[7], (255, 51, 51), linewidth)
    cv.line(depth_save, p[2], p[8], (230, 0, 230), linewidth)

    cv.imwrite("/home/ubuntu/workspace/@ICU/A2J/res/augmentation.jpg", depth_save)
