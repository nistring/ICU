import random
import math
import numpy as np
import torch


class RandomErasing(object):
    """Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.05, sh=0.4, r1=0.3, scale=0.5):
        self.probability = probability
        self.mean = 0
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.scale = scale

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            h, w = img.shape
            target_area = random.uniform(self.sl, self.sh) * img.size
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            noise_h = int(round(math.sqrt(target_area * aspect_ratio)))
            noise_w = int(round(math.sqrt(target_area / aspect_ratio)))

            if noise_w < w and noise_h < h:
                rand_patch = self.scale * torch.randn(noise_h, noise_w) + self.mean
                noise_y = random.randint(0, h - noise_h)
                noise_x = random.randint(0, w - noise_w)
                img[noise_y : noise_y + noise_h, noise_x : noise_x + noise_w] = rand_patch
                return img

        return img
