# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /home/atincuzun/Desktop/nn-dataset/ab/nn/transform/coco_detection_transform.py
# Bytecode version: 3.10.0rc2 (3439)
# Source timestamp: 2024-12-19 20:57:58 UTC (1734641878)

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            if target is None:
                image = t(image)
            else:
                image, target = t(image, target)
        if target is not None:
            return (image, target)
        return image

class Resize:

    def __init__(self, size):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self, image, target=None):
        orig_size = image.size[::-1]
        image = F.resize(image, self.size)
        if target is None:
            return image
        h_ratio = self.size[0] / orig_size[0]
        w_ratio = self.size[1] / orig_size[1]
        if 'boxes' in target:
            boxes = target['boxes']
            scaled_boxes = boxes.clone()
            scaled_boxes[:, [0, 2]] *= w_ratio
            scaled_boxes[:, [1, 3]] *= h_ratio
            target['boxes'] = scaled_boxes
        return (image, target)

class ToTensor:

    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        if target is not None:
            return (image, target)
        return image

class Normalize:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is not None:
            return (image, target)
        return image

def transform(**kwargs):
    """
    Returns transform for object detection:
    - Resizes image to 320x320 (SSDLite's expected input size)
    - Converts to tensor
    - Normalizes with ImageNet stats
    """
    return Compose([Resize((320, 320)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
