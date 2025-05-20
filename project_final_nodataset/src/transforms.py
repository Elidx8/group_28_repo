# src/transforms.py

import random
import torch
import torchvision.transforms.functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Resize:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
    def __call__(self, image, target):
        w, h = image.size
        
        scale = self.min_size / min(h, w)
        if max(h, w) * scale > self.max_size:
            scale = self.max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h))
        if "boxes" in target:
            target["boxes"] = target["boxes"] * scale
        return image, target
