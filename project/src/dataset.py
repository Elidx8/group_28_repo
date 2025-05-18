import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms

# class ChocolateDataset(Dataset):
#     def __init__(self, csv_file, img_path, transform=None, subset_fraction=0.5):
#         # Handle the case where csv_file is None
#         if csv_file is None:
#             self.labels = None
#         else:
#             self.labels = pd.read_csv(csv_file)
#             # Keep only the first subset_fraction of rows
#             subset_size = int(len(self.labels) * subset_fraction)
#             self.labels = self.labels.iloc[:subset_size]

#         self.img_path = img_path
#         self.transform = transform

#     def __len__(self):
#         # Update __len__ to handle no labels
#         if self.labels is None:
#             return len(os.listdir(self.img_path))  # Number of images in the directory
#         return len(self.labels)  # Number of rows in the labels DataFrame

#     def __getitem__(self, idx):
#         # Update __getitem__ to handle no labels
#         if self.labels is None:
#             img_id = os.listdir(self.img_path)[idx].split('.')[0]  # Extract ID from file name
#             img_file = os.path.join(self.img_path, os.listdir(self.img_path)[idx])
#             image = Image.open(img_file).convert("RGB")
#             if self.transform:
#                 image = self.transform(image)
#             return image, img_id  # Return image and ID only

#         # Get image ID from CSV
#         img_id = self.labels.iloc[idx, 0]
#         # Build image file path (assuming .jpg)
#         img_file = os.path.join(self.img_path, f"L{img_id}.jpg")
#         # Load image
#         image = Image.open(img_file).convert("RGB")
#         # Get label (all columns except the first)
#         label = self.labels.iloc[idx, 1:].values.astype('int')
        
#         if self.transform:
#             image = self.transform(image)
#         return image, label


class ChocolateDataset(Dataset):
    def __init__(self, csv_file, img_path, transform=None, subset_fraction=1.0, preload=False):
        """
        csv_file: path to the CSV (or None)
        img_path: folder with images
        transform: torchvision transforms
        subset_fraction: float between 0 and 1
        preload: if True, load and transform all images in __init__ (use for training)
                 if False, load image per sample (use for testing)
        """
        self.img_path = img_path
        self.transform = transform
        self.preload = preload

        if csv_file is not None:
            self.labels = pd.read_csv(csv_file)
            subset_size = int(len(self.labels) * subset_fraction)
            self.labels = self.labels.iloc[:subset_size]
        else:
            self.labels = None

        if self.preload:
            self.data = []
            if self.labels is not None:
                for idx in tqdm(range(len(self.labels)), desc="Preloading training data"):
                    img_id = self.labels.iloc[idx, 0]
                    img_file = os.path.join(self.img_path, f"L{img_id}.jpg")
                    image = Image.open(img_file).convert("RGB")
                    if self.transform:
                        image = self.transform(image)
                    label = self.labels.iloc[idx, 1:].values.astype('int')
                    self.data.append((image, label))
            else:
                all_files = sorted(os.listdir(self.img_path))
                for fname in tqdm(all_files, desc="Preloading unlabeled data"):
                    img_file = os.path.join(self.img_path, fname)
                    image = Image.open(img_file).convert("RGB")
                    if self.transform:
                        image = self.transform(image)
                    img_id = fname.split('.')[0]
                    self.data.append((image, img_id))

    def __len__(self):
        if self.preload:
            return len(self.data)
        elif self.labels is not None:
            return len(self.labels)
        else:
            return len(os.listdir(self.img_path))

    def __getitem__(self, idx):
        if self.preload:
            return self.data[idx]

        if self.labels is not None:
            img_id = self.labels.iloc[idx, 0]
            img_file = os.path.join(self.img_path, f"L{img_id}.jpg")
            image = Image.open(img_file).convert("RGB")
            label = self.labels.iloc[idx, 1:].values.astype('int')
            if self.transform:
                image = self.transform(image)
            return image, label
        else:
            fname = os.listdir(self.img_path)[idx]
            img_file = os.path.join(self.img_path, fname)
            image = Image.open(img_file).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else: # If no transform is provided, transform to tensor
                image = transforms.ToTensor()(image)
            img_id = fname.split('.')[0]
            return image, img_id