import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ChocolateDataset(Dataset):
    def __init__(self, csv_file, img_path, transform=None, subset_fraction=0.1):
        # Handle the case where csv_file is None
        if csv_file is None:
            self.labels = None
        else:
            self.labels = pd.read_csv(csv_file)
            # Keep only the first subset_fraction of rows
            subset_size = int(len(self.labels) * subset_fraction)
            self.labels = self.labels.iloc[:subset_size]

        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        # Update __len__ to handle no labels
        if self.labels is None:
            return len(os.listdir(self.img_path))  # Number of images in the directory
        return len(self.labels)  # Number of rows in the labels DataFrame

    def __getitem__(self, idx):
        # Update __getitem__ to handle no labels
        if self.labels is None:
            img_id = os.listdir(self.img_path)[idx].split('.')[0]  # Extract ID from file name
            img_file = os.path.join(self.img_path, os.listdir(self.img_path)[idx])
            image = Image.open(img_file).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, img_id  # Return image and ID only

        # Get image ID from CSV
        img_id = self.labels.iloc[idx, 0]
        # Build image file path (assuming .jpg)
        img_file = os.path.join(self.img_path, f"L{img_id}.jpg")
        # Load image
        image = Image.open(img_file).convert("RGB")
        # Get label (all columns except the first)
        label = self.labels.iloc[idx, 1:].values.astype('int')
        
        if self.transform:
            image = self.transform(image)
        return image, label