import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ChocolateDataset(Dataset):
    def __init__(self, csv_file, img_path, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_path = img_path
        self.transform = transform
    def __len__(self):
        return len(self.labels) # Returns the Number of images in the dataset class
    def __getitem__(self, idx):
        # Get image ID from CSV
        img_id = self.labels.iloc[idx, 0]
        # Build image file path (assuming .jpg)
        img_file = os.path.join(self.img_path, f"L{img_id}.jpg")
        # Load image
        image = Image.open(img_file).convert("RGB")
        # Get label (all columns except the first)
        label = self.labels.iloc[idx, 1:].values.astype('float32')
        if self.transform:
            image = self.transform(image)
        return image, label