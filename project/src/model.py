import torch
from torch import nn
import torch.nn.functional as F

# Add more layers to make the CNN larger
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 224 -> 112
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 112 -> 56
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 56 -> 28
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 28 -> 14
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return F.relu(x)  # Apply ReLU to ensure non-negative outputs