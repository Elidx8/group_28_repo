import torch
from torch import nn
import torch.nn.functional as F

# Further reduce the model size to have fewer trainable parameters
# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=13):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 224 -> 112
#             nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 112 -> 56
#             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 56 -> 28
#             nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 28 -> 14
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 14 * 14, 256), nn.ReLU(),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return F.relu(x)  # Apply ReLU to ensure non-negative outputs

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=13):
#         super().__init__()
#         # Single convolutional layer
#         self.conv = nn.Conv2d(3, 16, kernel_size=15, stride=15, padding=7)  # Adjust kernel size and stride
#         self.avg_pool = nn.AvgPool2d(kernel_size=50, stride=50)  # Average pooling to "count" chocolates
#         self.classifier = nn.Linear(16, num_classes)  # Fully connected layer

#     def forward(self, x):
#         x = self.conv(x)  # Apply convolution
#         x = F.relu(x)  # Apply ReLU activation
#         x = self.avg_pool(x)  # Apply average pooling
#         x = torch.mean(x, dim=(2, 3))  # Global average pooling (spatial dimensions)
#         x = self.classifier(x)  # Classifier
#         return x

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        # Convolutional layers with max pooling to reduce spatial dimensions
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)  # Reduce spatial size
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halve spatial dimensions

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)  # Further reduce spatial size
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halve spatial dimensions again

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Final convolutional layer
        self.avg_pool3 = nn.AvgPool2d(kernel_size=8)  # Average pooling to reduce spatial dimensions

        # Fully connected classifier
        self.classifier = nn.Linear(44160, num_classes)  # Adjust input size to match flattened dimensions

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        # print("After conv1:", x.shape, flush=True)
        x = F.relu(x)
        # print("After ReLU (conv1):", x.shape, flush=True)
        x = self.max_pool1(x)
        # print("After max_pool1:", x.shape, flush=True)

        # Second convolutional block
        x = self.conv2(x)
        # print("After conv2:", x.shape, flush=True)
        x = F.relu(x)
        # print("After ReLU (conv2):", x.shape, flush=True)
        x = self.max_pool2(x)
        # print("After max_pool2:", x.shape, flush=True)

        # Third convolutional block
        x = self.conv3(x)
        # print("After conv3:", x.shape, flush=True)
        x = F.relu(x)
        # print("After ReLU (conv3):", x.shape, flush=True)

        # Average pooling to reduce spatial dimensions
        x = self.avg_pool3(x)
        # print("After avg_pool3:", x.shape, flush=True)

        # Flatten the output for the classifier
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch size
        # print("After flatten:", x.shape, flush=True)

        # Fully connected layer
        x = self.classifier(x)
        # print("After classifier:", x.shape, flush=True)
        return x