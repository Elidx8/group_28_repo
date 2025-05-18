import torch
from torch import nn
import torch.nn.functional as F

class SimpleCNN_Light(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=2, padding=4)  # 128 → 64
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64 → 32

        # Convolutional blocks
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)  # 32 → 16
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 16 → 8
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1)  # 8 → 4
        self.bn4 = nn.BatchNorm2d(384)

        self.conv5 = nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=1)  # 4 → 2
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # 2 → 2
        self.bn6 = nn.BatchNorm2d(512)

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # MLP head (slightly wider)
        self.fc1 = nn.Linear(512, 768)
        self.fc2 = nn.Linear(768, 384)
        self.fc3 = nn.Linear(384, 256)
        self.fc4 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))

        x = self.avg_pool(x)
        x = x.squeeze(-1).squeeze(-1)  # Flatten to [batch_size, num_features]

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)

        return x
    
# class ChocolateCounterCNN(nn.Module):
#     def __init__(self, num_classes=13):
#         super().__init__()
#         # Initial BIG kernel (like ResNet50 stem)
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=11, stride=2, padding=5)  # Larger RF at start
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu = nn.ReLU(inplace=True)
#         self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # smoother downsampling

#         # Second block: medium kernel, aggressive stride
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3)
#         self.bn2 = nn.BatchNorm2d(64)

#         # Third block
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
#         self.bn3 = nn.BatchNorm2d(128)

#         # Fourth block
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
#         self.bn4 = nn.BatchNorm2d(256)

#         # Fifth block (deeper feature extraction, no downsampling)
#         self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(256)

#         # Global average pooling
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

#         # Final linear layer
#         self.classifier = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.max_pool1(x)

#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.relu(self.bn3(self.conv3(x)))
#         x = self.relu(self.bn4(self.conv4(x)))
#         x = self.relu(self.bn5(self.conv5(x)))

#         x = self.avg_pool(x)
#         # print(x.shape)
#         x = x.view(x.size(0), -1)  # flatten
#         # print(x.shape)
#         x = self.classifier(x)
#         return x

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=13):
#         super().__init__()
#         # Adjusted convolutional layers with max pooling to increase receptive field
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=17, stride=2, padding=8)  # Larger kernel size for faster RF growth
#         self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halve spatial dimensions

#         self.conv2 = nn.Conv2d(32, 64, kernel_size=15, stride=2, padding=7)  # Further increase RF
#         self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halve spatial dimensions again

#         self.conv3 = nn.Conv2d(64, 128, kernel_size=13, stride=2, padding=6)  # Final convolutional layer
#         self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halve spatial dimensions again

#         # Average pooling to reduce spatial dimensions
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to ensure output is a 1D feature vector

#         # Fully connected classifier
#         self.classifier = nn.Linear(128, num_classes)

#     def forward(self, x):
#         # First convolutional block
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.max_pool1(x)

#         # Second convolutional block
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.max_pool2(x)

#         # Third convolutional block
#         x = self.conv3(x)
#         x = F.relu(x)
#         # print(x.shape)
#         x = self.max_pool3(x)

#         # Average pooling to reduce spatial dimensions
#         # print(x.shape)
#         x = self.avg_pool(x)
#         # print(x.shape)
#         x = x.squeeze(-1).squeeze(-1)

#         # Fully connected layer
#         x = self.classifier(x)
#         return x

# class DeepChocolateCounter(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # output 400x400
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             # Residual block (just a few examples)
#             *[ResidualBlock(128) for _ in range(4)],

#             nn.Conv2d(128, 256, 3, padding=2, dilation=2),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.Conv2d(256, 256, 3, padding=4, dilation=4),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.Conv2d(256, 512, 3, padding=8, dilation=8),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),

#             nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
#         )
#         self.regressor = nn.Linear(512, 13)

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)  # flatten
#         return self.regressor(x)

# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(),
#             nn.Conv2d(channels, channels, 3, padding=1),
#             nn.BatchNorm2d(channels)
#         )

#     def forward(self, x):
#         return nn.ReLU()(x + self.block(x))
    
# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=13):
#         super().__init__()
#         # First convolutional block (largest kernel)
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=21, stride=2, padding=10)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Second convolutional block
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=19, stride=2, padding=9)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Third convolutional block
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=17, stride=2, padding=8)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Fourth convolutional block
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=15, stride=2, padding=7)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Average pooling to reduce spatial dimensions
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

#         # Fully connected classifier
#         self.classifier = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.max_pool1(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.max_pool2(x)

#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = F.relu(x)
#         x = self.max_pool3(x)

#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = F.relu(x)
#         x = self.max_pool4(x)

#         x = self.avg_pool(x)
#         x = x.squeeze(-1).squeeze(-1)
#         x = self.classifier(x)
#         return x
    
    

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=13):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=61, stride=2, padding=30)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.pool1 = nn.MaxPool2d(2, 2)

#         self.conv2 = nn.Conv2d(16, 32, kernel_size=41, stride=2, padding=20)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.pool2 = nn.MaxPool2d(2, 2)

#         self.conv3 = nn.Conv2d(32, 48, kernel_size=31, stride=2, padding=15)
#         self.bn3 = nn.BatchNorm2d(48)
#         self.pool3 = nn.MaxPool2d(2, 2)

#         self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.pool4 = nn.MaxPool2d(2, 2)

#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(64, num_classes)

#     def forward(self, x):
#         x = self.pool1(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool2(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool3(F.relu(self.bn3(self.conv3(x))))
#         x = self.pool4(F.relu(self.bn4(self.conv4(x))))
#         x = self.global_avg_pool(x)
#         x = x.squeeze(-1).squeeze(-1)  # Flatten to [batch_size, num_features]
#         x = self.fc(x)
#         return x #F.relu(x)  # Ensure counts are non-negative

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=13):
#         super(SimpleCNN, self).__init__()
        
#         # Feature Extractor
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=15, stride=1, padding=1)  # Stem
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling
        
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling
        
#         self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling
        
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling
        
#         self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # Final feature map
        
#         # Head
#         self.head_conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)  # 3×3 convolution
#         self.head_conv2 = nn.Conv2d(128, num_classes, kernel_size=1, stride=1)  # 1×1 output head
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

#     def forward(self, x):
#         print("Input:", x.shape)
#         # Feature Extractor
#         x = F.leaky_relu(self.conv1(x), negative_slope=0.3)
#         print("After conv1:", x.shape)
#         x = F.leaky_relu(self.conv2(x), negative_slope=0.3)
#         print("After conv2:", x.shape)
#         x = self.pool2(x)
#         print("After pool2:", x.shape)
        
#         x = F.leaky_relu(self.conv3(x), negative_slope=0.3)
#         print("After conv3:", x.shape)
#         x = self.pool3(x)
#         print("After pool3:", x.shape)
        
#         x = F.leaky_relu(self.conv4(x), negative_slope=0.3)
#         print("After conv4:", x.shape)
#         x = self.pool4(x)
#         print("After pool4:", x.shape)
        
#         x = F.leaky_relu(self.conv5(x), negative_slope=0.3)
#         print("After conv5:", x.shape)
#         x = self.pool5(x)
#         print("After pool5:", x.shape)
        
#         x = F.leaky_relu(self.conv6(x), negative_slope=0.3)
#         print("After conv6:", x.shape)
        
#         # Head
#         x = F.leaky_relu(self.head_conv1(x), negative_slope=0.3)
#         print("After head_conv1:", x.shape)
#         x = self.head_conv2(x)
#         print("After head_conv2:", x.shape)
#         x = self.global_avg_pool(x)
#         print("After global_avg_pool:", x.shape)
#         x = x.squeeze(-1).squeeze(-1)  # Flatten to [batch_size, num_classes]
#         print("After squeeze:", x.shape)
        
#         return x

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=13, num_bins=6):
#         super(SimpleCNN, self).__init__()

#         # Feature extractor with larger early kernels and progressive downsampling
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=12, stride=3, padding=3)       
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=7, stride=2, padding=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)    
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=7, stride=3, padding=2)     
#         self.conv4 = nn.Conv2d(64, 128, kernel_size=7, stride=3, padding=1)    
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=7, stride=3, padding=1)    
#         self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)    

#         # Head
#         self.head_conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
#         self.head_conv2 = nn.Conv2d(256, num_classes * num_bins, kernel_size=1, stride=1)
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  

#     def forward(self, x):
#         # print("Input:", x.shape)
#         x = F.leaky_relu(self.conv1(x), negative_slope=0.3)
#         # print("After conv1:", x.shape)
#         x = self.pool2(x)
#         # print("After pool2:", x.shape)
#         x = F.leaky_relu(self.conv2(x), negative_slope=0.3)
#         # print("After conv2:", x.shape)
#         x = F.leaky_relu(self.conv3(x), negative_slope=0.3)
#         # print("After conv3:", x.shape)
#         x = F.leaky_relu(self.conv4(x), negative_slope=0.3)
#         # print("After conv4:", x.shape)
#         x = F.leaky_relu(self.conv5(x), negative_slope=0.3)
#         # print("After conv5:", x.shape)
#         x = F.leaky_relu(self.conv6(x), negative_slope=0.3)
#         # print("After conv6:", x.shape)
        
#         x = F.leaky_relu(self.head_conv1(x), negative_slope=0.3)
#         # print("After head_conv1:", x.shape)
#         x = self.head_conv2(x)
#         # print("After head_conv2:", x.shape)
#         x = self.global_avg_pool(x)
#         # print("After global_avg_pool:", x.shape)

#         x = x.view(x.size(0), 13, 6)  # reshape to [batch_size, num_classes, num_bins]
#         # print("After reshape:", x.shape)
        
#         return x
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.3),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.block(x)
        return F.leaky_relu(out + identity, negative_slope=0.3)


class DeepResidualCNN(nn.Module):
    def __init__(self, num_classes=13, num_bins=6):
        super(DeepResidualCNN, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=4, padding=2),  # early downsampling
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.3)
        )
        
        # Residual blocks with increasing depth
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 512, stride=2)
        self.layer5 = ResidualBlock(512, 512, stride=1)
        self.layer6 = ResidualBlock(512, 512, stride=1)

        self.dropout = nn.Dropout(0.4)

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.3),
            nn.Conv2d(256, num_classes * num_bins, kernel_size=1)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.dropout(x)
        x = self.head(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), 13, 6)
        return x

    

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=13):
#         super(SimpleCNN, self).__init__()
        
#         # Feature Extractor
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling
        
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
#         # Head
#         self.head_conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)  # 3×3 convolution
#         self.head_conv2 = nn.Conv2d(128, num_classes * 6, kernel_size=1, stride=1)  # 1×1 output head
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

#     def forward(self, x):
#         # Feature Extractor
#         x = F.leaky_relu(self.conv1(x), negative_slope=0.3)
#         x = F.leaky_relu(self.conv2(x), negative_slope=0.3)
#         x = self.pool(x)
        
#         x = F.leaky_relu(self.conv3(x), negative_slope=0.3)
#         x = F.leaky_relu(self.conv4(x), negative_slope=0.3)
        
#         # Head
#         x = F.leaky_relu(self.head_conv1(x), negative_slope=0.3)
#         x = self.head_conv2(x)
#         x = self.global_avg_pool(x)
#         print(x.shape)
#         x = x.squeeze(-1).squeeze(-1)  # Reshape to [B, 13, 6]
#         print(x.shape)
        
#         return x

    
    
# Simpple ResNet

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample

#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = F.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         out += identity
#         out = F.relu(out)
#         return out


# class SimpleResNet(nn.Module):
#     def __init__(self, num_classes=13):
#         super(SimpleResNet, self).__init__()
#         # Initial convolutional layer
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         # Residual blocks
#         self.layer1 = self._make_layer(32, 64, stride=2)
#         self.layer2 = self._make_layer(64, 128, stride=2)
#         self.layer3 = self._make_layer(128, 256, stride=2)

#         # Average pooling and classifier
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(256, num_classes)

#     def _make_layer(self, in_channels, out_channels, stride=1):
#         downsample = None
#         if stride != 1 or in_channels != out_channels:
#             downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels),
#             )

#         layers = []
#         layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
#         layers.append(ResidualBlock(out_channels, out_channels))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         # Initial convolutional layer
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         # Residual blocks
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)

#         # Average pooling and classifier
#         x = self.avg_pool(x)
#         x = x.squeeze(-1).squeeze(-1)  # Flatten to [batch_size, num_features]
#         x = self.fc(x)
#         return x