# src/model.py

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models._utils import IntermediateLayerGetter


# Light Two‐layer head 
class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        return x

def get_model(num_classes):
    # Fresh ResNet-18, extracting layer2,3,4 → C3,C4,C5
    backbone_net = torchvision.models.resnet18(pretrained=False)
    return_layers = {'layer2':'feat2', 'layer3':'feat3', 'layer4':'feat4'}
    body = IntermediateLayerGetter(backbone_net, return_layers=return_layers)

    # Tiny FPN on those three maps, all reduced to 64 channels
    in_channels_list = [128, 256, 512]  # channels of layer2,3,4
    fpn_out_channels = 64
    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        out_channels=fpn_out_channels
    )

    # Gluing body + FPN
    backbone = nn.Sequential(body, fpn)
    backbone.out_channels = fpn_out_channels

    # Anchor sizes 
    anchor_sizes = (
        (256, 320, 384, 448),   # P3  for small objects
        (384, 448, 512, 576),   # P4  for medium
        (512, 576, 640, 704),   # P5  for large
    )
    anchor_aspect_ratios = (
        (0.75, 1.0, 1.33),      # P3
        (0.75, 1.0, 1.33),      # P4
        (0.75, 1.0, 1.33),      # P5
    )

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=anchor_aspect_ratios
    )

    # RoI pooling across the three FPN levels
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['feat2','feat3','feat4'],
        output_size=7,
        sampling_ratio=2
    )

    # Build the Faster R-CNN
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    # Slim down the box head to a small TwoMLPHead (32-dim)
    pooled_h, pooled_w = model.roi_heads.box_roi_pool.output_size
    in_channels = backbone.out_channels * pooled_h * pooled_w   # 64*7*7 = 3136
    rep_size   = 180

    model.roi_heads.box_head = TwoMLPHead(in_channels, rep_size)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=rep_size,
        num_classes=num_classes
    )

    return model
    