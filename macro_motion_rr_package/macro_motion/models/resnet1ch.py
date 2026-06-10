import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def make_frozen_resnet50_feature_extractor():
    """
    Built-in PyTorch pretrained ResNet-50, unchanged conv1=3 channels.
    Frozen feature extractor:
      input:  (N,3,H,W)
      output: (N,2048)
    """
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Identity()
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m

def make_frozen_resnet50_trunk():
    """
    Built-in PyTorch pretrained ResNet-50 trunk up to layer4.
    Unchanged conv1=3 channels, frozen.
      input:  (N,3,H,W)
      output: (N,2048,7,7) for 224x224 input
    """
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    trunk = nn.Sequential(
        m.conv1, m.bn1, m.relu, m.maxpool,
        m.layer1, m.layer2, m.layer3, m.layer4
    )
    trunk.eval()
    for p in trunk.parameters():
        p.requires_grad = False
    return trunk
