import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def make_resnet50_1ch(pretrained=True, luminance=False):
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = resnet50(weights=weights)
    w_rgb = m.conv1.weight.detach().clone()
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        if luminance:
            coeff = torch.tensor([0.2989, 0.5870, 0.1140]).view(1,3,1,1)
            m.conv1.weight.copy_((w_rgb * coeff).sum(dim=1, keepdim=True))
        else:
            m.conv1.weight.copy_(w_rgb.mean(dim=1, keepdim=True))
    m.fc = nn.Identity()
    return m

def freeze_resnet_until(model: nn.Module, mode: str):
    if not mode:
        return
    if mode == "layer3":
        freeze_names = ["conv1", "bn1", "layer1", "layer2"]
    elif mode == "layer4":
        freeze_names = ["conv1", "bn1", "layer1", "layer2", "layer3"]
    elif mode == "all":
        freeze_names = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
    else:
        raise ValueError(f"Unknown freeze mode: {mode}")
    for name, module in model.named_children():
        if name in freeze_names:
            for p in module.parameters():
                p.requires_grad = False
