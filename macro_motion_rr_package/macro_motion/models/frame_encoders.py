import torch
import torch.nn as nn
from .resnet1ch import make_resnet50_1ch, freeze_resnet_until

class ResNetFrameEncoder1Ch(nn.Module):
    def __init__(self, d_model=512, pretrained=True, freeze_until=""):
        super().__init__()
        self.backbone = make_resnet50_1ch(pretrained=pretrained)
        freeze_resnet_until(self.backbone, freeze_until)
        self.proj = nn.Linear(2048, d_model)

    def forward(self, x):
        B,T,C,H,W = x.shape
        feat = self.backbone(x.reshape(B*T, C, H, W))
        return self.proj(feat).view(B, T, -1)

class ResNetSpatialBackbone1Ch(nn.Module):
    def __init__(self, pretrained=True, freeze_until=""):
        super().__init__()
        m = make_resnet50_1ch(pretrained=pretrained)
        freeze_resnet_until(m, freeze_until)
        self.trunk = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4
        )
        self.out_channels = 2048

    def forward(self, x):
        return self.trunk(x)

class FaceAwareResNetEncoder1Ch(nn.Module):
    def __init__(self, d_model=512, pretrained=True, freeze_until=""):
        super().__init__()
        self.backbone = ResNetSpatialBackbone1Ch(pretrained, freeze_until)
        self.attn = nn.Conv2d(2048, 1, 1)
        self.softplus = nn.Softplus()
        self.proj = nn.Linear(2048, d_model)

    def forward(self, frames, face_mask_7x7=None):
        B,T,C,H,W = frames.shape
        fmap = self.backbone(frames.reshape(B*T, C, H, W))
        A = self.softplus(self.attn(fmap)) + 1e-6
        if face_mask_7x7 is None:
            M = torch.ones_like(A)
        else:
            M = face_mask_7x7.reshape(B*T, 1, 7, 7).to(A.device).clamp(0,1)
        Wm = A * M
        Wm = Wm / Wm.sum(dim=(2,3), keepdim=True).clamp_min(1e-6)
        pooled = (fmap * Wm).sum(dim=(2,3))
        return self.proj(pooled).view(B, T, -1)
