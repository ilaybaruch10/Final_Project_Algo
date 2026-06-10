import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet1ch import make_frozen_resnet50_feature_extractor, make_frozen_resnet50_trunk

class FrozenResNet50FrameEncoder(nn.Module):
    """
    Frozen built-in PyTorch pretrained ResNet-50 feature extractor.

    Input:
      x: (B,T,3,H,W) for the RR path.

    The built-in ResNet-50 is NOT modified and NOT fine-tuned.
    A fallback for 1-channel input still exists, but the RR path uses 3-channel input directly.

    Output:
      z: (B,T,d_model)
    """
    def __init__(self, d_model=512):
        super().__init__()
        self.backbone = make_frozen_resnet50_feature_extractor()
        self.proj = nn.Linear(2048, d_model)

    @staticmethod
    def to_3ch(x):
        # RR path should already be 3-channel and passes through unchanged.
        if x.shape[2] == 3:
            return x
        # Fallback only for optional modules / experiments.
        if x.shape[2] == 1:
            return x.repeat(1, 1, 3, 1, 1)
        raise ValueError(f"Expected 1 or 3 channels, got {x.shape[2]}")

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = self.to_3ch(x).reshape(B*T, 3, H, W)

        # ResNet is frozen; no gradients/activation storage through it.
        with torch.no_grad():
            feat = self.backbone(x)  # (B*T,2048)

        z = self.proj(feat).view(B, T, -1)
        return z


class FrozenResNet50SpatialBackbone(nn.Module):
    """
    Frozen built-in PyTorch pretrained ResNet-50 trunk up to layer4.

    Input:
      x: (N,1,H,W) or (N,3,H,W)

    If 1-channel, repeat to 3 channels.
    Output:
      fmap: (N,2048,7,7)
    """
    def __init__(self):
        super().__init__()
        self.trunk = make_frozen_resnet50_trunk()
        self.out_channels = 2048

    @staticmethod
    def to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        if x.shape[1] == 3:
            return x
        raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")

    def forward(self, x):
        x = self.to_3ch(x)
        with torch.no_grad():
            return self.trunk(x)


class FaceAwareFrozenResNet50Encoder(nn.Module):
    """
    Temp path encoder with frozen built-in pretrained ResNet-50 trunk
    and trainable face-aware attention pooling + projection.

    Input:
      frames: (B,T,1,H,W) or (B,T,3,H,W)
      face_mask: (B,T,1,h,w), optional. Resized to the ResNet feature-map grid
                 (H//32) if it doesn't already match.

    Output:
      z: (B,T,d_model)
    """
    def __init__(self, d_model=512):
        super().__init__()
        self.backbone = FrozenResNet50SpatialBackbone()
        self.attn = nn.Conv2d(2048, 1, 1)
        self.softplus = nn.Softplus()
        self.proj = nn.Linear(2048, d_model)

    def forward(self, frames, face_mask=None):
        B,T,C,H,W = frames.shape
        fmap = self.backbone(frames.reshape(B*T, C, H, W))  # (B*T,2048,h,w), h=w=H//32

        A = self.softplus(self.attn(fmap)) + 1e-6           # (B*T,1,h,w)

        if face_mask is None:
            M = torch.ones_like(A)
        else:
            M = face_mask.reshape(B*T, 1, *face_mask.shape[-2:]).to(A.device).clamp(0, 1)
            if M.shape[-2:] != A.shape[-2:]:                # defensive: match feature-map grid
                M = F.interpolate(M, size=A.shape[-2:], mode="nearest")

        Wm = A * M
        Wm = Wm / Wm.sum(dim=(2,3), keepdim=True).clamp_min(1e-6)

        pooled = (fmap * Wm).sum(dim=(2,3))                 # (B*T,2048)
        return self.proj(pooled).view(B, T, -1)
