import torch
import torch.nn as nn
import torch.nn.functional as F
from .frame_encoders import ResNetFrameEncoder1Ch, FaceAwareResNetEncoder1Ch
from .transformer import SharedMacroTransformer
from .unet import UNetNIR2IR

class MacroMotionModel(nn.Module):
    def __init__(self, d_model=512, n_heads=8, depth=6, dropout=0.1, ff_mult=4,
                 max_len=700, temporal_downsample=3, pretrained_resnet=True,
                 freeze_resnet_until="", include_temp_path=False, unet_base=32):
        super().__init__()
        self.temporal_downsample = temporal_downsample
        self.include_temp_path = include_temp_path
        self.rr_encoder = ResNetFrameEncoder1Ch(d_model, pretrained_resnet, freeze_resnet_until)

        if include_temp_path:
            self.unet = UNetNIR2IR(in_ch=3, out_ch=1, base=unet_base)
            self.temp_encoder = FaceAwareResNetEncoder1Ch(d_model, pretrained_resnet, freeze_resnet_until)
            self.temp_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        else:
            self.unet = None
            self.temp_encoder = None
            self.temp_head = None

        self.transformer = SharedMacroTransformer(d_model, n_heads, depth, dropout, ff_mult, max_len)
        self.rr_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    @staticmethod
    def nir_to_gray(x):
        return x.mean(dim=2, keepdim=True)

    def _downsample_time(self, x):
        if self.temporal_downsample is None or self.temporal_downsample <= 1:
            return x
        return x[:, ::self.temporal_downsample]

    def _unet_frames(self, x):
        B,T,C,H,W = x.shape
        y = self.unet(x.reshape(B*T, C, H, W))
        return y.view(B, T, 1, H, W)

    def _mask_to_7x7(self, mask, target_T):
        if mask is None:
            return None
        mask = self._downsample_time(mask)
        mask = mask[:, :target_T]
        B,T,C,H,W = mask.shape
        m = F.interpolate(mask.reshape(B*T,1,H,W), size=(7,7), mode="nearest")
        return m.view(B,T,1,7,7)

    def forward(self, clip, cond_id, face_mask=None, unet_no_grad=True):
        # RR path
        x_rr = self._downsample_time(clip)
        gray = self.nir_to_gray(x_rr)
        z_rr = self.rr_encoder(gray)
        h_rr = self.transformer.forward_once(
            z_rr, domain_id=SharedMacroTransformer.DOMAIN_RR, cond_id=cond_id
        )
        out = {"rr_pred": self.rr_head(h_rr).squeeze(-1)}

        # Optional Temp path
        if self.include_temp_path:
            x_temp = self._downsample_time(clip)
            if unet_no_grad:
                with torch.no_grad():
                    ir_hat = self._unet_frames(x_temp)
            else:
                ir_hat = self._unet_frames(x_temp)
            mask_7 = self._mask_to_7x7(face_mask, target_T=ir_hat.shape[1])
            z_temp = self.temp_encoder(ir_hat, face_mask_7x7=mask_7)
            h_temp = self.transformer.forward_once(
                z_temp, domain_id=SharedMacroTransformer.DOMAIN_TEMP, cond_id=cond_id
            )
            out["temp_pred"] = self.temp_head(h_temp).squeeze(-1)
            out["ir_hat"] = ir_hat
        return out
