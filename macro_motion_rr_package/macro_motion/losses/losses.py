import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssim import ssim_1ch

class UncertaintyWeightedRRTempLoss(nn.Module):
    def __init__(self, rr_loss="huber", huber_delta=2.0):
        super().__init__()
        self.rr_loss = rr_loss
        self.huber_delta = huber_delta
        self.logsig_rr = nn.Parameter(torch.tensor(0.0))
        self.logsig_temp = nn.Parameter(torch.tensor(0.0))

    def base_rr_loss(self, pred, target):
        if self.rr_loss == "huber":
            return F.huber_loss(pred, target, delta=self.huber_delta, reduction="none")
        if self.rr_loss == "mse":
            return F.mse_loss(pred, target, reduction="none")
        if self.rr_loss == "l1":
            return F.l1_loss(pred, target, reduction="none")
        raise ValueError(self.rr_loss)

    def forward(self, preds, rr, temp=None, temp_valid=None, w_temp=None):
        rr_loss_vec = self.base_rr_loss(preds["rr_pred"], rr)
        rr_unc = 0.5 * torch.exp(-self.logsig_rr) * rr_loss_vec + 0.5 * self.logsig_rr
        total = rr_unc.mean()
        terms = {"rr_loss": rr_loss_vec.mean().detach(), "rr_unc": rr_unc.mean().detach()}

        if "temp_pred" in preds and temp is not None:
            temp_loss_vec = F.mse_loss(preds["temp_pred"], temp, reduction="none")
            if temp_valid is None:
                temp_valid = torch.ones_like(temp_loss_vec)
            if w_temp is None:
                w_temp = torch.ones_like(temp_loss_vec)
            temp_unc = 0.5 * torch.exp(-self.logsig_temp) * temp_loss_vec + 0.5 * self.logsig_temp
            temp_unc = temp_unc * temp_valid * w_temp
            total = total + temp_unc.mean()
            terms["temp_loss"] = temp_loss_vec.mean().detach()
            terms["temp_unc"] = temp_unc.mean().detach()
        return total, terms

def unet_nir2ir_loss(ir_hat, ir_gt, mse_w=1.0, ssim_w=0.1):
    mse = F.mse_loss(ir_hat, ir_gt)
    ssim_loss = 1.0 - ssim_1ch(ir_hat, ir_gt)
    return mse_w*mse + ssim_w*ssim_loss, {"mse": mse.detach(), "ssim": (1.0-ssim_loss).detach()}
