import torch
import torch.nn.functional as F

def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords**2) / (2*sigma*sigma))
    g = g / g.sum()
    w = g[:, None] @ g[None, :]
    return (w / w.sum()).view(1,1,window_size,window_size)

def ssim_1ch(x, y, window_size=11, C1=0.01**2, C2=0.03**2):
    w = _gaussian_window(window_size, 1.5, x.device, x.dtype)
    mu_x = F.conv2d(x, w, padding=window_size//2)
    mu_y = F.conv2d(y, w, padding=window_size//2)
    mu_x2, mu_y2, mu_xy = mu_x*mu_x, mu_y*mu_y, mu_x*mu_y
    sx2 = F.conv2d(x*x, w, padding=window_size//2) - mu_x2
    sy2 = F.conv2d(y*y, w, padding=window_size//2) - mu_y2
    sxy = F.conv2d(x*y, w, padding=window_size//2) - mu_xy
    ssim_map = ((2*mu_xy + C1)*(2*sxy + C2)) / ((mu_x2+mu_y2+C1)*(sx2+sy2+C2) + 1e-12)
    return ssim_map.mean()
