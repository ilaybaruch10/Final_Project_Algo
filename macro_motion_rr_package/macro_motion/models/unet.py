import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNetNIR2IR(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base, base*2)
        self.d3 = DoubleConv(base*2, base*4)
        self.d4 = DoubleConv(base*4, base*8)
        self.b = DoubleConv(base*8, base*16)
        self.p = nn.MaxPool2d(2)
        self.u4 = nn.ConvTranspose2d(base*16, base*8, 2, 2)
        self.c4 = DoubleConv(base*16, base*8)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.c3 = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.c2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.c1 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p(d1))
        d3 = self.d3(self.p(d2))
        d4 = self.d4(self.p(d3))
        b = self.b(self.p(d4))
        x = self.c4(torch.cat([self.u4(b), d4], dim=1))
        x = self.c3(torch.cat([self.u3(x), d3], dim=1))
        x = self.c2(torch.cat([self.u2(x), d2], dim=1))
        x = self.c1(torch.cat([self.u1(x), d1], dim=1))
        return torch.sigmoid(self.out(x))
