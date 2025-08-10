# src/models/unet.py
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        # Output
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # [B, 32, H,   W]
        e2 = self.enc2(self.pool1(e1))  # [B, 64, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))  # [B,128, H/4, W/4]

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))  # [B,256, H/8, W/8]

        # Decoder with skip connections
        d3 = self.up3(b)                    # [B,128, H/4, W/4]
        d3 = torch.cat([d3, e3], dim=1)     # [B,256, H/4, W/4]
        d3 = self.dec3(d3)                  # [B,128, H/4, W/4]

        d2 = self.up2(d3)                   # [B, 64, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)     # [B,128, H/2, W/2]
        d2 = self.dec2(d2)                  # [B, 64, H/2, W/2]

        d1 = self.up1(d2)                   # [B, 32, H,   W]
        d1 = torch.cat([d1, e1], dim=1)     # [B, 64, H,   W]
        d1 = self.dec1(d1)                  # [B, 32, H,   W]

        return self.out(d1)                 # [B, 1,  H,   W]

