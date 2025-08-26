import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """
    Attention mechanism to focus on hotspot regions.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Generate attention maps
        query = self.conv1(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # B, HW, C//8
        key = self.conv2(x).view(batch_size, -1, H * W)  # B, C//8, HW
        value = self.conv3(x).view(batch_size, -1, H * W)  # B, C, HW
        
        # Compute attention weights
        attention = torch.bmm(query, key)  # B, HW, HW
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B, C, HW
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))

class ResUNetAttention(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        # Encoder
        self.enc1 = ResidualBlock(in_ch, 16)
        self.att1 = AttentionBlock(16)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ResidualBlock(16, 32)
        self.att2 = AttentionBlock(32)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ResidualBlock(32, 64)
        self.att3 = AttentionBlock(64)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(64, 128)
        self.att_bottleneck = AttentionBlock(128)

        # Decoder
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ResidualBlock(128, 64)
        self.att_dec3 = AttentionBlock(64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = ResidualBlock(64, 32)
        self.att_dec2 = AttentionBlock(32)
        
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = ResidualBlock(32, 16)
        self.att_dec1 = AttentionBlock(16)

        self.final_conv = nn.Conv2d(16, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e1 = self.att1(e1)
        e2 = self.enc2(self.pool1(e1))
        e2 = self.att2(e2)
        e3 = self.enc3(self.pool2(e2))
        e3 = self.att3(e3)

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        b = self.att_bottleneck(b)

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d3 = self.att_dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d2 = self.att_dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        d1 = self.att_dec1(d1)

        # Add ReLU activation to ensure non-negative output (IR drop >= 0)
        return torch.relu(self.final_conv(d1))
