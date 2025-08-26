import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

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

class AttentionBlock(nn.Module):
    """
    Attention mechanism for better hotspot detection.
    """
    def __init__(self, in_ch):
        super().__init__()
        c_ = max(1, in_ch // 8)
        self.conv1 = nn.Conv2d(in_ch, c_, kernel_size=1)
        self.conv2 = nn.Conv2d(in_ch, c_, kernel_size=1)
        self.conv3 = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Generate Q, K, V
        query = self.conv1(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # B x HW x C'
        key = self.conv2(x).view(batch_size, -1, H * W)  # B x C' x HW
        value = self.conv3(x).view(batch_size, -1, H * W)  # B x C x HW
        
        # Attention weights
        attention = torch.bmm(query, key)  # B x HW x HW
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x HW
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x

class HotspotDetectionBlock(nn.Module):
    """
    Specialized block for hotspot detection using multi-scale analysis.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=3)
        
        self.attention = AttentionBlock(out_ch * 3)
        self.fusion = nn.Conv2d(out_ch * 3, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Multi-scale feature extraction
        f1 = self.conv1(x)
        f2 = self.conv2(x)
        f3 = self.conv3(x)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([f1, f2, f3], dim=1)
        
        # Apply attention
        attended = self.attention(multi_scale)
        
        # Fusion
        out = self.fusion(attended)
        out = self.bn(out)
        out = self.relu(out)
        
        return out

class AdaptivePooling(nn.Module):
    """
    Adaptive pooling that works with variable input sizes.
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, self.output_size)

class ResUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_channels=16):
        super().__init__()
        self.base_channels = base_channels
        
        # Encoder
        self.enc1 = ResidualBlock(in_ch, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck with hotspot detection
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 8)
        self.hotspot_detector = HotspotDetectionBlock(base_channels * 8, base_channels * 8)

        # Decoder with attention
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 4)
        self.att3 = AttentionBlock(base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels * 2)
        self.att2 = AttentionBlock(base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels)
        self.att1 = AttentionBlock(base_channels)

        # Final layers
        self.final_conv = nn.Conv2d(base_channels, out_ch, kernel_size=1)
        
        # Hotspot refinement
        self.hotspot_refinement = nn.Sequential(
            nn.Conv2d(out_ch, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_ch, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck with hotspot detection
        b = self.bottleneck(self.pool3(e3))
        hotspot_features = self.hotspot_detector(b)

        # Decoder with attention
        d3 = self.up3(hotspot_features)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d3 = self.att3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d2 = self.att2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        d1 = self.att1(d1)

        # Final prediction
        out = self.final_conv(d1)
        
        # Hotspot refinement
        refined = self.hotspot_refinement(out)
        out = out + 0.1 * refined  # Residual connection for refinement
        
        # Ensure non-negative output (IR drop >= 0)
        return torch.relu(out)


class SafeAttention(nn.Module):
    def __init__(self, attn: nn.Module, max_tokens: int = 128 * 128):
        super().__init__()
        self.attn = attn
        self.max_tokens = max_tokens
    def forward(self, x):
        H, W = x.shape[-2:]
        if H * W > self.max_tokens:
            return x  # skip attention at high resolution
        return self.attn(x)




class VariableSizeResUNet(nn.Module):
    """
    ResUNet variant that can handle variable input sizes by using adaptive pooling.
    """
    def __init__(self, in_ch=3, out_ch=1, base_channels=16):
        super().__init__()
        self.base_channels = base_channels
        
        # Encoder
        self.enc1 = ResidualBlock(in_ch, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck with adaptive pooling
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 8)
        self.hotspot_detector = HotspotDetectionBlock(base_channels * 8, base_channels * 8)

        # Decoder with adaptive upsampling
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 4)
        #self.att3 = AttentionBlock(base_channels * 4)
        self.att3 = SafeAttention(AttentionBlock(base_channels * 4), max_tokens=128*128)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels * 2)
        #self.att2 = AttentionBlock(base_channels * 2)
        self.att2 = SafeAttention(AttentionBlock(base_channels * 2), max_tokens=128*128)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels)
        #self.att1 = AttentionBlock(base_channels)
        self.att1 = SafeAttention(AttentionBlock(base_channels), max_tokens=128*128)

        # Final layers
        self.final_conv = nn.Conv2d(base_channels, out_ch, kernel_size=1)
        self.hotspot_refinement = nn.Sequential(
            nn.Conv2d(out_ch, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_ch, kernel_size=1)
        )

    def forward(self, x):
        # Store original size for final upsampling
        original_size = x.shape[2:]
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck - no fixed size constraint
        b = self.bottleneck(self.pool3(e3))
        hotspot_features = self.hotspot_detector(b)

        # Decoder with adaptive skip connections
        d3 = self.up3(hotspot_features)
        # Adaptive resize for skip connection
        e3_resized = F.adaptive_avg_pool2d(e3, d3.shape[2:])
        d3 = torch.cat([d3, e3_resized], dim=1)
        d3 = self.dec3(d3)
        d3 = self.att3(d3)
        
        d2 = self.up2(d3)
        e2_resized = F.adaptive_avg_pool2d(e2, d2.shape[2:])
        d2 = torch.cat([d2, e2_resized], dim=1)
        d2 = self.dec2(d2)
        d2 = self.att2(d2)
        
        d1 = self.up1(d2)
        e1_resized = F.adaptive_avg_pool2d(e1, d1.shape[2:])
        d1 = torch.cat([d1, e1_resized], dim=1)
        d1 = self.dec1(d1)
        d1 = self.att1(d1)

        # Final prediction
        out = self.final_conv(d1)
        
        # Hotspot refinement
        refined = self.hotspot_refinement(out)
        out = out + 0.1 * refined
        
        # Resize to original input size
        out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
        
        # Ensure non-negative output
        return torch.relu(out)

def get_model(kind: str, variable_size=False):
    """
    Get model instance with specified configuration.
    
    Args:
        kind: Model type ("resunet", "unet", "variable_resunet")
        variable_size: Whether to use variable size model
    """
    kind = (kind or "resunet").lower()
    
    if kind == "unet":
        from models.unet import UNet
        return UNet()
    elif kind == "variable_resunet" or variable_size:
        return VariableSizeResUNet()  # No target_size parameter
    else:
        return ResUNet()

