#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedHotspotLoss(nn.Module):
    """
    Enhanced loss function specifically designed to address false positives
    and improve hotspot detection.
    """
    def __init__(self, alpha=10.0, beta=5.0, gamma=2.0, focal_alpha=2.0):
        super().__init__()
        self.alpha = float(alpha)  # hotspot MAE weight
        self.beta = float(beta)    # false positive penalty weight
        self.gamma = float(gamma)  # dice loss weight
        self.focal_alpha = float(focal_alpha)  # focal loss alpha
        self.l1 = nn.L1Loss()

    def forward(self, pred, true):
        # Base MAE loss
        base_mae = self.l1(pred, true)
        
        with torch.no_grad():
            # Create hotspot and non-hotspot masks
            max_true = torch.amax(true, dim=(2,3), keepdim=True)  # [B,1,1,1]
            hotspot_thr = 0.9 * max_true
            nonhotspot_thr = 0.1 * max_true
            
            hotspot_mask = (true >= hotspot_thr)  # [B,1,H,W]
            nonhotspot_mask = (true <= nonhotspot_thr)  # [B,1,H,W]
            
            # Create binary masks for dice loss
            hotspot_binary = hotspot_mask.float()
            pred_hotspot_binary = (pred >= hotspot_thr).float()

        # 1. Hotspot MAE loss (higher weight on hotspots)
        if hotspot_mask.any():
            hotspot_mae = torch.abs(pred - true) * hotspot_mask
            hotspot_mae = hotspot_mae.sum(dim=(2,3)) / hotspot_mask.sum(dim=(2,3)).clamp_min(1)
            hotspot_mae = hotspot_mae.mean()
        else:
            hotspot_mae = torch.tensor(0.0, device=pred.device)

        # 2. False Positive Penalty (FOCAL LOSS on non-hotspot regions)
        if nonhotspot_mask.any():
            # Penalize high predictions in non-hotspot regions
            nonhotspot_pred = pred * nonhotspot_mask
            # Focal loss: penalize more as prediction gets higher
            focal_loss = torch.pow(nonhotspot_pred, self.focal_alpha)
            focal_loss = focal_loss.sum(dim=(2,3)) / nonhotspot_mask.sum(dim=(2,3)).clamp_min(1)
            focal_loss = focal_loss.mean()
        else:
            focal_loss = torch.tensor(0.0, device=pred.device)

        # 3. Dice loss for hotspot segmentation
        if hotspot_mask.any():
            intersection = (pred_hotspot_binary * hotspot_binary).sum(dim=(2,3))
            union = pred_hotspot_binary.sum(dim=(2,3)) + hotspot_binary.sum(dim=(2,3))
            dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
            dice_loss = dice_loss.mean()
        else:
            dice_loss = torch.tensor(0.0, device=pred.device)

        total_loss = base_mae + self.alpha * hotspot_mae + self.beta * focal_loss + self.gamma * dice_loss
        
        return total_loss, {
            'base_mae': base_mae.item(),
            'hotspot_mae': hotspot_mae.item(),
            'focal_loss': focal_loss.item(),
            'dice_loss': dice_loss.item()
        }
