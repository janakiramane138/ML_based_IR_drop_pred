#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
import random
from collections import Counter

def augment_features(X, Y, p=0.5, augmentation_level="standard"):
    """
    Apply data augmentation to input features and labels.
    
    Args:
        X: Input features [B, 3, H, W]
        Y: Labels [B, 1, H, W]
        p: Probability of applying each augmentation
        augmentation_level: "standard", "aggressive", or "synthetic"
    
    Returns:
        Augmented X and Y
    """
    B, C, H, W = X.shape
    
    # Standard augmentations
    if augmentation_level in ["standard", "aggressive", "synthetic"]:
        # Random horizontal flip
        if np.random.random() < p:
            X = torch.flip(X, dims=[3])  # Flip width
            Y = torch.flip(Y, dims=[3])
        
        # Random vertical flip
        if np.random.random() < p:
            X = torch.flip(X, dims=[2])  # Flip height
            Y = torch.flip(Y, dims=[2])
        
        # Random rotation (90, 180, 270 degrees)
        if np.random.random() < p:
            k = np.random.choice([1, 2, 3])  # 90, 180, 270 degrees
            X = torch.rot90(X, k, dims=[2, 3])
            Y = torch.rot90(Y, k, dims=[2, 3])
        
        # Random brightness/contrast adjustment (for current and density maps)
        if np.random.random() < p:
            # Apply to current and density maps (channels 0 and 1)
            brightness = 0.8 + 0.4 * np.random.random()  # 0.8 to 1.2
            X[:, :2] = X[:, :2] * brightness
            X = torch.clamp(X, 0, 1)
        
        # Random noise injection
        if np.random.random() < p:
            noise_level = 0.02 * np.random.random()  # 0 to 0.02
            noise = torch.randn_like(X) * noise_level
            X = X + noise
            X = torch.clamp(X, 0, 1)
    
    # Aggressive augmentations
    if augmentation_level in ["aggressive", "synthetic"]:
        # Elastic deformation
        #if np.random.random() < p * 0.5:
        #    X, Y = elastic_deformation(X, Y, alpha=1, sigma=50)
        
        # Random scaling
        if np.random.random() < p * 0.5:
            scale_factor = 0.8 + 0.4 * np.random.random()  # 0.8 to 1.2
            X, Y = random_scale(X, Y, scale_factor)
        
        # Random cropping and padding
        if np.random.random() < p * 0.5:
            X, Y = random_crop_pad(X, Y, crop_ratio=0.8)
        
        # Gaussian blur (for current and density maps)
        #if np.random.random() < p * 0.3:
        #    X = apply_gaussian_blur(X, sigma=0.5 + np.random.random())
    
    # Synthetic augmentations
    if augmentation_level == "synthetic":
        # Hotspot-aware augmentation
        if np.random.random() < p * 0.7:
            X, Y = hotspot_aware_augmentation(X, Y)
        
        # Synthetic hotspot generation
        if np.random.random() < p * 0.3:
            X, Y = synthetic_hotspot_generation(X, Y)
        
        # PDN density variation
        if np.random.random() < p * 0.5:
            X = pdn_density_variation(X)
    
    return X, Y

def elastic_deformation(X, Y, alpha=1, sigma=50):
    """Apply elastic deformation to both input and output."""
    B, C, H, W = X.shape
    
    # Create random displacement fields
    dx = np.random.randn(H, W) * alpha
    dy = np.random.randn(H, W) * alpha
    
    # Smooth the displacement fields
    dx = ndimage.gaussian_filter(dx, sigma)
    dy = ndimage.gaussian_filter(dy, sigma)
    
    # Normalize displacement
    dx = dx * sigma / np.max(np.abs(dx)) if np.max(np.abs(dx)) > 0 else dx
    dy = dy * sigma / np.max(np.abs(dy)) if np.max(np.abs(dy)) > 0 else dy
    
    # Apply deformation
    X_deformed = torch.zeros_like(X)
    Y_deformed = torch.zeros_like(Y)
    
    # precompute coordinate grids once
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    map_y = (yy + dy).astype(np.float32)
    map_x = (xx + dx).astype(np.float32)

    for b in range(B):
        for c in range(C):
            x_np = X[b, c].detach().cpu().numpy()
            out = ndimage.map_coordinates(x_np, [map_y, map_x], order=1, mode='nearest')
            X_deformed[b, c] = torch.from_numpy(out).to(X.device, X.dtype)

        y_np = Y[b, 0].detach().cpu().numpy()
        out = ndimage.map_coordinates(y_np, [map_y, map_x], order=1, mode='nearest')
        Y_deformed[b, 0] = torch.from_numpy(out).to(Y.device, Y.dtype)

    return X_deformed, Y_deformed

def random_scale(X, Y, scale_factor):
    """Resize to scale_factor, then center crop/pad back to original (H,W)."""
    B, C, H, W = X.shape
    new_H = max(1, int(round(H * scale_factor)))
    new_W = max(1, int(round(W * scale_factor)))

    # 1) Resize whole batch
    Xr = F.interpolate(X, size=(new_H, new_W), mode='bilinear', align_corners=False)
    # If Y is a **continuous** IR-drop map, bilinear is fine; if it's a class mask, use mode='nearest'
    Yr = F.interpolate(Y, size=(new_H, new_W), mode='bilinear', align_corners=False)

    if new_H == H and new_W == W:
        return Xr, Yr

    # 2) Center pad or center crop to (H, W)
    if new_H < H or new_W < W:
        # pad to target using a mode that allows large pads
        pad_top    = (H - new_H) // 2
        pad_bottom = H - new_H - pad_top
        pad_left   = (W - new_W) // 2
        pad_right  = W - new_W - pad_left
        Xo = F.pad(Xr, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        Yo = F.pad(Yr, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
    else:
        # crop to target
        start_h = (new_H - H) // 2
        start_w = (new_W - W) // 2
        Xo = Xr[:, :, start_h:start_h+H, start_w:start_w+W]
        Yo = Yr[:, :, start_h:start_h+H, start_w:start_w+W]

    return Xo, Yo

def random_crop_pad(X, Y, crop_ratio=0.8):
    """Random crop and pad to maintain size."""
    B, C, H, W = X.shape
    
    crop_H = int(H * crop_ratio)
    crop_W = int(W * crop_ratio)
    
    # Random crop position
    start_h = np.random.randint(0, H - crop_H + 1)
    start_w = np.random.randint(0, W - crop_W + 1)
    
    X_cropped = X[:, :, start_h:start_h+crop_H, start_w:start_w+crop_W]
    Y_cropped = Y[:, :, start_h:start_h+crop_H, start_w:start_w+crop_W]
    
    # Pad back to original size
    pad_h = (H - crop_H) // 2
    pad_w = (W - crop_W) // 2
    
    X_padded = F.pad(X_cropped, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
    Y_padded = F.pad(Y_cropped, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
    
    return X_padded, Y_padded

def apply_gaussian_blur(X, sigma=1.0):
    """Apply Gaussian blur to current and density maps."""
    B, C, H, W = X.shape
    
    # Create Gaussian kernel
    kernel_size = max(3, int(sigma * 6))
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    kernel = torch.exp(-torch.arange(-(kernel_size//2), kernel_size//2+1)**2 / (2*sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1, 1).repeat(1, 1, 1, kernel_size)
    
    X_blurred = X.clone()
    
    # Apply blur to current and density maps (channels 0 and 1)
    for b in range(B):
        for c in [0, 1]:  # Only blur current and density maps
            X_blurred[b, c] = F.conv2d(
                X[b:b+1, c:c+1], kernel, padding=kernel_size//2
            ).squeeze(0, 1)
    
    return X_blurred

def hotspot_aware_augmentation(X, Y):
    """Apply augmentations that preserve hotspot characteristics."""
    B, C, H, W = X.shape
    
    # Identify hotspots in ground truth
    hotspot_masks = []
    for b in range(B):
        y_max = torch.max(Y[b, 0])
        hotspot_threshold = 0.8 * y_max
        hotspot_mask = (Y[b, 0] >= hotspot_threshold).float()
        hotspot_masks.append(hotspot_mask)
    
    # Apply hotspot-preserving transformations
    for b in range(B):
        if hotspot_masks[b].sum() > 0:
            # Enhance current density around hotspots
            hotspot_region = hotspot_masks[b].unsqueeze(0).unsqueeze(0)
            enhancement_factor = 1.2 + 0.3 * np.random.random()  # 1.2 to 1.5
            
            # Enhance current map in hotspot regions
            X[b, 0] = X[b, 0] * (1 + (enhancement_factor - 1) * hotspot_region.squeeze())
            
            # Enhance PDN density in hotspot regions
            X[b, 1] = X[b, 1] * (1 + (enhancement_factor - 1) * hotspot_region.squeeze())
    
    return X, Y

def synthetic_hotspot_generation(X, Y):
    """Generate synthetic hotspots based on current and density patterns."""
    B, C, H, W = X.shape
    
    for b in range(B):
        # Create synthetic hotspot based on current density
        current_map = X[b, 0].numpy()
        density_map = X[b, 1].numpy()
        
        # Find regions with high current and density
        current_threshold = np.percentile(current_map, 85)
        density_threshold = np.percentile(density_map, 85)
        
        hotspot_candidates = (current_map > current_threshold) & (density_map > density_threshold)
        
        if hotspot_candidates.sum() > 0:
            # Create synthetic hotspot
            hotspot_intensity = 0.3 + 0.4 * np.random.random()  # 0.3 to 0.7
            hotspot_size = 3 + int(5 * np.random.random())  # 3 to 8 pixels
            
            # Find center of hotspot candidate region
            candidate_coords = np.where(hotspot_candidates)
            if len(candidate_coords[0]) > 0:
                center_idx = np.random.randint(len(candidate_coords[0]))
                center_y, center_x = candidate_coords[0][center_idx], candidate_coords[1][center_idx]
                
                # Create Gaussian hotspot
                y_coords, x_coords = np.ogrid[:H, :W]
                distance = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
                synthetic_hotspot = np.exp(-distance**2 / (2 * hotspot_size**2))
                
                # Add to IR drop map
                print(Y[b, 0].shape)
                print(synthetic_hotspot.shape)
                #Y[b, 0] = torch.tensor(Y[b, 0] + hotspot_intensity * synthetic_hotspot)
                hotspot = torch.from_numpy(synthetic_hotspot).to(Y.device, Y.dtype)
                Y[b, 0].add_(hotspot_intensity * hotspot).clamp_(0, 1)
                #Y[b, 0] = torch.clamp(Y[b, 0], 0, 1)
    
    return X, Y

def pdn_density_variation(X):
    """Apply variations to PDN density map."""
    B, C, H, W = X.shape
    
    for b in range(B):
        # Add random density variations
        density_variation = torch.randn(H, W) * 0.1  # 10% variation
        X[b, 1] = torch.clamp(X[b, 1] + density_variation, 0, 1)
        
        # Add density gradients
        if np.random.random() < 0.3:
            gradient_direction = np.random.choice(['horizontal', 'vertical', 'diagonal'])
            if gradient_direction == 'horizontal':
                gradient = torch.linspace(0, 1, W).unsqueeze(0).repeat(H, 1)
            elif gradient_direction == 'vertical':
                gradient = torch.linspace(0, 1, H).unsqueeze(1).repeat(1, W)
            else:  # diagonal
                gradient = torch.zeros(H, W)
                for i in range(H):
                    for j in range(W):
                        gradient[i, j] = (i + j) / (H + W)
            
            gradient_factor = 0.1 * np.random.random()  # 0 to 0.1
            X[b, 1] = torch.clamp(X[b, 1] + gradient_factor * gradient, 0, 1)
    
    return X

class AugmentedIRDropDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that applies augmentation during training.
    """
    def __init__(self, base_dataset, augment=True, augmentation_level="standard"):
        self.base_dataset = base_dataset
        self.augment = augment
        self.augmentation_level = augmentation_level
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        X, Y, norm_factor = self.base_dataset[idx]
        
        if self.augment:
            X = X.unsqueeze(0)  # Add batch dimension
            Y = Y.unsqueeze(0)
            X, Y = augment_features(X, Y, p=0.7, augmentation_level=self.augmentation_level)
            X = X.squeeze(0)  # Remove batch dimension
            Y = Y.squeeze(0)
        
        return X, Y, norm_factor

class SyntheticDataGenerator:
    """
    Generate synthetic training data by combining and permuting existing samples.
    """
    def __init__(self, base_dataset, num_synthetic_samples=1000):
        self.base_dataset = base_dataset
        self.num_synthetic_samples = num_synthetic_samples
        self.synthetic_samples = []
        self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic samples by combining existing ones."""
        print(f"[INFO] Generating {self.num_synthetic_samples} synthetic samples...")
        
        for i in range(self.num_synthetic_samples):
            # Show progress every 50 samples
            if i % 50 == 0:
                print(f"[INFO] Generated {i}/{self.num_synthetic_samples} synthetic samples...")
            
            # Randomly select 2-3 base samples to combine
            num_samples = np.random.randint(2, 4)
            selected_indices = np.random.choice(len(self.base_dataset), num_samples, replace=False)
            
            # Load selected samples (keep original sizes)
            samples = []
            for idx in selected_indices:
                X, Y, norm_factor = self.base_dataset[idx]
                samples.append((X, Y, norm_factor))
            
            # Find the largest size among selected samples for combination
            max_h = max([X.shape[1] for X, _, _ in samples])
            max_w = max([X.shape[2] for X, _, _ in samples])
            
            # Combine samples using weighted averaging with padding to max size
            combined_X = torch.zeros(3, max_h, max_w)
            combined_Y = torch.zeros(1, max_h, max_w)
            combined_norm = 0
            
            weights = torch.softmax(torch.randn(num_samples), dim=0)
            
            for j, (X, Y, norm_factor) in enumerate(samples):
                # Pad smaller samples to max size
                if X.shape[1] < max_h or X.shape[2] < max_w:
                    pad_h = max_h - X.shape[1]
                    pad_w = max_w - X.shape[2]
                    X_padded = F.pad(X, (0, pad_w, 0, pad_h), mode='replicate')
                    Y_padded = F.pad(Y, (0, pad_w, 0, pad_h), mode='replicate')
                else:
                    X_padded = X
                    Y_padded = Y
                
                combined_X += weights[j] * X_padded
                combined_Y += weights[j] * Y_padded
                combined_norm += weights[j] * norm_factor
            
            # Apply additional transformations (simplified to avoid size issues)
            combined_X = combined_X.unsqueeze(0)
            combined_Y = combined_Y.unsqueeze(0)
            
            # Apply only safe augmentations that don't change size
            if np.random.random() < 0.8:
                # Random horizontal flip
                if np.random.random() < 0.5:
                    combined_X = torch.flip(combined_X, dims=[3])
                    combined_Y = torch.flip(combined_Y, dims=[3])
                
                # Random vertical flip
                if np.random.random() < 0.5:
                    combined_X = torch.flip(combined_X, dims=[2])
                    combined_Y = torch.flip(combined_Y, dims=[2])
                
                # Random rotation (90, 180, 270 degrees)
                if np.random.random() < 0.5:
                    k = np.random.choice([1, 2, 3])
                    combined_X = torch.rot90(combined_X, k, dims=[2, 3])
                    combined_Y = torch.rot90(combined_Y, k, dims=[2, 3])
                
                # Brightness adjustment
                if np.random.random() < 0.5:
                    brightness = 0.8 + 0.4 * np.random.random()
                    combined_X[:, :2] = combined_X[:, :2] * brightness
                    combined_X = torch.clamp(combined_X, 0, 1)
                
                # Noise injection
                if np.random.random() < 0.3:
                    noise_level = 0.02 * np.random.random()
                    noise = torch.randn_like(combined_X) * noise_level
                    combined_X = combined_X + noise
                    combined_X = torch.clamp(combined_X, 0, 1)
            
            combined_X = combined_X.squeeze(0)
            combined_Y = combined_Y.squeeze(0)
            
            self.synthetic_samples.append((combined_X, combined_Y, combined_norm))
        
        print(f"[INFO] Generated {len(self.synthetic_samples)} synthetic samples")
    
    def __len__(self):
        return len(self.synthetic_samples)
    
    def __getitem__(self, idx):
        return self.synthetic_samples[idx]

class CombinedDataset(torch.utils.data.Dataset):
    """
    Combine original and synthetic datasets.
    """
    def __init__(self, base_dataset, synthetic_generator=None):
        self.base_dataset = base_dataset
        self.synthetic_generator = synthetic_generator
    
    def __len__(self):
        base_len = len(self.base_dataset)
        synthetic_len = len(self.synthetic_generator) if self.synthetic_generator else 0
        return base_len + synthetic_len
    
    def __getitem__(self, idx):
        if idx < len(self.base_dataset):
            return self.base_dataset[idx]
        else:
            return self.synthetic_generator[idx - len(self.base_dataset)]
