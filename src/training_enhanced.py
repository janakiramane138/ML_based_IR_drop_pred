#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import torch.nn.functional as F

# robust local imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from data_generation import generate_maps, determine_grid_size
from ir_solver_sparse import solve_ir_drop
from data_augmentation import AugmentedIRDropDataset, SyntheticDataGenerator, CombinedDataset
from enhanced_loss import EnhancedHotspotLoss

# ---------------- Enhanced Dataset ----------------
class VariableSizeIRDropDataset(Dataset):
    def __init__(self, feature_dir):
        self.samples = []
        
        for fname in sorted(os.listdir(feature_dir)):
            if fname.startswith("current_map_") and fname.endswith(".csv"):
                base = fname[len("current_map_"):-4]
                paths = [os.path.join(feature_dir, pfx + base + ".csv")
                         for pfx in ("current_map_", "pdn_density_map_", "voltage_source_map_", "ir_drop_map_")]
                if all(os.path.exists(p) for p in paths):
                    self.samples.append(tuple(paths))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        c_path, d_path, v_path, y_path = self.samples[idx]
        cur  = np.loadtxt(c_path, delimiter=",")
        dens = np.loadtxt(d_path, delimiter=",")
        vsrc = np.loadtxt(v_path, delimiter=",")
        ir   = np.loadtxt(y_path, delimiter=",")

        # Keep original sizes - no resizing!
        # Normalize inputs and labels to stabilize training
        def norm(x):
            mx = np.max(np.abs(x))
            return x / (mx + 1e-12)

        X = np.stack([norm(cur), norm(dens), norm(vsrc)], axis=0)  # [3,H,W]
        
        # Normalize labels to 0-1 range to match input scale
        ir_max = np.max(ir) if np.max(ir) > 0 else 1.0
        Y = (ir / ir_max)[np.newaxis, ...]  # [1,H,W] (normalized to 0-1)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32), ir_max

# --------------- Enhanced Models ------------------
def get_enhanced_model(kind: str, variable_size=False):
    kind = (kind or "resunet").lower()
    if kind == "unet":
        from models.unet import UNet
        return UNet()
    elif kind == "variable_resunet" or variable_size:
        from models.resunet import VariableSizeResUNet
        return VariableSizeResUNet()  # No target_size parameter
    else:
        from models.resunet import ResUNet
        return ResUNet()

# --------------- Enhanced Visualization -----------
def save_enhanced_vis(epoch, batch_X, batch_Y, model, out_dir, max_samples=3, norm_factors=None):
    """
    Save enhanced visualization with multiple views.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        pred = model(batch_X).cpu().numpy()  # [B,1,H,W]
    true = batch_Y.cpu().numpy()

    os.makedirs(out_dir, exist_ok=True)
    B = min(pred.shape[0], max_samples)
    for i in range(B):
        p = pred[i,0]
        t = true[i,0]
        
        # Denormalize predictions and ground truth for visualization
        if norm_factors is not None and i < len(norm_factors):
            p = p * norm_factors[i]  # Convert back to original scale
            t = t * norm_factors[i]  # Convert back to original scale
        
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original heatmaps
        im0 = axes[0,0].imshow(p, cmap="hot")
        axes[0,0].set_title("Predicted IR Drop")
        fig.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.04)
        
        im1 = axes[0,1].imshow(t, cmap="hot")
        axes[0,1].set_title("Ground Truth IR Drop")
        fig.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04)
        
        # Hotspot masks
        max_true = np.max(t)
        hotspot_thr = 0.9 * max_true
        pred_hotspot = (p >= hotspot_thr).astype(float)
        true_hotspot = (t >= hotspot_thr).astype(float)
        
        axes[1,0].imshow(pred_hotspot, cmap="gray")
        axes[1,0].set_title("Predicted Hotspots")
        
        axes[1,1].imshow(true_hotspot, cmap="gray")
        axes[1,1].set_title("Ground Truth Hotspots")
        
        for ax in axes.flat: 
            ax.axis("off")
        
        fn = os.path.join(out_dir, f"epoch{epoch:03d}_sample{i}_enhanced.png")
        plt.tight_layout()
        plt.savefig(fn, dpi=160)
        plt.close(fig)

# --------------- Enhanced Training ----------------
def train_enhanced_model(input_dir, output_model, *,
                        model_name="resunet",
                        epochs=50, lr=3e-4, batch_size=8,
                        alpha=10.0, beta=5.0, gamma=2.0, focal_alpha=2.0,
                        use_augmentation=True, use_synthetic=True, num_synthetic=500,
                        augmentation_level="synthetic", variable_size=False,
                        viz_every=5, device="cpu"):

    device = torch.device(device)

    # Where to stash generated features from .sp/.voltage
    feature_dir = os.path.join(input_dir, "generated_features")
    os.makedirs(feature_dir, exist_ok=True)

    # 1) Ensure voltage + maps for each .sp with variable grid sizes
    grid_sizes = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".sp"):
            continue
        spice_path = os.path.join(input_dir, fname)
        base = os.path.splitext(fname)[0]
        voltage_path = os.path.join(feature_dir, f"{base}.voltage")
        
        if not os.path.exists(voltage_path):
            print(f"[INFO] Solving MNA for {fname} -> {voltage_path}")
            solve_ir_drop(spice_path, voltage_path)
        
        # Determine grid size for this netlist
        grid_size = determine_grid_size(spice_path)
        grid_sizes.append(grid_size)
        print(f"[INFO] Generating maps for {base} with grid size {grid_size}x{grid_size}")
        generate_maps(spice_path, voltage_path, feature_dir, grid_size)

    print(f"[INFO] Grid sizes range: {min(grid_sizes)}x{min(grid_sizes)} to {max(grid_sizes)}x{max(grid_sizes)}")

    # 2) Data with optional augmentation and synthetic data
    base_dataset = VariableSizeIRDropDataset(feature_dir)
    if len(base_dataset) == 0:
        raise RuntimeError(f"No training samples found in {feature_dir}.")
    
    print(f"[INFO] Found {len(base_dataset)} base samples")
    
    # Create synthetic data generator if requested
    synthetic_generator = None
    if use_synthetic and len(base_dataset) > 0:
        synthetic_generator = SyntheticDataGenerator(base_dataset, num_synthetic_samples=num_synthetic)
        print(f"[INFO] Generated {len(synthetic_generator)} synthetic samples")
    
    # Combine datasets
    if synthetic_generator:
        dataset = CombinedDataset(base_dataset, synthetic_generator)
        print(f"[INFO] Combined dataset size: {len(dataset)}")
    else:
        dataset = base_dataset
    
    # Apply augmentation if requested
    if use_augmentation:
        dataset = AugmentedIRDropDataset(dataset, augment=True, augmentation_level=augmentation_level)
        print(f"[INFO] Using {augmentation_level} data augmentation")
    
    # Create data loader with collate function to handle variable sizes
    def collate_fn(batch):
        # Find maximum dimensions in batch
        max_h = max([x[0].shape[1] for x in batch])
        max_w = max([x[0].shape[2] for x in batch])
        
        # Pad all samples to maximum size
        padded_batch = []
        for X, Y, norm_factor in batch:
            if X.shape[1] != max_h or X.shape[2] != max_w:
                # Pad to maximum size
                pad_h = max_h - X.shape[1]
                pad_w = max_w - X.shape[2]
                X = F.pad(X, (0, pad_w, 0, pad_h), mode='replicate')
                Y = F.pad(Y, (0, pad_w, 0, pad_h), mode='replicate')
            padded_batch.append((X, Y, norm_factor))
        
        # Stack into tensors
        X_batch = torch.stack([x[0] for x in padded_batch])
        Y_batch = torch.stack([x[1] for x in padded_batch])
        norm_factors = torch.tensor([x[2] for x in padded_batch])
        
        return X_batch, Y_batch, norm_factors
    
    def collate_pad_replicate(batch):
        """
        Batch samples of varying HxW by padding each to the max H/W in the batch.
        Pads symmetrically with 'replicate' to avoid reflect-size constraints.
        batch: list of (X:[C,H,W], Y:[1,H,W], norm)
        """
        Xs, Ys, norms = zip(*batch)

        # target size = per-batch max
        Htgt = max(x.shape[-2] for x in Xs)
        Wtgt = max(x.shape[-1] for x in Xs)

        Xp, Yp = [], []
        for X, Y in zip(Xs, Ys):
            # ensure float32 (saves memory)
            X = X.to(dtype=torch.float32)
            Y = Y.to(dtype=torch.float32)

            _, H, W = X.shape
            pad_top    = (Htgt - H) // 2
            pad_bottom = Htgt - H - pad_top
            pad_left   = (Wtgt - W) // 2
            pad_right  = Wtgt - W - pad_left

            if pad_top or pad_bottom or pad_left or pad_right:
                X = F.pad(X, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
                Y = F.pad(Y, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')

            Xp.append(X)
            Yp.append(Y)

        X = torch.stack(Xp, dim=0)                 # [B,C,Htgt,Wtgt]
        Y = torch.stack(Yp, dim=0)                 # [B,1,Htgt,Wtgt]
        norms = [n if torch.is_tensor(n) else torch.tensor(n, dtype=torch.float32) for n in norms]
        norms = torch.stack(norms, dim=0)
        return X, Y, norms

    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=1,                 # start small; raise later if stable
        shuffle=True,
        num_workers=0,                # WSL/multiprocessing + SciPy can explode RAM
        pin_memory=False,
        persistent_workers=False,
        #prefetch_factor=2,            #  # ← remove this line when num_workers=0
        drop_last=False,
        collate_fn=collate_pad_replicate
    )

    # 3) Model / Loss / Optimizer
    model = get_enhanced_model(model_name, variable_size=variable_size).to(device)
    criterion = EnhancedHotspotLoss(alpha=alpha, beta=beta, gamma=gamma, focal_alpha=focal_alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr/100
    )

    # 4) Train
    loss_hist, base_hist, hotspot_hist, focal_hist, dice_hist = [], [], [], [], []
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        base_sum, hotspot_sum, focal_sum, dice_sum = 0.0, 0.0, 0.0, 0.0

        for X, Y, norm_factors in loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss, parts = criterion(pred, Y)
            loss.backward()
            
            # Enhanced gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running += loss.item()
            base_sum += parts['base_mae']
            hotspot_sum += parts['hotspot_mae']
            focal_sum += parts['focal_loss']
            dice_sum += parts['dice_loss']

        scheduler.step()
        
        epoch_loss = running / len(loader)
        loss_hist.append(epoch_loss)
        base_hist.append(base_sum / len(loader))
        hotspot_hist.append(hotspot_sum / len(loader))
        focal_hist.append(focal_sum / len(loader))
        dice_hist.append(dice_sum / len(loader))
        
        print(f"Epoch {epoch:02d}/{epochs}  Loss={epoch_loss:.6f}  "
              f"(MAE={base_hist[-1]:.6f}  HotMAE={hotspot_hist[-1]:.6f}  "
              f"Focal={focal_hist[-1]:.6f}  Dice={dice_hist[-1]:.6f})")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), output_model.replace('.pt', '_best.pt'))
            print(f"[INFO] New best model saved: {output_model.replace('.pt', '_best.pt')}")

        # Enhanced visualization
        if viz_every and (epoch % viz_every == 0):
            X_s, Y_s, norm_factors = next(iter(loader))
            save_enhanced_vis(epoch, X_s.to(device), Y_s.to(device),
                             model, out_dir=os.path.join(feature_dir, "train_vis_enhanced"), 
                             norm_factors=norm_factors.numpy())

    # 5) Save final model + loss curves + normalization info
    torch.save(model.state_dict(), output_model)
    
    # Save normalization factors for inference
    norm_info = {
        "label_normalization": "per_sample_max",
        "input_normalization": "per_channel_max_abs",
        "output_activation": "relu",
        "enhanced_loss": True,
        "data_augmentation": use_augmentation,
        "synthetic_data": use_synthetic,
        "variable_size": variable_size,
        "grid_sizes": grid_sizes
    }
    with open(output_model.replace('.pt', '_norm_info.json'), 'w') as f:
        json.dump(norm_info, f, indent=2)
    
    # Save loss history
    loss_data = np.column_stack([loss_hist, base_hist, hotspot_hist, focal_hist, dice_hist])
    np.savetxt(os.path.join(feature_dir, "enhanced_training_loss.csv"), 
               loss_data, delimiter=",", 
               header="total_loss,base_mae,hotspot_mae,focal_loss,dice_loss")
    
    # Enhanced loss plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,6))
        plt.plot(loss_hist, label="Total Loss", linewidth=2)
        plt.plot(base_hist, label="Base MAE", alpha=0.7)
        plt.plot(hotspot_hist, label="Hotspot MAE", alpha=0.7)
        plt.plot(focal_hist, label="Focal Loss", alpha=0.7)
        plt.plot(dice_hist, label="Dice Loss", alpha=0.7)
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("Enhanced Training Loss Components")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(feature_dir, "enhanced_training_loss.png"), dpi=160)
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not save loss plot: {e}")
    
    print(f"[OK] Enhanced model saved: {output_model}")
    print(f"[OK] Best model saved: {output_model.replace('.pt', '_best.pt')}")
    print(f"[OK] Artifacts in: {feature_dir}")

# --------------- CLI ---------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train enhanced ML model for IR-drop prediction with better hotspot detection")
    ap.add_argument("-input", required=True, help="Directory containing training .sp files")
    ap.add_argument("-output", required=True, help="Path to save trained model (.pt)")
    ap.add_argument("--model", default="resunet", choices=["unet","resunet","variable_resunet"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=10.0, help="Weight for hotspot MAE")
    ap.add_argument("--beta", type=float, default=5.0, help="Weight for focal loss (false positive penalty)")
    ap.add_argument("--gamma", type=float, default=2.0, help="Weight for dice loss")
    ap.add_argument("--focal_alpha", type=float, default=2.0, help="Focal loss alpha")
    ap.add_argument("--no_augment", action="store_true", help="Disable data augmentation")
    ap.add_argument("--no_synthetic", action="store_true", help="Disable synthetic data generation")
    ap.add_argument("--num_synthetic", type=int, default=500, help="Number of synthetic samples to generate")
    ap.add_argument("--augmentation_level", default="synthetic", 
                    choices=["standard", "aggressive", "synthetic"], help="Augmentation level")
    ap.add_argument("--variable_size", action="store_true", help="Use variable size model")
    ap.add_argument("--viz_every", type=int, default=5, help="Save vis every N epochs (0=off)")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    args = ap.parse_args()

    train_enhanced_model(args.input, args.output,
                        model_name=args.model, epochs=args.epochs, lr=args.lr,
                        batch_size=args.batch, alpha=args.alpha, beta=args.beta,
                        gamma=args.gamma, focal_alpha=args.focal_alpha,
                        use_augmentation=not args.no_augment,
                        use_synthetic=not args.no_synthetic,
                        num_synthetic=args.num_synthetic,
                        augmentation_level=args.augmentation_level,
                        variable_size=args.variable_size,
                        viz_every=args.viz_every, device=args.device)
