#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json

# robust local imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from data_generation import generate_maps
from ir_solver_sparse import solve_ir_drop

# ---------------- Dataset ----------------
class IRDropDataset(Dataset):
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

        # Normalize inputs and labels to stabilize training
        def norm(x):
            mx = np.max(np.abs(x))
            return x / (mx + 1e-12)

        X = np.stack([norm(cur), norm(dens), norm(vsrc)], axis=0)  # [3,H,W]
        
        # Normalize labels to 0-1 range to match input scale
        ir_max = np.max(ir) if np.max(ir) > 0 else 1.0
        Y = (ir / ir_max)[np.newaxis, ...]  # [1,H,W] (normalized to 0-1)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32), ir_max

# --------------- Models ------------------
def get_model(kind: str):
    kind = (kind or "resunet").lower()
    if kind == "unet":
        from models.unet import UNet
        return UNet()
    else:
        from models.resunet import ResUNet
        return ResUNet()

# --------------- Losses ------------------
class HotspotMAE(nn.Module):
    """
    Composite loss: MAE + alpha * MAE on hotspots (>=90% of max true IR drop per-sample).
    """
    def __init__(self, alpha=5.0):
        super().__init__()
        self.alpha = float(alpha)
        self.l1 = nn.L1Loss()

    def forward(self, pred, true):
        base = self.l1(pred, true)
        # per-sample hotspot mask
        with torch.no_grad():
            # max over H,W (dim=2,3); keep dims for broadcasting
            max_true = torch.amax(true, dim=(2,3), keepdim=True)  # [B,1,1,1]
            thr = 0.9 * max_true
            mask = (true >= thr)  # [B,1,H,W]
            any_hot = mask.any(dim=(2,3), keepdim=True)  # [B,1,1,1]

        # avoid empty mask -> weight 0 for that sample
        if mask.any():
            # compute masked MAE sample-wise, average over valid samples
            # add small eps in denom to avoid /0
            diff = torch.abs(pred - true) * mask
            masked_loss_per_sample = diff.sum(dim=(2,3)) / (mask.sum(dim=(2,3)).clamp_min(1))
            masked_loss = masked_loss_per_sample.mean()
        else:
            masked_loss = torch.tensor(0.0, device=pred.device)

        return base + self.alpha * masked_loss, {'base': base.item(), 'hot': masked_loss.item()}

# --------------- Visualization -----------
def save_vis(epoch, batch_X, batch_Y, model, out_dir, max_samples=3, norm_factors=None):
    """
    Save side-by-side predicted vs true heatmaps for a few samples.
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
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(p, cmap="hot")
        axes[0].set_title("Predicted IR Drop")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(t, cmap="hot")
        axes[1].set_title("Ground Truth IR Drop")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        for ax in axes: ax.axis("off")
        fn = os.path.join(out_dir, f"epoch{epoch:03d}_sample{i}.png")
        plt.tight_layout()
        plt.savefig(fn, dpi=160)
        plt.close(fig)

# --------------- Training ----------------
def train_model(input_dir, output_model, *,
                model_name="resunet",
                epochs=25, lr=1e-4, batch_size=4,
                alpha=5.0, viz_every=2, device="cpu"):

    device = torch.device(device)

    # Where to stash generated features from .sp/.voltage
    feature_dir = os.path.join(input_dir, "generated_features")
    os.makedirs(feature_dir, exist_ok=True)

    # 1) Ensure voltage + maps for each .sp
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".sp"):
            continue
        spice_path = os.path.join(input_dir, fname)
        base = os.path.splitext(fname)[0]
        voltage_path = os.path.join(feature_dir, f"{base}.voltage")
        if not os.path.exists(voltage_path):
            print(f"[INFO] Solving MNA for {fname} -> {voltage_path}")
            solve_ir_drop(spice_path, voltage_path)
        print(f"[INFO] Generating maps for {base}")
        generate_maps(spice_path, voltage_path, feature_dir)

    # 2) Data
    dataset = IRDropDataset(feature_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No training samples found in {feature_dir}.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3) Model / Loss / Optimizer
    model = get_model(model_name).to(device)
    criterion = HotspotMAE(alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                           patience=3)

    # 4) Train
    loss_hist, base_hist, hot_hist = [], [], []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        base_sum, hot_sum = 0.0, 0.0

        for X, Y, norm_factors in loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss, parts = criterion(pred, Y)
            loss.backward()
            # clip to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running += loss.item()
            base_sum += parts['base']
            hot_sum  += parts['hot']

        epoch_loss = running / len(loader)
        loss_hist.append(epoch_loss)
        base_hist.append(base_sum / len(loader))
        hot_hist.append(hot_sum  / len(loader))
        print(f"Epoch {epoch:02d}/{epochs}  Loss={epoch_loss:.6f}  "
              f"(MAE={base_hist[-1]:.6f}  HotMAE={hot_hist[-1]:.6f})")

        scheduler.step(epoch_loss)

        # quick visual check
        if viz_every and (epoch % viz_every == 0):
            X_s, Y_s, norm_factors = next(iter(loader))
            save_vis(epoch, X_s.to(device), Y_s.to(device),
                     model, out_dir=os.path.join(feature_dir, "train_vis"), 
                     norm_factors=norm_factors.numpy())

    # 5) Save model + loss curves + normalization info
    torch.save(model.state_dict(), output_model)
    
    # Save normalization factors for inference
    norm_info = {
        "label_normalization": "per_sample_max",
        "input_normalization": "per_channel_max_abs",
        "output_activation": "relu"
    }
    with open(output_model.replace('.pt', '_norm_info.json'), 'w') as f:
        json.dump(norm_info, f, indent=2)
    
    np.savetxt(os.path.join(feature_dir, "training_loss.csv"), np.array(loss_hist), delimiter=",")
    # loss plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5,3))
        plt.plot(loss_hist, label="Total")
        plt.plot(base_hist, label="MAE")
        plt.plot(hot_hist, label="Hotspot MAE")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(feature_dir, "training_loss.png"), dpi=160)
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not save loss plot: {e}")
    print(f"[OK] Model saved: {output_model}")
    print(f"[OK] Artifacts in: {feature_dir}")

# --------------- CLI ---------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train ML model for IR-drop prediction (hotspot-aware)")
    ap.add_argument("-input", required=True, help="Directory containing training .sp files")
    ap.add_argument("-output", required=True, help="Path to save trained model (.pt)")
    ap.add_argument("--model", default="resunet", choices=["unet","resunet"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=5.0, help="Weight for hotspot MAE")
    ap.add_argument("--viz_every", type=int, default=2, help="Save vis every N epochs (0=off)")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    args = ap.parse_args()

    train_model(args.input, args.output,
                model_name=args.model, epochs=args.epochs, lr=args.lr,
                batch_size=args.batch, alpha=args.alpha,
                viz_every=args.viz_every, device=args.device)

