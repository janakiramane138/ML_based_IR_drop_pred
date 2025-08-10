#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# make src/ imports robust
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from data_generation import generate_maps
from ir_solver_sparse import solve_ir_drop
# choose your model:
from models.unet import UNet as Net
#from models.resunet import ResUNet as Net  # if you prefer ResUNet

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
        c, d, v, y = self.samples[idx]
        cur  = np.loadtxt(c, delimiter=",")
        dens = np.loadtxt(d, delimiter=",")
        vsrc = np.loadtxt(v, delimiter=",")
        ir   = np.loadtxt(y, delimiter=",")
        X = np.stack([cur, dens, vsrc], axis=0)
        Y = ir[np.newaxis, ...]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def train_model(input_dir, output_model, epochs=10, lr=1e-4, batch_size=2):
    # where to stash generated stuff
    feature_dir = os.path.join(input_dir, "generated_features")
    os.makedirs(feature_dir, exist_ok=True)

    # 1) For each .sp, ensure we have a .voltage (run solver if missing), then generate maps
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".sp"):
            continue
        spice_path = os.path.join(input_dir, fname)
        base = os.path.splitext(fname)[0]

        # put the .voltage in feature_dir to avoid clutter
        voltage_path = os.path.join(feature_dir, f"{base}.voltage")
        if not os.path.exists(voltage_path):
            print(f"[INFO] Solving MNA for {fname} -> {voltage_path}")
            solve_ir_drop(spice_path, voltage_path)

        print(f"[INFO] Generating maps for {base}")
        generate_maps(spice_path, voltage_path, feature_dir)

    # 2) Dataset / Loader
    dataset = IRDropDataset(feature_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No training samples found in {feature_dir}.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3) Model / Loss / Optimizer
    model = Net()
    criterion = nn.L1Loss()  # MAE aligns with contest metric
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4) Train
    loss_hist = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for X, Y in loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg = running / len(loader)
        loss_hist.append(avg)
        print(f"Epoch {epoch+1:02d}/{epochs} - Loss(MAE): {avg:.6f}")

    # 5) Save model + loss curve data for report
    torch.save(model.state_dict(), output_model)
    np.savetxt(os.path.join(feature_dir, "training_loss.csv"), np.array(loss_hist), delimiter=",")
    print(f"[OK] Model saved: {output_model}")
    print(f"[OK] Loss history saved: {feature_dir}/training_loss.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train ML model for IR-drop prediction")
    ap.add_argument("-input", required=True, help="Directory containing training .sp files")
    ap.add_argument("-output", required=True, help="Path to save trained model (.pt)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    train_model(args.input, args.output, epochs=args.epochs, lr=args.lr, batch_size=args.batch)

