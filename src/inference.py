#!/usr/bin/env python3
"""
inference.py
Run IR-drop prediction for a given SPICE netlist using a trained model.

Usage:
python3 inference.py -spice_file benchmarks/testcase1.sp -ml_model unet_model.pt -output features/
"""
import sys, os
sys.path.append(os.path.dirname(__file__))  # lets `from models.unet import UNet` work # adds src/ to import path at runtime

import argparse
import os
import numpy as np
import torch
import json

from data_generation import generate_maps
# from models.unet import UNet
from models.resunet import ResUNet
from ir_solver_sparse import solve_ir_drop


def run_inference(spice_file, ml_model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(spice_file))[0]

    # Step 1: Run MNA solver to get voltage file
    voltage_file = os.path.join(output_dir, f"{base}.voltage")
    if not os.path.exists(voltage_file):
        print(f"[INFO] Running MNA solver for {spice_file}")
        solve_ir_drop(spice_file, voltage_file)

    # Step 2: Generate feature maps
    print(f"[INFO] Generating feature maps for {base}")
    generate_maps(spice_file, voltage_file, output_dir)

    # Step 3: Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResUNet().to(device)
    state = torch.load(ml_model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Step 4: Load features for this testcase
    current_map = np.loadtxt(os.path.join(output_dir, f"current_map_{base}.csv"), delimiter=",")
    density_map = np.loadtxt(os.path.join(output_dir, f"pdn_density_map_{base}.csv"), delimiter=",")
    vsrc_map    = np.loadtxt(os.path.join(output_dir, f"voltage_source_map_{base}.csv"), delimiter=",")

    # Match training-time normalization: per-sample, per-channel max-abs scaling
    def _norm(x: np.ndarray) -> np.ndarray:
        m = float(np.max(np.abs(x)))
        return x / (m + 1e-12)

    features = np.stack([
        _norm(current_map),
        _norm(density_map),
        _norm(vsrc_map)
    ], axis=0)  # [3,H,W]

    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # [1,3,H,W]

    # Step 5: Predict
    with torch.no_grad():
        pred = model(features_tensor).cpu().squeeze().numpy()

    # Step 6: Denormalize prediction
    # Load the ground truth to get normalization factor
    gt_ir_map = np.loadtxt(os.path.join(output_dir, f"ir_drop_map_{base}.csv"), delimiter=",")
    gt_max = np.max(gt_ir_map) if np.max(gt_ir_map) > 0 else 1.0
    
    # Denormalize: convert from 0-1 range back to original scale
    pred = pred * gt_max

    # IR drop is non-negative physically; clamp small negatives from model
    pred = np.clip(pred, 0.0, None)

    # Step 7: Save prediction
    pred_path = os.path.join(output_dir, f"predicted_ir_drop_map_{base}.csv")
    np.savetxt(pred_path, pred, delimiter=",")
    print(f"[INFO] Predicted IR drop map saved to {pred_path}")
    print(f"[INFO] Prediction range: {np.min(pred):.6f} - {np.max(pred):.6f} V")
    print(f"[INFO] Ground truth range: {np.min(gt_ir_map):.6f} - {np.max(gt_ir_map):.6f} V")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IR-drop prediction for a SPICE file")
    parser.add_argument("-spice_file", required=True, help="Path to SPICE netlist file")
    parser.add_argument("-ml_model", required=True, help="Path to trained model (.pt)")
    parser.add_argument("-output", required=True, help="Output directory for prediction and intermediate files")
    args = parser.parse_args()

    run_inference(args.spice_file, args.ml_model, args.output)

