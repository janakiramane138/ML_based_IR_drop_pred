#!/usr/bin/env python3
import argparse, os, sys, time, csv
import numpy as np
import torch

# local imports w/o PYTHONPATH hassles
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from ir_solver_sparse import solve_ir_drop
from data_generation import generate_maps

# --------- helpers ---------
def select_model():
    # prefer ResUNet if present, else UNet
    try:
        from models.resunet import ResUNet as Net
        return Net()
    except Exception:
        from models.unet import UNet as Net
        return Net()

def load_model(model_path, device):
    model = select_model()
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model

def load_map(path):
    arr = np.loadtxt(path, delimiter=",")
    if np.isnan(arr).any() or np.isinf(arr).any():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))  # volts

def f1_90pct_of_max(pred, true):
    max_true = float(np.max(true)) if true.size else 0.0
    if max_true == 0.0:
        stats = dict(max_true=0.0, thr=0.0, n_pos_true=0, n_pos_pred=0,
                     TP=0, FP=0, FN=0, precision=1.0, recall=1.0, f1=1.0)
        return 1.0, stats
    thr = 0.9 * max_true
    true_hot = (true >= thr)
    pred_hot = (pred >= thr)
    TP = int(np.sum(pred_hot & true_hot))
    FP = int(np.sum(pred_hot & (~true_hot)))
    FN = int(np.sum((~pred_hot) & true_hot))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    stats = dict(max_true=max_true, thr=thr,
                 n_pos_true=int(np.sum(true_hot)),
                 n_pos_pred=int(np.sum(pred_hot)),
                 TP=TP, FP=FP, FN=FN,
                 precision=precision, recall=recall, f1=f1)
    return f1, stats

def ensure_golden(spice_path, gt_dir, debug=False):
    """Ensure .voltage and ir_drop_map exist in gt_dir for this spice file."""
    base = os.path.splitext(os.path.basename(spice_path))[0]
    os.makedirs(gt_dir, exist_ok=True)
    vfile = os.path.join(gt_dir, f"{base}.voltage")
    if not os.path.exists(vfile):
        if debug: print(f"[DEBUG] {base}: solving MNA -> {vfile}")
        solve_ir_drop(spice_path, vfile)
    gt_ir = os.path.join(gt_dir, f"ir_drop_map_{base}.csv")
    if not os.path.exists(gt_ir):
        if debug: print(f"[DEBUG] {base}: generating GT maps in {gt_dir}")
        generate_maps(spice_path, vfile, gt_dir)
    return base, vfile, gt_ir

def _norm(x: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(x)))
    return x / (m + 1e-12)

def load_features(base, feature_dir):
    cur  = load_map(os.path.join(feature_dir, f"current_map_{base}.csv"))
    dens = load_map(os.path.join(feature_dir, f"pdn_density_map_{base}.csv"))
    vsrc = load_map(os.path.join(feature_dir, f"voltage_source_map_{base}.csv"))
    # Match training-time normalization: per-channel max-abs scaling
    X = np.stack([_norm(cur), _norm(dens), _norm(vsrc)], axis=0)  # [3,H,W]
    return X

def forward_once(model, X, device, gt_ir_map):
    t0 = time.perf_counter()
    with torch.no_grad():
        inp = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)  # [1,3,H,W]
        out = model(inp).squeeze().cpu().numpy()  # [H,W]
    
    # Denormalize prediction
    gt_max = np.max(gt_ir_map) if np.max(gt_ir_map) > 0 else 1.0
    out = out * gt_max
    
    # Physically IR-drop is non-negative
    out = np.clip(out, 0.0, None)
    dt = time.perf_counter() - t0
    return out, dt

# --------- modes ---------
def eval_from_spice(spice_dir, model_path, pred_out, gt_out, debug=False, csv_out=None, device="cpu"):
    device = torch.device(device)
    model = load_model(model_path, device)
    os.makedirs(pred_out, exist_ok=True)
    rows = []

    for fname in sorted(os.listdir(spice_dir)):
        if not fname.endswith(".sp"): 
            continue
        spice_path = os.path.join(spice_dir, fname)
        base, _vfile, gt_ir_path = ensure_golden(spice_path, gt_out, debug=debug)

        # features are deterministic from netlist+voltages; reuse GT dir to load them
        X = load_features(base, gt_out)
        gt_ir_map = load_map(gt_ir_path)

        # time only the model forward (inference time)
        pred, runtime = forward_once(model, X, device, gt_ir_map)
        if debug:
            print(f"[DEBUG] {base}: model inference runtime = {runtime:.4f}s")

        # save prediction
        pred_path = os.path.join(pred_out, f"predicted_ir_drop_map_{base}.csv")
        np.savetxt(pred_path, pred, delimiter=",")

        # evaluate
        true = gt_ir_map
        m = mae(pred, true)
        f1, stats = f1_90pct_of_max(pred, true)

        if debug:
            print(f"[DEBUG] {base}: max_true={stats['max_true']*1000:.3f} mV  thr={stats['thr']*1000:.3f} mV  "
                  f"pos_true={stats['n_pos_true']} pos_pred={stats['n_pos_pred']}  "
                  f"TP={stats['TP']} FP={stats['FP']} FN={stats['FN']}  "
                  f"P={stats['precision']:.3f} R={stats['recall']:.3f} F1={stats['f1']:.3f}")

        rows.append((base, m, f1, runtime, stats))

    print_table(rows)
    if csv_out:
        write_csv(rows, csv_out)
        print(f"[INFO] Wrote CSV: {csv_out}")

def eval_from_dirs(pred_dir, true_dir, debug=False, csv_out=None):
    rows = []
    for fname in sorted(os.listdir(pred_dir)):
        if not (fname.startswith("predicted_ir_drop_map_") and fname.endswith(".csv")):
            continue
        base = fname[len("predicted_ir_drop_map_"):-4]
        pred = load_map(os.path.join(pred_dir, fname))
        true = load_map(os.path.join(true_dir, f"ir_drop_map_{base}.csv"))
        m = mae(pred, true)
        f1, stats = f1_90pct_of_max(pred, true)
        if debug:
            print(f"[DEBUG] {base}: max_true={stats['max_true']*1000:.3f} mV  thr={stats['thr']*1000:.3f} mV  "
                  f"pos_true={stats['n_pos_true']} pos_pred={stats['n_pos_pred']}  "
                  f"TP={stats['TP']} FP={stats['FP']} FN={stats['FN']}  "
                  f"P={stats['precision']:.3f} R={stats['recall']:.3f} F1={stats['f1']:.3f}")
        rows.append((base, m, f1, None, stats))
    print_table(rows)
    if csv_out:
        write_csv(rows, csv_out)
        print(f"[INFO] Wrote CSV: {csv_out}")

# --------- reporting ---------
def print_table(rows):
    print("\nEvaluation Results")
    print(f"{'Testcase':<16} {'MAE (mV)':>10} {'F1 (>=90% max)':>15} {'Runtime (s)':>12} {'Threshold (mV)':>15}")
    print("-" * 70)
    for base, m, f1, rt, s in rows:
        rt_str = f"{rt:.3f}" if rt is not None else "-"
        thr_mv = s['thr'] * 1000.0
        print(f"{base:<16} {m*1000:>10.3f} {f1:>15.3f} {rt_str:>12} {thr_mv:>15.3f}")

    # optional averages
    if rows:
        ms = np.mean([m for _, m, *_ in rows]) * 1000.0
        f1s = np.mean([f1 for *_, f1, _, _ in rows])
        rts = [rt for *_, rt, _ in rows if rt is not None]
        rt_avg = np.mean(rts) if rts else None
        print("-" * 70)
        print(f"{'Average':<16} {ms:>10.3f} {f1s:>15.3f} {('-' if rt_avg is None else f'{rt_avg:.3f}'):>12} {'':>15}")

def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["testcase","mae_mV","f1","runtime_s","thr_mV",
                    "max_true_mV","pos_true","pos_pred","TP","FP","FN","precision","recall"])
        for base, m, f1, rt, s in rows:
            w.writerow([base, m*1000.0, f1, ("" if rt is None else rt),
                        s['thr']*1000.0, s['max_true']*1000.0, s['n_pos_true'], s['n_pos_pred'],
                        s['TP'], s['FP'], s['FN'], s['precision'], s['recall']])

# --------- CLI ---------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Evaluate IR-drop predictions: MAE, F1 (>=90% per testcase), and runtime (model forward)."
    )
    sub = ap.add_subparsers(dest="mode")

    a = sub.add_parser("from_spice", help="Run full evaluation from .sp files (build GT, run model, time inference).")
    a.add_argument("--spice_dir", required=True)
    a.add_argument("--ml_model", required=True)
    a.add_argument("--pred_out", required=True)
    a.add_argument("--gt_out", required=True)
    a.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    a.add_argument("--csv_out")
    a.add_argument("--debug", action="store_true")

    b = sub.add_parser("from_dirs", help="Evaluate from precomputed CSVs.")
    b.add_argument("-pred_dir", required=True)
    b.add_argument("-true_dir", required=True)
    b.add_argument("--csv_out")
    b.add_argument("--debug", action="store_true")

    args = ap.parse_args()
    if args.mode == "from_spice":
        eval_from_spice(args.spice_dir, args.ml_model, args.pred_out, args.gt_out,
                        debug=args.debug, csv_out=args.csv_out, device=args.device)
    elif args.mode == "from_dirs":
        eval_from_dirs(args.pred_dir, args.true_dir, debug=args.debug, csv_out=args.csv_out)
    else:
        ap.print_help()
        sys.exit(1)

