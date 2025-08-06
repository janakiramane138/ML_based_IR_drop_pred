
import argparse
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def load_maps(pred_file, true_file):
    pred = np.loadtxt(pred_file, delimiter=",")
    true = np.loadtxt(true_file, delimiter=",")
    return pred, true

def compute_mae(pred, true):
    return np.mean(np.abs(pred - true))

def compute_f1(pred, true):
    threshold = 0.9 * np.max(true)
    pred_hot = (pred > threshold).astype(int).flatten()
    true_hot = (true > threshold).astype(int).flatten()
    if np.sum(true_hot) == 0 and np.sum(pred_hot) == 0:
        return 1.0  # perfect match with no hotspots
    return f1_score(true_hot, pred_hot, zero_division=1)

def evaluate_all(pred_dir, true_dir):
    results = []
    for file in os.listdir(pred_dir):
        if file.startswith("predicted_ir_drop_map_") and file.endswith(".csv"):
            base = file.replace("predicted_ir_drop_map_", "").replace(".csv", "")
            pred_path = os.path.join(pred_dir, file)
            true_path = os.path.join(true_dir, f"ir_drop_map_{base}.csv")
            if not os.path.exists(true_path):
                print(f"Missing ground truth for {base}, skipping.")
                continue
            pred, true = load_maps(pred_path, true_path)
            mae = compute_mae(pred, true)
            f1 = compute_f1(pred, true)
            results.append((base, mae, f1))

    print("\nEvaluation Results:")
    print(f"{'Testcase':<12} {'MAE (mV)':<10} {'F1 Score':<10}")
    print("-" * 34)
    for base, mae, f1 in results:
        print(f"{base:<12} {mae*1000:<10.3f} {f1:<10.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pred_dir', required=True, help="Directory containing predicted_ir_drop_map_*.csv")
    parser.add_argument('-true_dir', required=True, help="Directory containing ir_drop_map_*.csv (ground truth)")
    args = parser.parse_args()

    evaluate_all(args.pred_dir, args.true_dir)
