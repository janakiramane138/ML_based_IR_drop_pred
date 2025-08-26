#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import cv2

# robust local imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from data_generation import generate_maps, determine_grid_size
from ir_solver_sparse import solve_ir_drop

def load_model(model_path, norm_info_path=None):
    """
    Load trained model with proper configuration.
    """
    # Load normalization info
    norm_info = {}
    if norm_info_path and os.path.exists(norm_info_path):
        with open(norm_info_path, 'r') as f:
            norm_info = json.load(f)
    
    # Determine model type from norm_info
    variable_size = norm_info.get('variable_size', False)
    target_size = norm_info.get('target_size', 256)
    
    # Import appropriate model
    if variable_size:
        from models.resunet import VariableSizeResUNet
        model = VariableSizeResUNet(target_size=target_size)
    else:
        from models.resunet import ResUNet
        model = ResUNet()
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, norm_info

def preprocess_for_evaluation(spice_netlist, voltage_file, output_dir, target_size=256):
    """
    Preprocess input data for evaluation.
    """
    # Determine grid size for this netlist
    grid_size = determine_grid_size(spice_netlist)
    
    # Generate maps
    generate_maps(spice_netlist, voltage_file, output_dir, grid_size)
    
    # Load maps
    base = os.path.splitext(os.path.basename(spice_netlist))[0]
    
    current_map = np.loadtxt(os.path.join(output_dir, f"current_map_{base}.csv"), delimiter=",")
    density_map = np.loadtxt(os.path.join(output_dir, f"pdn_density_map_{base}.csv"), delimiter=",")
    vsrc_map = np.loadtxt(os.path.join(output_dir, f"voltage_source_map_{base}.csv"), delimiter=",")
    ground_truth = np.loadtxt(os.path.join(output_dir, f"ir_drop_map_{base}.csv"), delimiter=",")
    
    # Resize to target size if needed
    if current_map.shape[0] != target_size or current_map.shape[1] != target_size:
        current_map = cv2.resize(current_map, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        density_map = cv2.resize(density_map, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        vsrc_map = cv2.resize(vsrc_map, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        ground_truth = cv2.resize(ground_truth, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Normalize inputs
    def norm(x):
        mx = np.max(np.abs(x))
        return x / (mx + 1e-12)
    
    X = np.stack([norm(current_map), norm(density_map), norm(vsrc_map)], axis=0)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    return X, ground_truth, base, grid_size

def calculate_metrics(prediction, ground_truth):
    """
    Calculate comprehensive evaluation metrics.
    """
    # Flatten arrays for metric calculation
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()
    
    # Basic regression metrics
    mae = mean_absolute_error(gt_flat, pred_flat)
    mse = mean_squared_error(gt_flat, pred_flat)
    rmse = np.sqrt(mse)
    
    # Correlation
    correlation, p_value = pearsonr(gt_flat, pred_flat)
    
    # Relative errors
    rel_mae = mae / (np.max(gt_flat) + 1e-12)
    rel_rmse = rmse / (np.max(gt_flat) + 1e-12)
    
    # Peak error
    peak_error = np.max(np.abs(pred_flat - gt_flat))
    
    # Hotspot detection metrics
    hotspot_metrics = calculate_hotspot_metrics(prediction, ground_truth)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'correlation': correlation,
        'p_value': p_value,
        'relative_mae': rel_mae,
        'relative_rmse': rel_rmse,
        'peak_error': peak_error,
        'hotspot_metrics': hotspot_metrics
    }

def calculate_hotspot_metrics(prediction, ground_truth, threshold_percentile=90):
    """
    Calculate hotspot detection accuracy metrics.
    """
    # Create hotspot masks
    gt_threshold = np.percentile(ground_truth, threshold_percentile)
    pred_threshold = np.percentile(prediction, threshold_percentile)
    
    gt_hotspots = ground_truth >= gt_threshold
    pred_hotspots = prediction >= pred_threshold
    
    # Calculate intersection over union (IoU)
    intersection = np.logical_and(gt_hotspots, pred_hotspots).sum()
    union = np.logical_or(gt_hotspots, pred_hotspots).sum()
    iou = intersection / (union + 1e-12)
    
    # Calculate precision and recall
    tp = intersection
    fp = pred_hotspots.sum() - intersection
    fn = gt_hotspots.sum() - intersection
    
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-12)
    
    # Calculate hotspot area accuracy
    gt_area = gt_hotspots.sum()
    pred_area = pred_hotspots.sum()
    area_accuracy = 1 - abs(gt_area - pred_area) / (gt_area + 1e-12)
    
    # Calculate hotspot intensity correlation
    if gt_hotspots.sum() > 0 and pred_hotspots.sum() > 0:
        gt_hotspot_intensities = ground_truth[gt_hotspots]
        pred_hotspot_intensities = prediction[pred_hotspots]
        
        # Resample to same size for correlation
        min_size = min(len(gt_hotspot_intensities), len(pred_hotspot_intensities))
        if min_size > 1:
            gt_sample = np.random.choice(gt_hotspot_intensities, min_size, replace=False)
            pred_sample = np.random.choice(pred_hotspot_intensities, min_size, replace=False)
            hotspot_correlation, _ = pearsonr(gt_sample, pred_sample)
        else:
            hotspot_correlation = 0.0
    else:
        hotspot_correlation = 0.0
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'area_accuracy': area_accuracy,
        'hotspot_correlation': hotspot_correlation,
        'gt_hotspot_area': int(gt_hotspots.sum()),
        'pred_hotspot_area': int(pred_hotspots.sum())
    }

def evaluate_single_file(spice_netlist, voltage_file, model_path, output_dir, 
                        norm_info_path=None, save_visualization=True):
    """
    Evaluate model performance on a single file.
    """
    print(f"[INFO] Evaluating {os.path.basename(spice_netlist)}...")
    
    # Load model
    model, norm_info = load_model(model_path, norm_info_path)
    target_size = norm_info.get('target_size', 256)
    
    # Preprocess data
    X, ground_truth, base, original_grid_size = preprocess_for_evaluation(
        spice_netlist, voltage_file, output_dir, target_size
    )
    
    # Run prediction
    with torch.no_grad():
        prediction = model(X)
        prediction = prediction.squeeze().numpy()
    
    # Denormalize if needed
    if norm_info.get('label_normalization') == 'per_sample_max':
        # Use ground truth max for denormalization
        gt_max = np.max(ground_truth) if np.max(ground_truth) > 0 else 1.0
        prediction = prediction * gt_max
    
    # Calculate metrics
    metrics = calculate_metrics(prediction, ground_truth)
    
    # Save results
    results = {
        'file': base,
        'original_grid_size': original_grid_size,
        'processed_grid_size': target_size,
        'metrics': metrics,
        'prediction_range': [float(np.min(prediction)), float(np.max(prediction))],
        'ground_truth_range': [float(np.min(ground_truth)), float(np.max(ground_truth))]
    }
    
    # Save prediction
    prediction_path = os.path.join(output_dir, f"predicted_ir_drop_{base}.csv")
    np.savetxt(prediction_path, prediction, delimiter=",")
    
    # Save visualization
    if save_visualization:
        save_evaluation_visualization(prediction, ground_truth, metrics, 
                                    os.path.join(output_dir, f"evaluation_{base}.png"), base)
    
    # Print summary
    print(f"  - MAE: {metrics['mae']:.6f}")
    print(f"  - RMSE: {metrics['rmse']:.6f}")
    print(f"  - Correlation: {metrics['correlation']:.4f}")
    print(f"  - Hotspot IoU: {metrics['hotspot_metrics']['iou']:.4f}")
    print(f"  - Hotspot F1: {metrics['hotspot_metrics']['f1_score']:.4f}")
    
    return results

def save_evaluation_visualization(prediction, ground_truth, metrics, save_path, title):
    """
    Create comprehensive evaluation visualization.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Main predictions
    im1 = axes[0,0].imshow(prediction, cmap='hot', interpolation='nearest')
    axes[0,0].set_title(f"Prediction\nRange: [{metrics['prediction_range'][0]:.4f}, {metrics['prediction_range'][1]:.4f}]")
    plt.colorbar(im1, ax=axes[0,0], fraction=0.046, pad=0.04)
    
    im2 = axes[0,1].imshow(ground_truth, cmap='hot', interpolation='nearest')
    axes[0,1].set_title(f"Ground Truth\nRange: [{metrics['ground_truth_range'][0]:.4f}, {metrics['ground_truth_range'][1]:.4f}]")
    plt.colorbar(im2, ax=axes[0,1], fraction=0.046, pad=0.04)
    
    # Error map
    error_map = np.abs(prediction - ground_truth)
    im3 = axes[0,2].imshow(error_map, cmap='viridis', interpolation='nearest')
    axes[0,2].set_title(f"Absolute Error\nMAE: {metrics['mae']:.6f}")
    plt.colorbar(im3, ax=axes[0,2], fraction=0.046, pad=0.04)
    
    # Hotspot comparison
    gt_threshold = np.percentile(ground_truth, 90)
    pred_threshold = np.percentile(prediction, 90)
    
    gt_hotspots = ground_truth >= gt_threshold
    pred_hotspots = prediction >= pred_threshold
    
    axes[1,0].imshow(gt_hotspots, cmap='gray', interpolation='nearest')
    axes[1,0].set_title(f"Ground Truth Hotspots\nArea: {metrics['hotspot_metrics']['gt_hotspot_area']}")
    
    axes[1,1].imshow(pred_hotspots, cmap='gray', interpolation='nearest')
    axes[1,1].set_title(f"Predicted Hotspots\nArea: {metrics['hotspot_metrics']['pred_hotspot_area']}")
    
    # Scatter plot
    axes[1,2].scatter(ground_truth.flatten(), prediction.flatten(), alpha=0.1, s=1)
    axes[1,2].plot([0, np.max(ground_truth)], [0, np.max(ground_truth)], 'r--', alpha=0.8)
    axes[1,2].set_xlabel('Ground Truth')
    axes[1,2].set_ylabel('Prediction')
    axes[1,2].set_title(f'Correlation: {metrics["correlation"]:.4f}')
    axes[1,2].grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.suptitle(f'Evaluation Results: {title}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def batch_evaluate(input_dir, model_path, output_dir, norm_info_path=None):
    """
    Evaluate model performance on all files in a directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all SPICE files
    spice_files = [f for f in os.listdir(input_dir) if f.endswith('.sp')]
    
    if not spice_files:
        print(f"[ERROR] No SPICE files found in {input_dir}")
        return
    
    print(f"[INFO] Found {len(spice_files)} SPICE files for evaluation")
    
    all_results = []
    
    for spice_file in spice_files:
        spice_path = os.path.join(input_dir, spice_file)
        base = os.path.splitext(spice_file)[0]
        voltage_file = os.path.join(input_dir, "generated_features", f"{base}.voltage")
        
        # Check if voltage file exists, if not solve MNA
        if not os.path.exists(voltage_file):
            print(f"[INFO] Solving MNA for {spice_file}...")
            solve_ir_drop(spice_path, voltage_file)
        
        try:
            results = evaluate_single_file(
                spice_path, voltage_file, model_path, output_dir,
                norm_info_path, save_visualization=True
            )
            all_results.append(results)
            
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {spice_file}: {e}")
            continue
    
    # Calculate aggregate metrics
    if all_results:
        aggregate_metrics = calculate_aggregate_metrics(all_results)
        
        # Save comprehensive results
        save_comprehensive_results(all_results, aggregate_metrics, output_dir)
        
        # Print summary
        print(f"\n[SUMMARY] Evaluation completed:")
        print(f"  - Total files: {len(spice_files)}")
        print(f"  - Successful evaluations: {len(all_results)}")
        print(f"  - Average MAE: {aggregate_metrics['avg_mae']:.6f}")
        print(f"  - Average RMSE: {aggregate_metrics['avg_rmse']:.6f}")
        print(f"  - Average Correlation: {aggregate_metrics['avg_correlation']:.4f}")
        print(f"  - Average Hotspot IoU: {aggregate_metrics['avg_hotspot_iou']:.4f}")
        print(f"  - Average Hotspot F1: {aggregate_metrics['avg_hotspot_f1']:.4f}")

def calculate_aggregate_metrics(results):
    """
    Calculate aggregate metrics across all results.
    """
    metrics_list = [r['metrics'] for r in results]
    
    aggregate = {
        'avg_mae': np.mean([m['mae'] for m in metrics_list]),
        'avg_mse': np.mean([m['mse'] for m in metrics_list]),
        'avg_rmse': np.mean([m['rmse'] for m in metrics_list]),
        'avg_correlation': np.mean([m['correlation'] for m in metrics_list]),
        'avg_relative_mae': np.mean([m['relative_mae'] for m in metrics_list]),
        'avg_relative_rmse': np.mean([m['relative_rmse'] for m in metrics_list]),
        'avg_peak_error': np.mean([m['peak_error'] for m in metrics_list]),
        'avg_hotspot_iou': np.mean([m['hotspot_metrics']['iou'] for m in metrics_list]),
        'avg_hotspot_precision': np.mean([m['hotspot_metrics']['precision'] for m in metrics_list]),
        'avg_hotspot_recall': np.mean([m['hotspot_metrics']['recall'] for m in metrics_list]),
        'avg_hotspot_f1': np.mean([m['hotspot_metrics']['f1_score'] for m in metrics_list]),
        'avg_hotspot_area_accuracy': np.mean([m['hotspot_metrics']['area_accuracy'] for m in metrics_list]),
        'avg_hotspot_correlation': np.mean([m['hotspot_metrics']['hotspot_correlation'] for m in metrics_list])
    }
    
    return aggregate

def save_comprehensive_results(results, aggregate_metrics, output_dir):
    """
    Save comprehensive evaluation results.
    """
    # Save detailed results
    detailed_results = {
        'individual_results': results,
        'aggregate_metrics': aggregate_metrics,
        'evaluation_summary': {
            'total_files': len(results),
            'evaluation_timestamp': str(np.datetime64('now'))
        }
    }
    
    with open(os.path.join(output_dir, 'comprehensive_evaluation_results.json'), 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Save metrics summary as CSV
    import pandas as pd
    
    # Individual file metrics
    individual_data = []
    for result in results:
        row = {
            'file': result['file'],
            'mae': result['metrics']['mae'],
            'rmse': result['metrics']['rmse'],
            'correlation': result['metrics']['correlation'],
            'hotspot_iou': result['metrics']['hotspot_metrics']['iou'],
            'hotspot_f1': result['metrics']['hotspot_metrics']['f1_score'],
            'hotspot_precision': result['metrics']['hotspot_metrics']['precision'],
            'hotspot_recall': result['metrics']['hotspot_metrics']['recall']
        }
        individual_data.append(row)
    
    df_individual = pd.DataFrame(individual_data)
    df_individual.to_csv(os.path.join(output_dir, 'individual_metrics.csv'), index=False)
    
    # Aggregate metrics
    aggregate_data = [aggregate_metrics]
    df_aggregate = pd.DataFrame(aggregate_data)
    df_aggregate.to_csv(os.path.join(output_dir, 'aggregate_metrics.csv'), index=False)
    
    print(f"[INFO] Comprehensive results saved to {output_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate enhanced IR drop prediction model")
    ap.add_argument("-spice", type=str, help="Path to SPICE netlist file (.sp)")
    ap.add_argument("-voltage", type=str, help="Path to voltage output file from MNA solver")
    ap.add_argument("-model", type=str, required=True, help="Path to trained model (.pt)")
    ap.add_argument("-output", type=str, required=True, help="Directory to save evaluation results")
    ap.add_argument("-norm_info", type=str, help="Path to normalization info file (.json)")
    ap.add_argument("--batch", action="store_true", help="Run batch evaluation on directory")
    ap.add_argument("--input_dir", type=str, help="Input directory for batch evaluation")
    ap.add_argument("--no_viz", action="store_true", help="Disable visualization")
    args = ap.parse_args()

    if args.batch:
        if not args.input_dir:
            print("[ERROR] --input_dir required for batch evaluation")
            sys.exit(1)
        batch_evaluate(args.input_dir, args.model, args.output, args.norm_info)
    else:
        if not args.spice or not args.voltage:
            print("[ERROR] -spice and -voltage required for single file evaluation")
            sys.exit(1)
        evaluate_single_file(args.spice, args.voltage, args.model, args.output,
                           args.norm_info, save_visualization=not args.no_viz)

