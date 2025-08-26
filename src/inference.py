#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import torch
import json
import matplotlib.pyplot as plt

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

def preprocess_input(spice_netlist, voltage_file, output_dir, target_size=256):
    """
    Preprocess input data for inference.
    """
    # Determine grid size for this netlist
    grid_size = determine_grid_size(spice_netlist)
    print(f"[INFO] Determined grid size: {grid_size}x{grid_size}")
    
    # Generate maps
    generate_maps(spice_netlist, voltage_file, output_dir, grid_size)
    
    # Load and resize maps to target size
    base = os.path.splitext(os.path.basename(spice_netlist))[0]
    
    current_map = np.loadtxt(os.path.join(output_dir, f"current_map_{base}.csv"), delimiter=",")
    density_map = np.loadtxt(os.path.join(output_dir, f"pdn_density_map_{base}.csv"), delimiter=",")
    vsrc_map = np.loadtxt(os.path.join(output_dir, f"voltage_source_map_{base}.csv"), delimiter=",")
    
    # Resize to target size if needed
    if current_map.shape[0] != target_size or current_map.shape[1] != target_size:
        import cv2
        current_map = cv2.resize(current_map, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        density_map = cv2.resize(density_map, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        vsrc_map = cv2.resize(vsrc_map, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        print(f"[INFO] Resized maps to {target_size}x{target_size}")
    
    # Normalize inputs
    def norm(x):
        mx = np.max(np.abs(x))
        return x / (mx + 1e-12)
    
    X = np.stack([norm(current_map), norm(density_map), norm(vsrc_map)], axis=0)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    return X, base, grid_size

def detect_hotspots(prediction, threshold_percentile=90):
    """
    Detect hotspots in the prediction using adaptive thresholding.
    """
    # Use percentile-based thresholding
    threshold = np.percentile(prediction, threshold_percentile)
    hotspot_mask = prediction >= threshold
    
    # Find connected components
    from scipy import ndimage
    labeled_hotspots, num_hotspots = ndimage.label(hotspot_mask)
    
    # Get hotspot properties
    hotspot_properties = []
    for i in range(1, num_hotspots + 1):
        hotspot_region = (labeled_hotspots == i)
        hotspot_intensity = np.max(prediction[hotspot_region])
        hotspot_area = np.sum(hotspot_region)
        hotspot_center = ndimage.center_of_mass(prediction, labels=labeled_hotspots, index=i)
        
        hotspot_properties.append({
            'id': i,
            'intensity': hotspot_intensity,
            'area': hotspot_area,
            'center': hotspot_center,
            'mask': hotspot_region
        })
    
    return hotspot_mask, hotspot_properties

def visualize_results(prediction, ground_truth=None, hotspot_properties=None, 
                     save_path=None, title="IR Drop Prediction"):
    """
    Create comprehensive visualization of results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Main prediction
    im1 = axes[0,0].imshow(prediction, cmap='hot', interpolation='nearest')
    axes[0,0].set_title(f"{title}")
    plt.colorbar(im1, ax=axes[0,0], fraction=0.046, pad=0.04)
    
    # Ground truth comparison
    if ground_truth is not None:
        im2 = axes[0,1].imshow(ground_truth, cmap='hot', interpolation='nearest')
        axes[0,1].set_title("Ground Truth")
        plt.colorbar(im2, ax=axes[0,1], fraction=0.046, pad=0.04)
    else:
        axes[0,1].text(0.5, 0.5, 'No Ground Truth\nAvailable', 
                      ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title("Ground Truth")
    
    # Hotspot detection
    if hotspot_properties:
        hotspot_mask = np.zeros_like(prediction)
        for prop in hotspot_properties:
            hotspot_mask = hotspot_mask | prop['mask']
        
        im3 = axes[1,0].imshow(hotspot_mask, cmap='gray', interpolation='nearest')
        axes[1,0].set_title(f"Detected Hotspots ({len(hotspot_properties)} regions)")
        
        # Overlay hotspots on prediction
        overlay = prediction.copy()
        overlay[hotspot_mask] = np.max(prediction) * 1.2  # Highlight hotspots
        im4 = axes[1,1].imshow(overlay, cmap='hot', interpolation='nearest')
        axes[1,1].set_title("Prediction with Hotspot Overlay")
        plt.colorbar(im4, ax=axes[1,1], fraction=0.046, pad=0.04)
    else:
        axes[1,0].text(0.5, 0.5, 'No Hotspots\nDetected', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title("Detected Hotspots")
        axes[1,1].text(0.5, 0.5, 'No Hotspots\nDetected', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title("Prediction with Hotspot Overlay")
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Visualization saved to {save_path}")
    
    plt.show()

def run_inference(spice_netlist, voltage_file, model_path, output_dir, 
                 norm_info_path=None, visualize=True, save_predictions=True):
    """
    Run inference on a SPICE netlist.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"[INFO] Loading model from {model_path}")
    model, norm_info = load_model(model_path, norm_info_path)
    
    # Get target size from norm_info
    target_size = norm_info.get('target_size', 256)
    
    # Preprocess input
    print(f"[INFO] Preprocessing input data...")
    X, base, original_grid_size = preprocess_input(spice_netlist, voltage_file, output_dir, target_size)
    
    # Run inference
    print(f"[INFO] Running inference...")
    with torch.no_grad():
        prediction = model(X)
        prediction = prediction.squeeze().numpy()
    
    # Denormalize if needed
    if norm_info.get('label_normalization') == 'per_sample_max':
        # For now, we'll use a reasonable scaling factor
        # In practice, you'd want to store the normalization factors during training
        prediction = prediction * 0.1  # Approximate scaling
    
    # Detect hotspots
    print(f"[INFO] Detecting hotspots...")
    hotspot_mask, hotspot_properties = detect_hotspots(prediction)
    
    # Save results
    if save_predictions:
        prediction_path = os.path.join(output_dir, f"predicted_ir_drop_{base}.csv")
        np.savetxt(prediction_path, prediction, delimiter=",")
        print(f"[INFO] Prediction saved to {prediction_path}")
        
        # Save hotspot information
        hotspot_info = {
            'num_hotspots': len(hotspot_properties),
            'hotspot_properties': [
                {
                    'id': prop['id'],
                    'intensity': float(prop['intensity']),
                    'area': int(prop['area']),
                    'center': [float(prop['center'][0]), float(prop['center'][1])]
                }
                for prop in hotspot_properties
            ]
        }
        
        hotspot_path = os.path.join(output_dir, f"hotspot_info_{base}.json")
        with open(hotspot_path, 'w') as f:
            json.dump(hotspot_info, f, indent=2)
        print(f"[INFO] Hotspot information saved to {hotspot_path}")
    
    # Visualize results
    if visualize:
        vis_path = os.path.join(output_dir, f"visualization_{base}.png")
        visualize_results(prediction, hotspot_properties=hotspot_properties, 
                         save_path=vis_path, title=f"IR Drop Prediction - {base}")
    
    # Print summary
    print(f"\n[SUMMARY] Inference Results for {base}:")
    print(f"  - Original grid size: {original_grid_size}x{original_grid_size}")
    print(f"  - Processed grid size: {target_size}x{target_size}")
    print(f"  - Max IR drop: {np.max(prediction):.6f}")
    print(f"  - Mean IR drop: {np.mean(prediction):.6f}")
    print(f"  - Detected hotspots: {len(hotspot_properties)}")
    
    if hotspot_properties:
        print(f"  - Hotspot details:")
        for prop in hotspot_properties:
            print(f"    * Hotspot {prop['id']}: intensity={prop['intensity']:.6f}, "
                  f"area={prop['area']}, center=({prop['center'][0]:.1f}, {prop['center'][1]:.1f})")
    
    return prediction, hotspot_properties

def batch_inference(input_dir, model_path, output_dir, norm_info_path=None):
    """
    Run inference on all SPICE files in a directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all SPICE files
    spice_files = [f for f in os.listdir(input_dir) if f.endswith('.sp')]
    
    if not spice_files:
        print(f"[ERROR] No SPICE files found in {input_dir}")
        return
    
    print(f"[INFO] Found {len(spice_files)} SPICE files for batch inference")
    
    results = {}
    
    for spice_file in spice_files:
        print(f"\n[INFO] Processing {spice_file}...")
        
        spice_path = os.path.join(input_dir, spice_file)
        base = os.path.splitext(spice_file)[0]
        voltage_file = os.path.join(input_dir, "generated_features", f"{base}.voltage")
        
        # Check if voltage file exists, if not solve MNA
        if not os.path.exists(voltage_file):
            print(f"[INFO] Solving MNA for {spice_file}...")
            from ir_solver_sparse import solve_ir_drop
            solve_ir_drop(spice_path, voltage_file)
        
        # Run inference
        try:
            prediction, hotspots = run_inference(
                spice_path, voltage_file, model_path, output_dir,
                norm_info_path, visualize=False, save_predictions=True
            )
            
            results[base] = {
                'prediction': prediction,
                'hotspots': hotspots,
                'max_ir_drop': float(np.max(prediction)),
                'mean_ir_drop': float(np.mean(prediction)),
                'num_hotspots': len(hotspots)
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to process {spice_file}: {e}")
            continue
    
    # Save batch results summary
    summary_path = os.path.join(output_dir, "batch_inference_summary.json")
    summary = {
        'total_files': len(spice_files),
        'successful_inferences': len(results),
        'results': results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[SUMMARY] Batch inference completed:")
    print(f"  - Total files: {len(spice_files)}")
    print(f"  - Successful: {len(results)}")
    print(f"  - Summary saved to: {summary_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run inference with enhanced IR drop prediction model")
    ap.add_argument("-spice", type=str, help="Path to SPICE netlist file (.sp)")
    ap.add_argument("-voltage", type=str, help="Path to voltage output file from MNA solver")
    ap.add_argument("-model", type=str, required=True, help="Path to trained model (.pt)")
    ap.add_argument("-output", type=str, required=True, help="Directory to save results")
    ap.add_argument("-norm_info", type=str, help="Path to normalization info file (.json)")
    ap.add_argument("--batch", action="store_true", help="Run batch inference on directory")
    ap.add_argument("--input_dir", type=str, help="Input directory for batch inference")
    ap.add_argument("--no_viz", action="store_true", help="Disable visualization")
    ap.add_argument("--no_save", action="store_true", help="Disable saving predictions")
    args = ap.parse_args()

    if args.batch:
        if not args.input_dir:
            print("[ERROR] --input_dir required for batch inference")
            sys.exit(1)
        batch_inference(args.input_dir, args.model, args.output, args.norm_info)
    else:
        if not args.spice or not args.voltage:
            print("[ERROR] -spice and -voltage required for single file inference")
            sys.exit(1)
        run_inference(args.spice, args.voltage, args.model, args.output,
                     args.norm_info, visualize=not args.no_viz, save_predictions=not args.no_save)

