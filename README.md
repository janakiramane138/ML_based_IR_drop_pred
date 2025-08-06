# ML_based_ir_drop_pred
IR drop prediction framework combining Modified Nodal Analysis (MNA) for ground truth voltage solving and a UNet-based deep learning model to predict IR drop heatmaps from circuit features, enabling fast and accurate estimation for VLSI power grids.

# IR Drop ML Prediction Pipeline

## Phase 1: Static IR Drop Solver
- Run with:
  python3 ir_solver_sparse.py --input_file <netlist.sp> --output_file <output.voltage>

## Phase 2: Data Generation
- Generate maps:
  python3 data_generation_fixed.py -spice_netlist <file.sp> -voltage_file <file.voltage> -output <output_dir>

## Training
- Train U-Net on all CSVs in a directory:
  python3 training.py -input features -output unet_model.pt

## Inference
- Predict IR drop map:
  python3 inference.py -spice_file <test.sp> -ml_model unet_model.pt -output features

## Evaluation
- Evaluate model predictions:
  python3 evaluate.py -pred_dir features -true_dir features


## Outputs
- Predicted: `predicted_ir_drop_map_*.csv`
- Ground truth: `ir_drop_map_*.csv`
