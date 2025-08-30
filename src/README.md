# Source Code

Core implementation for ML-based IR drop prediction: data processing, training, evaluation, inference, and models.

## Contents
- `training_enhanced.py`: Enhanced training with variable-size support and custom loss.
- `training.py`: Baseline training loop.
- `data_generation.py`: Convert `.sp` to feature CSVs; determine grid sizes.
- `data_augmentation.py`: Augmented datasets, synthetic data generator, dataset combiners.
- `evaluate.py`: Metrics and evaluation utilities.
- `inference.py`: Batch inference over feature maps.
- `enhanced_loss.py`: Hotspot-oriented/custom loss terms.
- `ir_solver_sparse.py`: Sparse solver utilities.
- `models/`: Architectures and checkpoints.
- `tools/`: Visualization utilities.
- `__init__.py`: Module marker.

## Setup
```bash
pip install -r requirements.txt
```

## Training
### Enhanced Training (Recommended)
```bash
python training_enhanced.py -input training_data -output src/models/enhanced_model.pt --model variable_resunet --epochs 50 --batch 8 --augmentation_level synthetic --variable_size --num_synthetic 100 --device cpu
```

### Basic Training
```bash
python training.py --feature_dir ../training_data/generated_features --epochs 50
```

## Inference
### Single File Inference
```bash
python inference.py -spice benchmark/testcase1.sp -voltage benchmark/generated_features/testcase1.voltage -model models/enhanced_model_best.pt -output benchmark_predictions
```

### Batch Inference
```bash
python inference.py -model models/enhanced_model_best.pt -output benchmark_predictions --batch --input_dir benchmark
```

## Evaluation
### Single File Evaluation
```bash
python evaluate.py -spice benchmark/testcase1.sp -voltage benchmark/generated_features/testcase1.voltage -model models/enhanced_model_best.pt -output evaluation_results
```

### Batch Evaluation
```bash
python evaluate.py -model models/enhanced_model_best.pt -output evaluation_results --batch --input_dir benchmark
```

## Data Preparation
### Generate Features from SPICE Files
```bash
python data_generation.py -spice_netlist training_data/data_point01.sp -voltage_file training_data/generated_features/data_point01.voltage -output training_data/generated_features
```

### Batch Feature Generation
```bash
# For training data
for file in training_data/*.sp; do
    base=$(basename "$file" .sp)
    python data_generation.py -spice_netlist "$file" -voltage_file "training_data/generated_features/${base}.voltage" -output training_data/generated_features
done

# For benchmark data
for file in benchmark/*.sp; do
    base=$(basename "$file" .sp)
    python data_generation.py -spice_netlist "$file" -voltage_file "benchmark/generated_features/${base}.voltage" -output benchmark/generated_features
done
```

## Key Parameters

### Training Parameters
- `-input`: Directory containing training SPICE files
- `-output`: Path to save trained model (.pt)
- `--model`: Model architecture (unet, resunet, variable_resunet)
- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--augmentation_level`: Data augmentation level (standard, aggressive, synthetic)
- `--variable_size`: Enable variable size model support
- `--num_synthetic`: Number of synthetic samples to generate
- `--device`: Device to use (cpu, cuda)

### Inference Parameters
- `-spice`: Path to SPICE netlist file (.sp)
- `-voltage`: Path to voltage output file from MNA solver
- `-model`: Path to trained model (.pt)
- `-output`: Directory to save results
- `--batch`: Run batch inference on directory
- `--input_dir`: Input directory for batch inference

### Evaluation Parameters
- `-spice`: Path to SPICE netlist file (.sp)
- `-voltage`: Path to voltage output file from MNA solver
- `-model`: Path to trained model (.pt)
- `-output`: Directory to save evaluation results
- `--batch`: Run batch evaluation on directory
- `--input_dir`: Input directory for batch evaluation
