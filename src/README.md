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
pip install -r ../requirements.txt
```

## Training
- Enhanced:
```bash
python training_enhanced.py --feature_dir ../training_data/generated_features --model resunet --epochs 100 --batch_size 4
```
- Baseline:
```bash
python training.py --feature_dir ../training_data/generated_features --epochs 50
```

## Inference
```bash
python inference.py --model_path models/enhanced_model_best.pt --input_dir ../benchmark/generated_features --out_dir ../benchmark_predictions
```

## Evaluation
```bash
python evaluate.py --model_path models/enhanced_model_best.pt --test_dir ../benchmark/generated_features
```

## Data Preparation
```bash
python data_generation.py --sp_dir ../training_data --out_dir ../training_data/generated_features
```
