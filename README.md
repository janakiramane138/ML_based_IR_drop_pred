## ML-based IR Drop Prediction

End-to-end pipeline for IR drop prediction from SPICE netlists using deep learning. Includes data generation, training, evaluation, inference, benchmarks, and pretrained checkpoints.

### Repository structure
- `src/`: Source code (training, inference, evaluation, data generation, models, tools)
- `training_data/`: Raw SPICE files and generated features/labels
- `benchmark/`: Held-out SPICE testcases and generated features
- `benchmark_predictions/`: Predicted maps and reference inputs/labels
- `doc/`: Reports and write-ups

### Setup
```bash
# Python 3.9+ recommended
pip install -r src/requirements.txt
```

### Quickstart
- Train (enhanced, variable-size):
```bash
python src/training_enhanced.py -input training_data -output src/models/enhanced_model.pt --model variable_resunet --epochs 50 --batch 8 --augmentation_level synthetic --variable_size --num_synthetic 100 --device cpu
```

- Inference (batch over benchmark):
```bash
python src/inference.py -model src/models/enhanced_model_best.pt -output benchmark_predictions --batch --input_dir benchmark
```

- Evaluation (batch over benchmark):
```bash
python src/evaluate.py -model src/models/enhanced_model_best.pt -output evaluation_results --batch --input_dir benchmark
```

### Data preparation
Generate features from SPICE files:
```bash
python src/data_generation.py -spice_netlist training_data/data_point01.sp -voltage_file training_data/generated_features/data_point01.voltage -output training_data/generated_features
```

Batch examples (bash):
```bash
# Training set
for file in training_data/*.sp; do
  base=$(basename "$file" .sp)
  python src/data_generation.py -spice_netlist "$file" -voltage_file "training_data/generated_features/${base}.voltage" -output training_data/generated_features
done

# Benchmark set
for file in benchmark/*.sp; do
  base=$(basename "$file" .sp)
  python src/data_generation.py -spice_netlist "$file" -voltage_file "benchmark/generated_features/${base}.voltage" -output benchmark/generated_features
done
```

### Models
Pretrained and newly trained checkpoints in `src/models/`:
- `enhanced_model.pt`, `enhanced_model_best.pt`
- `hotspot_enhanced_resunet.pt`, `hotspot_enhanced_resunet_best.pt`

### Visualization
```bash
python src/tools/visualize_heatmap.py --input benchmark_predictions/predicted_ir_drop_map_testcase1.csv --output doc/pred_testcase1.png
```

### Benchmarks
Benchmark SPICE files are in `benchmark/` with corresponding generated features in `benchmark/generated_features/`. Example predictions and labels are in `benchmark_predictions/`.

### Documentation
See additional write-ups in `doc/`.

### Citation
If you use this repository in your work, please cite appropriately (add citation details here).
