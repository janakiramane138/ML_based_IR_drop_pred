# Benchmark Predictions

Predicted IR drop maps for benchmark testcases, alongside the corresponding inputs and ground truth for convenience.

## Contents
- `predicted_ir_drop_map_testcaseX.csv`: Model outputs.
- `ir_drop_map_testcaseX.csv`: Ground truth labels.
- `current_map_testcaseX.csv`, `pdn_density_map_testcaseX.csv`, `voltage_source_map_testcaseX.csv`: Input features.
- `testcaseX.voltage`: Reference voltage solutions.

## Reproduce
```bash
cd ../src
python inference.py --model_path models/enhanced_model_best.pt --input_dir ../benchmark/generated_features --out_dir ../benchmark_predictions
```

## Compare
```bash
cd ../src
python evaluate.py --model_path models/enhanced_model_best.pt --test_dir ../benchmark/generated_features
```
