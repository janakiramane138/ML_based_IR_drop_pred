# Generated Features (Benchmark)

Features and labels for benchmark testcases.

## Files per testcase
- `current_map_testcaseX.csv`
- `pdn_density_map_testcaseX.csv`
- `voltage_source_map_testcaseX.csv`
- `ir_drop_map_testcaseX.csv` (ground truth)
- `testcaseX.voltage` (reference voltage solution)

## Format
- CSV grids `[H, W]` (float), `.voltage` text solution aligned to the same grid.
- Naming aligns with loaders used in `training_enhanced.py`.

## Usage
Point inference/evaluation here:
```bash
python ../src/inference.py --model_path ../src/models/enhanced_model_best.pt --input_dir . --out_dir ../../benchmark_predictions
```
