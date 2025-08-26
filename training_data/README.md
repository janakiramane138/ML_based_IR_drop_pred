# Training Data

Raw SPICE testcases and generated feature maps for training.

## Contents
- `data_pointXX.sp`: SPICE circuit files (01–99).
- `generated_features/`: Preprocessed features and labels for each data point.

## Generate Features
Use the generator to create CSV maps and grid info:
```bash
cd ../src
python data_generation.py --sp_dir ../training_data --out_dir ../training_data/generated_features
```

## Feature Naming (in `generated_features/`)
- `current_map_data_pointXX.csv`
- `pdn_density_map_data_pointXX.csv`
- `voltage_source_map_data_pointXX.csv`
- `ir_drop_map_data_pointXX.csv` (label)
- `grid_info_data_pointXX.txt` (H, W, grid config)
