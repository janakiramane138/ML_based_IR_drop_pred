# Benchmark Set

Held-out SPICE testcases for model evaluation and reproducible comparisons.

## Contents
- `testcaseX.sp`: Benchmark circuits (1,2,3,4,5,6,11,12,17,18).
- `generated_features/`: Feature maps and `.voltage` solutions derived from each testcase.

## Prepare Features
If not present, generate from `.sp` files:
```bash
cd ../src
python data_generation.py --sp_dir ../benchmark --out_dir ../benchmark/generated_features
```

## Evaluate
```bash
cd ../src
python evaluate.py --model_path models/enhanced_model_best.pt --test_dir ../benchmark/generated_features
```
