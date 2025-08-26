# Tools

Utility scripts for visualization and analysis.

## Scripts
- `visualize_heatmap.py`: Render heatmaps for predicted/ground-truth IR drop.

## Example
```bash
python -m tools.visualize_heatmap --pred ../benchmark_predictions/predicted_ir_drop_map_testcase6.csv --out heatmap_test6.png
```

Args (typical):
- `--pred`: path to predicted IR drop map CSV.
- `--gt` (optional): path to ground-truth CSV for side-by-side view.
- `--cmap`: matplotlib colormap (default: hot).
- `--out`: output image path.
