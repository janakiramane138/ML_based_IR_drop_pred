# Generated Features (Training)

Preprocessed feature tensors and labels for each training data point.

## Files per datapoint
- `current_map_data_pointXX.csv`: Current distribution.
- `pdn_density_map_data_pointXX.csv`: PDN density.
- `voltage_source_map_data_pointXX.csv`: Voltage source map.
- `ir_drop_map_data_pointXX.csv`: Ground truth IR drop.
- `grid_info_data_pointXX.txt`: Grid size/config.

## Format
- CSV grids with shape `[H, W]` (float).
- Typical model input stack: `[3, H, W]` = [current, density, vsrc].
- Labels: `[1, H, W]` IR drop.

## Notes
- Some datapoints are large (up to ~25MB per CSV). Ensure adequate disk and RAM.
- Keep file naming consistent for dataloader discovery.
