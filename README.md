# ML_based_IR_drop_pred

This repository provides a **Machine Learning–based IR drop prediction framework** for VLSI power grids.  
It combines **Modified Nodal Analysis (MNA)** for ground truth voltage solving with a **ResU-Net** deep learning model to predict IR drop heatmaps from circuit features.

## RUN commands
#train
python src/training.py -input training_data -output src/models/hotspot_fixed_resunet.pt --epochs 50 --lr 3e-4 --batch 8 --alpha 10.0 --device cpu
#evaluate/inference
python src/evaluate.py from_spice --spice_dir benchmark --ml_model src/models/hotspot_fixed_resunet.pt --pred_out benchmark_predictions --gt_out benchmark_predictions --device cpu
