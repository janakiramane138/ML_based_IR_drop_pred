# Models

Neural network architectures and trained checkpoints for IR drop prediction.

## Files
- `resunet.py`: ResUNet and `VariableSizeResUNet` implementations.
- `resunet_attention.py`: ResUNet with attention.
- `unet.py`: Standard UNet.
- `*_norm_info.json`: Normalization parameters for corresponding checkpoints.
- `*.pt`: Trained model weights (e.g., `enhanced_model_best.pt`, `unet_model.pt`, `hotspot_*` variants).
- `__init__.py`: Module marker.

## Usage
```python
import torch
from models.resunet import ResUNet, VariableSizeResUNet
model = ResUNet()  # or VariableSizeResUNet()
model.load_state_dict(torch.load("models/enhanced_model_best.pt", map_location="cpu"))
model.eval()
```

## Notes
- Variable-size models accept inputs of shape `[B, 3, H, W]` with arbitrary `H, W`.
- Keep `*_norm_info.json` alongside its `.pt` checkpoint for consistent scaling at inference.
