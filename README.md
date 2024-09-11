# bitvit
Code implementation for Efficient Quantized Vision Transformers for Deepfake Detection

## Features

- `models.bitlinear`: code implementations of the BitLinear layer, and RMSNorm
- `models.dataloader`: custom dataloader class for faster training
- `models.deepfake`: custom deepfake classifier, ViT wrapper, and EfficientNet baseline detector
- `models.distill`: code implementation for knowledge distillation with hard-label loss
- `models.utils`: utility functions for parameter stats and model size estimation
- `models.vit`: code implementation of LinViT (ViT) and BitViT models using SPT, sincos2d, and GAP

## Requirements

Models:
  - `torch` >= v2.1.0
  - `einops` >= 0.8.0

See `environment.yml` for full environment used for the thesis research.

## License
MIT
