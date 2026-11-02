# Review and restoration notes

The original archive was inspected as text before execution. No obvious credential access, shell execution, or data exfiltration was found in the two Python scripts. The HTML export referenced external JavaScript libraries and was omitted from the executable project.

## Repairs in November 2026

- Training now honors the requested epoch count.
- Validation uses the selected device instead of forcing CUDA.
- GPU arguments accept an explicit false value; CPU is the default.
- Both supported architectures locate their first linear classifier layer correctly, including AlexNet's leading dropout.
- Classifier dropout layers have unique names.
- Saved metadata records the selected architecture and hidden size.
- Optimizer state is obtained by calling `state_dict()`.
- Checkpoints contain the full model state dictionary and basic metadata rather than serialized model objects.
- Prediction uses restricted `torch.load(..., weights_only=True)` and never retries with unsafe pickle loading. Legacy checkpoints require retraining.
- Image preprocessing handles grayscale and alpha-channel images by converting to RGB and returns float32 data.
- The supported top-k range is checked.
- Unused dependencies were removed; requirements record the versions used for CPU verification.

## Validation and limits

Five regression tests passed on Linux/Python 3.12/CPU, including model save/reload checks for AlexNet and VGG16. A separate small real-image CLI run completed training, validation, testing, checkpoint saving, and prediction using pretrained AlexNet.

These checks do not establish full-dataset accuracy, CUDA behavior, or Windows compatibility. The tiny training subset was intended to verify execution only. Restricted checkpoint loading reduces pickle-related risk but is not a guarantee that arbitrary files are harmless; use checkpoints from a trusted source.

The original source and archive were preserved. Endpoint-protection settings were not changed.
