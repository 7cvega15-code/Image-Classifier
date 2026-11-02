# Image Classifier Project

A 2024 image-classification project using PyTorch and torchvision transfer learning to classify flower images. The original `train.py` and `predict.py` from the supplied archive are included unchanged.

## Archive contents and omissions

The source archive also contained an exported HTML project report. It is not included here because it is not needed to run the Python scripts and loads JavaScript libraries from external CDNs. The archive remains unchanged in Google Drive.

The archive did not include the flower image dataset, `cat_to_name.json`, or a trained checkpoint. Those inputs are required by the scripts and should be added only after reviewing their source and licensing.

## Requirements

Python 3.

Install compatible versions of:

- PyTorch
- torchvision
- NumPy
- Pillow
- Matplotlib
- pandas (imported by `predict.py`)

Version pins are intentionally omitted because PyTorch/torchvision compatibility depends on the operating system and CPU/GPU setup.

## Expected layout

```text
flowers/
  train/<class_name>/*.jpg
  valid/<class_name>/*.jpg
  test/<class_name>/*.jpg
cat_to_name.json
```

## Usage

The commands below describe the scripts' expected interface; this repository package has not been executed. Review the security notes and known issues before running either script.

```bash
python train.py --dir flowers --arch vgg16 --epochs 5 --save_dir checkpoint.pth
python predict.py --image flowers/test/1/image_06764.jpg --checkpoint checkpoint.pth --cat_names cat_to_name.json --top_k 5
```

## Security note

`predict.py` calls `torch.load` on a checkpoint path. PyTorch checkpoints may use Python pickle deserialization, which can execute code when loading a malicious checkpoint. Load only checkpoints from a trusted source, and use a safer weights-only format/API where supported. Do not commit checkpoints or install/run anything from this project before reviewing it.

See [SECURITY_REVIEW.md](SECURITY_REVIEW.md) for the static review and functional issues found.
