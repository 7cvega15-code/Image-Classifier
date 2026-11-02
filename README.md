# Flower Image Classifier

A recovered Udacity image-classification project using PyTorch and torchvision transfer learning to classify 102 flower categories. The 2024 source was restored and repaired in November 2026.

**Status:** CPU workflow checks pass. Training, validation, checkpoint saving/loading, and prediction have been exercised on a small sample of real flower images. A full training run and held-out accuracy benchmark are still pending; this repository does not claim a production-ready or high-accuracy model.

## What the project demonstrates

- Image augmentation, resizing, cropping, RGB conversion, and ImageNet normalization
- Transfer learning with pretrained AlexNet or VGG16
- Training a replacement classifier while freezing the feature extractor
- CPU execution, optional CUDA, validation, and top-k flower predictions
- Checkpoints containing model weights and basic metadata, loaded with `weights_only=True`
- Regression tests for preprocessing, device selection, epoch count, and checkpoint round trips

## Setup

Tested on Linux with Python 3.12 and CPU PyTorch. Create and activate a virtual environment, then install the tested dependencies:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell activation instead:
# .venv\Scripts\Activate.ps1

python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

CUDA and Windows execution have not been tested in this restoration. Use the appropriate PyTorch installation for your platform if the CPU wheels above are unavailable.

## Dataset and category names

Download [Udacity's flower archive](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz) and extract its contents into `flowers/`. Download [Udacity's category-name mapping](https://raw.githubusercontent.com/udacity/aipnd-project/master/cat_to_name.json) as `cat_to_name.json` beside the scripts.

```text
flowers/
  train/<category-number>/*.jpg
  valid/<category-number>/*.jpg
  test/<category-number>/*.jpg
cat_to_name.json
train.py
predict.py
```

The inspected archive contained 6,552 training images, 818 validation images, and 819 test images, with 102 categories in each split. All 8,189 JPEGs passed Pillow's image verification. This inspection used the original public Udacity download, which matched the uploaded archive's byte length; the Drive copy exceeded the connection's download limit and was not byte-compared.

Image provenance: [Oxford 102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), distributed for the Udacity coursework. Follow the dataset provider's usage terms. Data, downloaded pretrained weights, category mappings, and generated checkpoints are not committed to this repository.

## Train and predict

```bash
python train.py --dir flowers --arch alexnet --hidden_units 512 --epochs 5 --save_dir checkpoint.pth
python predict.py --image flowers/test/1/image_06743.jpg --checkpoint checkpoint.pth --cat_names cat_to_name.json --top_k 5
```

Replace the example image path with a file that exists locally. `--arch vgg16` is also supported. CPU is the default; add `--gpu` to request CUDA when available, or use `--gpu false` explicitly. The first training run downloads ImageNet weights from PyTorch. `--no_pretrained` disables that download and uses random initialization, intended for offline workflow checks.

Prediction rebuilds the network from checkpoint metadata and does not download pretrained weights. Legacy checkpoints that serialize model objects are unsupported; retrain with these scripts instead of enabling unsafe pickle loading.

## Verification completed November 2, 2026

```bash
python -m unittest discover -s tests -v
```

Five regression tests passed, including subcases for both AlexNet and VGG16. They cover GPU argument parsing, RGB/grayscale/alpha image preprocessing, CPU validation, honoring the epoch argument, checkpoint metadata and optimizer serialization, matching predictions before/after reload, valid top-k output, and rejection of an unsupported legacy checkpoint. These tests use random weights and make no network requests.

A separate CLI smoke run used pretrained AlexNet, one epoch, 32 hidden units, and one image from each of the 102 categories in each split (306 real images total). Training, CPU validation/testing, checkpoint saving, and a separate prediction process completed successfully. Both scripts' `--help` commands also passed.

**The small CLI run tests the software workflow, not classification quality.** It is insufficient training for useful predictions. Full-dataset training, held-out accuracy, and example predictions from a properly trained model remain future work.

## Restoration notes

The original exported HTML notebook report was omitted from the executable project because it loads external JavaScript libraries. The original source remains in Git history. See [SECURITY_REVIEW.md](SECURITY_REVIEW.md) for the repaired issues and remaining limits.
