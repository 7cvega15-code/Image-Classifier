# Flower Image Classifier

A Udacity image-classification project using PyTorch and torchvision transfer learning to classify 102 flower categories. Originally completed in 2024 and updated in November 2026.

**Status:** Trained and evaluated on the full Udacity split. Pretrained AlexNet achieved 64.10% first-choice and 86.57% top-five accuracy on 819 held-out test images. See [measured results](results/README.md) for the protocol and limitations.

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
python train.py --dir flowers --arch alexnet --hidden_units 512 --epochs 5 --seed 42 --save_dir checkpoint.pth
python predict.py --image flowers/test/1/image_06743.jpg --checkpoint checkpoint.pth --cat_names cat_to_name.json --top_k 5
```

Replace the example image path with a file that exists locally. `--arch vgg16` is also supported. CPU is the default; add `--gpu` to request CUDA when available, or use `--gpu false` explicitly. The first training run downloads ImageNet weights from PyTorch. `--no_pretrained` disables that download and uses random initialization, intended for offline workflow checks.

Prediction rebuilds the network from checkpoint metadata and does not download pretrained weights. Legacy checkpoints that serialize model objects are unsupported; retrain with these scripts instead of enabling unsafe pickle loading.

## Verification completed November 2, 2026

```bash
python -m unittest discover -s tests -v
```

Eight regression tests passed. Checks cover GPU argument parsing, training seeds, RGB/grayscale/alpha preprocessing, CPU validation, honoring epochs, checkpoint round trips for AlexNet and VGG16, top-k output, legacy-checkpoint rejection, class-mapping validation, and evaluation metrics against known predictions. Tests use random weights and make no network requests.

A small real-image smoke run first checked the full CLI workflow. The subsequent full-data run trained pretrained AlexNet on all 6,552 training images for five epochs and evaluated the final model on all 819 test images.

| Held-out test metric | Result |
| --- | ---: |
| First-choice accuracy | 64.10% (525/819) |
| Top-five accuracy | 86.57% (709/819) |

```bash
python evaluate.py --checkpoint checkpoint.pth --data_dir flowers --cat_names cat_to_name.json --output evaluation.json
```

See [results and example predictions](results/README.md) and [machine-readable metrics](results/evaluation.json). The test set has no exact image duplicates against training or validation. One train/validation duplicate is documented in the data-integrity report. Results apply to this run and split; CUDA, Windows, and arbitrary-photo performance have not been validated.

## Project notes

The original exported HTML notebook report was omitted from the executable project because it loads external JavaScript libraries. The original source remains in Git history. See [SECURITY_REVIEW.md](SECURITY_REVIEW.md) for the repaired issues and remaining limits.
