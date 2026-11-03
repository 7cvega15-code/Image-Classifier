# Flower Image Classifier

A Udacity image-classification project using PyTorch and torchvision transfer learning to classify 102 flower categories. Originally completed in 2024 and updated in November 2026.

**Status:** The publicly downloadable ResNet18 model achieved **91.09% first-choice** and **98.29% top-five accuracy** on 819 flower test images. Fourteen regression tests passed on a clean GitHub CPU runner. See [the model download](https://github.com/7cvega15-code/Image-Classifier/releases/tag/model-v2-41674454513) and [measured results](results/README.md).

## What the project demonstrates

- Image augmentation, resizing, cropping, RGB conversion, and ImageNet normalization
- Transfer learning with pretrained ResNet18, AlexNet, or VGG16
- Validation-only model selection with cached features and exact-duplicate filtering
- Training a replacement classifier while freezing the feature extractor
- CPU execution, optional CUDA, validation, and top-k flower predictions
- Checkpoints containing model weights and basic metadata, loaded with `weights_only=True`
- Regression tests for preprocessing, device selection, epoch count, and checkpoint round trips

## Try the trained model

Download or clone this repository, then follow the Setup section below to install Python dependencies. You can run predictions without downloading the training dataset or training a model:

1. Download [flower-classifier-resnet18.pth](https://github.com/7cvega15-code/Image-Classifier/releases/download/model-v2-41674454513/flower-classifier-resnet18.pth) from the [public model release](https://github.com/7cvega15-code/Image-Classifier/releases/tag/model-v2-41674454513).
2. Download [cat_to_name.json](https://raw.githubusercontent.com/udacity/aipnd-project/master/cat_to_name.json) and save it beside the scripts.
3. Supply a flower photo and run:

```bash
python predict.py --checkpoint flower-classifier-resnet18.pth --image path/to/flower.jpg --cat_names cat_to_name.json --top_k 5
```

Replace the image path with your own file. The release includes [SHA-256 checksums](https://github.com/7cvega15-code/Image-Classifier/releases/download/model-v2-41674454513/SHA256SUMS), exact evaluation counts, and the validation-search history. This download is a public model prerelease; its predictions were verified on GitHub's Linux/CPU runner. Windows and GPU execution remain unverified.

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

For the current ResNet18 workflow, compare the three predefined classifier heads using validation data and save the selected model:

```bash
python tune.py --data_dir flowers --checkpoint checkpoint-resnet18.pth --report validation-search.json --seed 42
python predict.py --image flowers/test/1/image_06743.jpg --checkpoint checkpoint-resnet18.pth --cat_names cat_to_name.json --top_k 5
python evaluate.py --checkpoint checkpoint-resnet18.pth --data_dir flowers --cat_names cat_to_name.json --output evaluation.json
```

Replace the example image path with an existing local file. `tune.py` uses CPU, downloads official PyTorch ResNet18 ImageNet weights, and reads only the training and validation folders. It excludes validation images that exactly duplicate training images without changing the files on disk. Test evaluation is a separate command run after model selection.

The original coursework-style AlexNet/VGG16 workflow remains available:

```bash
python train.py --dir flowers --arch alexnet --hidden_units 512 --epochs 5 --seed 42 --save_dir checkpoint-alexnet.pth
```

CPU is the default. `train.py` and `predict.py` accept `--gpu` to request CUDA when available, or `--gpu false`. In `train.py`, `--no_pretrained` uses random initialization for offline software checks. Prediction rebuilds the network from saved weights and metadata without downloading weights. Unsafe legacy model-object checkpoints require retraining.

## Verification completed November 2, 2026

```bash
python -m unittest discover -s tests -v
```

Fourteen regression tests passed. Checks cover argument parsing, seeds, image preprocessing, CPU validation, epoch handling, AlexNet/VGG16/ResNet18 checkpoint round trips, top-k predictions, unsupported legacy checkpoints, category mapping, evaluation metrics against known predictions, exact-duplicate filtering, and reproduction of the selected classifier head's validation score. Tests make no network requests.

Full-data training and test evaluation are complete for the AlexNet baseline and the validation-selected ResNet18 models. A subsequent bounded comparison and public model release were verified on GitHub Actions. See [measured results and example predictions](results/README.md) for the current accuracy, comparison, data-quality caveats, and reproduction commands.

## Project notes

The original exported HTML notebook report was omitted from the executable project because it loads external JavaScript libraries. The original source remains in Git history. See [SECURITY_REVIEW.md](SECURITY_REVIEW.md) for the repaired issues and remaining limits.

