# Public trained model

Downloadable validation-selected ResNet18 checkpoint.

| Measure | Measured result |
| --- | ---: |
| First-choice test accuracy | 91.09% (746/819) |
| Top-five test accuracy | 98.29% (805/819) |
| Selected validation accuracy | 94.37% |
| Change from previously reported first-choice accuracy | +0.12 percentage points |

Four predefined configurations were compared on the cleaned validation set. Horizontal-flip training views and longer training were explored. The test split was evaluated after selection. Earlier test results were already known; this is not a new independent dataset. Results describe one seed and this coursework split.

Download the model asset, install the repository dependencies, and download the category-name mapping from https://raw.githubusercontent.com/udacity/aipnd-project/master/cat_to_name.json. Supply your own flower photo:

    python predict.py --checkpoint flower-classifier-resnet18.pth --image YOUR_FLOWER.jpg --cat_names cat_to_name.json --top_k 5

Prediction requires the checkpoint, category mapping, and one photo. The training dataset is unnecessary for prediction. The checkpoint uses restricted weights-only loading. Check SHA256SUMS after downloading. Regression tests and a real-image prediction command passed before publication.

No flower images are included. Dataset provenance: Udacity distribution of Oxford Flowers 102. Initial model weights: torchvision ResNet18 ImageNet weights. Attached JSON files record configuration, class-level metrics, learning history, and the source commit.

## Verification record

- Successful run: https://github.com/7cvega15-code/Image-Classifier/actions/runs/41674454513
- Source commit: d485e85abcc79a10af9c37b5987c96612be2ed0c
- Release: https://github.com/7cvega15-code/Image-Classifier/releases/tag/model-v2-41674454513
- Model asset size: 47308141 bytes
- Fourteen regression tests passed, followed by training, validation-selected checkpoint evaluation, real-image CLI prediction, checksum verification, and successful asset upload.

The previous report's first-choice test result differs by one image. This small change should not be described as proven performance improvement on new datasets. The downloaded model is compatible with the existing predict.py command and uses restricted weights-only loading.
