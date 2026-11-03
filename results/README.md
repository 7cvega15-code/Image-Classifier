# Flower classifier: measured results

## Public model release: November 3, 2026

[Download the trained model](https://github.com/7cvega15-code/Image-Classifier/releases/download/model-v2-41674454513/flower-classifier-resnet18.pth) · [Release and raw metrics](https://github.com/7cvega15-code/Image-Classifier/releases/tag/model-v2-41674454513) · [Successful training/test workflow](https://github.com/7cvega15-code/Image-Classifier/actions/runs/41674454513)

| Measure | Previous run | Published model |
| --- | ---: | ---: |
| First-choice test accuracy | 90.96% (745/819) | **91.09% (746/819)** |
| Top-five test accuracy | 97.44% (798/819) | **98.29% (805/819)** |

The first-choice improvement is **one additional correctly classified image**, or 0.12 percentage points. Top-five accuracy improved by seven images, or 0.85 percentage points. These are descriptive differences on the same coursework test split, not evidence of a statistically established generalization improvement.

Four predefined configurations compared the previous head with horizontal-flip training views, lower dropout, a wider head, and longer training limits. Candidate and epoch selection used the cleaned validation set only. The selected validation accuracy was 94.37%. Test evaluation followed selection; earlier test results were already known.

All fourteen regression tests, a real-image CLI prediction command, and release-file checksum checks passed on a clean GitHub CPU runner. The public prerelease contains the trained checkpoint, evaluation.json, validation-search.json, and SHA256SUMS. See [published-model.md](published-model.md) for the release record. The historical results below are preserved.


## November 2 model: ResNet18

The November 2 model achieved **90.96% first-choice accuracy** and **97.44% top-five accuracy** on all 819 images in the supplied test split, across 102 flower categories.

| Model | First-choice test accuracy | Top-five test accuracy |
| --- | ---: | ---: |
| Initial AlexNet baseline | 64.10% (525/819) | 86.57% (709/819) |
| Validation-selected ResNet18 | 90.96% (745/819) | 97.44% (798/819) |

The first-choice improvement is **26.86 percentage points**. Architecture, classifier head, preprocessing, and training procedure changed together. This is not an isolated experiment on the effect of additional epochs alone.

## Training and selection protocol

- Frozen ImageNet-pretrained ResNet18 encoder with deterministic resize/center-crop preprocessing and cached 512-dimensional image features.
- All 6,552 training images used for fitting classifier heads.
- Three predefined head configurations compared, with up to 40 epochs and validation-based early stopping.
- One exact training-image duplicate excluded from validation, leaving 817 validation images. Original image files were preserved.
- Highest validation accuracy selected the candidate and epoch; lower validation loss broke ties.
- Selected head: 512 hidden units, dropout 0.2, Adam learning rate 0.001, weight decay 0.0001, training batch size 128, seed 42, epoch 27.
- Selected validation accuracy: 93.88%.
- The tuning program never loads the test folder. The selected checkpoint was then evaluated on the unchanged 819-image test split.
- No test image was an exact byte duplicate of a training or validation image in the preceding dataset-integrity check.

The baseline test result was already known from the prior restoration run. It was not used to choose the ResNet18 head or epoch; selection used validation results. The test set was not used for repeated hyperparameter searching.

Mean test accuracy across categories: **90.11%**. Mean negative log likelihood: **0.3341**. Feature extraction and validation search took 153.4 seconds in this CPU environment.

## Example predictions

Six examples selected using random seed 42, without choosing for correctness. Probabilities are model outputs and are not calibrated confidence guarantees.

| Image | Actual flower | First prediction | Probability | Correct |
| --- | --- | --- | ---: | --- |
| `test/102/image_08042.jpg` | blackberry lily | blackberry lily | 99.9% | Yes |
| `test/26/image_06526.jpg` | corn poppy | corn poppy | 56.7% | Yes |
| `test/43/image_02433.jpg` | sword lily | bougainvillea | 60.8% | No |
| `test/46/image_01090.jpg` | wallflower | wallflower | 99.0% | Yes |
| `test/83/image_01755.jpg` | hibiscus | hibiscus | 97.2% | Yes |
| `test/93/image_06017.jpg` | ball moss | ball moss | 99.9% | Yes |

## Reproduce

```bash
python tune.py --data_dir flowers --checkpoint checkpoint-resnet18.pth --report validation-search.json --seed 42
python evaluate.py --checkpoint checkpoint-resnet18.pth --data_dir flowers --cat_names cat_to_name.json --output resnet18-evaluation.json
```

The first command downloads official PyTorch pretrained weights and reads only `train/` and `valid/`. The second command evaluates `test/` after selection. The feature cache is held in memory and shared across the three candidate runs. Eleven regression tests passed, including checkpoint loading, inference preprocessing, validation duplicate filtering, selected-head score reproduction, and metric aggregation.

## Scope and limitations

This is one seed and one supplied dataset split. The comparison is useful coursework evidence, not a controlled study of every changed factor. It does not establish accuracy on arbitrary photos, calibrated probabilities, Windows, or GPU execution. Original splits are preserved on disk; one train/validation duplicate was filtered from this selection run. Near-duplicate images were not assessed.

Machine-readable outputs: [current test metrics](resnet18-evaluation.json), [validation search and learning history](validation-search.json), and [original AlexNet metrics](evaluation.json). The [original AlexNet report](alexnet-baseline.md) preserves the baseline protocol and its data-quality caveat. The saved model's SHA-256 is in the current test-metrics file. Model files and image data are excluded from Git.
