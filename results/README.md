# Full-data training and evaluation: November 2, 2026

A pretrained AlexNet feature extractor was frozen and a replacement classifier was trained on all 6,552 training images for five epochs. The classifier used 512 hidden units, Adam with learning rate 0.001, batch size 32, and random seed 42. Validation monitored progress on 818 images. The final five-epoch checkpoint was evaluated on all 819 test images, without tuning against test results.

| Measure | Result |
| --- | ---: |
| Test first-choice accuracy | 64.10% (525/819) |
| Test top-five accuracy | 86.57% (709/819) |
| Mean accuracy across categories | 58.95% |
| Mean negative log likelihood | 1.4826 |
| Test flower categories | 102 |
| Training time in this environment | 403.3 seconds |

First-choice accuracy means the model's highest-probability label was correct. Top-five accuracy means the correct label appeared among its five highest-probability predictions. These are results for the provided Udacity test split and this single run, not a claim about performance on arbitrary photos or other environments.

## Example predictions

These six test examples were selected using random seed 42, without selecting for correct predictions. Probabilities are model outputs and are not calibrated confidence guarantees.

| Image | Actual flower | First prediction | Probability | Correct |
| --- | --- | --- | ---: | --- |
| `test/102/image_08042.jpg` | blackberry lily | blackberry lily | 79.1% | Yes |
| `test/26/image_06526.jpg` | corn poppy | rose | 26.0% | No |
| `test/43/image_02433.jpg` | sword lily | rose | 12.2% | No |
| `test/46/image_01090.jpg` | wallflower | watercress | 56.9% | No |
| `test/83/image_01755.jpg` | hibiscus | hibiscus | 78.2% | Yes |
| `test/93/image_06017.jpg` | ball moss | ball moss | 5.2% | Yes |

## Data quality and limitations

All 8,189 JPEGs passed image verification during restoration. A SHA-256 comparison found one identical image shared by the training and validation splits (`train/31/image_08077.jpg` and `valid/31/image_08067.jpg`). The original splits were retained. No test image was an exact byte duplicate of an image in either training or validation. This check does not detect resized or visually similar duplicates.

The results cover one architecture, one seed, and the provided split. VGG16 checkpoint handling was regression-tested, but no full VGG16 training benchmark was run. GPU, Windows, out-of-distribution images, and probability calibration remain unverified.

Eight software regression tests passed, including tests that verify evaluation counts and accuracy aggregation against deliberately known predictions. See `tests/`.

## Reproduce evaluation

After training or obtaining the matching checkpoint and downloading the category mapping:

```bash
python evaluate.py --checkpoint checkpoint.pth --data_dir flowers --cat_names cat_to_name.json --output evaluation.json
```

The inference checkpoint contains the same trained model weights with optimizer state omitted to reduce its download size. Its SHA-256 is recorded in `evaluation.json`. Large model and dataset files are excluded from Git.

`evaluation.json` contains per-category counts and the six example predictions. `dataset-integrity.json` records the source archive checksum and duplicate finding. `training-log.txt` records the training progress; its validation figures are batch-averaged monitoring values, while the test metrics above are computed per image.
