"""Bounded validation-only comparison, then one held-out test evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torchvision import datasets
from evaluate import evaluate
from resnet_classifier import build_resnet18, image_transform, make_head
from tune import encode, fit_head, remove_training_duplicates

DATA_URL = "https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz"
DATA_SHA256 = "19f2cbbcd6cbbe977ff5baa19143e3a9132a1fceb2a2f61c8558573f49fbb1b3"
LABEL_URL = "https://raw.githubusercontent.com/udacity/aipnd-project/master/cat_to_name.json"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_archive_members(members):
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
            raise ValueError("Unsafe archive member")
    return members


@torch.no_grad()
def encode_training_views(encoder, dataset, batch_size):
    encoder.eval()
    original, flipped, labels = [], [], []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for index, (images, target) in enumerate(loader):
        original.append(encoder(images).flatten(1).cpu())
        flipped.append(encoder(images.flip(-1)).flatten(1).cpu())
        labels.append(target.cpu())
        if index % 40 == 0:
            print(f"Training views: {min((index + 1) * batch_size, len(dataset))}/{len(dataset)}", flush=True)
    return torch.cat(original), torch.cat(flipped), torch.cat(labels)


def validation_rank(summary):
    return summary["best_validation_accuracy"], -summary["best_validation_loss"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    random.seed(42)
    torch.manual_seed(42)
    start = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = output.parent / ".experiment-data"
    work.mkdir(exist_ok=True)
    archive = work / "flower_data.tar.gz"
    urllib.request.urlretrieve(DATA_URL, archive)
    if file_hash(archive) != DATA_SHA256:
        raise ValueError("Dataset checksum differs from verified archive")
    data = work / "flowers"
    data.mkdir(exist_ok=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(data, members=valid_archive_members(bundle.getmembers()), filter="data")
    label_path = work / "cat_to_name.json"
    urllib.request.urlretrieve(LABEL_URL, label_path)
    if set(json.loads(label_path.read_text())) != {str(i) for i in range(1, 103)}:
        raise ValueError("Unexpected category mapping")
    training = datasets.ImageFolder(data / "train", transform=image_transform())
    validation = datasets.ImageFolder(data / "valid", transform=image_transform())
    if training.class_to_idx != validation.class_to_idx:
        raise ValueError("Training/validation categories differ")
    validation_clean, excluded = remove_training_duplicates(training, validation)
    print(f"Training={len(training)}, clean validation={len(validation_clean)}", flush=True)
    model = build_resnet18(pretrained=True)
    encoder = nn.Sequential(*list(model.children())[:-1])
    original, flipped, labels = encode_training_views(encoder, training, 32)
    valid_features, valid_labels = encode(encoder, validation_clean, 32)
    two_views, two_labels = torch.cat([original, flipped]), torch.cat([labels, labels])
    candidates = [
        {"name": "baseline-rebuild", "hidden_units": 512, "dropout": 0.2, "learning_rate": 0.001, "views": 1, "epochs": 40, "patience": 8},
        {"name": "two-view-512", "hidden_units": 512, "dropout": 0.2, "learning_rate": 0.001, "views": 2, "epochs": 60, "patience": 12},
        {"name": "two-view-lower-dropout", "hidden_units": 512, "dropout": 0.1, "learning_rate": 0.0003, "views": 2, "epochs": 60, "patience": 12},
        {"name": "two-view-wider-head", "hidden_units": 1024, "dropout": 0.2, "learning_rate": 0.0003, "views": 2, "epochs": 60, "patience": 12},
    ]
    summaries, best, best_state = [], None, None
    for config in candidates:
        print(f"Candidate: {config}", flush=True)
        features, target = (original, labels) if config["views"] == 1 else (two_views, two_labels)
        state, summary = fit_head(features, target, valid_features, valid_labels, config, config["epochs"], config["patience"], 42, len(training.classes))
        summaries.append(summary)
        if best is None or validation_rank(summary) > validation_rank(best):
            best, best_state = summary, state
    config = best["config"]
    model.fc = make_head(config["hidden_units"], config["dropout"], len(training.classes))
    model.fc.load_state_dict(best_state)
    checkpoint = output / "flower-classifier-resnet18.pth"
    torch.save({
        "format_version": 3, "arch": "resnet18", "hidden_units": config["hidden_units"],
        "dropout": config["dropout"], "num_classes": len(training.classes), "resize_size": 256,
        "class_to_idx": dict(training.class_to_idx), "state_dict": model.state_dict(),
        "epochs": best["best_epoch"], "seed": 42,
        "validation_accuracy": best["best_validation_accuracy"],
        "training_protocol": "Validation-only selection across baseline and two-view frozen ResNet18 heads",
    }, checkpoint)
    search = {
        "architecture": "resnet18", "seed": 42, "training_images": len(training),
        "validation_images": len(validation_clean), "excluded_validation_duplicates": excluded,
        "test_data_accessed_during_search": False,
        "selection_rule": "Highest validation accuracy, then lowest validation loss; no test-based selection",
        "candidates": summaries, "selected_config": config, "selected_epoch": best["best_epoch"],
        "selected_validation_accuracy": best["best_validation_accuracy"],
        "dataset_url": DATA_URL, "dataset_sha256": DATA_SHA256,
        "elapsed_training_seconds": round(time.monotonic() - start, 2),
        "source_commit": os.environ.get("GITHUB_SHA"),
    }
    (output / "validation-search.json").write_text(json.dumps(search, indent=2) + "\n")
    print("Selection complete; evaluating test split once.", flush=True)
    report = evaluate(checkpoint, data, label_path, batch_size=32)
    report.update({
        "selection_protocol": search["selection_rule"], "selected_config": config,
        "selected_epoch": best["best_epoch"], "selected_validation_accuracy": best["best_validation_accuracy"],
        "source_commit": os.environ.get("GITHUB_SHA"), "previous_reported_top1_accuracy": 745 / 819,
        "previous_reported_top5_accuracy": 798 / 819,
        "comparison_note": "Earlier test results were already known. Candidate and epoch selection here used validation data only.",
    })
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    image = data / report["examples"][0]["image"]
    checked = subprocess.run([sys.executable, str(ROOT / "predict.py"), "--checkpoint", str(checkpoint), "--image", str(image), "--cat_names", str(label_path), "--top_k", "5", "--gpu", "false"], check=True, text=True, capture_output=True)
    print(checked.stdout, flush=True)
    filenames = ("flower-classifier-resnet18.pth", "evaluation.json", "validation-search.json")
    (output / "SHA256SUMS").write_text("".join(f"{file_hash(output / name)}  {name}\n" for name in filenames))
    delta = (report["top1_accuracy"] - 745 / 819) * 100
    notes = f"""Downloadable validation-selected ResNet18 checkpoint.

| Measure | Measured result |
| --- | ---: |
| First-choice test accuracy | {report['top1_accuracy']:.2%} ({report['top1_correct']}/{report['images']}) |
| Top-five test accuracy | {report['top5_accuracy']:.2%} ({report['top5_correct']}/{report['images']}) |
| Selected validation accuracy | {best['best_validation_accuracy']:.2%} |
| Change from previously reported first-choice accuracy | {delta:+.2f} percentage points |

Four predefined configurations were compared on the cleaned validation set. Horizontal-flip training views and longer training were explored. The test split was evaluated after selection. Earlier test results were already known; this is not a new independent dataset. Results describe one seed and this coursework split.

Download the model asset, install the repository dependencies, and download the category-name mapping from {LABEL_URL}. Supply your own flower photo:

    python predict.py --checkpoint flower-classifier-resnet18.pth --image YOUR_FLOWER.jpg --cat_names cat_to_name.json --top_k 5

Prediction requires the checkpoint, category mapping, and one photo. The training dataset is unnecessary for prediction. The checkpoint uses restricted weights-only loading. Check SHA256SUMS after downloading. Regression tests and a real-image prediction command passed before publication.

No flower images are included. Dataset provenance: Udacity distribution of Oxford Flowers 102. Initial model weights: torchvision ResNet18 ImageNet weights. Attached JSON files record configuration, class-level metrics, learning history, and the source commit.
"""
    (output / "release-notes.md").write_text(notes)
    print(f"FINAL top-1={report['top1_accuracy']:.2%}; top-5={report['top5_accuracy']:.2%}; delta={delta:+.2f} percentage points", flush=True)


if __name__ == "__main__":
    main()
