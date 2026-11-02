"""Evaluate a restored classifier on a separate ImageFolder test split."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
from torchvision import datasets, transforms

from predict import load_checkpoint


def evaluate(checkpoint, data_dir, category_names, batch_size=32):
    model = load_checkpoint(checkpoint)
    model.eval()
    # Matches train.py's validation/test preprocessing.
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = datasets.ImageFolder(Path(data_dir) / 'test', transform=transform)
    if dataset.class_to_idx != model.class_to_idx:
        raise ValueError('Test categories do not match the checkpoint class mapping.')
    names = json.loads(Path(category_names).read_text())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.NLLLoss(reduction='sum')
    k = min(5, len(dataset.classes))
    selected = set(random.Random(42).sample(range(len(dataset)), min(6, len(dataset))))
    hits = {name: 0 for name in dataset.classes}
    totals = {name: 0 for name in dataset.classes}
    correct = top5 = count = 0
    loss = 0.0
    examples = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            ranking = outputs.topk(k, dim=1).indices
            probabilities = outputs.exp()
            loss += criterion(outputs, labels).item()
            correct += (ranking[:, 0] == labels).sum().item()
            top5 += (ranking == labels[:, None]).any(dim=1).sum().item()
            for j, label in enumerate(labels.tolist()):
                actual = dataset.classes[label]
                predicted = dataset.classes[ranking[j, 0].item()]
                totals[actual] += 1
                hits[actual] += int(actual == predicted)
                if count + j in selected:
                    image_path = Path(dataset.samples[count + j][0])
                    examples.append({
                        'image': image_path.relative_to(data_dir).as_posix(),
                        'actual': names[actual],
                        'predicted': names[predicted],
                        'top1_probability': round(probabilities[j, ranking[j, 0]].item(), 6),
                        'correct': actual == predicted,
                    })
            count += len(labels)
    per_class = {c: {'name': names[c], 'correct': hits[c], 'total': totals[c]} for c in dataset.classes}
    return {
        'split': 'test', 'images': count, 'categories': len(dataset.classes),
        'top1_correct': correct, 'top1_accuracy': correct / count,
        'top5_correct': top5, 'top5_accuracy': top5 / count,
        'mean_negative_log_likelihood': loss / count,
        'macro_class_accuracy': sum(hits[c] / totals[c] for c in dataset.classes) / len(dataset.classes),
        'device': 'cpu', 'torch_version': str(torch.__version__),
        'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'example_selection': 'Six test indices sampled without replacement using random seed 42',
        'examples': examples, 'per_class': per_class,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--cat_names', default='cat_to_name.json')
    parser.add_argument('--output', default='evaluation.json')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.data_dir, args.cat_names, args.batch_size)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(f"Test images: {report['images']}; top-1: {report['top1_accuracy']:.2%}; top-5: {report['top5_accuracy']:.2%}")


if __name__ == '__main__':
    main()
