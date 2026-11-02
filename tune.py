"""Compare frozen-feature ResNet18 heads using validation data only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torchvision import datasets

from resnet_classifier import build_resnet18, image_transform, make_head


def remove_training_duplicates(train_data, valid_data):
    """Filter validation indices; leave every original image on disk intact."""
    training_hashes = {hashlib.sha256(Path(path).read_bytes()).digest() for path, _ in train_data.samples}
    kept, excluded = [], []
    for index, (path, _) in enumerate(valid_data.samples):
        if hashlib.sha256(Path(path).read_bytes()).digest() in training_hashes:
            excluded.append(Path(path).name)
        else:
            kept.append(index)
    return torch.utils.data.Subset(valid_data, kept), excluded


@torch.no_grad()
def encode(encoder, dataset, batch_size):
    encoder.eval()
    features, labels = [], []
    for index, (images, target) in enumerate(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)):
        features.append(encoder(images).flatten(1).cpu())
        labels.append(target.cpu())
        if index % 40 == 0:
            print(f'Encoded {min((index + 1) * batch_size, len(dataset))}/{len(dataset)} images', flush=True)
    return torch.cat(features), torch.cat(labels)


def fit_head(train_features, train_labels, valid_features, valid_labels, config, epochs, patience, seed, num_classes):
    torch.manual_seed(seed)
    head = make_head(config['hidden_units'], config['dropout'], num_classes)
    optimizer = torch.optim.Adam(head.parameters(), lr=config['learning_rate'], weight_decay=0.0001)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_features, train_labels), batch_size=128, shuffle=True, generator=generator)
    criterion = nn.NLLLoss()
    history = []
    best_score = (-1.0, float('-inf'))
    best_state = None
    best_epoch = stale = 0
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        for features, labels in loader:
            optimizer.zero_grad()
            loss = criterion(head(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        head.eval()
        with torch.no_grad():
            outputs = head(valid_features)
            valid_loss = criterion(outputs, valid_labels).item()
            accuracy = (outputs.argmax(1) == valid_labels).float().mean().item()
        row = {'epoch': epoch, 'train_loss': total_loss / len(train_labels), 'validation_loss': valid_loss, 'validation_accuracy': accuracy}
        history.append(row)
        score = (accuracy, -valid_loss)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f'Epoch {epoch}: validation accuracy={accuracy:.4f}, loss={valid_loss:.4f}', flush=True)
        if stale >= patience:
            break
    return best_state, {'config': config, 'best_epoch': best_epoch, 'best_validation_accuracy': best_score[0], 'best_validation_loss': -best_score[1], 'history': history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--checkpoint', default='checkpoint-resnet18.pth')
    parser.add_argument('--report', default='validation-search.json')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1:
        parser.error('epochs, patience, batch_size, and threads must be positive')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    data = Path(args.data_dir)
    training = datasets.ImageFolder(data / 'train', transform=image_transform())
    validation = datasets.ImageFolder(data / 'valid', transform=image_transform())
    if training.class_to_idx != validation.class_to_idx:
        raise ValueError('Training and validation categories differ')
    validation_clean, excluded = remove_training_duplicates(training, validation)
    print(f'Training: {len(training)}; validation: {len(validation_clean)}; excluded duplicates: {excluded}', flush=True)
    model = build_resnet18(pretrained=True)
    encoder = nn.Sequential(*list(model.children())[:-1])
    train_features, train_labels = encode(encoder, training, args.batch_size)
    valid_features, valid_labels = encode(encoder, validation_clean, args.batch_size)
    candidates = [
        {'hidden_units': 256, 'dropout': 0.2, 'learning_rate': 0.001},
        {'hidden_units': 512, 'dropout': 0.2, 'learning_rate': 0.001},
        {'hidden_units': 256, 'dropout': 0.5, 'learning_rate': 0.0003},
    ]
    summaries = []
    best = None
    best_state = None
    for index, config in enumerate(candidates):
        print(f'Candidate {index + 1}/{len(candidates)}: {config}', flush=True)
        state, summary = fit_head(train_features, train_labels, valid_features, valid_labels, config, args.epochs, args.patience, args.seed, len(training.classes))
        summaries.append(summary)
        if best is None or (summary['best_validation_accuracy'], -summary['best_validation_loss']) > (best['best_validation_accuracy'], -best['best_validation_loss']):
            best, best_state = summary, state
    config = best['config']
    model.fc = make_head(config['hidden_units'], config['dropout'], len(training.classes))
    model.fc.load_state_dict(best_state)
    checkpoint = {
        'format_version': 3, 'arch': 'resnet18', 'hidden_units': config['hidden_units'],
        'dropout': config['dropout'], 'num_classes': len(training.classes), 'resize_size': 256,
        'class_to_idx': dict(training.class_to_idx), 'state_dict': model.state_dict(),
        'epochs': best['best_epoch'], 'seed': args.seed,
        'validation_accuracy': best['best_validation_accuracy'],
        'training_protocol': 'Frozen pretrained ResNet18; deterministic center-cropped features; validation-only head selection',
    }
    torch.save(checkpoint, args.checkpoint)
    report = {
        'architecture': 'resnet18', 'seed': args.seed, 'training_images': len(training),
        'validation_images': len(validation_clean), 'excluded_validation_duplicates': excluded,
        'test_data_accessed_during_search': False, 'max_epochs': args.epochs, 'patience': args.patience,
        'selection_rule': 'Highest validation accuracy; lower validation loss breaks ties',
        'candidates': summaries, 'selected_config': config, 'selected_epoch': best['best_epoch'],
        'selected_validation_accuracy': best['best_validation_accuracy'],
        'elapsed_seconds': round(time.monotonic() - started, 2),
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
    print(f"Selected validation accuracy: {best['best_validation_accuracy']:.2%}; epoch {best['best_epoch']}", flush=True)
    print(f'Saved {args.checkpoint}', flush=True)


if __name__ == '__main__':
    main()
