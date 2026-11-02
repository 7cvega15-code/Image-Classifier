"""Frozen ResNet18 features and a small flower classifier head."""
from collections import OrderedDict

from torch import nn
from torchvision import models, transforms


def make_head(hidden_units=256, dropout=0.2, num_classes=102):
    if not 1 <= hidden_units <= 4096 or not 1 <= num_classes <= 102:
        raise ValueError('Invalid classifier dimensions')
    if not 0 <= dropout < 1:
        raise ValueError('dropout must be in [0, 1)')
    return nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(512, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_units, num_classes)),
        ('output', nn.LogSoftmax(dim=1)),
    ]))


def build_resnet18(hidden_units=256, dropout=0.2, num_classes=102, pretrained=False):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = make_head(hidden_units, dropout, num_classes)
    model.preprocess_resize = 256
    model.architecture = 'resnet18'
    return model


def image_transform():
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
