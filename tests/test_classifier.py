"""CPU regression checks; random weights avoid network/model downloads."""
import gc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

import train
import predict

torch.set_num_threads(2)


class ClassifierTests(unittest.TestCase):
    def test_gpu_arguments(self):
        for module in (train, predict):
            for args, expected in (([], False), (['--gpu'], True), (['--gpu', 'false'], False)):
                with self.subTest(module=module.__name__, args=args):
                    with patch.object(sys, 'argv', ['script'] + args):
                        self.assertEqual(module.get_input_args().gpu, expected)

    def test_training_seed_argument(self):
        with patch.object(sys, 'argv', ['train.py', '--seed', '7']):
            self.assertEqual(train.get_input_args().seed, 7)

    def test_preprocess_rgb_grayscale_and_alpha(self):
        with tempfile.TemporaryDirectory() as folder:
            for mode in ('RGB', 'L', 'RGBA'):
                path = Path(folder) / (mode + '.png')
                Image.new(mode, (320, 270)).save(path)
                result = predict.process_image(path)
                self.assertEqual(result.shape, (3, 224, 224))
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.isfinite(result).all())

    def test_validation_on_cpu_and_epochs_respected(self):
        model = nn.Sequential(nn.Linear(3, 2), nn.LogSoftmax(dim=1))
        loader = [(torch.randn(2, 3), torch.tensor([0, 1]))]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()
        with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
            train.train_model(2, loader, loader, torch.device('cpu'), model, optimizer, criterion)
            self.assertEqual(step.call_count, 2)
        loss, accuracy = train.validation(model, loader, torch.device('cpu'), criterion)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(0 <= accuracy <= 1)

    def test_checkpoint_roundtrip_both_architectures_and_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            image = folder / 'image.png'
            Image.new('RGB', (270, 320), (100, 180, 90)).save(image)
            mapping = {str(i): i - 1 for i in range(1, 103)}
            labels = folder / 'labels.json'
            labels.write_text(json.dumps({str(i): f'flower {i}' for i in range(1, 103)}))
            for arch in ('alexnet', 'vgg16'):
                with self.subTest(arch=arch):
                    model = train.classifier(train.load_pretrained_network(arch, pretrained=False), 16)
                    model.eval()
                    optimizer = torch.optim.Adam(model.classifier.parameters())
                    checkpoint = folder / (arch + '.pth')
                    train.save_checkpoint(2, model, SimpleNamespace(class_to_idx=mapping), checkpoint, arch, optimizer)
                    stored = torch.load(checkpoint, weights_only=True)
                    self.assertEqual(stored['arch'], arch)
                    self.assertIsInstance(stored['optimizer_state'], dict)
                    restored = predict.load_checkpoint(checkpoint)
                    restored.eval()
                    inputs = torch.from_numpy(predict.process_image(image)).unsqueeze(0)
                    with torch.no_grad():
                        self.assertTrue(torch.allclose(model(inputs), restored(inputs)))
                    probability, classes, names = predict.predict(image, restored, torch.device('cpu'), labels, 3)
                    self.assertEqual(len(classes), 3)
                    self.assertEqual(names, [f'flower {c}' for c in classes])
                    self.assertTrue(np.isfinite(probability).all())
                    with self.assertRaises(ValueError):
                        predict.predict(image, restored, torch.device('cpu'), labels, 103)
                    del model, restored, optimizer, stored, inputs
                    gc.collect()

    def test_legacy_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.pth'
            torch.save({'arch': 'vgg16'}, path)
            with self.assertRaisesRegex(ValueError, 'legacy'):
                predict.load_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
