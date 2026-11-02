import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

import evaluate


class ConstantClassifier(nn.Module):
    class_to_idx = {'1': 0, '2': 1, '3': 2}

    def forward(self, images):
        return torch.log_softmax(torch.tensor([[3.0, 2.0, 1.0]]).expand(len(images), -1), dim=1)


class EvaluationTests(unittest.TestCase):
    def test_known_predictions_have_expected_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for category in ('1', '2', '3'):
                target = root / 'test' / category
                target.mkdir(parents=True)
                for i in range(2):
                    Image.new('RGB', (256, 256)).save(target / f'{i}.jpg')
            labels = root / 'names.json'
            labels.write_text(json.dumps({'1': 'one', '2': 'two', '3': 'three'}))
            checkpoint = root / 'fake-checkpoint.pth'
            checkpoint.write_bytes(b'evaluation fixture, not a real checkpoint')
            with patch.object(evaluate, 'load_checkpoint', return_value=ConstantClassifier()):
                result = evaluate.evaluate(checkpoint, root, labels, batch_size=4)
            self.assertEqual(result['images'], 6)
            self.assertEqual(result['top1_correct'], 2)
            self.assertAlmostEqual(result['top1_accuracy'], 1 / 3)
            self.assertEqual(result['top5_accuracy'], 1)
            self.assertAlmostEqual(result['macro_class_accuracy'], 1 / 3)
            self.assertEqual(len(result['examples']), 6)
            self.assertEqual(sum(row['correct'] for row in result['examples']), 2)
            expected_loss = -torch.log_softmax(torch.tensor([3.0, 2.0, 1.0]), dim=0).mean().item()
            self.assertAlmostEqual(result['mean_negative_log_likelihood'], expected_loss, places=6)

    def test_category_mapping_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'test' / 'unexpected-category').mkdir(parents=True)
            Image.new('RGB', (256, 256)).save(root / 'test' / 'unexpected-category' / '0.jpg')
            with patch.object(evaluate, 'load_checkpoint', return_value=ConstantClassifier()):
                with self.assertRaisesRegex(ValueError, 'categories'):
                    evaluate.evaluate('unused-checkpoint', root, 'unused-label-file')


if __name__ == '__main__':
    unittest.main()
