import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from torchvision import datasets

from predict import load_checkpoint, predict
from resnet_classifier import build_resnet18, image_transform, make_head
from tune import fit_head, remove_training_duplicates


class ResNetTests(unittest.TestCase):
    def test_checkpoint_and_prediction_roundtrip(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / 'flower.png'
            Image.new('RGBA', (300, 250), (80, 140, 90, 120)).save(image)
            model = build_resnet18(16, 0.2, 3)
            model.eval()
            mapping = {'1': 0, '2': 1, '3': 2}
            path = root / 'model.pth'
            torch.save({'format_version': 3, 'arch': 'resnet18', 'hidden_units': 16, 'dropout': 0.2, 'num_classes': 3, 'state_dict': model.state_dict(), 'class_to_idx': mapping}, path)
            restored = load_checkpoint(path)
            restored.eval()
            with Image.open(image) as source:
                inputs = image_transform()(source.convert('RGB')).unsqueeze(0)
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(inputs), restored(inputs)))
                expected = restored(inputs).exp().topk(3).values[0].numpy()
            labels = root / 'names.json'
            labels.write_text(json.dumps({'1': 'one', '2': 'two', '3': 'three'}))
            probabilities, classes, names = predict(image, restored, torch.device('cpu'), labels, 3)
            self.assertTrue(torch.allclose(torch.tensor(probabilities), torch.tensor(expected)))
            self.assertEqual(len(classes), 3)
            self.assertEqual(restored.preprocess_resize, 256)

    def test_exact_training_duplicate_excluded_from_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ('train', 'valid'):
                (root / split / '1').mkdir(parents=True)
            Image.new('RGB', (10, 10), 'red').save(root / 'train/1/a.png')
            (root / 'valid/1/duplicate.png').write_bytes((root / 'train/1/a.png').read_bytes())
            Image.new('RGB', (10, 10), 'blue').save(root / 'valid/1/new.png')
            train = datasets.ImageFolder(root / 'train')
            valid = datasets.ImageFolder(root / 'valid')
            kept, excluded = remove_training_duplicates(train, valid)
            self.assertEqual(len(kept), 1)
            self.assertEqual(excluded, ['duplicate.png'])
            self.assertTrue((root / 'valid/1/duplicate.png').exists())

    def test_selected_head_reproduces_recorded_validation_score(self):
        torch.manual_seed(12)
        features = torch.randn(24, 512)
        labels = (features[:, 0] > 0).long()
        config = {'hidden_units': 16, 'dropout': 0.2, 'learning_rate': 0.001}
        state, result = fit_head(features, labels, features[:8], labels[:8], config, 4, 3, 42, 2)
        head = make_head(16, 0.2, 2)
        head.load_state_dict(state)
        head.eval()
        with torch.no_grad():
            score = (head(features[:8]).argmax(1) == labels[:8]).float().mean().item()
        self.assertEqual(score, result['best_validation_accuracy'])
        self.assertGreaterEqual(result['best_epoch'], 1)
        self.assertLessEqual(result['best_epoch'], 4)


if __name__ == '__main__':
    unittest.main()
