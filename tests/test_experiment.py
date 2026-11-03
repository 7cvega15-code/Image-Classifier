import tarfile
import unittest
import torch
from torch import nn
from experiments.improve import encode_training_views, valid_archive_members, validation_rank


class FirstColumn(nn.Module):
    def forward(self, images):
        return images[:, :, :, :1]


class ExperimentTests(unittest.TestCase):
    def test_flipped_features_preserve_labels_and_use_right_column(self):
        images = torch.tensor([[[[1.0, 2.0, 3.0]]], [[[4.0, 5.0, 6.0]]]])
        labels = torch.tensor([2, 7])
        dataset = torch.utils.data.TensorDataset(images, labels)
        original, flipped, returned = encode_training_views(FirstColumn(), dataset, 1)
        self.assertEqual(original[:, 0].tolist(), [1.0, 4.0])
        self.assertEqual(flipped[:, 0].tolist(), [3.0, 6.0])
        self.assertEqual(returned.tolist(), [2, 7])

    def test_selection_uses_validation_accuracy_then_loss(self):
        a = {"best_validation_accuracy": 0.9, "best_validation_loss": 0.3, "test_accuracy": 0.99}
        b = {"best_validation_accuracy": 0.91, "best_validation_loss": 0.5, "test_accuracy": 0.1}
        c = {"best_validation_accuracy": 0.91, "best_validation_loss": 0.2}
        self.assertGreater(validation_rank(b), validation_rank(a))
        self.assertGreater(validation_rank(c), validation_rank(b))

    def test_archive_traversal_and_links_are_rejected(self):
        for path in ("../outside", "/absolute"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    valid_archive_members([tarfile.TarInfo(path)])
        link = tarfile.TarInfo("link")
        link.type = tarfile.SYMTYPE
        with self.assertRaises(ValueError):
            valid_archive_members([link])


if __name__ == "__main__":
    unittest.main()
