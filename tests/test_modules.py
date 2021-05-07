from unittest import TestCase

import numpy as np
import torch

from molgym.modules import to_one_hot, masked_softmax


class TestModules(TestCase):
    def test_one_hot(self):
        positions = np.array([[1], [3], [2]])
        indices = torch.from_numpy(positions)

        result = to_one_hot(indices=indices, num_classes=4).detach()
        expected = [
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
        self.assertTrue(np.allclose(expected, result))

    def test_one_hot_wrong_index(self):
        positions = np.array([
            [5],
        ])
        indices = torch.from_numpy(positions)

        with self.assertRaises(RuntimeError):
            to_one_hot(indices=indices, num_classes=3).detach()

    def test_softmax(self):
        logits = torch.from_numpy(np.array([
            [0.5, 0.5],
            [1.0, 0.5],
        ], dtype=np.float))

        mask_1 = torch.ones(size=logits.shape, dtype=torch.bool)

        y1 = masked_softmax(logits=logits, mask=mask_1)
        self.assertEqual(y1.shape, (2, 2))
        self.assertAlmostEqual(y1.sum().item(), 2.0)

        mask_2 = torch.from_numpy(np.array([[1, 0], [1, 0]], dtype=np.bool))
        y2 = masked_softmax(logits=logits, mask=mask_2)

        total = y2.sum(dim=0, keepdim=False)
        self.assertTrue(np.allclose(total, np.array([2, 0])))
