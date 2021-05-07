from unittest import TestCase

import torch

from molgym.agents.covariant.tools import pad_sequence


class ToolsTest(TestCase):
    def test_pad_sequence(self):
        a = torch.rand(size=(3, 2))
        b = torch.rand(size=(4, 2))
        max_length = 5
        c = pad_sequence(sequences=[a, b], max_length=max_length, padding_value=0.0)
        self.assertEqual(c.shape, (2, max_length, 2))
        self.assertTrue(torch.all(c[0, 3:] == 0.0))
        self.assertTrue(torch.all(c[0, 4:] == 0.0))

    def test_pad_sequence_too_small(self):
        a = torch.rand(size=(3, 2))
        b = torch.rand(size=(4, 3))
        max_length = 3
        with self.assertRaises(RuntimeError):
            pad_sequence(sequences=[a, b], max_length=max_length, padding_value=0.0)

    def test_pad_sequence_mismatch(self):
        a = torch.rand(size=(3, 2))
        b = torch.rand(size=(4, 3))
        max_length = 5
        with self.assertRaises(RuntimeError):
            pad_sequence(sequences=[a, b], max_length=max_length, padding_value=0.0)
