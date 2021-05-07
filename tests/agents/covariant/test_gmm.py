from unittest import TestCase

import numpy as np
import torch

from molgym.agents.covariant.gmm import GaussianMixtureModel
from molgym.tools.util import to_numpy


class GaussianMixtureModelTest(TestCase):
    def setUp(self):
        self.log_probs = torch.log(torch.tensor([[0.7, 0.3], [0.5, 0.5]]))
        self.means = torch.tensor([[-0.5, 0.3], [0.0, 0.2]])
        self.log_stds = torch.log(torch.tensor([[0.2, 0.5], [0.3, 0.2]]))
        self.distr = GaussianMixtureModel(log_probs=self.log_probs, means=self.means, stds=torch.exp(self.log_stds))

    def test_samples(self):
        s = self.distr.sample(torch.Size((3, )))
        self.assertEqual(s.shape, (3, 2))

    def test_argmax(self):
        torch.manual_seed(1)
        argmax = self.distr.argmax(128)
        self.assertEqual(argmax.shape, (2, ))
        self.assertTrue(np.allclose(to_numpy(argmax), np.array([-0.495, 0.156]), atol=1.e-2))
