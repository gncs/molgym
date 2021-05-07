from unittest import TestCase

import numpy as np
import torch
from cormorant.cg_lib import SphericalHarmonics

from molgym.agents.covariant.so3_tools import spherical_to_cartesian


class TestSphericalHarmonics(TestCase):
    def test_conversion(self):
        theta_phi = np.array([np.pi / 3, np.pi / 4])
        pos = spherical_to_cartesian(theta_phi)

        expected = np.array([0.612372, 0.612372, 0.5])
        self.assertTrue(np.allclose(pos, expected))

    def test_l_1(self):
        theta_phi = np.array([np.pi / 2, 0.0])
        pos = spherical_to_cartesian(theta_phi)
        pos_tensor = torch.tensor(pos, dtype=torch.float32)

        # To match the definition Mathematica uses sh_norm='qm' is required
        sph = SphericalHarmonics(maxl=1, normalize=True, sh_norm='qm')
        output = sph.forward(pos_tensor)

        # Mathematica output:
        expected = np.array([
            [0.345494, 0],
            [0, 0],
            [-0.345494, 0],
        ], dtype=np.float32)

        self.assertTrue(np.allclose(output[1].cpu().detach().numpy(), expected))

    def test_l_2(self):
        theta_phi = np.array([np.pi / 3, np.pi / 4])
        pos = spherical_to_cartesian(theta_phi)
        pos_tensor = torch.tensor(pos, dtype=torch.float32)

        # To match the definition Mathematica uses sh_norm='qm' is required
        sph = SphericalHarmonics(maxl=2, normalize=False, sh_norm='qm')
        output = sph.forward(pos_tensor)

        # Mathematica output:
        expected = np.array([
            [0, -0.289706],
            [0.236544, -0.236544],
            [-0.0788479, 0],
            [-0.236544, -0.236544],
            [0, 0.289706],
        ],
                            dtype=np.float32)

        self.assertTrue(np.allclose(output[2].cpu().detach().numpy(), expected))
