from unittest import TestCase

import numpy as np
import torch
from cormorant.cg_lib import SphericalHarmonics

from molgym.agents.covariant.so3_tools import (spherical_to_cartesian, estimate_alms, concat_so3vecs,
                                               cartesian_to_spherical, generate_fibonacci_grid)
from molgym.agents.covariant.spherical_dists import SphericalUniform, SO3Distribution, ExpSO3Distribution
from molgym.tools.util import to_numpy


class SphericalUniformTest(TestCase):
    def setUp(self):
        self.dist = SphericalUniform()

    def test_shape(self):
        num_samples = 1000
        samples = self.dist.sample(torch.Size((num_samples, )))
        self.assertTrue(samples.shape == (num_samples, 3))

    def test_min_max(self):
        self.dist = SphericalUniform(batch_shape=(3, ))
        self.assertTrue(self.dist.get_max_prob().shape == (3, ))

    def test_distance(self):
        num_samples = 1000
        samples = self.dist.sample(torch.Size((num_samples, )))
        self.assertTrue(np.allclose(samples.norm(dim=-1), 1.))

    def test_mean(self):
        torch.manual_seed(1)
        num_samples = 200_000
        self.dist = SphericalUniform()
        samples = self.dist.sample(torch.Size((num_samples, )))
        self.assertAlmostEqual(samples.mean(0).norm().item(), 0, places=2)

    def test_argmax(self):
        dist = SphericalUniform(batch_shape=torch.Size((3, )))
        arg_maxes = dist.argmax()
        self.assertEqual(arg_maxes.shape, dist.batch_shape + dist.event_shape)


class SphericalDistributionTest(TestCase):
    def setUp(self):
        self.maxl = 3
        self.sphs = SphericalHarmonics(maxl=self.maxl, sh_norm='qm')
        sphs_conj = SphericalHarmonics(maxl=self.maxl, conj=True, sh_norm='qm')

        # Generate some reference point(s)
        phi_refs = np.array([
            np.pi / 2,
            -np.pi / 2,
        ])
        theta_refs = np.pi / 2 * np.ones_like(phi_refs)
        theta_phi_refs = np.stack([theta_refs, phi_refs], axis=-1)
        xyz_refs = spherical_to_cartesian(theta_phi_refs)
        y_lms_conj = sphs_conj.forward(torch.tensor(xyz_refs, dtype=torch.float))
        self.a_lms_1 = estimate_alms(y_lms_conj)

        # Another set of a_lms
        phi_refs = np.array([np.pi / 3])
        theta_refs = np.pi / 3 * np.ones_like(phi_refs)
        theta_phi_refs = np.stack([theta_refs, phi_refs], axis=-1)
        xyz_refs = spherical_to_cartesian(theta_phi_refs)
        y_lms_conj = sphs_conj.forward(torch.tensor(xyz_refs, dtype=torch.float))
        self.a_lms_2 = estimate_alms(y_lms_conj)

    def test_max(self):
        a_lms = concat_so3vecs([self.a_lms_1, self.a_lms_2, self.a_lms_1])
        so3_distr = SO3Distribution(a_lms=a_lms, sphs=self.sphs)
        self.assertEqual(so3_distr.get_max_prob().shape, (3, ))

    def test_sample(self):
        torch.manual_seed(1)
        samples_shape = (2048, )

        a_lms = concat_so3vecs([self.a_lms_1, self.a_lms_2])
        so3_distr = SO3Distribution(a_lms=a_lms, sphs=self.sphs)
        samples = so3_distr.sample(samples_shape)

        self.assertEqual(samples.shape, samples_shape + so3_distr.batch_shape + so3_distr.event_shape)

        angles = cartesian_to_spherical(to_numpy(samples))  # [S, B, 2]
        mean_angles = np.mean(angles, axis=0)  # [B, 2]

        self.assertEqual(mean_angles.shape, (2, 2))

        so3_distr_1 = SO3Distribution(a_lms=self.a_lms_1, sphs=self.sphs)
        samples_1 = so3_distr_1.sample(samples_shape)
        angles_1 = cartesian_to_spherical(to_numpy(samples_1))  # [S, 1, 2]
        mean_angles_1 = np.mean(angles_1, axis=0)  # [1, 2]

        so3_distr_2 = SO3Distribution(a_lms=self.a_lms_2, sphs=self.sphs)
        samples_2 = so3_distr_2.sample(samples_shape)
        angles_2 = cartesian_to_spherical(to_numpy(samples_2))  # [S, 1, 2]
        mean_angles_2 = np.mean(angles_2, axis=0)  # [1, 2]

        # Assert that batching does not affect the result
        self.assertTrue(np.allclose(mean_angles[0], mean_angles_1, atol=0.1))
        self.assertTrue(np.allclose(mean_angles[1], mean_angles_2, atol=0.1))

    def test_prob(self):
        a_lms = concat_so3vecs([self.a_lms_1, self.a_lms_2, self.a_lms_1])
        so3_distr = SO3Distribution(a_lms=a_lms, sphs=self.sphs)
        samples = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ])

        self.assertEqual(so3_distr.log_prob(samples).shape, (3, ))
        self.assertEqual(so3_distr.log_prob(samples[[0]]).shape, (3, ))

        with self.assertRaises(RuntimeError):
            so3_distr.log_prob(samples[:2])

    def test_max_sample(self):
        a_lms = concat_so3vecs([self.a_lms_1, self.a_lms_2])
        so3_distr = SO3Distribution(a_lms=a_lms, sphs=self.sphs, dtype=torch.float)
        samples = so3_distr.argmax(count=17)
        self.assertEqual(samples.shape, (2, 3))

    def test_normalization(self):
        a_lms = concat_so3vecs([self.a_lms_1, self.a_lms_2])
        so3_distr = SO3Distribution(a_lms=a_lms, sphs=self.sphs, dtype=torch.float)
        grid = generate_fibonacci_grid(n=1024)
        grid_t = torch.tensor(grid, dtype=torch.float).unsqueeze(1)
        probs = so3_distr.prob(grid_t)
        integral = 4 * np.pi * torch.mean(probs, dim=0)
        self.assertTrue(np.allclose(to_numpy(integral), 1.0))


class ExpSphericalDistributionTest(TestCase):
    def setUp(self) -> None:
        self.maxl = 3
        self.sphs = SphericalHarmonics(maxl=self.maxl, sh_norm='qm')
        sphs_conj = SphericalHarmonics(maxl=self.maxl, conj=True, sh_norm='qm')

        # Generate some reference point(s)
        phi_refs = np.array([
            np.pi / 2,
            -np.pi / 2,
        ])
        theta_refs = np.pi / 2 * np.ones_like(phi_refs)
        theta_phi_refs = np.stack([theta_refs, phi_refs], axis=-1)
        xyz_refs = spherical_to_cartesian(theta_phi_refs)
        y_lms_conj = sphs_conj.forward(torch.tensor(xyz_refs, dtype=torch.float))
        self.a_lms_1 = estimate_alms(y_lms_conj)

        # Another set of a_lms
        phi_refs = np.array([np.pi / 3])
        theta_refs = np.pi / 3 * np.ones_like(phi_refs)
        theta_phi_refs = np.stack([theta_refs, phi_refs], axis=-1)
        xyz_refs = spherical_to_cartesian(theta_phi_refs)
        y_lms_conj = sphs_conj.forward(torch.tensor(xyz_refs, dtype=torch.float))
        self.a_lms_2 = estimate_alms(y_lms_conj)

    def test_max(self):
        a_lms = concat_so3vecs([self.a_lms_1, self.a_lms_2, self.a_lms_1])
        distr = ExpSO3Distribution(a_lms=a_lms, sphs=self.sphs, beta=100)
        self.assertEqual(distr.get_max_log_prob().shape, (3, ))

    def test_sample(self):
        torch.manual_seed(1)
        samples_shape = (2048, )

        a_lms = concat_so3vecs([self.a_lms_1, self.a_lms_2])
        distr = ExpSO3Distribution(a_lms=a_lms, sphs=self.sphs, beta=100)
        samples = distr.sample(samples_shape)

        self.assertEqual(samples.shape, samples_shape + distr.batch_shape + distr.event_shape)

        angles = cartesian_to_spherical(to_numpy(samples))  # [S, B, 2]
        mean_angles = np.mean(angles, axis=0)  # [B, 2]

        self.assertEqual(mean_angles.shape, (2, 2))

        distr_1 = ExpSO3Distribution(a_lms=self.a_lms_1, sphs=self.sphs, beta=100)
        samples_1 = distr_1.sample(samples_shape)
        angles_1 = cartesian_to_spherical(to_numpy(samples_1))  # [S, 1, 2]
        mean_angles_1 = np.mean(angles_1, axis=0)  # [1, 2]

        distr_2 = ExpSO3Distribution(a_lms=self.a_lms_2, sphs=self.sphs, beta=100)
        samples_2 = distr_2.sample(samples_shape)
        angles_2 = cartesian_to_spherical(to_numpy(samples_2))  # [S, 1, 2]
        mean_angles_2 = np.mean(angles_2, axis=0)  # [1, 2]

        # Assert that batching does not affect the result
        self.assertTrue(np.allclose(mean_angles[0], mean_angles_1, atol=0.1))
        self.assertTrue(np.allclose(mean_angles[1], mean_angles_2, atol=0.1))

    def test_prob(self):
        a_lms = concat_so3vecs([self.a_lms_1, self.a_lms_2, self.a_lms_1])
        distr = ExpSO3Distribution(a_lms=a_lms, sphs=self.sphs, beta=100)
        samples = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ])

        self.assertEqual(distr.log_prob(samples).shape, (3, ))
        self.assertEqual(distr.log_prob(samples[[0]]).shape, (3, ))

        with self.assertRaises(RuntimeError):
            distr.log_prob(samples[:2])

    def test_max_sample(self):
        a_lms = concat_so3vecs([self.a_lms_1, self.a_lms_2])
        distr = ExpSO3Distribution(a_lms=a_lms, sphs=self.sphs, dtype=torch.float, beta=100)
        samples = distr.argmax(count=17)
        self.assertEqual(samples.shape, (2, 3))

    def test_normalization(self):
        a_lms = concat_so3vecs([self.a_lms_1, self.a_lms_2])
        distr = ExpSO3Distribution(a_lms=a_lms, sphs=self.sphs, dtype=torch.float, beta=100)
        grid = generate_fibonacci_grid(n=1024)
        grid_t = torch.tensor(grid, dtype=torch.float).unsqueeze(1)
        probs = torch.exp(distr.log_prob(grid_t))
        integral = 4 * np.pi * torch.mean(probs, dim=0)
        self.assertTrue(np.allclose(to_numpy(integral), 1.0, atol=5e-3))
