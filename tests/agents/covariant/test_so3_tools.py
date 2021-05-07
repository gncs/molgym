from unittest import TestCase

import numpy as np
import torch
from cormorant.cg_lib import SphericalHarmonics
from cormorant.so3_lib import SO3WignerD
from cormorant.so3_lib.rotations import rotate_rep

from molgym.agents.covariant.so3_tools import (spherical_to_cartesian, estimate_alms, concat_so3vecs, AtomicScalars,
                                               generate_fibonacci_grid, cartesian_to_spherical, complex_product,
                                               get_normalization_constant, normalize_alms)
from molgym.tools.util import to_numpy


class FibonacciGridTest(TestCase):
    def test_generation(self):
        count = 10
        grid = generate_fibonacci_grid(n=count)
        self.assertEqual(grid.shape, (count, 3))

    def test_empty(self):
        count = 0
        grid = generate_fibonacci_grid(n=count)
        self.assertEqual(grid.shape, (count, 3))


class SphericalCartesianTransformationTest(TestCase):
    def test_spherical_to_cartesian(self):
        theta_phi = np.array([[np.pi / 2, np.pi]])
        xyz = spherical_to_cartesian(theta_phi)
        self.assertTrue(np.all(np.isclose(xyz, np.array([[-1.0, 0.0, 0.0]]))))

    def test_spherical_to_cartesian_2(self):
        theta_phi = np.array([[np.pi / 2, 3 / 2 * np.pi]])
        xyz = spherical_to_cartesian(theta_phi)
        self.assertTrue(np.all(np.isclose(xyz, np.array([[0.0, -1.0, 0.0]]))))

    def test_cartesian_to_spherical(self):
        xyz = np.array([[0.0, -1.0, 0.0]])
        theta_phi = cartesian_to_spherical(xyz)
        self.assertTrue(np.all(np.isclose(theta_phi, np.array([[np.pi / 2, -np.pi / 2]]))))

    def test_cycle(self):
        xyz = np.array([[0.0, -1.0, 0.0]])
        theta_phi = cartesian_to_spherical(xyz)
        xyz_new = spherical_to_cartesian(theta_phi)
        self.assertTrue(np.all(np.isclose(xyz, xyz_new)))

    def test_cycle_2(self):
        theta_phi = np.array([[0.3, -1.2]])
        xyz = spherical_to_cartesian(theta_phi)
        theta_phi_2 = cartesian_to_spherical(xyz)
        self.assertTrue(np.all(np.isclose(theta_phi, theta_phi_2)))


class ComplexNumbersTest(TestCase):
    def test_multiplication(self):
        a = torch.tensor([2.0, -1.0], dtype=torch.float)
        b = torch.tensor([3.0, -2.0], dtype=torch.float)
        c = to_numpy(complex_product(a, b))
        expected = np.array([4.0, -7.0])
        self.assertTrue(np.allclose(c, expected))

    def test_multiplication_2(self):
        a = torch.tensor([2.0, 0.0], dtype=torch.float)
        b = torch.tensor([3.0, 0.0], dtype=torch.float)
        c = to_numpy(complex_product(a, b))
        expected = np.array([6.0, 0.0])
        self.assertTrue(np.allclose(c, expected))


class NormalizationTest(TestCase):
    def setUp(self):
        self.maxl = 4
        self.sphs = SphericalHarmonics(maxl=self.maxl)
        self.sphs_conj = SphericalHarmonics(maxl=self.maxl, conj=True, sh_norm='unit')

    def test_concat(self):
        theta_phi = np.array([[np.pi / 2, np.pi / 2]])
        xyz_refs = spherical_to_cartesian(theta_phi)
        y_lms_conj = self.sphs_conj.forward(torch.tensor(xyz_refs, dtype=torch.float))

        a_lms = estimate_alms(y_lms_conj)
        a_lms = concat_so3vecs([a_lms] * 3)

        self.assertTrue(all(a_lm.shape[0] == 3 for a_lm in a_lms))

    def test_normalization(self):
        theta_phi = np.array([[np.pi / 2, np.pi / 2]])
        xyz_refs = spherical_to_cartesian(theta_phi)
        y_lms_conj = self.sphs_conj.forward(torch.tensor(xyz_refs, dtype=torch.float))

        a_lms = estimate_alms(y_lms_conj)
        k1 = get_normalization_constant(a_lms)

        self.assertTrue(k1.shape, (1, ))

        # If sh_norm='unit', sum over m = 1.
        self.assertTrue(k1.item(), self.maxl + 1)

        normalized_a_lms = normalize_alms(a_lms)

        k2 = get_normalization_constant(normalized_a_lms)
        self.assertAlmostEqual(k2.item(), 1.0)


class AtomicScalarsTest(TestCase):
    def test_invariant(self):
        max_ell = 4
        sphs_conj = SphericalHarmonics(maxl=max_ell, conj=True, sh_norm='unit')
        atomic_scalars = AtomicScalars(maxl=max_ell)

        theta_phi = np.array([[np.pi / 3, np.pi / 4], [2 * np.pi / 3, np.pi / 2]])
        xyz_refs = spherical_to_cartesian(theta_phi)
        y_lms_conj = sphs_conj.forward(torch.tensor(xyz_refs, dtype=torch.float))

        a_lms = estimate_alms(y_lms_conj)

        invariant = atomic_scalars(a_lms)

        self.assertTrue(invariant.shape[-1], atomic_scalars.get_output_dim(channels=1))

        random_rotation = SO3WignerD.euler(maxl=max_ell, dtype=torch.float)
        a_lms_rotated = rotate_rep(random_rotation, a_lms)

        self.assertFalse(np.allclose(to_numpy(a_lms[1]), to_numpy(a_lms_rotated[1])))

        invariant_rotated = atomic_scalars(a_lms_rotated)

        self.assertTrue(np.allclose(invariant, invariant_rotated))
