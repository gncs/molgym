import io
from unittest import TestCase

import ase.io
import numpy as np

from molgym.agents.internal.zmat import get_distance, get_angle, get_dihedral, position_point


class TestZMat(TestCase):
    def test_distance(self):
        p1 = np.array([0, 0, 0], dtype=np.float)
        p2 = np.array([0, 1, 0], dtype=np.float)
        p3 = np.array([1, 0, 0], dtype=np.float)

        self.assertAlmostEqual(get_distance(p1, p1), 0)
        self.assertAlmostEqual(get_distance(p1, p2), 1)
        self.assertAlmostEqual(get_distance(p1, p3), 1)
        self.assertAlmostEqual(get_distance(p2, p3), np.sqrt(2))

    def test_angle(self):
        p1 = np.array([1, 0, 0], dtype=np.float)
        p2 = np.array([0, 0, 0], dtype=np.float)
        p3 = np.array([0, 1, 0], dtype=np.float)
        p4 = np.array([-1, 0, 0], dtype=np.float)

        self.assertAlmostEqual(get_angle(p1, p2, p1), 0)
        self.assertAlmostEqual(get_angle(p1, p2, p3), np.pi / 2)
        self.assertAlmostEqual(get_angle(p1, p2, p4), np.pi)

    def test_dihedral(self):
        p1 = np.array([0, 0, 1.5], dtype=np.float)
        p2 = np.array([0, 0, 0], dtype=np.float)
        p3 = np.array([0, 0.5, 0], dtype=np.float)

        for psi in np.arange(start=-np.pi, stop=np.pi, step=np.pi / 17):
            p4 = np.array([np.sin(psi), 0.5, np.cos(psi)], dtype=np.float)
            dihedral = get_dihedral(p1, p2, p3, p4)
            self.assertAlmostEqual(psi, dihedral)

    def test_dihedral_2(self):
        p1 = np.array([0, 0, 1.5], dtype=np.float)
        p2 = np.array([0, 0, 0], dtype=np.float)
        p3 = np.array([0, 0.5, 0], dtype=np.float)

        # Add delta so that the corner case of -180, 180 goes away
        delta = 1E-4
        for psi in np.arange(start=-np.pi + delta, stop=np.pi - delta, step=np.pi / 17):
            p4 = np.array([np.sin(2 * np.pi + psi), 0.5, np.cos(2 * np.pi + psi)], dtype=np.float)
            dihedral = get_dihedral(p1, p2, p3, p4)
            self.assertAlmostEqual(psi, dihedral)

    def test_dihedral_sign(self):
        p0 = np.array([0, 0, 1], dtype=np.float)
        p1 = np.array([0, 0, 0], dtype=np.float)
        p2 = np.array([0, 1, 0], dtype=np.float)

        p3_1 = np.array([1, 0, 0], dtype=np.float)
        dihedral = get_dihedral(p0, p1, p2, p3_1)
        self.assertEqual(dihedral, np.pi / 2)

        p3_2 = np.array([-1, 0, 0], dtype=np.float)
        dihedral = get_dihedral(p0, p1, p2, p3_2)
        self.assertEqual(dihedral, -np.pi / 2)

    def test_dihedral_nan(self):
        string = '4\n\nC 0.5995394918 0.0 1.0\nC -0.5995394918 0.0 1.0\nH -1.6616385861 0.0 1.0\nH 1.6616385861 0.0 1.0'
        atoms = ase.io.read(io.StringIO(string), format='xyz')
        dihedral = get_dihedral(*(a.position for a in atoms))
        self.assertTrue(np.isnan(dihedral))

    def test_positioning(self):
        p0 = np.array([0, 0, 1], dtype=np.float)
        p1 = np.array([0, 0, 0], dtype=np.float)
        p2 = np.array([0, 1, 0], dtype=np.float)

        distance = 2.5
        angle = 2 * np.pi / 3

        # Add delta so that the corner case of -180, 180 goes away
        delta = 1E-4
        for psi in np.arange(start=-np.pi + delta, stop=np.pi - delta, step=np.pi / 17):
            p_new = position_point(p0=p0, p1=p1, p2=p2, distance=distance, angle=angle, dihedral=psi)

            self.assertAlmostEqual(get_distance(p2, p_new), distance)
            self.assertAlmostEqual(get_angle(p1, p2, p_new), angle)
            self.assertAlmostEqual(get_dihedral(p0, p1, p2, p_new), psi)

    def test_neg_angles(self):
        p0 = np.array([0, 0, 1], dtype=np.float)
        p1 = np.array([0, 0, 0], dtype=np.float)
        p2 = np.array([0, 1, 0], dtype=np.float)

        angle = 1 * np.pi / 3
        p_neg = position_point(p0=p0, p1=p1, p2=p2, distance=2.5, angle=-1 * angle, dihedral=np.pi)
        self.assertAlmostEqual(get_angle(p1, p2, p_neg), angle)

    def test_neg_distance(self):
        p0 = np.array([0, 0, 1], dtype=np.float)
        p1 = np.array([0, 0, 0], dtype=np.float)
        p2 = np.array([0, 1, 0], dtype=np.float)

        distance = 2.5
        angle = 3 * np.pi / 2
        dihedral = 3 * np.pi / 2

        p_new = position_point(p0=p0, p1=p1, p2=p2, distance=-1 * distance, angle=angle, dihedral=dihedral)

        self.assertAlmostEqual(get_distance(p2, p_new), distance)

        # If the distance is negative, the angle is messed up!
        self.assertNotAlmostEqual(get_angle(p1, p2, p_new), angle)
