from unittest import TestCase

import numpy as np

from molgym.tools.util import discount_cumsum, split_formula_strings, zs_to_formula


class TestTools(TestCase):
    def test_parse_formula(self):
        s = 'H2O, CH4, O2'
        formulas = split_formula_strings(s)

        self.assertEqual(len(formulas), 3)

    def test_zs_to_formula(self):
        formula = zs_to_formula([1, 1, 2, 4])
        self.assertEqual(len(formula), 3)

    def test_cumsum(self):
        discount = 0.5
        x = np.ones(3, dtype=np.float32)
        y = discount_cumsum(x, discount=discount)

        self.assertAlmostEqual(y[0], x[0] + discount * x[1] + discount**2 * x[2])
        self.assertAlmostEqual(y[1], x[1] + discount * x[2])
        self.assertAlmostEqual(y[2], x[2])
