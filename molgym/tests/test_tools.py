from unittest import TestCase

import numpy as np
from ase.formula import Formula

from molgym.tools.util import remove_from_formula, discount_cumsum, parse_formulas


class TestTools(TestCase):
    def test_formula(self):
        f = Formula('HCO')
        f2 = remove_from_formula(f, 'H')

        self.assertEqual(f2.count()['H'], 0)

        with self.assertRaises(KeyError):
            remove_from_formula(f, 'He')

    def test_parse_formula(self):
        s = 'H2O, CH4, O2'
        formulas = parse_formulas(s)

        self.assertEqual(len(formulas), 3)

    def test_cumsum(self):
        discount = 0.5
        x = np.ones(3, dtype=np.float32)
        y = discount_cumsum(x, discount=discount)

        self.assertAlmostEqual(y[0], x[0] + discount * x[1] + discount**2 * x[2])
        self.assertAlmostEqual(y[1], x[1] + discount * x[2])
        self.assertAlmostEqual(y[2], x[2])
