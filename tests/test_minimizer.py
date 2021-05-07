from unittest import TestCase

import numpy as np
from ase import Atoms

from molgym.calculator import Sparrow
from molgym.minimizer import minimize


class TestMinimizer(TestCase):
    def setUp(self):
        self.atoms = Atoms(symbols='OHH',
                           positions=[
                               (-0.27939703, 0.83823215, 0.00973345),
                               (-0.52040310, 1.77677325, 0.21391146),
                               (0.54473632, 0.90669722, -0.53501306),
                           ])

        self.charge = 0
        self.spin_multiplicity = 1

    def test_minimize(self):
        calculator = Sparrow('PM6')

        calculator.set_elements(list(self.atoms.symbols))
        calculator.set_positions(self.atoms.positions)
        calculator.set_settings({'molecular_charge': self.charge, 'spin_multiplicity': self.spin_multiplicity})
        energy1 = calculator.calculate_energy()
        gradients1 = calculator.calculate_gradients()

        opt_atoms, success = minimize(calculator=calculator,
                                      atoms=self.atoms,
                                      charge=self.charge,
                                      spin_multiplicity=self.spin_multiplicity)

        calculator.set_positions(opt_atoms.positions)
        energy2 = calculator.calculate_energy()
        gradients2 = calculator.calculate_gradients()

        self.assertTrue(energy1 > energy2)
        self.assertTrue(np.sum(np.square(gradients1)) > np.sum(np.square(gradients2)))
        self.assertTrue(np.all(gradients2 < 1E-3))

    def test_minimize_fail(self):
        calculator = Sparrow('PM6')
        calculator.set_elements(list(self.atoms.symbols))
        calculator.set_positions(self.atoms.positions)
        calculator.set_settings({'molecular_charge': self.charge, 'spin_multiplicity': self.spin_multiplicity})

        opt_atoms, success = minimize(
            calculator=calculator,
            atoms=self.atoms,
            charge=self.charge,
            spin_multiplicity=self.spin_multiplicity,
            max_iter=1,
        )

        self.assertFalse(success)

    def test_minimize_fixed(self):
        calculator = Sparrow('PM6')

        calculator.set_elements(list(self.atoms.symbols))
        calculator.set_positions(self.atoms.positions)
        calculator.set_settings({'molecular_charge': self.charge, 'spin_multiplicity': self.spin_multiplicity})

        fixed_index = 2
        opt_atoms, success = minimize(
            calculator=calculator,
            atoms=self.atoms,
            charge=self.charge,
            spin_multiplicity=self.spin_multiplicity,
            fixed_indices=[fixed_index],
        )

        self.assertTrue(np.all((self.atoms.positions - opt_atoms.positions)[fixed_index] < 1E-6))
