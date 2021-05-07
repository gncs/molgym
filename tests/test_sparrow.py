import os
from unittest import TestCase

import ase.io
import numpy as np
import pkg_resources
from ase import Atoms

from molgym.calculator import Sparrow

RESOURCES_FOLDER = 'resources'


class TestSparrow(TestCase):
    RESOURCES = pkg_resources.resource_filename(__package__, RESOURCES_FOLDER)

    def setUp(self):
        self.atoms = Atoms(symbols='HH', positions=[(0, 0, 0), (1.2, 0, 0)])
        self.charge = 0
        self.spin_multiplicity = 1

    def test_calculator(self):
        calculator = Sparrow('PM6')
        calculator.set_elements(list(self.atoms.symbols))
        calculator.set_positions(self.atoms.positions)
        calculator.set_settings({'molecular_charge': 0, 'spin_multiplicity': 1})

        gradients = calculator.calculate_gradients()
        energy = calculator.calculate_energy()

        self.assertAlmostEqual(energy, -0.9379853016)
        self.assertEqual(gradients.shape, (2, 3))

    def test_atomic_energies(self):
        calculator = Sparrow('PM6')
        calculator.set_positions([(0, 0, 0)])

        calculator.set_elements(['H'])
        calculator.set_settings({'molecular_charge': 0, 'spin_multiplicity': 2})
        self.assertAlmostEqual(calculator.calculate_energy(), -0.4133180865)

        calculator.set_elements(['C'])
        calculator.set_settings({'molecular_charge': 0, 'spin_multiplicity': 1})
        self.assertAlmostEqual(calculator.calculate_energy(), -4.162353543)

        calculator.set_elements(['O'])
        calculator.set_settings({'molecular_charge': 0, 'spin_multiplicity': 1})
        self.assertAlmostEqual(calculator.calculate_energy(), -10.37062419)

    def test_energy_gradients(self):
        calculator = Sparrow('PM6')
        atoms = ase.io.read(filename=os.path.join(self.RESOURCES, 'h2o.xyz'), format='xyz', index=0)
        calculator.set_positions(atoms.positions)
        calculator.set_elements(list(atoms.symbols))
        calculator.set_settings({'molecular_charge': 0, 'spin_multiplicity': 1})

        energy = calculator.calculate_energy()
        gradients = calculator.calculate_gradients()

        energy_file = os.path.join(self.RESOURCES, 'energy.dat')
        expected_energy = float(np.genfromtxt(energy_file))
        self.assertAlmostEqual(energy, expected_energy)

        gradients_file = os.path.join(self.RESOURCES, 'gradients.dat')
        expected_gradients = np.genfromtxt(gradients_file)
        self.assertTrue(np.allclose(gradients, expected_gradients))
