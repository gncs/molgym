from unittest import TestCase

import pkg_resources
from ase import Atoms, Atom

from molgym.reward import InteractionReward

RESOURCES_FOLDER = 'resources'


class TestReward(TestCase):
    RESOURCES = pkg_resources.resource_filename(__package__, RESOURCES_FOLDER)

    def setUp(self):
        self.reward = InteractionReward()

    def test_calculation(self):
        reward, info = self.reward.calculate(Atoms(), Atom('H'))
        self.assertEqual(reward, 0)

    def test_h2(self):
        atom1 = Atom('H', position=(0, 0, 0))
        atom2 = Atom('H', position=(1, 0, 0))

        atoms = Atoms()
        atoms.append(atom1)

        reward, info = self.reward.calculate(atoms, atom2)

        self.assertAlmostEqual(reward, 0.1696435)

    def test_addition(self):
        atom1 = Atom('H', position=(0, 0, 0))
        atom2 = Atom('H', position=(1, 0, 0))
        atom3 = Atom('H', position=(2, 0, 0))

        atoms = Atoms()
        atoms.append(atom1)

        reward1, _ = self.reward.calculate(atoms, atom2)
        atoms.append(atom2)

        reward2, _ = self.reward.calculate(atoms, atom3)
        atoms.append(atom3)

        self.assertAlmostEqual(reward1 + reward2, 0.2141968)
