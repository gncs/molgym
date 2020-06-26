from unittest import TestCase

import numpy as np
from ase import Atoms, Atom
from ase.formula import Formula

from molgym.spaces import ObservationSpace, AtomicSpace, ActionSpace, MolecularSpace, BagSpace


class TestAtomicSpace(TestCase):
    def test_atom(self):
        space = AtomicSpace()
        sample = space.sample()
        element, position = sample

        self.assertTrue(isinstance(element, int))
        self.assertEqual(len(position), 3)

        atom = AtomicSpace.to_atom(sample)
        tup = AtomicSpace.from_atom(atom)

        self.assertEqual(element, tup[0])
        self.assertTrue(np.isclose(position, tup[1]).all())

    def test_invalid_atom(self):
        with self.assertRaises(RuntimeError):
            invalid = (-1, (0, 0, 0))
            AtomicSpace.to_atom(invalid)

        with self.assertRaises(ValueError):
            invalid = (1, ('H', 0, 0))  # type: ignore
            AtomicSpace.to_atom(invalid)


class TestMolecularSpace(TestCase):
    def test_atoms(self):
        space = MolecularSpace(size=5)
        sample = space.sample()

        self.assertEqual(len(sample), 5)

        atoms = space.to_atoms(sample)
        self.assertLessEqual(len(atoms), 5)

        atoms = Atoms()
        tup = space.from_atoms(atoms)

        self.assertEqual(len(tup), 5)
        for element, position in tup:
            self.assertEqual(element, 0)

        parsed = space.to_atoms(tup)
        self.assertEqual(len(parsed), 0)

    def test_invalid_atoms(self):
        space = MolecularSpace(size=2)
        atoms = Atoms(symbols='HHH')
        with self.assertRaises(RuntimeError):
            space.from_atoms(atoms)


class TestBagSpace(TestCase):
    def setUp(self):
        self.symbols = ['H', 'C', 'N', 'O']

    def test_bag(self):
        bag_space = BagSpace(self.symbols)
        # This is an ASE bug
        self.assertEqual(bag_space.to_formula((0, 0, 2, 0, 0)).format(), 'XHC2NO')
        self.assertEqual(len(bag_space.sample()), 5)

    def test_invalid_bag(self):
        bag_space = BagSpace(symbols=self.symbols)
        with self.assertRaises(ValueError):
            bag_space.from_formula(Formula('F'))


class TestActionSpace(TestCase):
    def setUp(self) -> None:
        self.space = ActionSpace()

    def test_action(self):
        self.assertIsNone(self.space.shape)

    def test_shape(self):
        action = self.space.sample()
        self.assertEqual(len(action), 2)
        self.assertEqual(len(action[1]), 3)

    def test_build(self):
        symbol = 'C'
        action = self.space.from_atom(Atom(symbol=symbol))
        self.assertEqual(action[0], 6)
        self.assertEqual(self.space.to_atom(action).symbol, symbol)


class TestObservationSpace(TestCase):
    def setUp(self):
        self.symbols = ['H', 'C', 'N']

    def test_molecular_observation(self):
        space = ObservationSpace(canvas_size=5, symbols=self.symbols)
        canvas, bag = space.sample()

        self.assertEqual(len(canvas), 5)
        self.assertEqual(len(bag), len(self.symbols) + 1)  # +1 for 'X'
