from unittest import TestCase

import numpy as np
from ase import Atoms, Atom

from molgym.spaces import ObservationSpace, CanvasItemSpace, ActionSpace, CanvasSpace, BagSpace


class TestAtomicSpace(TestCase):
    def test_atom(self):
        space = CanvasItemSpace(zs=[0, 1, 6])
        sample = space.sample()
        element, position = sample

        self.assertTrue(isinstance(element, int))
        self.assertEqual(len(position), 3)

        atom = space.to_atom(sample)
        tup = space.from_atom(atom)

        self.assertEqual(element, tup[0])
        self.assertTrue(np.isclose(position, tup[1]).all())

    def test_invalid_atom(self):
        space = CanvasItemSpace(zs=[1, 6])

        with self.assertRaises(IndexError):
            space.to_atom((2, (0, 0, 0)))

        with self.assertRaises(RuntimeError):
            space.to_atom((-1, (0, 0, 0)))

        with self.assertRaises(ValueError):
            space.to_atom((1, ('H', 0, 0)))  # type: ignore


class TestMolecularSpace(TestCase):
    def test_atoms(self):
        space = CanvasSpace(size=5, zs=[0, 1])
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
        space = CanvasSpace(size=2, zs=[0, 1])
        atoms = Atoms(symbols='HHH')
        with self.assertRaises(RuntimeError):
            space.from_atoms(atoms)


class TestBagSpace(TestCase):
    def setUp(self):
        self.atomic_numbers = [1, 6, 7, 8]
        self.bag_space = BagSpace(self.atomic_numbers)

    def test_bag(self):
        for item in self.bag_space.sample():
            self.assertIsInstance(item, int)
        self.assertEqual(len(self.bag_space.sample()), len(self.atomic_numbers))

    def test_invalid_bag(self):
        with self.assertRaises(AssertionError):
            self.bag_space.to_formula((1, 0, 0, 0, 0))


class TestActionSpace(TestCase):
    def setUp(self) -> None:
        self.space = ActionSpace(zs=[0, 1, 6])

    def test_action(self):
        self.assertIsNone(self.space.shape)

    def test_shape(self):
        action = self.space.sample()
        self.assertEqual(len(action), 2)
        self.assertEqual(len(action[1]), 3)

    def test_build(self):
        symbol = 'C'
        action = self.space.from_atom(Atom(symbol=symbol))
        self.assertEqual(action[0], 2)
        self.assertEqual(self.space.to_atom(action).symbol, symbol)


class TestObservationSpace(TestCase):
    def setUp(self):
        self.atomic_numbers = [0, 1, 6, 7]

    def test_molecular_observation(self):
        space = ObservationSpace(canvas_size=5, zs=self.atomic_numbers)
        canvas, bag = space.sample()

        self.assertEqual(len(canvas), 5)
        self.assertEqual(len(bag), len(self.atomic_numbers))
