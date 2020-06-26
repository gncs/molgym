from unittest import TestCase

from ase import Atom
from ase.formula import Formula

from molgym.environment import MolecularEnvironment
from molgym.reward import InteractionReward
from molgym.spaces import ObservationSpace, ActionSpace


class TestEnvironment(TestCase):
    def setUp(self):
        self.reward = InteractionReward()
        self.symbols = ['H', 'C', 'N', 'O']
        self.observation_space = ObservationSpace(canvas_size=5, symbols=self.symbols)
        self.action_space = ActionSpace()

    def test_addition(self):
        formula = Formula('H2CO')
        env = MolecularEnvironment(reward=self.reward,
                                   observation_space=self.observation_space,
                                   action_space=self.action_space,
                                   formulas=[formula])
        action = self.action_space.from_atom(Atom(symbol='H', position=(0.0, 1.0, 0.0)))
        obs, reward, done, info = env.step(action=action)

        atoms1, f1 = self.observation_space.parse(obs)

        self.assertEqual(atoms1[0].symbol, 'H')
        self.assertDictEqual(f1.count(), {'H': 1, 'C': 1, 'O': 1, 'N': 0, 'X': 0})
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)

    def test_invalid_action(self):
        formula = Formula('H2CO')
        env = MolecularEnvironment(reward=self.reward,
                                   observation_space=self.observation_space,
                                   action_space=self.action_space,
                                   formulas=[formula])
        action = self.action_space.from_atom(Atom(symbol='He', position=(0, 1, 0)))
        with self.assertRaises(KeyError):
            env.step(action)

    def test_invalid_formula(self):
        formula = Formula('He2')

        with self.assertRaises(ValueError):
            MolecularEnvironment(reward=self.reward,
                                 observation_space=self.observation_space,
                                 action_space=self.action_space,
                                 formulas=[formula])

    def test_h_distance(self):
        formula = Formula('H2CO')
        env = MolecularEnvironment(
            reward=self.reward,
            observation_space=self.observation_space,
            action_space=self.action_space,
            formulas=[formula],
            max_h_distance=1.0,
        )

        # First H can be on its own
        action = self.action_space.from_atom(atom=Atom(symbol='H', position=(0, 0, 0)))
        obs, reward, done, info = env.step(action=action)
        self.assertFalse(done)

        # Second H cannot
        action = self.action_space.from_atom(atom=Atom(symbol='H', position=(0, 1.5, 0)))
        obs, reward, done, info = env.step(action=action)
        self.assertTrue(done)
