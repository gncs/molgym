from unittest import TestCase

from ase import Atom

from molgym.environment import MolecularEnvironment
from molgym.reward import InteractionReward
from molgym.spaces import ObservationSpace, ActionSpace
from molgym.tools.util import string_to_formula


class TestEnvironment(TestCase):
    def setUp(self):
        self.reward = InteractionReward()
        self.zs = [0, 1, 6, 7, 8]
        self.observation_space = ObservationSpace(canvas_size=5, zs=self.zs)
        self.action_space = ActionSpace(zs=self.zs)

    def test_addition(self):
        formula = string_to_formula('H2CO')
        env = MolecularEnvironment(reward=self.reward,
                                   observation_space=self.observation_space,
                                   action_space=self.action_space,
                                   formulas=[formula])
        action = self.action_space.from_atom(Atom(symbol='H', position=(0.0, 1.0, 0.0)))
        obs, reward, done, info = env.step(action=action)

        atoms1, formula = self.observation_space.parse(obs)

        self.assertEqual(atoms1[0].symbol, 'H')
        self.assertEqual(formula, ((0, 0), (1, 1), (6, 1), (7, 0), (8, 1)))
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)

    def test_invalid_action(self):
        formula = string_to_formula('H2CO')
        env = MolecularEnvironment(reward=self.reward,
                                   observation_space=self.observation_space,
                                   action_space=self.action_space,
                                   formulas=[formula])
        action = self.action_space.from_atom(Atom(symbol='N', position=(0, 1, 0)))
        with self.assertRaises(RuntimeError):
            env.step(action)

    def test_invalid_formula(self):
        formula = string_to_formula('He2')
        with self.assertRaises(AssertionError):
            self.observation_space.bag_space.from_formula(formula)

    def test_solo_distance(self):
        formula = string_to_formula('H2CO')
        env = MolecularEnvironment(
            reward=self.reward,
            observation_space=self.observation_space,
            action_space=self.action_space,
            formulas=[formula],
            max_solo_distance=1.0,
        )

        # First H can be on its own
        action = self.action_space.from_atom(atom=Atom(symbol='H', position=(0, 0, 0)))
        obs, reward, done, info = env.step(action=action)
        self.assertFalse(done)

        # Second H cannot
        action = self.action_space.from_atom(atom=Atom(symbol='H', position=(0, 1.5, 0)))
        obs, reward, done, info = env.step(action=action)
        self.assertTrue(done)
