import abc
import itertools
import logging
from typing import Tuple, List

import ase.data
import gym
import numpy as np
from ase import Atoms, Atom
from scipy.spatial.qhull import ConvexHull, Delaunay

from molgym.reward import InteractionReward
from molgym.spaces import ActionSpace, ObservationSpace, ActionType, ObservationType, FormulaType
from molgym.tools.util import remove_atom_from_formula, get_formula_size, zs_to_formula


class AbstractMolecularEnvironment(gym.Env, abc.ABC):
    # Negative reward should be on the same order of magnitude as the positive ones.
    # Memory agent on QM9: mean 0.26, std 0.13, min -0.54, max 1.23 (negative reward indeed possible
    # but avoidable and probably due to PM6)

    def __init__(
        self,
        reward: InteractionReward,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        min_atomic_distance=0.6,  # Angstrom
        max_solo_distance=2.0,  # Angstrom
        min_reward=-0.6,  # Hartree
        seed=0,
    ):
        self.reward = reward
        self.observation_space = observation_space
        self.action_space = action_space

        self.random_state = np.random.RandomState(seed=seed)

        self.min_atomic_distance = min_atomic_distance
        self.max_solo_distance = max_solo_distance
        self.min_reward = min_reward

        self.current_atoms = Atoms()
        self.current_formula: FormulaType = tuple()

    @abc.abstractmethod
    def reset(self) -> ObservationType:
        raise NotImplementedError

    def step(self, action: ActionType) -> Tuple[ObservationType, float, bool, dict]:
        atomic_number_index, position = action
        atomic_number = self.action_space.zs[atomic_number_index]
        done = atomic_number == 0

        if done:
            return self.observation_space.build(self.current_atoms, self.current_formula), 0.0, done, {}

        new_atom = self.action_space.to_atom(action)
        if not self._is_valid_action(current_atoms=self.current_atoms, new_atom=new_atom):
            return (
                self.observation_space.build(self.current_atoms, self.current_formula),
                self.min_reward,
                True,
                {},
            )

        reward, info = self._calculate_reward(new_atom)

        if reward < self.min_reward:
            done = True
            reward = self.min_reward

        self.current_atoms.append(new_atom)
        self.current_formula = remove_atom_from_formula(self.current_formula, atomic_number)

        # Check if state is terminal
        if self._is_terminal():
            done = True

        return self.observation_space.build(self.current_atoms, self.current_formula), reward, done, info

    def _is_terminal(self) -> bool:
        return len(self.current_atoms) == self.observation_space.canvas_space.size or get_formula_size(
            self.current_formula) == 0

    def _is_valid_action(self, current_atoms: Atoms, new_atom: Atom) -> bool:
        if self._is_too_close(current_atoms, new_atom):
            return False

        return self._all_covered(current_atoms, new_atom)

    def _is_too_close(self, existing_atoms: Atoms, new_atom: Atom) -> bool:
        # Check distances between new and old atoms
        for existing_atom in existing_atoms:
            if np.linalg.norm(existing_atom.position - new_atom.position) < self.min_atomic_distance:
                logging.debug('Atoms are too close')
                return True

        return False

    def _calculate_reward(self, new_atom: Atom) -> Tuple[float, dict]:
        return self.reward.calculate(self.current_atoms, new_atom)

    def _all_covered(self, existing_atoms: Atoms, new_atom: Atom) -> bool:
        # Ensure that certain atoms are not too far away from the nearest heavy atom to avoid H2, F2,... formation
        candidates = ['H', 'F', 'Cl', 'Br']
        if len(existing_atoms) == 0 or new_atom.symbol not in candidates:
            return True

        for existing_atom in existing_atoms:
            if existing_atom.symbol in candidates:
                continue

            distance = np.linalg.norm(existing_atom.position - new_atom.position)
            if distance < self.max_solo_distance:
                return True

        logging.debug('There is a single atom floating around')
        return False

    def render(self, mode='human'):
        pass

    def seed(self, seed=None) -> int:
        seed = seed or np.random.randint(int(1e5))
        self.random_state = np.random.RandomState(seed)
        return seed


class MolecularEnvironment(AbstractMolecularEnvironment):
    def __init__(self, formulas: List[FormulaType], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.formulas = formulas
        self.formula_cycle = itertools.cycle(self.formulas)
        self.reset()

    def reset(self) -> ObservationType:
        self.current_atoms = Atoms()
        self.current_formula = next(self.formula_cycle)
        return self.observation_space.build(self.current_atoms, self.current_formula)


class ConstrainedMolecularEnvironment(MolecularEnvironment):
    def __init__(self, scaffold: Atoms, scaffold_z: int, *args, **kwargs):
        self.scaffold = scaffold
        self.scaffold_z = scaffold_z

        super().__init__(*args, **kwargs)

    def reset(self) -> ObservationType:
        self.current_atoms = self.scaffold.copy()
        self.current_formula = next(self.formula_cycle)
        return self.observation_space.build(self.current_atoms, self.current_formula)

    def _is_valid_action(self, current_atoms: Atoms, new_atom: Atom) -> bool:
        is_scaffold = list(ase.data.atomic_numbers[symbol] == self.scaffold_z for symbol in current_atoms.symbols)
        scaffold_atoms = current_atoms[is_scaffold]

        if not self._is_inside_scaffold(scaffold_positions=scaffold_atoms.positions, new_position=new_atom.position):
            logging.debug(f'Atom {new_atom} is not inside scaffold')
            return False

        # Make sure atom is not too close to _any_ other atom (also scaffold atoms)
        return super()._is_valid_action(current_atoms=current_atoms, new_atom=new_atom)

    @staticmethod
    def _is_inside_scaffold(scaffold_positions: np.ndarray, new_position: np.ndarray):
        hull = ConvexHull(scaffold_positions, incremental=False)
        vertices = scaffold_positions[hull.vertices]
        delaunay = Delaunay(vertices)
        return delaunay.find_simplex(new_position) >= 0

    def _calculate_reward(self, new_atom: Atom) -> Tuple[float, dict]:
        is_scaffold = list(ase.data.atomic_numbers[symbol] == self.scaffold_z for symbol in self.current_atoms.symbols)
        return self.reward.calculate(self.current_atoms[np.logical_not(is_scaffold)], new_atom)


class RefillableMolecularEnvironment(AbstractMolecularEnvironment):
    def __init__(self, formulas: List[FormulaType], initial_structure: Atoms, num_refills: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.formulas = formulas
        self.atoms = initial_structure.copy()
        self.num_refills = num_refills
        self.formulas_cycle = itertools.cycle(self.formulas)

        self.current_refill_counter = 0
        self.reset()

    def _is_terminal(self) -> bool:
        if len(self.current_atoms) == self.observation_space.canvas_space.size:
            return True

        if get_formula_size(self.current_formula) == 0:
            if self.current_refill_counter < self.num_refills:
                self.current_formula = next(self.formulas_cycle)
                self.current_refill_counter += 1
            else:
                return True

        return False

    def reset(self) -> ObservationType:
        self.current_refill_counter = 0
        self.current_atoms = self.atoms.copy()
        self.current_formula = next(self.formulas_cycle)
        return self.observation_space.build(self.current_atoms, self.current_formula)


class StochasticEnvironment(AbstractMolecularEnvironment):
    def __init__(self, formula: FormulaType, size_range: Tuple[int, int], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.formula = formula
        self.min_size, self.max_size = size_range

        formula_size = get_formula_size(formula)
        self.zs = [z for z, count in formula]
        self.z_probs = [count / formula_size for z, count in formula]

        self.z_to_bond_count = {
            1: 1,
            5: 3,
            6: 4,
            7: 3,
            8: 2,
            9: 1,
        }

        self.reset()

    def reset(self) -> ObservationType:
        self.current_atoms = Atoms()
        self.current_formula = self.sample_formula()
        while not self.is_valid_formula(self.current_formula):
            self.current_formula = self.sample_formula()

        return self.observation_space.build(self.current_atoms, self.current_formula)

    def sample_formula(self) -> FormulaType:
        if self.min_size < self.max_size:
            size = self.random_state.randint(low=self.min_size, high=self.max_size, size=1)
        else:
            size = self.max_size
        zs = np.random.choice(self.zs, size=size, replace=True, p=self.z_probs)
        return zs_to_formula(zs)

    def is_valid_formula(self, formula: FormulaType) -> bool:
        return sum(count * self.z_to_bond_count[z] for z, count in formula) % 2 == 0
