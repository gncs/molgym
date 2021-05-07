import abc
import time
from typing import Tuple, Dict

import ase.data
import numpy as np
from ase import Atoms, Atom

from molgym.calculator import Sparrow


class MolecularReward(abc.ABC):
    @abc.abstractmethod
    def calculate(self, atoms: Atoms, new_atom: Atom) -> Tuple[float, dict]:
        raise NotImplementedError

    @staticmethod
    def get_minimum_spin_multiplicity(atoms) -> int:
        return sum(ase.data.atomic_numbers[atom.symbol] for atom in atoms) % 2 + 1


class InteractionReward(MolecularReward):
    def __init__(self) -> None:
        # Due to some mysterious bug in Sparrow, calculations get slower and slower over time.
        # Therefore, we generate a new Sparrow object every time.
        self.calculator = Sparrow('PM6')

        self.settings = {
            'molecular_charge': 0,
            'max_scf_iterations': 128,
            'unrestricted_calculation': 1,
        }

        self.atom_energies: Dict[str, float] = {}

    def calculate(self, atoms: Atoms, new_atom: Atom) -> Tuple[float, dict]:
        start = time.time()
        self.calculator = Sparrow('PM6')

        all_atoms = atoms.copy()
        all_atoms.append(new_atom)

        e_tot = self._calculate_energy(all_atoms)
        e_parts = self._calculate_energy(atoms) + self._calculate_atomic_energy(new_atom)
        delta_e = e_tot - e_parts

        elapsed = time.time() - start

        reward = -1 * delta_e

        info = {
            'elapsed_time': elapsed,
        }

        return reward, info

    def _calculate_atomic_energy(self, atom: Atom) -> float:
        if atom.symbol not in self.atom_energies:
            atoms = Atoms()
            atoms.append(atom)
            self.atom_energies[atom.symbol] = self._calculate_energy(atoms)
        return self.atom_energies[atom.symbol]

    def _calculate_energy(self, atoms: Atoms) -> float:
        if len(atoms) == 0:
            return 0.0

        self.calculator.set_elements(list(atoms.symbols))
        self.calculator.set_positions(atoms.positions)
        self.settings['spin_multiplicity'] = self.get_minimum_spin_multiplicity(atoms)
        self.calculator.set_settings(self.settings)
        return self.calculator.calculate_energy()


class SolvationReward(InteractionReward):
    def __init__(self, distance_penalty=0.01) -> None:
        super().__init__()

        self.distance_penalty = distance_penalty

    def calculate(self, atoms: Atoms, new_atom: Atom) -> Tuple[float, dict]:
        start_time = time.time()
        self.calculator = Sparrow('PM6')

        all_atoms = atoms.copy()
        all_atoms.append(new_atom)

        e_tot = self._calculate_energy(all_atoms)
        e_parts = self._calculate_energy(atoms) + self._calculate_atomic_energy(new_atom)
        delta_e = e_tot - e_parts

        distance = np.linalg.norm(new_atom.position)

        reward = -1 * (delta_e + self.distance_penalty * distance)

        info = {
            'elapsed_time': time.time() - start_time,
        }

        return reward, info
