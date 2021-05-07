import sys
from collections import defaultdict
from typing import Tuple, List, Dict

import ase.data
import gym
import numpy as np
from ase import Atom, Atoms

CanvasItemType = Tuple[int, Tuple[float, float, float]]
ActionType = CanvasItemType
CanvasType = Tuple[CanvasItemType, ...]
BagType = Tuple[int, ...]
ObservationType = Tuple[CanvasType, BagType]

FormulaType = Tuple[Tuple[int, int], ...]

NULL_SYMBOL = 'X'


class CanvasItemSpace(gym.spaces.Tuple):
    def __init__(self, zs: List[int]) -> None:
        self.zs = zs

        label = gym.spaces.Discrete(n=len(zs))

        low = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float)
        high = np.array([np.inf, np.inf, np.inf], dtype=np.float)
        position = gym.spaces.Box(low=low, high=high, dtype=np.float)

        super().__init__((label, position))

    def to_atom(self, canvas_item: CanvasItemType) -> Atom:
        label, position = canvas_item
        if label < 0:
            raise RuntimeError(f'Invalid atomic number: {label}')

        return Atom(symbol=self.zs[label], position=position)

    def from_atom(self, atom: Atom) -> CanvasItemType:
        return self.zs.index(ase.data.atomic_numbers[atom.symbol]), tuple(atom.position)  # type: ignore


ActionSpace = CanvasItemSpace


class CanvasSpace(gym.spaces.Tuple):
    def __init__(self, size: int, zs: List[int]) -> None:
        assert 0 in zs, '0 has to be in the list of atomic numbers'
        self.size = size
        self.zs = zs
        self.canvas_item_space = CanvasItemSpace(zs)
        super().__init__((self.canvas_item_space, ) * self.size)

    def to_atoms(self, canvas: CanvasType) -> Atoms:
        atoms = Atoms()
        for canvas_item in canvas:
            atom = self.canvas_item_space.to_atom(canvas_item)
            if atom.symbol != NULL_SYMBOL:
                atoms.append(atom)
        return atoms

    def from_atoms(self, atoms: Atoms) -> CanvasType:
        if len(atoms) > self.size:
            raise RuntimeError(f'Too many atoms: {len(atoms)} > {self.size}')

        elif len(atoms) < self.size:
            atoms = atoms.copy()

            dummy = Atom(symbol=NULL_SYMBOL, position=(0, 0, 0))
            while len(atoms) < self.size:
                atoms.append(dummy)

        return tuple(self.canvas_item_space.from_atom(atom) for atom in atoms)


class BagSpace(gym.spaces.Tuple):
    def __init__(self, zs: List[int]):
        self.zs = zs
        self.size = len(zs)
        self.bag_item_space = gym.spaces.Discrete(n=sys.maxsize)

        super().__init__((self.bag_item_space, ) * self.size)

    def to_formula(self, bag: BagType) -> FormulaType:
        assert len(bag) == self.size
        return tuple(zip(self.zs, bag))

    def from_formula(self, formula: FormulaType) -> BagType:
        assert all(z in self.zs for z, count in formula)
        formula_dict: Dict[int, int] = defaultdict(int)
        formula_dict.update(formula)
        return tuple(formula_dict[z] for z in self.zs)


class ObservationSpace(gym.spaces.Tuple):
    def __init__(self, canvas_size: int, zs: List[int]):
        self.zs = zs
        self.canvas_space = CanvasSpace(size=canvas_size, zs=zs)
        self.bag_space = BagSpace(zs=zs)
        super().__init__((self.canvas_space, self.bag_space))

    def build(self, atoms: Atoms, formula: FormulaType) -> ObservationType:
        return self.canvas_space.from_atoms(atoms), self.bag_space.from_formula(formula)

    def parse(self, observation: ObservationType) -> Tuple[Atoms, FormulaType]:
        return self.canvas_space.to_atoms(observation[0]), self.bag_space.to_formula(observation[1])
