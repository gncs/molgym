import sys
from typing import Tuple, List

import gym
import numpy as np
from ase import Atom, Atoms
from ase.data import chemical_symbols, atomic_numbers
from ase.formula import Formula

AtomicType = Tuple[int, Tuple[float, float, float]]
MolecularType = Tuple[AtomicType, ...]
BagType = Tuple[int, ...]
ActionType = AtomicType
ObservationType = Tuple[MolecularType, BagType]

NULL_SYMBOL = 'X'


class AtomicSpace(gym.spaces.Tuple):
    def __init__(self) -> None:
        element = gym.spaces.Discrete(n=len(atomic_numbers))

        low = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float)
        high = np.array([np.inf, np.inf, np.inf], dtype=np.float)
        position = gym.spaces.Box(low=low, high=high, dtype=np.float)

        super().__init__((element, position))

    @staticmethod
    def to_atom(sample: AtomicType) -> Atom:
        atomic_number, position = sample
        if atomic_number < 0:
            raise RuntimeError(f'Invalid atomic number: {atomic_number}')

        return Atom(symbol=chemical_symbols[atomic_number], position=position)

    @staticmethod
    def from_atom(atom: Atom) -> AtomicType:
        return int(atomic_numbers[atom.symbol]), tuple(atom.position)  # type: ignore


class MolecularSpace(gym.spaces.Tuple):
    def __init__(self, size: int) -> None:
        self.size = size
        super().__init__((AtomicSpace(), ) * self.size)

    @staticmethod
    def to_atoms(sample: MolecularType) -> Atoms:
        atoms = Atoms()
        for atomic_sample in sample:
            atom = AtomicSpace.to_atom(atomic_sample)
            if atom.symbol == NULL_SYMBOL:
                break
            atoms.append(atom)
        return atoms

    def from_atoms(self, atoms: Atoms) -> MolecularType:
        if len(atoms) > self.size:
            raise RuntimeError(f'Too many atoms: {len(atoms)} > {self.size}')

        elif len(atoms) < self.size:
            atoms = atoms.copy()

            dummy = Atom(symbol=NULL_SYMBOL, position=(0, 0, 0))
            while len(atoms) < self.size:
                atoms.append(dummy)

        return tuple(AtomicSpace.from_atom(atom) for atom in atoms)


ActionSpace = AtomicSpace


class SymbolTable:
    def __init__(self, symbols: List[str]):
        if NULL_SYMBOL in symbols:
            raise RuntimeError(f'Place holder symbol {NULL_SYMBOL} cannot be in list of symbols')

        if len(symbols) < 1:
            raise RuntimeError('List of symbols cannot be empty')

        # Ensure that all symbols are valid
        for symbol in symbols:
            chemical_symbols.index(symbol)

        # Ensure that there are no duplicates
        if len(set(symbols)) != len(symbols):
            raise RuntimeError(f'List of symbols {symbols} cannot contain duplicates')

        self._symbols = [NULL_SYMBOL] + symbols

    def get_index(self, symbol: str) -> int:
        return self._symbols.index(symbol)

    def get_symbol(self, index: int) -> str:
        if index < 0:
            raise ValueError(f'Index ({index}) cannot be less than zero')
        return self._symbols[index]

    def count(self) -> int:
        return len(self._symbols)


class BagSpace(gym.spaces.Tuple):
    def __init__(self, symbols: List[str]):
        self.symbol_table = SymbolTable(symbols)
        self.size = self.symbol_table.count()

        super().__init__((gym.spaces.Discrete(n=sys.maxsize), ) * self.size)

    def from_formula(self, formula: Formula) -> BagType:
        formula_dict = formula.count()

        bag = [0] * self.symbol_table.count()
        for symbol, value in formula_dict.items():
            bag[self.symbol_table.get_index(symbol)] = value

        return tuple(bag)

    def to_formula(self, bag: 'BagType') -> Formula:
        if len(bag) != self.symbol_table.count():
            raise ValueError(f'Bag {bag} does not fit symbol table')

        d = {self.symbol_table.get_symbol(index): count for index, count in enumerate(bag)}
        return Formula.from_dict(d)

    def get_symbol(self, index: int) -> str:
        return self.symbol_table.get_symbol(index)

    def get_index(self, symbol: str) -> int:
        return self.symbol_table.get_index(symbol)


class ObservationSpace(gym.spaces.Tuple):
    def __init__(self, canvas_size: int, symbols: List[str]):
        self.canvas_space = MolecularSpace(size=canvas_size)
        self.bag_space = BagSpace(symbols=symbols)
        super().__init__((self.canvas_space, self.bag_space))

    def build(self, atoms: Atoms, formula: Formula) -> ObservationType:
        return self.canvas_space.from_atoms(atoms), self.bag_space.from_formula(formula)

    def parse(self, observation: ObservationType) -> Tuple[Atoms, Formula]:
        return self.canvas_space.to_atoms(observation[0]), self.bag_space.to_formula(observation[1])
