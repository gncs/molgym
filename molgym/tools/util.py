import collections
import json
import logging
import os
import pickle
import sys
from typing import Optional, List, Iterable, Tuple, Dict

import ase.data
import ase.formula
import numpy as np
import scipy.signal
import torch
from ase.formula import Formula
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from molgym.spaces import FormulaType


def string_to_formula(string: str) -> FormulaType:
    d = Formula(string).count().items()
    return tuple((ase.data.atomic_numbers[symbol], count) for symbol, count in d)


def zs_to_formula(zs: List[int]) -> FormulaType:
    counter: Dict[int, int] = collections.Counter()
    for z in zs:
        counter[z] += 1
    return tuple(counter.items())


def remove_atom_from_formula(formula: FormulaType, atomic_number: int) -> FormulaType:
    copy = list(formula)
    for i, (z, count) in enumerate(formula):
        if z == atomic_number and count >= 1:
            copy[i] = (z, count - 1)
            return tuple(copy)

    raise RuntimeError(f"Could not remove atomic number {atomic_number} from bag {formula}")


def get_formula_size(formula: FormulaType) -> int:
    return sum(count for z, count in formula)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


def combined_shape(length: int, shape: Optional[tuple] = None) -> tuple:
    if shape is None:
        return length,
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module: torch.nn.Module) -> int:
    return sum(np.prod(p.shape) for p in module.parameters())


def compute_gradient_norm(parameters: Iterable[torch.nn.Parameter], norm_type: int = 2) -> float:
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    if len(parameters) == 0:
        return 0.0
    device = parameters[0].grad.device  # type: ignore
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),  # type: ignore
        norm_type)
    return total_norm.item()


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_formula_strings(formulas: str) -> List[str]:
    return formulas.split(',')


def parse_size_range(size_range: str) -> Tuple[int, int]:
    parsed_range = [int(i) for i in size_range.split(',')]
    assert len(parsed_range) == 2
    return parsed_range[0], parsed_range[1]


def get_tag(config: dict) -> str:
    return '{exp}_run-{seed}'.format(exp=config['name'], seed=config['seed'])


def save_config(config: dict, directory: str, tag: str, verbose=True):
    formatted = json.dumps(config, indent=4, sort_keys=True)

    if verbose:
        logging.info(formatted)

    path = os.path.join(directory, tag + '.json')
    with open(file=path, mode='w') as f:
        f.write(formatted)


def create_directories(directories: List[str]):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def setup_logger(config: dict, directory, tag: str):
    logger = logging.getLogger()
    logger.setLevel(config['log_level'])

    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    path = os.path.join(directory, tag + '.log')
    fh = logging.FileHandler(path)
    fh.setFormatter(formatter)

    logger.addHandler(fh)


def setup_simple_logger(path: str = None, log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(message)s')

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if path:
        fh = logging.FileHandler(path, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)


class RolloutSaver:
    def __init__(self, directory: str, tag: str):
        self.directory = directory
        self.tag = tag
        self._suffix = '.pkl'

    def save(self, obj: object, num_steps: int, info: str):
        added = f'steps-{num_steps}'

        path = os.path.join(self.directory, self.tag + '_' + added + '_' + info + self._suffix)
        logging.debug(f'Saving rollout: {path}')
        with open(path, mode='wb') as f:
            pickle.dump(obj, f)


class InfoSaver:
    def __init__(self, directory: str, tag: str):
        self.directory = directory
        self.tag = tag
        self._suffix = '.txt'

    def save(self, obj: object, name: str):
        path = os.path.join(self.directory, self.tag + '_' + name + self._suffix)
        logging.debug(f'Saving info: {path}')
        with open(path, mode='a') as f:
            f.write(json.dumps(obj))
            f.write('\n')


def init_device(device_str: str) -> torch.device:
    if device_str == 'cuda':
        assert (torch.cuda.is_available()), 'No CUDA device available!'
        logging.info('CUDA Device: {}'.format(torch.cuda.current_device()))
        torch.cuda.init()
        return torch.device('cuda')
    else:
        logging.info('Using CPU')
        return torch.device('cpu')


def get_optimizer(name: str, learning_rate: float, parameters: Iterable[torch.Tensor]) -> Optimizer:
    if name == 'adam':
        amsgrad = False
    elif name == 'amsgrad':
        amsgrad = True
    else:
        raise RuntimeError(f"Unknown optimizer '{name}'")

    return Adam(parameters, lr=learning_rate, amsgrad=amsgrad)
