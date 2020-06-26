import json
import logging
import os
import pickle
import sys
from typing import Optional, List, Tuple

import ase.formula
import numpy as np
import scipy.signal
import torch
from ase.formula import Formula

from molgym.agents.base import AbstractActorCritic
from molgym.tools import mpi


def remove_from_formula(formula: Formula, symbol: str) -> Formula:
    d = formula.count()
    d[symbol] -= 1
    return Formula.from_dict(d)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


def combined_shape(length: int, shape: Optional[tuple] = None) -> tuple:
    if shape is None:
        return length,
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module: torch.nn.Module) -> int:
    return sum(np.prod(p.shape) for p in module.parameters())


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


def set_one_thread():
    # Avoid certain slowdowns from PyTorch + MPI combo.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_formulas(formulas: str) -> List[ase.formula.Formula]:
    return [ase.formula.Formula(s.strip()) for s in formulas.split(',')]


def get_tag(config: dict) -> str:
    return '{exp}_run-{seed}'.format(exp=config['name'], seed=config['seed'])


def save_config(config: dict, directory: str, tag: str, verbose=True):
    if not mpi.is_main_proc():
        return

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

    if not mpi.is_main_proc() and not config['all_ranks']:
        # Set level to a something higher than logging.CRITICAL to silence all messages
        logger.setLevel(logging.CRITICAL + 1)
    else:
        logger.setLevel(config['log_level'])

    name = ''
    if mpi.get_num_procs() > 1:
        name = f'rank[{mpi.get_proc_rank()}] '

    formatter = logging.Formatter('%(asctime)s.%(msecs)03d ' + name + '%(levelname)s: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    path = os.path.join(directory, tag + '.log')
    fh = mpi.MPIFileHandler(path)
    fh.setFormatter(formatter)

    logger.addHandler(fh)


def setup_simple_logger(path: str, log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(message)s')

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(path, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class ModelIO:
    def __init__(self, directory: str, tag: str) -> None:
        self.directory = directory
        self.root_name = tag
        self._suffix = '.model'
        self._iter_suffix = '.txt'

    def _get_model_path(self) -> str:
        return os.path.join(self.directory, self.root_name + self._suffix)

    def _get_info_path(self) -> str:
        return os.path.join(self.directory, self.root_name + self._iter_suffix)

    def save(self, module: AbstractActorCritic, num_steps: int):
        if not mpi.is_main_proc():
            return

        # Save model
        model_path = self._get_model_path()
        logging.debug(f'Saving model: {model_path}')
        torch.save(obj=module, f=model_path)

        # Save iteration
        info_path = self._get_info_path()
        with open(info_path, mode='w') as f:
            f.write(str(num_steps))

    def load(self) -> Tuple[AbstractActorCritic, int]:
        # Load model
        model_path = self._get_model_path()
        logging.info(f'Loading model: {model_path}')
        model = torch.load(f=model_path)

        # Load number of steps
        info_path = self._get_info_path()
        with open(info_path, mode='r') as f:
            num_steps = int(f.read())

        return model, num_steps


class RolloutSaver:
    def __init__(self, directory: str, tag: str, all_ranks=False):
        self.directory = directory
        self.tag = tag
        self._suffix = '.pkl'

        self.all_ranks = all_ranks

    def save(self, obj: object, num_steps: int, info: str):
        if not self.all_ranks and not mpi.is_main_proc():
            return

        added = f'steps-{num_steps}_rank-{mpi.get_proc_rank()}'

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
        if not mpi.is_main_proc():
            return

        path = os.path.join(self.directory, self.tag + '_' + name + self._suffix)
        logging.debug(f'Saving info: {path}')
        with open(path, mode='a') as f:
            f.write(json.dumps(obj))
            f.write('\n')
