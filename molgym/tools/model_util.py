import logging
import os
import re
from dataclasses import dataclass
from typing import Tuple, Optional, Sequence

import torch

from molgym.agents.base import AbstractActorCritic
from molgym.agents.covariant.agent import CovariantAC
from molgym.agents.internal.agent import SchNetAC
from molgym.spaces import ObservationSpace, ActionSpace


def build_model(config: dict, observation_space: ObservationSpace, action_space: ActionSpace,
                device: torch.device) -> AbstractActorCritic:
    if config['model'] == 'internal':
        return SchNetAC(
            observation_space=observation_space,
            action_space=action_space,
            min_max_distance=(config['min_mean_distance'], config['max_mean_distance']),
            network_width=config['network_width'],
            device=device,
        )
    elif config['model'] == 'covariant':
        return CovariantAC(
            observation_space=observation_space,
            action_space=action_space,
            min_max_distance=(config['min_mean_distance'], config['max_mean_distance']),
            network_width=config['network_width'],
            maxl=config['maxl'],
            num_cg_levels=config['num_cg_levels'],
            num_channels_hidden=config['num_channels_hidden'],
            num_channels_per_element=config['num_channels_per_element'],
            num_gaussians=config['num_gaussians'],
            bag_scale=config['bag_scale'],
            beta=float(config['beta']) if config['beta'] is not None else config['beta'],
            device=device,
        )
    else:
        raise RuntimeError(f'Model \'{config["model"]}\' is not available.')


@dataclass
class ModelPathInfo:
    path: str
    tag: str
    num_steps: int


class ModelIO:
    def __init__(self, directory: str, tag: str, keep: bool = False) -> None:
        self.directory = directory
        self.tag = tag
        self.keep = keep
        self.old_path: Optional[str] = None

        self._steps_string = '_steps-'
        self._suffix = '.model'
        self._iter_suffix = '.txt'

    def _get_model_filename(self, num_steps: int) -> str:
        return self.tag + self._steps_string + str(num_steps) + self._suffix

    def _list_file_paths(self) -> Sequence[str]:
        all_paths = [os.path.join(self.directory, f) for f in os.listdir(self.directory)]
        return [path for path in all_paths if os.path.isfile(path)]

    def _parse_model_path(self, path: str) -> Optional[ModelPathInfo]:
        filename = os.path.basename(path)
        regex = re.compile(rf'(?P<tag>.+){self._steps_string}(?P<num_steps>\d+){self._suffix}')
        match = regex.match(filename)
        if not match:
            return None

        return ModelPathInfo(
            path=path,
            tag=match.group('tag'),
            num_steps=int(match.group('num_steps')),
        )

    def save(self, module: AbstractActorCritic, num_steps: int) -> None:
        if not self.keep and self.old_path:
            logging.debug(f'Deleting old model: {self.old_path}')
            os.remove(self.old_path)

        filename = self._get_model_filename(num_steps)
        path = os.path.join(self.directory, filename)
        logging.debug(f'Saving model: {path}')
        torch.save(obj=module, f=path)
        self.old_path = path

    def load(self, device: torch.device, path: str) -> Tuple[AbstractActorCritic, int]:
        model_info = self._parse_model_path(path)

        if model_info is None:
            raise RuntimeError(f"Cannot find model '{path}'")

        logging.info(f'Loading model: {model_info.path}')
        model = torch.load(f=model_info.path, map_location=device)

        return model, model_info.num_steps

    def load_latest(self, device: torch.device) -> Tuple[AbstractActorCritic, int]:
        all_file_paths = self._list_file_paths()
        model_infos = [self._parse_model_path(path) for path in all_file_paths]
        selected_model_infos = [info for info in model_infos if info and info.tag == self.tag]

        if len(selected_model_infos) == 0:
            raise RuntimeError(f"Cannot find model to load in '{self.directory}'")

        latest_model_info = max(selected_model_infos, key=lambda info: info.num_steps)

        logging.info(f'Loading model: {latest_model_info.path}')
        model = torch.load(f=latest_model_info.path, map_location=device)

        return model, latest_model_info.num_steps
