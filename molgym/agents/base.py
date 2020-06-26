import abc
from typing import List, Optional

import numpy as np
import torch.distributions

from molgym.spaces import ObservationType, ActionType, ObservationSpace, ActionSpace


class AbstractActorCritic(torch.nn.Module, abc.ABC):
    def __init__(self, observation_space: ObservationSpace, action_space: ActionSpace, internal_action_dim: int):
        super().__init__()

        self.internal_action_dim = internal_action_dim
        self.observation_space = observation_space
        self.action_space = action_space

    @abc.abstractmethod
    def step(self, observations: List[ObservationType], action: Optional[np.ndarray] = None) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def to_action_space(self, action: np.ndarray, observation: ObservationType) -> ActionType:
        raise NotImplementedError
