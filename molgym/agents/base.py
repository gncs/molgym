import abc
from typing import List, Optional

import numpy as np
import torch.distributions

from molgym.spaces import ObservationType, ObservationSpace, ActionSpace


class AbstractActorCritic(torch.nn.Module, abc.ABC):
    def __init__(self, observation_space: ObservationSpace, action_space: ActionSpace):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

    @abc.abstractmethod
    def step(self, observations: List[ObservationType], actions: Optional[np.ndarray] = None) -> dict:
        raise NotImplementedError
