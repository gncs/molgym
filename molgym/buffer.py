# The content of this file is based on: OpenAI Spinning Up https://spinningup.openai.com/.
from typing import Optional, List

import numpy as np

from molgym.spaces import ObservationType
from molgym.tools import util
from molgym.tools.mpi import mpi_mean_std


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, int_act_dim: int, size: int, gamma=0.99, lam=0.95) -> None:
        self.obs_buf: List[Optional[ObservationType]] = [None] * size
        self.act_buf = np.empty((size, int_act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.next_obs_buf: List[Optional[ObservationType]] = [None] * size
        self.term_buf = np.zeros(size, dtype=np.bool)

        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        # Filled when path is finished
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)

        self.gamma = gamma
        self.lam = lam

        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size

    def store(self, obs: ObservationType, act: np.ndarray, reward: float, next_obs: ObservationType, terminal: bool,
              value: float, logp: float):
        """Append one time step of agent-environment interaction to the buffer."""
        assert self.ptr < self.max_size  # buffer has to have room so you can store

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.term_buf[self.ptr] = terminal

        self.val_buf[self.ptr] = value
        self.logp_buf[self.ptr] = logp

        self.ptr += 1

    def finish_path(self, last_val: float) -> float:
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = util.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = util.discount_cumsum(rews, self.gamma)[:-1]
        episodic_return = self.ret_buf[self.path_start_idx]

        self.path_start_idx = self.ptr

        return episodic_return

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.is_full()  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_mean_std(self.adv_buf, axis=-1)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        return dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)

    def is_full(self) -> bool:
        return self.ptr == self.max_size
