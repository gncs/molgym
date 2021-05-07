# The content of this file is based on: OpenAI Spinning Up https://spinningup.openai.com/.
from typing import Optional, List, Tuple

import numpy as np

from molgym.spaces import ObservationType
from molgym.tools import util


class DynamicPPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    BUFFER_FIELDS = [
        'obs_buf', 'act_buf', 'rew_buf', 'next_obs_buf', 'term_buf', 'val_buf', 'logp_buf', 'adv_buf', 'ret_buf'
    ]

    def __init__(self, gamma=0.99, lam=0.95) -> None:
        self.obs_buf: List[ObservationType] = []
        self.act_buf: List[np.ndarray] = []
        self.rew_buf: List[float] = []
        self.next_obs_buf: List[ObservationType] = []
        self.term_buf: List[bool] = []

        self.val_buf: List[float] = []
        self.logp_buf: List[float] = []

        # Filled when path is finished
        self.adv_buf: List[float] = []
        self.ret_buf: List[float] = []

        self.gamma = gamma
        self.lam = lam

        self.current_index = 0
        self.start_index = 0

    def store(self, obs: ObservationType, act: np.ndarray, reward: float, next_obs: ObservationType, terminal: bool,
              value: float, logp: float) -> None:
        """Append one time step of agent-environment interaction to the buffer."""
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(reward)
        self.next_obs_buf.append(next_obs)
        self.term_buf.append(terminal)

        self.val_buf.append(value)
        self.logp_buf.append(logp)

        self.current_index += 1

    def finish_path(self, last_val: float) -> Tuple[Optional[float], int]:
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

        if self.is_finished():
            return None, 0

        path_slice = slice(self.start_index, self.current_index)
        rews = np.array(self.rew_buf[path_slice] + [last_val])
        vals = np.array(self.val_buf[path_slice] + [last_val])

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf += util.discount_cumsum(deltas, self.gamma * self.lam).tolist()

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf += util.discount_cumsum(rews, self.gamma).tolist()[:-1]

        episodic_return = self.ret_buf[self.start_index]
        episode_length = self.current_index - self.start_index

        self.start_index = self.current_index

        # Ensure that all buffer fields have the same length
        assert all(len(getattr(self, field)) == self.current_index for field in DynamicPPOBuffer.BUFFER_FIELDS)

        return episodic_return, episode_length

    def is_finished(self) -> bool:
        return self.start_index == self.current_index

    def get_data(self) -> dict:
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.is_finished()

        # advantage normalization trick
        adv_buf = np.array(self.adv_buf)
        adv_mean = np.mean(adv_buf)
        adv_std = np.std(adv_buf)

        adv_buf_standard = (adv_buf - adv_mean) / adv_std

        return dict(obs=self.obs_buf,
                    act=np.array(self.act_buf),
                    ret=np.array(self.ret_buf),
                    adv=adv_buf_standard,
                    logp=np.array(self.logp_buf))
