import itertools
from typing import List

import numpy as np

from molgym.buffer import DynamicPPOBuffer
from molgym.spaces import ObservationType


class PPOBufferContainer:
    def __init__(self, size: int, gamma: float, lam: float) -> None:
        super().__init__()

        self.gamma = gamma
        self.lam = lam
        self.size = size

        self.buffers = [DynamicPPOBuffer(gamma=self.gamma, lam=self.lam) for _ in range(self.size)]

        self.episodic_returns: List[float] = []
        self.episode_lengths: List[int] = []

    def get_num_episodes(self) -> int:
        num_returns = len(self.episodic_returns)
        assert num_returns == len(self.episode_lengths)
        return num_returns

    def store(
        self,
        observations: List[ObservationType],
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: List[ObservationType],
        terminals: np.ndarray,
        values: np.ndarray,
        logps: np.ndarray,
    ) -> None:
        assert len(observations) == actions.shape[0] == rewards.shape[0] == len(
            next_observations) == terminals.shape[0] == values.shape[0] == logps.shape[0] == len(self.buffers)

        for i, buffer in enumerate(self.buffers):
            buffer.store(
                obs=observations[i],
                act=actions[i],
                reward=rewards[i],
                next_obs=next_observations[i],
                terminal=terminals[i],
                value=values[i],
                logp=logps[i],
            )

            if terminals[i]:
                episodic_ret, episode_length = buffer.finish_path(0.0)
                assert episodic_ret is not None and episode_length > 0
                self.episodic_returns.append(episodic_ret)
                self.episode_lengths.append(episode_length)

    def finish_paths(self, values: np.ndarray):
        assert values.shape[0] == self.size

        for buffer, value in zip(self.buffers, values):
            # the buffer could be already finished so we have to check
            if not buffer.is_finished():
                # Don't record unfinished paths
                buffer.finish_path(value)

    def merge(self) -> DynamicPPOBuffer:
        new = DynamicPPOBuffer(gamma=self.gamma, lam=self.lam)

        assert all(buffer.is_finished() for buffer in self.buffers)

        for field in DynamicPPOBuffer.BUFFER_FIELDS:
            setattr(new, field, list(itertools.chain.from_iterable(getattr(buffer, field) for buffer in self.buffers)))

        return new
