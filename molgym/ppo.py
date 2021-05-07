# The content of this file is based on: DeepRL https://github.com/ShangtongZhang/DeepRL.
import logging
import time
from typing import Dict, Optional, Tuple, Sequence, List, Iterator

import numpy as np
import torch
from torch.optim.optimizer import Optimizer

from molgym.agents.base import AbstractActorCritic
from molgym.buffer import DynamicPPOBuffer
from molgym.buffer_container import PPOBufferContainer
from molgym.env_container import VecEnv
from molgym.tools.model_util import ModelIO
from molgym.tools.util import RolloutSaver, to_numpy, InfoSaver, compute_gradient_norm


def compute_loss(
    ac: AbstractActorCritic,
    data: dict,
    clip_ratio: float,
    vf_coef: float,
    entropy_coef: float,
    device=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred = ac.step(data['obs'], data['act'])

    old_logp = torch.as_tensor(data['logp'], device=device)
    adv = torch.as_tensor(data['adv'], device=device)
    ret = torch.as_tensor(data['ret'], device=device)

    # Policy loss
    ratio = torch.exp(pred['logp'] - old_logp)
    obj = ratio * adv
    clipped_obj = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
    policy_loss = -torch.min(obj, clipped_obj).mean()

    # Entropy loss
    entropy_loss = -entropy_coef * pred['ent'].mean()

    # Value loss
    vf_loss = vf_coef * (pred['v'] - ret).pow(2).mean()

    # Total loss
    loss = policy_loss + entropy_loss + vf_loss

    # Approximate KL for early stopping
    approx_kl = (old_logp - pred['logp']).mean()

    # Extra info
    clipped = ratio.lt(1 - clip_ratio) | ratio.gt(1 + clip_ratio)
    clip_fraction = torch.as_tensor(clipped, dtype=torch.float32).mean()

    info = dict(
        policy_loss=to_numpy(policy_loss).item(),
        entropy_loss=to_numpy(entropy_loss).item(),
        vf_loss=to_numpy(vf_loss).item(),
        total_loss=to_numpy(loss).item(),
        approx_kl=to_numpy(approx_kl).item(),
        clip_fraction=to_numpy(clip_fraction).item(),
    )

    return loss, info


def get_batch_generator(indices: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    assert len(indices.shape) == 1
    indices = np.random.permutation(indices)
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    remainder = len(indices) % batch_size
    if remainder:
        yield indices[-remainder:]


def collect_data_batch(data: Dict[str, Sequence], indices: np.ndarray) -> Dict[str, Sequence]:
    batch: Dict[str, Sequence] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            batch[key] = value[indices]
        elif isinstance(value, list):
            items = []
            for index in indices:
                items.append(value[index])
            batch[key] = items
        else:
            ValueError(value)
    return batch


def compute_mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    # Assert all dicts have the same keys
    assert (d.keys() == dicts[0].keys() for d in dicts)
    return {key: np.mean([d[key] for d in dicts]) for key in dicts[0].keys()}


# Train policy with multiple steps of gradient descent
def train(
    ac: AbstractActorCritic,
    optimizer: Optimizer,
    data: Dict[str, Sequence],
    mini_batch_size: int,
    clip_ratio: float,
    target_kl: float,
    vf_coef: float,
    entropy_coef: float,
    gradient_clip: float,
    max_num_steps: int,
    device=None,
) -> dict:
    infos = {}

    start_time = time.time()

    num_epochs = 0
    for i in range(max_num_steps):
        optimizer.zero_grad()

        batch_infos = []
        batch_generator = get_batch_generator(indices=np.arange(len(data['obs'])), batch_size=mini_batch_size)
        for batch_indices in batch_generator:
            data_batch = collect_data_batch(data, indices=batch_indices)
            batch_loss, batch_info = compute_loss(ac,
                                                  data=data_batch,
                                                  clip_ratio=clip_ratio,
                                                  vf_coef=vf_coef,
                                                  entropy_coef=entropy_coef,
                                                  device=device)

            batch_loss.backward(retain_graph=False)  # type: ignore
            batch_infos.append(batch_info)

        loss_info = compute_mean_dict(batch_infos)
        loss_info['grad_norm'] = compute_gradient_norm(ac.parameters())

        # Check KL
        if loss_info['approx_kl'] > 1.5 * target_kl:
            logging.debug(f'Early stopping at step {i} for reaching max KL.')
            break

        # Take gradient step
        logging.debug('Taking gradient step')
        torch.nn.utils.clip_grad_norm_(ac.parameters(), max_norm=gradient_clip)
        optimizer.step()
        optimizer.zero_grad()

        num_epochs += 1

        # Logging
        logging.debug(f'Loss {i}: {loss_info}')
        infos.update(loss_info)

    infos['num_opt_steps'] = num_epochs
    infos['time'] = time.time() - start_time

    if num_epochs > 0:
        logging.info(f'Optimization: policy loss={infos["policy_loss"]:.3f}, vf loss={infos["vf_loss"]:.3f}, '
                     f'entropy loss={infos["entropy_loss"]:.3f}, total loss={infos["total_loss"]:.3f}, '
                     f'num steps={num_epochs}')
    return infos


def batch_rollout(ac: AbstractActorCritic,
                  envs: VecEnv,
                  buffer_container: PPOBufferContainer,
                  num_steps: int = None,
                  num_episodes: int = None) -> dict:
    assert num_steps is not None or num_episodes is not None

    if num_steps is not None:
        assert num_steps % envs.get_size() == 0
        num_iters = num_steps // envs.get_size()
    else:
        num_iters = np.inf

    if num_episodes is not None:
        assert envs.get_size() == 1
    else:
        num_episodes = np.inf

    start_time = time.time()

    counter = 0
    observations = envs.reset()

    while counter < num_iters and buffer_container.get_num_episodes() < num_episodes:
        predictions = ac.step(observations)

        next_observations, rewards, terminals, _ = envs.step(predictions['actions'])

        buffer_container.store(observations=observations,
                               actions=to_numpy(predictions['a']),
                               rewards=rewards,
                               next_observations=next_observations,
                               terminals=terminals,
                               values=to_numpy(predictions['v']),
                               logps=to_numpy(predictions['logp']))

        # Reset environment if state is terminal to get valid next observation
        observations = envs.reset_if_terminal(next_observations, terminals)

        if counter == num_iters - 1:
            # Note: finished trajectories will not be affected by this
            predictions = ac.step(observations)
            buffer_container.finish_paths(to_numpy(predictions['v']))

        counter += 1

    info = {
        'time': time.time() - start_time,
        'return_mean': np.mean(buffer_container.episodic_returns).item(),
        'return_std': np.std(buffer_container.episodic_returns).item(),
        'episode_length_mean': np.mean(buffer_container.episode_lengths).item(),
        'episode_length_std': np.std(buffer_container.episode_lengths).item(),
    }

    return info


def compute_buffer_stats(buffer: DynamicPPOBuffer) -> Dict[str, float]:
    return {
        'value_mean': np.mean(buffer.val_buf).item(),
        'value_std': np.std(buffer.val_buf).item(),
        'logp_mean': np.mean(buffer.logp_buf).item(),
        'logp_std': np.std(buffer.logp_buf).item(),
    }


def batch_ppo(
    envs: VecEnv,
    eval_envs: VecEnv,
    ac: AbstractActorCritic,
    optimizer: Optimizer,
    gamma=0.99,
    start_num_steps=0,
    max_num_steps=4096,
    num_steps_per_iter=200,
    mini_batch_size=64,
    clip_ratio=0.2,
    vf_coef=0.5,
    entropy_coef=0.0,
    max_num_train_iters=80,
    lam=0.97,
    target_kl=0.01,
    gradient_clip=0.5,
    save_freq=5,
    model_handler: Optional[ModelIO] = None,
    eval_freq=10,
    num_eval_episodes=1,
    rollout_saver: Optional[RolloutSaver] = None,
    save_train_rollout=False,
    save_eval_rollout=True,
    info_saver: Optional[InfoSaver] = None,
    device=None,
):
    """
    Proximal Policy Optimization (by clipping), with early stopping based on approximate KL

    Args:
        :param envs: VecEnv for training.
        :param eval_envs: VecEnv for evaluation.
        :param ac: Instance of an AbstractActorCritic
        :param optimizer: Optimizer to optimize agent's parameters
        :param num_steps_per_iter: Number of agent-environment interaction steps per iteration.
        :param start_num_steps: Initial number of steps
        :param max_num_steps: Maximum number of steps
        :param mini_batch_size: mini batch size for loss calculation
        :param gamma: Discount factor. (Always between 0 and 1.)
        :param clip_ratio: Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.)
        :param vf_coef: coefficient for value function loss term
        :param entropy_coef: coefficient for entropy loss term
        :param gradient_clip: clip norm of gradients before optimization step is taken
        :param max_num_train_iters: Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer to take fewer than this.)
        :param lam: Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)
        :param target_kl: Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)
        :param eval_freq: How often to evaluate the policy
        :param num_eval_episodes: Number of evaluation episodes
        :param model_handler: Save model to file
        :param save_freq: How often the model is saved
        :param rollout_saver: Saves rollout buffers
        :param save_train_rollout: Save training rollout
        :param save_eval_rollout: Save evaluation rollout
        :param info_saver: Save statistics
        :param device: device on which to run the calculations
    """

    # Total number of steps
    total_num_steps = start_num_steps
    num_iterations = (max_num_steps - total_num_steps) // num_steps_per_iter

    logging.info('Starting PPO')

    # Main loop
    for iteration in range(num_iterations):
        logging.info(f'Iteration: {iteration}/{num_iterations - 1}, steps: {total_num_steps}')

        # Training rollout
        train_container = PPOBufferContainer(size=envs.get_size(), gamma=gamma, lam=lam)
        train_rollout = batch_rollout(ac=ac, envs=envs, buffer_container=train_container, num_steps=num_steps_per_iter)
        logging.info(
            f'Training rollout: return={train_rollout["return_mean"]:.3f} ({train_rollout["return_std"]:.1f}), '
            f'episode length={train_rollout["episode_length_mean"]:.1f}')

        train_buffer = train_container.merge()

        if info_saver:
            train_rollout['total_num_steps'] = total_num_steps
            train_rollout.update(compute_buffer_stats(train_buffer))
            info_saver.save(train_rollout, name='train')

        # Save training buffer
        if rollout_saver and save_train_rollout:
            rollout_saver.save(train_buffer, num_steps=total_num_steps, info='train')

        # Obtain (standardized) data for training
        data = train_buffer.get_data()

        # Train policy
        opt_info = train(
            ac=ac,
            optimizer=optimizer,
            data=data,
            mini_batch_size=mini_batch_size,
            clip_ratio=clip_ratio,
            vf_coef=vf_coef,
            entropy_coef=entropy_coef,
            target_kl=target_kl,
            gradient_clip=gradient_clip,
            max_num_steps=max_num_train_iters,
            device=device,
        )

        if info_saver:
            opt_info['total_num_steps'] = total_num_steps
            info_saver.save(opt_info, name='opt')

        # Update number of steps taken / trained
        total_num_steps += num_steps_per_iter

        # Evaluate policy
        if (iteration % eval_freq == 0) or (iteration == num_iterations - 1):
            eval_container = PPOBufferContainer(size=eval_envs.get_size(), gamma=gamma, lam=lam)

            with torch.no_grad():
                ac.training = False
                eval_rollout = batch_rollout(ac,
                                             eval_envs,
                                             buffer_container=eval_container,
                                             num_episodes=num_eval_episodes)
                logging.info(
                    f'Evaluation rollout: return={eval_rollout["return_mean"]:.3f} ({eval_rollout["return_std"]:.1f}), '
                    f'episode length={eval_rollout["episode_length_mean"]:.1f}')
                ac.training = True

            eval_buffer = eval_container.merge()

            # Log information
            if info_saver:
                eval_rollout['total_num_steps'] = total_num_steps
                eval_rollout.update(compute_buffer_stats(eval_buffer))
                info_saver.save(eval_rollout, name='eval')

            # Safe evaluation buffer
            if rollout_saver and save_eval_rollout:
                rollout_saver.save(eval_buffer, num_steps=total_num_steps, info='eval')

        # Save model
        if model_handler and ((iteration % save_freq == 0) or (iteration == num_iterations - 1)):
            model_handler.save(ac, num_steps=total_num_steps)

    logging.info('Finished PPO')
