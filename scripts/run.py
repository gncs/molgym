import argparse
import logging

import torch

from molgym.agents.base import AbstractActorCritic
from molgym.agents.internal import InternalAC
from molgym.environment import MolecularEnvironment
from molgym.ppo import ppo
from molgym.reward import InteractionReward
from molgym.spaces import ActionSpace, ObservationSpace
from molgym.tools import mpi, util
from molgym.tools.util import RolloutSaver, InfoSaver, parse_formulas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Agent in MolGym')

    # Name and seed
    parser.add_argument('--name', help='experiment name', required=True)
    parser.add_argument('--seed', help='run ID', type=int, default=0)

    # Directories
    parser.add_argument('--log_dir', help='directory for log files', type=str, default='logs')
    parser.add_argument('--model_dir', help='directory for model files', type=str, default='models')
    parser.add_argument('--data_dir', help='directory for saved rollouts', type=str, default='data')
    parser.add_argument('--results_dir', help='directory for results', type=str, default='results')

    # Spaces
    parser.add_argument('--canvas_size',
                        help='maximum number of atoms that can be placed on the canvas',
                        type=int,
                        default=25)
    parser.add_argument('--bag_symbols',
                        help='symbols representing elements in bag (comma separated)',
                        type=str,
                        default='H,He,Li,Be,B,C,N,O,F')

    # Environment
    parser.add_argument('--formulas',
                        help='list of formulas for environment (comma separated)',
                        type=str,
                        required=True)
    parser.add_argument('--eval_formulas',
                        help='list of formulas for environment (comma separated) used for evaluation',
                        type=str,
                        required=False)
    parser.add_argument('--min_atomic_distance', help='minimum allowed atomic distance', type=float, default=0.6)
    parser.add_argument('--max_h_distance',
                        help='maximum distance a H atom can be away from the nearest heavy atom',
                        type=float,
                        default=2.0)
    parser.add_argument('--min_reward', help='minimum reward given by environment', type=float, default=-0.6)

    # Model
    parser.add_argument('--min_mean_distance', help='minimum mean distance', type=float, default=0.8)
    parser.add_argument('--max_mean_distance', help='maximum mean distance', type=float, default=1.8)
    parser.add_argument('--network_width', help='width of FC layers', type=int, default=128)

    parser.add_argument('--load_model', help='load latest checkpoint file', action='store_true', default=False)
    parser.add_argument('--save_freq', help='save model every <n> iterations', type=int, default=5)
    parser.add_argument('--eval_freq', help='evaluate model every <n> iterations', type=int, default=5)
    parser.add_argument('--num_eval_episodes', help='number of episodes per evaluation', type=int, default=None)

    # Training algorithm
    parser.add_argument('--discount', help='discount factor', type=float, default=1.0)
    parser.add_argument('--num_steps', dest='max_num_steps', help='maximum number of steps', type=int, default=50000)
    parser.add_argument('--num_steps_per_iter',
                        help='number of optimization steps per iteration',
                        type=int,
                        default=128)
    parser.add_argument('--clip_ratio', help='PPO clip ratio', type=float, default=0.2)
    parser.add_argument('--learning_rate', help='Learning rate of Adam optimizer', type=float, default=3e-4)
    parser.add_argument('--vf_coef', help='Coefficient for value function loss', type=float, default=0.5)
    parser.add_argument('--entropy_coef', help='Coefficient for entropy loss', type=float, default=0.01)
    parser.add_argument('--max_num_train_iters', help='Maximum number of training iterations', type=int, default=7)
    parser.add_argument('--gradient_clip', help='maximum norm of gradients', type=float, default=0.5)
    parser.add_argument('--lam', help='Lambda for GAE-Lambda', type=float, default=0.97)
    parser.add_argument('--target_kl',
                        help='KL divergence between new and old policies after an update for early stopping',
                        type=float,
                        default=0.01)

    # Logging
    parser.add_argument('--log_level', help='log level', type=str, default='INFO')
    parser.add_argument('--save_rollouts',
                        help='which rollouts to save',
                        type=str,
                        default='none',
                        choices=['none', 'train', 'eval', 'all'])
    parser.add_argument('--all_ranks', help='print log of all ranks', action='store_true', default=False)

    return parser.parse_args()


def get_config() -> dict:
    config = vars(parse_args())

    config['num_procs'] = mpi.get_num_procs()

    return config


def build_model(config: dict, observation_space: ObservationSpace, action_space: ActionSpace) -> AbstractActorCritic:
    return InternalAC(
        observation_space=observation_space,
        action_space=action_space,
        min_max_distance=(config['min_mean_distance'], config['max_mean_distance']),
        network_width=config['network_width'],
        device=torch.device('cpu'),
    )


def main() -> None:
    util.set_one_thread()

    config = get_config()

    util.create_directories([config['log_dir'], config['model_dir'], config['data_dir'], config['results_dir']])

    tag = util.get_tag(config)
    util.setup_logger(config, directory=config['log_dir'], tag=tag)
    util.save_config(config, directory=config['log_dir'], tag=tag)

    util.set_seeds(seed=config['seed'] + mpi.get_proc_rank())

    model_handler = util.ModelIO(directory=config['model_dir'], tag=tag)

    bag_symbols = config['bag_symbols'].split(',')
    action_space = ActionSpace()
    observation_space = ObservationSpace(canvas_size=config['canvas_size'], symbols=bag_symbols)

    start_num_steps = 0
    if not config['load_model']:
        model = build_model(config, observation_space=observation_space, action_space=action_space)
    else:
        model, start_num_steps = model_handler.load()
        model.action_space = action_space
        model.observation_space = observation_space

    mpi.sync_params(model)

    var_counts = util.count_vars(model)
    logging.info(f'Number of parameters: {var_counts}')

    reward = InteractionReward()

    # Evaluation formulas
    if not config['eval_formulas']:
        config['eval_formulas'] = config['formulas']

    train_formulas = parse_formulas(config['formulas'])
    eval_formulas = parse_formulas(config['eval_formulas'])

    logging.info(f'Training bags: {train_formulas}')
    logging.info(f'Evaluation bags: {eval_formulas}')

    # Number of episodes during evaluation
    if not config['num_eval_episodes']:
        config['num_eval_episodes'] = len(eval_formulas)

    env = MolecularEnvironment(
        reward=reward,
        observation_space=observation_space,
        action_space=action_space,
        formulas=train_formulas,
        min_atomic_distance=config['min_atomic_distance'],
        max_h_distance=config['max_h_distance'],
        min_reward=config['min_reward'],
    )

    eval_env = MolecularEnvironment(
        reward=reward,
        observation_space=observation_space,
        action_space=action_space,
        formulas=eval_formulas,
        min_atomic_distance=config['min_atomic_distance'],
        max_h_distance=config['max_h_distance'],
        min_reward=config['min_reward'],
    )

    rollout_saver = RolloutSaver(directory=config['data_dir'], tag=tag, all_ranks=config['all_ranks'])
    info_saver = InfoSaver(directory=config['results_dir'], tag=tag)

    ppo(
        env=env,
        eval_env=eval_env,
        ac=model,
        gamma=config['discount'],
        start_num_steps=start_num_steps,
        max_num_steps=config['max_num_steps'],
        num_steps_per_iter=config['num_steps_per_iter'],
        clip_ratio=config['clip_ratio'],
        learning_rate=config['learning_rate'],
        vf_coef=config['vf_coef'],
        entropy_coef=config['entropy_coef'],
        max_num_train_iters=config['max_num_train_iters'],
        lam=config['lam'],
        target_kl=config['target_kl'],
        gradient_clip=config['gradient_clip'],
        eval_freq=config['eval_freq'],
        model_handler=model_handler,
        save_freq=config['save_freq'],
        num_eval_episodes=config['num_eval_episodes'],
        rollout_saver=rollout_saver,
        save_train_rollout=config['save_rollouts'] == 'train' or config['save_rollouts'] == 'all',
        save_eval_rollout=config['save_rollouts'] == 'eval' or config['save_rollouts'] == 'all',
        info_saver=info_saver,
    )


if __name__ == '__main__':
    main()
