import logging

import ase.data
import ase.io

from molgym.env_container import SimpleEnvContainer
from molgym.environment import RefillableMolecularEnvironment
from molgym.ppo import batch_ppo
from molgym.reward import SolvationReward
from molgym.spaces import ActionSpace, ObservationSpace
from molgym.tools import util
from molgym.tools.arg_parser import build_default_argparser
from molgym.tools.model_util import ModelIO, build_model


def get_config() -> dict:
    parser = build_default_argparser()
    parser.add_argument('--num_refills',
                        help='number of times the bag gets refilled by the environment',
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument('--initial_structure', help='path to initial structure', type=str, required=False)
    args = parser.parse_args()
    config = vars(args)
    return config


def main() -> None:
    config = get_config()

    util.create_directories([config['log_dir'], config['model_dir'], config['data_dir'], config['results_dir']])

    tag = util.get_tag(config)
    util.setup_logger(config, directory=config['log_dir'], tag=tag)
    util.save_config(config, directory=config['log_dir'], tag=tag)

    util.set_seeds(seed=config['seed'])
    device = util.init_device(config['device'])

    zs = [ase.data.atomic_numbers[s] for s in config['symbols'].split(',')]
    action_space = ActionSpace(zs=zs)
    observation_space = ObservationSpace(canvas_size=config['canvas_size'], zs=zs)

    # Evaluation formulas
    if not config['eval_formulas']:
        config['eval_formulas'] = config['formulas']

    train_formulas = util.split_formula_strings(config['formulas'])
    eval_formulas = util.split_formula_strings(config['eval_formulas'])

    logging.info(f'Training bags: {train_formulas}')
    logging.info(f'Evaluation bags: {eval_formulas}')

    model_handler = ModelIO(directory=config['model_dir'], tag=tag, keep=config['keep_models'])

    if config['load_latest']:
        model, start_num_steps = model_handler.load_latest(device=device)
        model.action_space = action_space
        model.observation_space = observation_space
    elif config['load_model'] is not None:
        model, start_num_steps = model_handler.load(device=device, path=config['load_model'])
        model.action_space = action_space
        model.observation_space = observation_space
    else:
        model = build_model(config, observation_space=observation_space, action_space=action_space, device=device)
        start_num_steps = 0

    var_counts = util.count_vars(model)
    logging.info(f'Number of parameters: {var_counts}')

    reward = SolvationReward()

    # Number of episodes during evaluation
    if not config['num_eval_episodes']:
        config['num_eval_episodes'] = len(eval_formulas)

    if config['initial_structure']:
        initial_structure = ase.io.read(config['initial_structure'], index=0, format='xyz')
    else:
        initial_structure = ase.Atoms()

    training_envs = SimpleEnvContainer([
        RefillableMolecularEnvironment(
            reward=reward,
            observation_space=observation_space,
            action_space=action_space,
            formulas=[util.string_to_formula(f) for f in train_formulas],
            initial_structure=initial_structure,
            num_refills=config['num_refills'],
            min_atomic_distance=config['min_atomic_distance'],
            max_solo_distance=config['max_solo_distance'],
            min_reward=config['min_reward'],
        ) for _ in range(config['num_envs'])
    ])

    eval_envs = SimpleEnvContainer([
        RefillableMolecularEnvironment(
            reward=reward,
            observation_space=observation_space,
            action_space=action_space,
            formulas=[util.string_to_formula(f) for f in eval_formulas],
            initial_structure=initial_structure,
            num_refills=config['num_refills'],
            min_atomic_distance=config['min_atomic_distance'],
            max_solo_distance=config['max_solo_distance'],
            min_reward=config['min_reward'],
        )
    ])

    batch_ppo(
        envs=training_envs,
        eval_envs=eval_envs,
        ac=model,
        optimizer=util.get_optimizer(name=config['optimizer'],
                                     learning_rate=config['learning_rate'],
                                     parameters=model.parameters()),
        gamma=config['discount'],
        start_num_steps=start_num_steps,
        max_num_steps=config['max_num_steps'],
        num_steps_per_iter=config['num_steps_per_iter'],
        mini_batch_size=config['mini_batch_size'],
        clip_ratio=config['clip_ratio'],
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
        rollout_saver=util.RolloutSaver(directory=config['data_dir'], tag=tag),
        save_train_rollout=config['save_rollouts'] == 'train' or config['save_rollouts'] == 'all',
        save_eval_rollout=config['save_rollouts'] == 'eval' or config['save_rollouts'] == 'all',
        info_saver=util.InfoSaver(directory=config['results_dir'], tag=tag),
        device=device,
    )


if __name__ == '__main__':
    main()
