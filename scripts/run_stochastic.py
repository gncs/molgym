import logging

import ase.data
import ase.io

from molgym.env_container import SimpleEnvContainer
from molgym.environment import StochasticEnvironment, MolecularEnvironment
from molgym.ppo import batch_ppo
from molgym.reward import InteractionReward
from molgym.spaces import ActionSpace, ObservationSpace
from molgym.tools import util
from molgym.tools.arg_parser import build_default_argparser
from molgym.tools.model_util import ModelIO, build_model


def get_config() -> dict:
    parser = build_default_argparser()
    parser.add_argument('--size_range', help='minimum and maximum bag size (comma-separated)', type=str, required=True)
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

    train_formula = util.split_formula_strings(config['formulas'])[0]
    eval_formulas = util.split_formula_strings(config['eval_formulas'])
    size_range = util.parse_size_range(config['size_range'])

    logging.info(f'Statistical training bag: {train_formula}, size range: {size_range}')
    logging.info(f'Evaluation bag(s): {eval_formulas}')

    model_handler = ModelIO(directory=config['model_dir'], tag=tag, keep=config['keep_models'])

    start_num_steps = 0
    if not config['load_latest']:
        model = build_model(config, observation_space=observation_space, action_space=action_space, device=device)
    else:
        model, start_num_steps = model_handler.load_latest(device=device)
        model.action_space = action_space
        model.observation_space = observation_space

    var_counts = util.count_vars(model)
    logging.info(f'Number of parameters: {var_counts}')

    reward = InteractionReward()

    # Number of episodes during evaluation
    if not config['num_eval_episodes']:
        config['num_eval_episodes'] = len(eval_formulas)

    training_envs = SimpleEnvContainer([
        StochasticEnvironment(
            reward=reward,
            observation_space=observation_space,
            action_space=action_space,
            formula=util.string_to_formula(train_formula),
            size_range=size_range,
            min_atomic_distance=config['min_atomic_distance'],
            max_solo_distance=config['max_solo_distance'],
            min_reward=config['min_reward'],
        ) for _ in range(config['num_envs'])
    ])

    eval_envs = SimpleEnvContainer([
        MolecularEnvironment(
            reward=reward,
            observation_space=observation_space,
            action_space=action_space,
            formulas=[util.string_to_formula(formula) for formula in eval_formulas],
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
