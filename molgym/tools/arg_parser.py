import argparse


def build_default_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Command line tool of MolGym')

    # Name and seed
    parser.add_argument('--name', help='experiment name', required=True)
    parser.add_argument('--seed', help='run ID', type=int, default=0)

    # Directories
    parser.add_argument('--log_dir', help='directory for log files', type=str, default='logs')
    parser.add_argument('--model_dir', help='directory for model files', type=str, default='models')
    parser.add_argument('--data_dir', help='directory for saved rollouts', type=str, default='data')
    parser.add_argument('--results_dir', help='directory for results', type=str, default='results')

    # Device
    parser.add_argument('--device', help='select device', type=str, choices=['cpu', 'cuda'], default='cpu')

    # Spaces
    parser.add_argument('--canvas_size',
                        help='maximum number of atoms that can be placed on the canvas',
                        type=int,
                        default=25)
    parser.add_argument('--symbols',
                        help='chemical symbols available on canvas and in bag (comma separated)',
                        type=str,
                        default='X,H,C,N,O,F')

    # Environment
    parser.add_argument('--formulas',
                        help='list of formulas for environment (comma separated)',
                        type=str,
                        required=True)
    parser.add_argument('--eval_formulas',
                        help='list of formulas for environment (comma separated) used for evaluation',
                        type=str,
                        required=False)
    parser.add_argument('--bag_scale', help='maximum bag size', type=int, required=True)
    parser.add_argument('--min_atomic_distance', help='minimum allowed atomic distance', type=float, default=0.6)
    parser.add_argument('--max_solo_distance',
                        help='maximum distance hydrogen or halogens can be away from the nearest heavy atom',
                        type=float,
                        default=2.0)
    parser.add_argument('--min_reward', help='minimum reward given by environment', type=float, default=-0.6)

    # Model
    parser.add_argument('--model',
                        help='model representation',
                        type=str,
                        default='internal',
                        choices=['internal', 'covariant'])
    parser.add_argument('--min_mean_distance', help='minimum mean distance', type=float, default=0.8)
    parser.add_argument('--max_mean_distance', help='maximum mean distance', type=float, default=1.8)
    parser.add_argument('--network_width', help='width of FC layers', type=int, default=128)
    parser.add_argument('--maxl', help='maximum L in spherical harmonics expansion', type=int, default=4)
    parser.add_argument('--num_cg_levels', help='number of CG layers', type=int, default=3)
    parser.add_argument('--num_channels_hidden', help='number of channels in hidden layers', type=int, default=10)
    parser.add_argument('--num_channels_per_element', help='number of channels per element', type=int, default=4)
    parser.add_argument('--num_gaussians', help='number of Gaussians in GMM', type=int, default=3)
    parser.add_argument('--beta', help='set beta parameter of spherical distribution', required=False, default=None)

    parser.add_argument('--load_latest', help='load latest checkpoint file', action='store_true', default=False)
    parser.add_argument('--load_model', help='load checkpoint file', type=str, default=None)
    parser.add_argument('--save_freq', help='save model every <n> iterations', type=int, default=10)
    parser.add_argument('--eval_freq', help='evaluate model every <n> iterations', type=int, default=10)
    parser.add_argument('--num_eval_episodes', help='number of episodes per evaluation', type=int, default=None)

    # Training algorithm
    parser.add_argument('--optimizer',
                        help='Optimizer for parameter optimization',
                        type=str,
                        default='adam',
                        choices=['adam', 'amsgrad'])
    parser.add_argument('--discount', help='discount factor', type=float, default=1.0)
    parser.add_argument('--num_steps', dest='max_num_steps', help='maximum number of steps', type=int, default=50000)
    parser.add_argument('--num_steps_per_iter',
                        help='number of optimization steps per iteration',
                        type=int,
                        default=128)
    parser.add_argument('--mini_batch_size', help='mini batch size for training', type=int, default=64)
    parser.add_argument('--num_envs', help='number of environment copies', type=int, default=8)
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
    parser.add_argument('--keep_models', help='keep all models', action='store_true', default=False)
    parser.add_argument('--save_rollouts',
                        help='which rollouts to save',
                        type=str,
                        default='none',
                        choices=['none', 'train', 'eval', 'all'])

    return parser
