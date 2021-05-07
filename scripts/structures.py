import argparse
import os
import pickle
from typing import Tuple

import ase.data
import ase.io

from molgym.spaces import CanvasSpace
from molgym.tools.analysis import parse_buffer_filename, collect_buffer_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyse MolGym output')

    parser.add_argument('--dir', help='path to data directory of experiment(s)', required=True)
    parser.add_argument('--symbols',
                        help='symbols representing elements on canvas (comma separated)',
                        type=str,
                        required=True)
    parser.add_argument('--canvas_size',
                        help='maximum number of atoms that can be placed on the canvas',
                        type=int,
                        default=128)
    parser.add_argument('--mode', help='select from train or eval mode', default='eval', choices=['train', 'eval'])
    parser.add_argument('--name', help='name of experiment')

    return parser.parse_args()


def path_to_sort_key(path: str) -> Tuple:
    parsed_path = parse_buffer_filename(os.path.basename(path))
    return parsed_path['steps'], parsed_path['mode'], parsed_path['name'], parsed_path['seed'], parsed_path['rank']


def main() -> None:
    args = parse_args()

    paths = collect_buffer_paths(args.dir, mode=args.mode)
    print(f'Parsed paths: {len(paths)}')

    canvas_space = CanvasSpace(size=args.canvas_size, zs=[ase.data.atomic_numbers[s] for s in args.symbols.split(',')])

    # Sort paths
    paths = sorted(paths, key=path_to_sort_key)

    atoms_list = []
    for path in paths:
        info = parse_buffer_filename(os.path.basename(path))

        if args.name and info['name'] != args.name:
            continue

        with open(path, mode='rb') as f:
            buffer = pickle.load(f)

        atoms = [
            canvas_space.to_atoms(obs[0]) for terminal, obs in zip(buffer.term_buf, buffer.next_obs_buf)
            if obs and terminal
        ]
        for atom in atoms:
            atom.info = info

        atoms_list += atoms

    if args.name:
        filename = f'structures_{args.name}_{args.mode}.xyz'
    else:
        filename = f'structures_{args.mode}.xyz'

    ase.io.write(filename, images=atoms_list)


if __name__ == '__main__':
    main()
