import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from molgym.tools.analysis import parse_json_lines_file, parse_results_filename, collect_results_paths

# Styling
fig_width = 0.45 * 5.50107
fig_height = 2.1

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 6})

colors = [
    '#1f77b4',  # muted blue
    '#d62728',  # brick red
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',  # blue-teal
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot MolGym output')

    parser.add_argument('--dir', help='path to results directory (repeatable)', required=True, action='append')
    parser.add_argument('--baseline', help='baseline (repeatable)', required=False, action='append')
    parser.add_argument('--max_num_steps', help='analyse up to maximum number of steps', required=False, type=int)
    parser.add_argument('--min_num_steps', help='analyse after minimum number of steps', required=False, type=int)
    parser.add_argument('--mode',
                        help='train or eval mode',
                        required=False,
                        type=str,
                        choices=['train', 'eval'],
                        default='eval')

    return parser.parse_args()


def get_data(directories: List[str], mode: str) -> pd.DataFrame:
    paths = []
    for directory in directories:
        paths += collect_results_paths(directory=directory, mode=mode)

    assert len(paths) > 0

    frames = []
    for path in paths:
        df = pd.DataFrame(parse_json_lines_file(path))

        info = parse_results_filename(os.path.basename(path))
        df['seed'] = info['seed']
        df['name'] = info['name']
        df['mode'] = info['mode']

        frames.append(df)

    data = pd.concat(frames)

    # Compute average and std over seeds
    data = data.groupby(['name', 'mode', 'total_num_steps']).agg([np.mean, np.std]).reset_index()

    return data


def main() -> None:
    args = parse_args()
    data = get_data(directories=args.dir, mode=args.mode)

    if args.max_num_steps:
        data = data[data['total_num_steps'] <= args.max_num_steps]

    if args.min_num_steps:
        data = data[data['total_num_steps'] >= args.min_num_steps]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height), constrained_layout=True)
    color_iter = iter(colors)

    prop = 'return_mean'
    for j, (name, group) in enumerate(data.groupby('name')):
        color = next(color_iter)

        if group[prop]['mean'].isna().all():
            continue
        ax.plot(
            group['total_num_steps'] / 1000,
            group[prop]['mean'],
            zorder=2 * j + 3,
            label=name,
            color=color,
        )
        ax.fill_between(
            x=group['total_num_steps'] / 1000,
            y1=group[prop]['mean'] - group[prop]['std'],
            y2=group[prop]['mean'] + group[prop]['std'],
            alpha=0.5,
            zorder=2 * j + 2,
            color=color,
        )

    color_iter = iter(colors)
    if args.baseline:
        for baseline in args.baseline:
            color = next(color_iter)
            ax.axhline(float(baseline), color=color, linestyle='dashed', zorder=1)

    ax.set_ylabel('Average Return')
    ax.set_xlabel('Steps x 1000')

    ax.legend(loc='lower right')

    fig.savefig('average_return.pdf')


if __name__ == '__main__':
    main()
