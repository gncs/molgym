import glob
import json
import os
import re
from typing import List


def parse_json_lines_file(path: str) -> List[dict]:
    dicts = []
    with open(path, mode='r') as f:
        for line in f:
            dicts.append(json.loads(line))
    return dicts


def parse_buffer_filename(filename: str) -> dict:
    regex = re.compile(fr'(?P<name>.*?)_run-(?P<seed>\d+)_steps-(?P<steps>\d+)_rank-(?P<rank>\d+)_(?P<mode>.*?)\.pkl')
    match = regex.match(filename)
    if not match:
        raise RuntimeError(f'Cannot parse filename: {filename}')
    return {
        'name': match.group('name'),
        'seed': int(match.group('seed')),
        'steps': int(match.group('steps')),
        'rank': int(match.group('rank')),
        'mode': match.group('mode'),
    }


def parse_results_filename(filename: str) -> dict:
    regex = re.compile(fr'(?P<name>.*?)_run-(?P<seed>\d+)_(?P<mode>.*?)\.txt')
    match = regex.match(filename)
    if not match:
        raise RuntimeError(f'Cannot parse filename: {filename}')
    return {
        'name': match.group('name'),
        'seed': int(match.group('seed')),
        'mode': match.group('mode'),
    }


def collect_results_paths(directory: str, mode: str) -> List[str]:
    return glob.glob(os.path.join(directory, f'*_run-*_{mode}.txt'))


def collect_buffer_paths(directory: str, mode: str) -> List[str]:
    return glob.glob(os.path.join(directory, f'*_{mode}.pkl'))
