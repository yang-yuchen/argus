"""Validate and summarize the sparse compiler sweep for plot_from_csv.py."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

LENGTHS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
METHODS = {'tilelang': 'TileLang', 'flex': 'FlexAttn', 'helion': 'Helion'}


def validated(path, *, inputs_hash=None):
    data = json.loads(path.read_text())
    if not data.get('correct') or not data.get('measurement_complete'):
        raise ValueError(f'Invalid/incomplete result: {path}')
    samples = data.get('samples_us', [])
    if data['repeats'] != 50 or len(samples) != 50:
        raise ValueError(f'Expected 50 measurements: {path}')
    if not all(math.isfinite(x) and x > 0 for x in samples):
        raise ValueError(f'Nonfinite latency: {path}')
    if not math.isclose(data['mean_us'], statistics.mean(samples), rel_tol=1e-12):
        raise ValueError(f'Inconsistent mean: {path}')
    if inputs_hash is not None and data.get('inputs_sha256') != inputs_hash:
        raise ValueError(f'Inputs differ: {path}')
    return data


def summarize(directory):
    plots, records = [], []
    for length in LENGTHS:
        case = f'sparse_L{length}'
        nautilus = validated(directory / 'nautilus' / f'{case}.json')
        row = {'Operator': 'SPARSE', 'SeqLen': length, 'Ours': nautilus['mean_us']}
        for method, label in METHODS.items():
            path = directory / method / f'{case}.json'
            record = {'case': case, 'method': label, 'status': 'unavailable', 'mean_us': '',
                      'nautilus_us': nautilus['mean_us'], 'relative_performance': ''}
            row[label] = ''
            if path.exists():
                data = json.loads(path.read_text())
                if data.get('measurement_complete'):
                    data = validated(path, inputs_hash=nautilus['inputs_sha256'])
                    if method == 'flex' and (data['source'].get('flex_search_space') != 'DEFAULT'
                                             or data['source'].get('kernel_options',{}).get('BLOCK_M') != 32
                                             or data['source'].get('kernel_options',{}).get('BLOCK_N') != 64
                                             or data['source'].get('kernel_options',{}).get('num_warps') != 8):
                        raise ValueError(f'Flex result needs uniform-search replay: {path}')
                    if method == 'tilelang' and not isinstance(data['source'].get('autotune'), dict):
                        raise ValueError(f'TileLang result needs calibrated replay: {path}')
                    if method == 'helion' and data['source'].get('configuration',{}).get('autotune_budget_seconds') != 180:
                        raise ValueError(f'Helion result needs uniform-budget replay: {path}')
                    if method in ('flex', 'helion') and data['source'].get('shared_mla_kv') is not True:
                        raise ValueError(f'{label} result needs shared-KV replay: {path}')
                    row[label] = data['mean_us']
                    record.update(status='passed', mean_us=data['mean_us'],
                                  relative_performance=nautilus['mean_us']/data['mean_us'])
            records.append(record)
        plots.append(row)
    present = {'Ours'} | {r['method'] for r in records if r['status'] == 'passed'}
    fields = ['Operator', 'SeqLen', 'Ours'] + [x for x in METHODS.values() if x in present]
    with (directory / 'plot_input.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(plots)
    with (directory / 'comparison.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return plots, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    plots, records = summarize(args.directory)
    print(json.dumps({'lengths': len(plots), 'passed': sum(r['status']=='passed' for r in records),
                      'unavailable': sum(r['status']!='passed' for r in records)}, indent=2))


if __name__ == '__main__':
    main()
