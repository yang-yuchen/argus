"""Summarize causal-attention shapes using the paper's colors and normalization."""
import argparse
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_from_csv as paper

LENGTHS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
STAGES = ('decode', 'mtp2', 'mtp4', 'mtp8', 'mtp16', 'mtp32', 'mtp64', 'prefill')
METHODS = ('Ours', 'FlexAttn', 'TileLang')
POSITIONS = tuple(range(len(STAGES)))


def summarize(data):
    if set(data['stage']) != set(STAGES):
        raise ValueError('Require decode, six MTP stages and causal prefill')
    for stage, group in data.groupby('stage'):
        if len(group) != 8 or sorted(group['kv_length']) != list(LENGTHS):
            raise ValueError(f'{stage}: require each common KV length once')
        heads = 4 if stage.startswith('mtp') else 32
        q = group['kv_length'] if stage == 'prefill' else (
            1 if stage == 'decode' else int(stage[3:]))
        if not group['heads'].eq(heads).all() or not group['q_length'].eq(q).all():
            raise ValueError(f'{stage}: unexpected Q/head shape')
    values = data[list(METHODS)]
    if not np.isfinite(values).all().all() or not values.gt(0).all().all():
        raise ValueError('Require finite positive measured latencies')
    latencies = values.copy()
    latencies.insert(0, 'Operator', 'MTP_CAUSAL')
    latencies.insert(1, 'SeqLen', data['kv_length'])
    normalized = paper.normalize_by_baseline(latencies, 'Ours')
    normalized['stage'] = data['stage']
    normalized['kv_length'] = data['kv_length']
    rows = []
    for stage in STAGES:
        group = normalized[normalized['stage'] == stage]
        for method in METHODS:
            rows.append(dict(stage=stage, method=method,
                             geomean=statistics.geometric_mean(group[method]),
                             min=group[method].min(), max=group[method].max()))
    return normalized, pd.DataFrame(rows)


def plot(data, output):
    normalized, summary = summarize(data)
    colors = paper.build_colormap([paper.METHOD_DISPLAY_NAMES[m] for m in METHODS])
    fig, ax = plt.subplots(figsize=(12, 4.6))
    offsets = np.linspace(-0.12, 0.12, len(LENGTHS))
    for method in METHODS:
        label = paper.METHOD_DISPLAY_NAMES[method]
        color = colors[label]
        means = summary[summary['method'] == method]['geomean'].to_numpy()
        ax.plot(POSITIONS, means, color=color, linewidth=2.4, label=label)
        if method != 'Ours':
            for stage, x in zip(STAGES, POSITIONS):
                group = normalized[normalized['stage'] == stage].sort_values('kv_length')
                ax.scatter(x + offsets, group[method], color=color, s=14, alpha=0.16,
                           edgecolors='none', zorder=2)
        ax.scatter(POSITIONS, means, color=color, s=52, zorder=4)
    ax.set_xticks(POSITIONS, ['1', '2', '4', '8', '16', '32', '64', 'KV'])
    ax.set_xlabel('Query tokens (Q)', fontsize=20, labelpad=46)
    ax.set_ylabel('Relative Performance', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.set_xlim(POSITIONS[0] - 0.42, POSITIONS[-1] + 0.42)
    ax.set_ylim(0, max(1.12, float(normalized[list(METHODS)].max().max()) * 1.08))
    ax.grid(True, linestyle=':', alpha=0.45)
    ax.set_axisbelow(True)
    for x, label in ((POSITIONS[0], 'Decode'),
                     ((POSITIONS[1] + POSITIONS[6]) / 2, 'Causal MTP'),
                     (POSITIONS[7], 'Causal Prefill')):
        ax.annotate(label, xy=(x, 0), xycoords=ax.get_xaxis_transform(),
                    xytext=(0, -39), textcoords='offset points',
                    ha='center', va='top', fontsize=13, annotation_clip=False)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=18,
               frameon=True)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'), bbox_inches='tight', dpi=180)
    summary.to_csv(output.with_suffix('.csv'), index=False)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(pd.read_csv(args.input), args.output)
