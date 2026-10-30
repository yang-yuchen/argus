"""Redraw existing E2E experiments with the paper's original throughput bars."""

import argparse
import importlib.util
from pathlib import Path
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.e2e_model_flops import model_flops

SOURCE = ROOT / 'paper_results/prefill_final_repro/scripts/csv_figures.py'
METHODS = {'nautilus': 'neptune_tawa', 'tileops': 'tilelang',
           'tileops_bound': 'tilelang_prebound', 'flex': 'flex_attention'}
ORDER = ('neptune_tawa', 'tilelang', 'tilelang_prebound', 'flex_attention')


def figure_title(query_length):
    operator = 'Decode' if query_length == 1 else 'Causal MTP'
    return f'Llama-7B · FP16 · B1/Q{query_length} · {operator} · GH200'


def load_data(path, query_length, kv_length=None):
    data = pd.read_csv(path)
    if 'seq_len' in data and 'neptune_tawa' in data:
        panel = data.copy()
    else:
        data = data.copy()
        if 'kv_length' not in data:
            if kv_length is None:
                raise ValueError('A fixed KV length is required for this single-shape CSV')
            data['kv_length'] = kv_length
        if not set(data['implementation']).issubset(METHODS):
            raise ValueError('Unknown implementation; refusing to drop a measured method')
        if data.duplicated(['kv_length', 'implementation']).any():
            raise ValueError('Duplicate method/length measurements')
        if not np.isfinite(data['mean_ms']).all() or not data['mean_ms'].gt(0).all():
            raise ValueError('Require finite positive whole-model latency')
        data['throughput_tflops'] = [
            model_flops(int(length), query_length=query_length) / float(ms) / 1e9
            for length, ms in zip(data['kv_length'], data['mean_ms'])
        ]
        panel = data.pivot(index='kv_length', columns='implementation',
                           values='throughput_tflops').rename(columns=METHODS)
        # A lone prebound route is the TileLang baseline; expose both when both were measured.
        if 'tilelang_prebound' in panel and 'tilelang' not in panel:
            panel = panel.rename(columns={'tilelang_prebound': 'tilelang'})
        panel = panel.reset_index().rename(columns={'kv_length': 'seq_len'})
    columns = [name for name in ORDER if name in panel]
    if 'neptune_tawa' not in columns or len(columns) < 2:
        raise ValueError('Require Nautilus and at least one measured baseline')
    panel = panel[['seq_len', *columns]].sort_values('seq_len').reset_index(drop=True)
    values = panel[columns]
    if (panel['seq_len'].duplicated().any() or not np.isfinite(values).all().all()
            or not values.gt(0).all().all()):
        raise ValueError('Missing, duplicate or invalid throughput values')
    return panel


def plot(data, output, title):
    spec = importlib.util.spec_from_file_location('paper_throughput_style', SOURCE)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    original.COLORS['tilelang_prebound'] = '#b39bd4'
    original.LEGEND_LABELS['tilelang_prebound'] = 'TileLang prebound'
    methods = [name for name in ORDER if name in data]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    original.plot_subplot(ax, data, methods, title)
    ax.set_xlabel('Sequence Length', fontsize=18)
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.995),
               fontsize=14 if len(methods) > 3 else 16, frameon=True, ncol=len(methods))
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.18, top=0.79)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output.with_suffix('.csv'), index=False)
    for suffix in ('pdf', 'png'):
        fig.savefig(output.with_suffix('.' + suffix), bbox_inches='tight', dpi=180)
    return fig, ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--query-length', type=int, choices=(1, 16), required=True)
    parser.add_argument('--kv-length', type=int)
    parser.add_argument('--title')
    args = parser.parse_args()
    figure, _ = plot(load_data(args.input, args.query_length, args.kv_length),
                     args.output, args.title or figure_title(args.query_length))
    plt.close(figure)
