"""Validate the FP8 decode8 / mtpc8 sweep and reuse plot_from_csv.py for the PDFs.

Usage (repo root): python profiles_fp8_compilers_20261029/make_plots_fp8_sweep.py RUN_DIR
"""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project))
import plot_from_csv as plotting  # noqa: E402

run = Path(sys.argv[1]).resolve()
GROUPS = {
    'decode8': ('DC_CAUSAL', 'FP8 Causal Attention (Decode)',
                [f'decode8_L{L}' for L in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)]),
    'mtpc8': ('MTP_CAUSAL', 'FP8 Causal MTP (Q=16)',
              [f'mtpc8_Q16_L{L}' for L in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)]),
}


def load(folder, case):
    path = run / folder / f'{case}.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    samples = data.get('samples_us', [])
    if not (data.get('correct') and data.get('measurement_complete')
            and data.get('repeats') == len(samples) == 50
            and all(math.isfinite(x) and x > 0 for x in samples)):
        return None
    return data


records, plots = [], []
for name, (operator, title, cases) in GROUPS.items():
    rows = []
    for case in cases:
        ours, tile = load('nautilus', case), load('tilelang', case)
        if ours is None:
            print(f'missing Nautilus result: {case}')
            continue
        if tile is not None:
            assert tile['inputs_sha256'] == ours['inputs_sha256'], case
        kv = int(case.rsplit('_L', 1)[1])
        rows.append({'Operator': operator, 'SeqLen': kv, 'Ours': ours['mean_us'],
                     'TileLang': tile['mean_us'] if tile else ''})
        records.append({'case': case, 'kv_length': kv, 'Nautilus_us': ours['mean_us'],
                        'TileLang_us': tile['mean_us'] if tile else '',
                        'nautilus_speedup_vs_tilelang': tile['mean_us'] / ours['mean_us'] if tile else '',
                        'host': ours['host'], 'same_host': bool(tile) and tile['host'] == ours['host'],
                        'kernel_sha256': ours['kernel_sha256']})
    csv_path = run / f'plot_input_{name}.csv'
    with csv_path.open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=['Operator', 'SeqLen', 'Ours', 'TileLang'])
        writer.writeheader()
        writer.writerows(rows)
    normalized = plotting.normalize_by_baseline(plotting.load_data(csv_path), 'Ours')
    plotting.OPERATOR_DISPLAY_NAMES[operator] = title
    suffix = f'-fp8-tc1-{name}-4e06b84'
    plotting.plot_multi_operator_row(normalized, [operator], 'Ours', project / 'plots_from_csv', suffix=suffix)
    plots.append({'group': name, 'csv': str(csv_path.relative_to(project)),
                  'pdf': f'plots_from_csv/gh200{suffix}.pdf', 'lengths': [r['SeqLen'] for r in rows]})

with (run / 'comparison.csv').open('w', newline='') as fp:
    writer = csv.DictWriter(fp, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
(run / 'plot_manifest.json').write_text(json.dumps({
    'kernels': 'neptune-internal nautilus-gh200 4e06b84 (source_4e06b84/paper_kernels)',
    'tilelang_label': 'official TileOPs FP8 (pinned 2c54caec)',
    'normalization': 'Nautilus mean latency / baseline mean latency; higher=faster',
    'plots': plots}, indent=2) + '\n')
for name in GROUPS:
    ratios = [r['nautilus_speedup_vs_tilelang'] for r in records
              if r['case'].startswith(name) and r['nautilus_speedup_vs_tilelang'] != '']
    if ratios:
        print(f'{name}: {len(ratios)} pairs, Nautilus vs TileLang geomean '
              f'{statistics.geometric_mean(ratios):.3f}x (min {min(ratios):.3f}, max {max(ratios):.3f})')
