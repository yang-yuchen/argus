"""Validate this sweep and reuse plot_from_csv.py for each query-token count."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

root = Path(__file__).resolve().parent
project = root.parents[2]
sys.path.insert(0, str(project / "paper_results/nautilus_mlsys27/performance_sources/scripts"))
import plot_from_csv as plotting

source_manifest = json.loads((root / 'source_manifest.json').read_text())
records = []
by_query = {}
hosts = set()
for entry in source_manifest['cases']:
    case = entry['case']
    measurements = {}
    for folder, label in [('nautilus', 'Ours'), ('flex', 'FlexAttn'), ('tileops', 'TileLang')]:
        data = json.loads((root / folder / f'{case}.json').read_text())
        assert data['case'] == case and data['correct'] and data['measurement_complete'], (case, folder)
        samples = data['samples_us']
        assert data['repeats'] == len(samples) == 50, (case, folder)
        assert all(math.isfinite(x) and x > 0 for x in samples), (case, folder)
        assert math.isclose(data['mean_us'], statistics.mean(samples), rel_tol=1e-12)
        assert math.isfinite(data['relative_l2']) and data['relative_l2'] < 1e-2
        assert data['l2_flush_bytes'] == 200 * 1024**2 and data['warmup'] == 10
        hosts.add(data['host'])
        measurements[label] = data
    original = measurements['Ours']
    for label, data in measurements.items():
        assert data['inputs_sha256'] == original['inputs_sha256'], (case, label)
    kernel = project / entry['export'] / 'kernel.so'
    assert original['kernel_sha256'] == entry['sha256']['kernel.so']
    assert hashlib.sha256(kernel.read_bytes()).hexdigest() == original['kernel_sha256']
    q, length = entry['query_length'], entry['kv_length']
    row = {'Operator': 'MTP_CAUSAL', 'SeqLen': length,
           **{label: data['mean_us'] for label, data in measurements.items()}}
    by_query.setdefault(q, []).append(row)
    records.append({'case': case, 'q_length': q, 'kv_length': length,
                    'Nautilus_us': original['mean_us'],
                    'FlexAttn_us': measurements['FlexAttn']['mean_us'],
                    'TileLang_us': measurements['TileLang']['mean_us'],
                    'nautilus_speedup_vs_flex': measurements['FlexAttn']['mean_us'] / original['mean_us'],
                    'nautilus_speedup_vs_tilelang': measurements['TileLang']['mean_us'] / original['mean_us']})
assert len(records) == 29 and len(hosts) == 1, hosts
plots = []
for q, rows in sorted(by_query.items()):
    rows.sort(key=lambda row: row['SeqLen'])
    expected = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] if q == 16 else [1024, 4096, 16384, 65536]
    assert [row['SeqLen'] for row in rows] == expected
    csv_path = root / f'plot_input_Q{q}.csv'
    with csv_path.open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=['Operator', 'SeqLen', 'Ours', 'FlexAttn', 'TileLang'])
        writer.writeheader()
        writer.writerows(rows)
    frame = plotting.load_data(csv_path)
    normalized = plotting.normalize_by_baseline(frame, 'Ours')
    assert normalized['Ours'].eq(1.0).all()
    for label in ('FlexAttn', 'TileLang'):
        for value, row in zip(normalized[label], rows):
            assert math.isclose(value, row['Ours'] / row[label], rel_tol=1e-12)
    plotting.OPERATOR_DISPLAY_NAMES['MTP_CAUSAL'] = f'Causal MTP (Q={q})'
    suffix = f'-fp16-tc1-mtpc-Q{q}-f8beba1'
    plotting.plot_multi_operator_row(normalized, ['MTP_CAUSAL'], 'Ours', project / 'plots_from_csv', suffix=suffix)
    pdf = project / 'plots_from_csv' / f'gh200{suffix}.pdf'
    assert pdf.read_bytes().startswith(b'%PDF-')
    plots.append({'q_length': q, 'csv': str(csv_path.relative_to(project)),
                  'pdf': str(pdf.relative_to(project))})
with (root / 'comparison.csv').open('w', newline='') as fp:
    writer = csv.DictWriter(fp, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
manifest = {'status': 'completed', 'method_length_pairs': 87, 'cases': 29,
            'hosts': sorted(hosts), 'slurm_job': '3478832', 'slurm_exit_code': '0:0',
            'plots': plots, 'tilelang_label': 'official TileOPs GQADenseFwdOp',
            'normalization': 'Nautilus mean latency / baseline mean latency; higher=faster'}
(root / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
source_manifest.update(status='complete', measured_method_case_pairs=87, nodes_used=sorted(hosts))
(root / 'source_manifest.json').write_text(json.dumps(source_manifest, indent=2) + '\n')
for q in sorted(by_query):
    subset = [r for r in records if r['q_length'] == q]
    tilelang = statistics.geometric_mean(r['nautilus_speedup_vs_tilelang'] for r in subset)
    flex = statistics.geometric_mean(r['nautilus_speedup_vs_flex'] for r in subset)
    print(f'Q={q}: vs TileLang {tilelang:.4f}x, vs Flex {flex:.4f}x')
print('87/87 results validated, same host/input hashes, six existing-style PDFs generated.')
