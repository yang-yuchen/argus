import csv
import json
from pathlib import Path

root = Path(__file__).parent
cases = [x.strip() for x in (root / 'cases.txt').read_text().splitlines() if x.strip()]
rows = []
summary = {'cases': cases, 'methods': {}, 'status': 'completed'}
for method in ('nautilus', 'flex', 'tileops'):
    good, bad = [], []
    for case in cases:
        path = root / method / f'{case}.json'
        if not path.exists():
            bad.append(case)
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            bad.append(case)
            continue
        if not (data.get('correct') and data.get('measurement_complete')):
            bad.append(case)
            continue
        good.append(case)
        rows.append({
            'case': case, 'method': method, 'mean_us': data.get('mean_us'),
            'median_us': data.get('median_us'), 'min_us': data.get('min_us'),
            'relative_l2': data.get('relative_l2'),
            'inputs_sha256': data.get('inputs_sha256'),
            'host': data.get('host'), 'gpu': data.get('gpu'),
        })
    summary['methods'][method] = {'complete': good, 'missing_or_failed': bad}
    if bad:
        summary['status'] = 'incomplete'
with (root / 'measurement_summary.json').open('w') as f:
    json.dump(summary, f, indent=2)
with (root / 'latencies_us.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else
                            ['case', 'method', 'mean_us', 'median_us', 'min_us',
                             'relative_l2', 'inputs_sha256', 'host', 'gpu'])
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({m: len(v['complete']) for m, v in summary['methods'].items()}))
if summary['status'] != 'completed':
    raise SystemExit(1)
