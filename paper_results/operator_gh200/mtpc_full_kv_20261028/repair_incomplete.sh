#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
root=/u/yucheny8/argus/profiles_paper_kernels/mtpc_full_kv_20261028
while read -r case; do
    if /work/nvme/beqg/yucheny8/envs/argus-gh200-latest/bin/python - "$root" "$case" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); case=sys.argv[2]
for method in ('nautilus','flex','tileops'):
    p=root/method/f'{case}.json'
    if not p.exists(): raise SystemExit(1)
    try:d=json.loads(p.read_text())
    except json.JSONDecodeError:raise SystemExit(1)
    if not (d.get('correct') and d.get('measurement_complete') and len(d.get('samples_us',[]))==50):raise SystemExit(1)
PY
    then continue; fi
    echo "Recollecting incomplete shape $case: all three methods in this allocation"
    FORCE_REMEASURE=1 bash "$root/run_missing_case.sh" "$case"
done < "$root/missing_cases.txt"
echo "All missing shapes now have complete triples"
