#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
set +u
source envs/activate_gh200_latest.sh
set -u
export PYTHONPATH=/u/yucheny8/argus:/u/yucheny8/argus/.tileops_work/src
unset TVM_LIBRARY_PATH
export PATH=/u/yucheny8/nsight-systems-2025.5/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
root=${SPARSE_OUTPUT_ROOT:-profiles_sparse_compilers_20261027}
if [[ $# -eq 0 ]]; then set -- tilelang flex helion; fi
for method in "$@"; do
    mkdir -p "$root/$method"
    for length in 256 512 1024 2048 4096 8192 16384 32768; do
        prefix="$root/$method/sparse_L${length}"
        if python - "$prefix.json" "$method" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);d=json.loads(p.read_text()) if p.exists() else {}
complete=d.get('measurement_complete') and d.get('correct')
if sys.argv[2] in ('flex','helion'):complete=complete and d.get('source',{}).get('shared_mla_kv') is True
if sys.argv[2]=='tilelang':complete=complete and isinstance(d.get('source',{}).get('autotune'),dict)
if sys.argv[2]=='helion':complete=complete and 'hl.load' in d.get('source',{}).get('mask_callback','') and d.get('source',{}).get('configuration',{}).get('autotune_budget_seconds')==180
if sys.argv[2]=='flex':complete=complete and d.get('source',{}).get('flex_search_space')=='DEFAULT' and d.get('source',{}).get('kernel_options',{}).get('BLOCK_M')==32 and d.get('source',{}).get('kernel_options',{}).get('BLOCK_N')==64 and d.get('source',{}).get('kernel_options',{}).get('num_warps')==8
sys.exit(0 if complete else 1)
PY
        then
            continue
        fi
        mkdir -p "$root/${method}_attempts"
        archive=$(mktemp -d "$root/${method}_attempts/sparse_L${length}.XXXXXX")
        for extension in log json csv nsys-rep sqlite; do
            if [[ -f "$prefix.$extension" ]]; then mv "$prefix.$extension" "$archive/"; fi
        done
        python -u -m scripts.bench_sparse_official "$method" "$length" \
            --inputs "$root/nautilus" --output "$prefix" --repeat 50 \
            >"$prefix.log" 2>&1 || echo "FAILED $method L$length: $prefix.log" >&2
    done
done
