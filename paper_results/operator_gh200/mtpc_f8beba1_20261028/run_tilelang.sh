#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
root=profiles_paper_kernels/mtpc_f8beba1_20261028
set +u
source envs/activate_gh200_latest.sh
set -u
export PYTHONPATH=/u/yucheny8/argus
export PATH=/u/yucheny8/nsight-systems-2025.5/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
ulimit -c 0
mkdir -p "$root/official_tilelang"
while read -r case; do
    q=${case#mtpc_Q}; q=${q%%_L*}
    length=${case##*_L}
    prefix="$root/official_tilelang/$case"
    if [[ -f "$prefix.json" ]] && python - "$prefix.json" <<'PY2'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
sys.exit(0 if d.get('correct') and d.get('measurement_complete') else 1)
PY2
    then continue; fi
    for ext in json csv nsys-rep sqlite log; do
        if [[ -f "$prefix.$ext" ]]; then
            mkdir -p "$root/official_tilelang_attempts/$case"
            mv "$prefix.$ext" "$root/official_tilelang_attempts/$case/"
        fi
    done
    echo "Running official TileLang $case"
    python -u -m scripts.bench_official_tilelang_mtpc "$length" \
        --q-length "$q" --input-file "$root/nautilus/${case}_inputs.pt" \
        --output "$prefix" --repeat 50 >"$prefix.log" 2>&1 \
        || echo "FAILED official_tilelang $case" >&2
done < <(rg '^mtpc_Q(2|4|8|32|64)_L' "$root/cases.txt")
