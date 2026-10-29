#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
opt=/u/yucheny8/argus/profiles_paper_kernels/mtpc_optimize_20261028
case=${1:-$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$opt/cases.txt")}
source "$opt/env.sh"
cd "$root/source/upstream/python"
records="$opt/retune/$case/tilelang/db/database_tuning_record.json"
count=$(wc -l < "$records")
while (( count < 256 )); do
    remaining=$((256-count))
    trials=$((remaining < 64 ? remaining : 64))
    seed=$((7 + 4*((count-64)/64)))
    echo "RESUME $case: $count committed records; requesting $trials more, seed=$seed"
    python -u -c 'import sys,runpy,tvm; path=sys.argv.pop(1); runpy.run_path(path,run_name="__main__")' attn_tune.py "$case" --backend tilelang --fusion auto --trials "$trials" --population 64 --evolve-iters 1 --seed "$seed" --resume --out "$opt/retune/$case/tilelang" >"$opt/retune/$case/resume-from-${count}.log" 2>&1
    updated=$(wc -l < "$records")
    if (( updated <= count )); then echo "No new tuning records" >&2; exit 1; fi
    count=$updated
done
echo "RETUNE COMPLETE $case: $count committed records"
cd /u/yucheny8/argus
python "$opt/select_checked.py" "$case" >"$opt/retune/$case/selection.log" 2>&1
echo "STRICT SELECTION COMPLETE $case"
