#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
opt=/u/yucheny8/argus/profiles_paper_kernels/mtpc_optimize_20261028
while IFS= read -r candidate; do
    bash "$opt/run_measure_candidate.sh" "$candidate"
done < "$opt/manual_cases.txt"
case=mtpc_Q4_L32768
source "$opt/env.sh"
mkdir -p "$opt/manual/$case"
python -u "$opt/trace_probe.py" >"$opt/manual/$case/probe.log" 2>&1
python -u -m scripts.summarize_paper_sweep "$opt/measurement/manual"
