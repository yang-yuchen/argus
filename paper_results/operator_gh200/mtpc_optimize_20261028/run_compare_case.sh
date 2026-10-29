#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
opt=/u/yucheny8/argus/profiles_paper_kernels/mtpc_optimize_20261028
export COMPARE_CASES=${1:-$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$opt/manual_cases.txt")}
bash "$opt/run_compare_manual.sh"
