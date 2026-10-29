#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
opt=/u/yucheny8/argus/profiles_paper_kernels/mtpc_optimize_20261028
case=mtpc_Q4_L32768
source "$opt/env.sh"
python -u "$opt/select_checked.py" "$case" >"$opt/retune/$case/checkpoint-selection.log" 2>&1
