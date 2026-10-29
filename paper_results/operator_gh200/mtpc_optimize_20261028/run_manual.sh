#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
opt=/u/yucheny8/argus/profiles_paper_kernels/mtpc_optimize_20261028
for case in mtpc_Q64_L256 mtpc_Q32_L32768 mtpc_Q64_L32768; do
    source "$opt/env.sh"
    mkdir -p "$opt/manual/$case"
    python -u "$opt/manual_probe.py" "$case" >"$opt/manual/$case/probe.log" 2>&1
    echo "MANUAL COMPLETE $case"
done
