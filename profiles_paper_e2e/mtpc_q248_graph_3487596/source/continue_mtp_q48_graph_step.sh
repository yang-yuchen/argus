#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
output=/u/yucheny8/argus/profiles_paper_e2e/mtpc_q248_graph_3487596
while [[ ! -f "$output/Q2/results.csv" ]]; do sleep 5; done
printf 'Q4/Q8 continuation step %s started %s\n' "$SLURM_STEP_ID" "$(date -Is)" | tee -a "$output/run.log"
cp scripts/continue_mtp_q48_graph_step.sh "$output/source/"
bash scripts/run_mtp_q248_graph_ghx4.sh 4 8
