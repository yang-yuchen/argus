#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
output=/u/yucheny8/argus/profiles_paper_e2e/mtpc_q248_graph_3487596
cp profiles_paper_e2e/mtpc_q248_hf_storage_checks_20261029.json "$output/hf_storage_checks.json"
cp profiles_paper_e2e/mtpc_q248_hf_storage_links_20261029.json "$output/hf_storage_links.json"
cp scripts/recover_mtp_q2_graph_step.sh scripts/run_mtpc_graph_sweep_ghx4.sh "$output/source/"
printf 'Recovery measurement step %s started %s\n' "$SLURM_STEP_ID" "$(date -Is)" | tee -a "$output/run.log"
bash scripts/run_mtpc_graph_sweep_ghx4.sh "$output/Q2" 2
export CONDA_PREFIX=/work/nvme/beqg/yucheny8/envs/argus-gh200-latest
export PATH="$CONDA_PREFIX/bin:$PATH"
source <(sed '/^source .*conda.sh$/d; /^conda activate /d' envs/activate_gh200_latest.sh)
export PYTHONPATH=/u/yucheny8/argus/.paper_e2e_hf:/u/yucheny8/argus
unset TVM_LIBRARY_PATH
python -u -m scripts.summarize_mtpc_graph_sweep "$output/Q2" --query-length 2
printf 'Recovery Q2 measurements completed %s\n' "$(date -Is)" | tee -a "$output/run.log"
