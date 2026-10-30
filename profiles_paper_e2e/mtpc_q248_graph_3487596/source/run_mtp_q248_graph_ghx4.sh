#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
output="/u/yucheny8/argus/profiles_paper_e2e/mtpc_q248_graph_${SLURM_JOB_ID:?}"
mkdir -p "$output"
cp profiles_paper_e2e/mtpc_q248_export_manifest_20261029.json "$output/kernel_manifest.json"
mkdir -p "$output/source"
for source_file in scripts/bench_paper_e2e.py scripts/bench_mtpc_e2e_graph_sweep.py \
    scripts/run_mtpc_graph_sweep_ghx4.sh scripts/summarize_mtpc_graph_sweep.py \
    scripts/e2e_model_flops.py argus/paper_baselines/workloads.py \
    argus/paper_baselines/e2e_graph.py argus/paper_baselines/e2e_native_cache.py \
    argus/paper_baselines/e2e_mtp_baselines.py; do
    cp "$source_file" "$output/source/$(basename "$source_file")"
done
exec > >(tee -a "$output/run.log") 2>&1
printf 'Started %s on %s job %s\n' "$(date -Is)" "$(hostname)" "$SLURM_JOB_ID"
if [[ $# == 0 ]]; then set -- 2; fi
for query_length in "$@"; do
    bash scripts/run_mtpc_graph_sweep_ghx4.sh "$output/Q${query_length}" "$query_length"
    (
        # Use the same environment directly; conda activation scans stalled shared metadata.
export CONDA_PREFIX=/work/nvme/beqg/yucheny8/envs/argus-gh200-latest
export PATH="$CONDA_PREFIX/bin:$PATH"
source <(sed '/^source .*conda.sh$/d; /^conda activate /d' envs/activate_gh200_latest.sh)
        export PYTHONPATH=/u/yucheny8/argus/.paper_e2e_hf:/u/yucheny8/argus
        unset TVM_LIBRARY_PATH
        python -u -m scripts.summarize_mtpc_graph_sweep "$output/Q${query_length}" --query-length "$query_length"
    )
    printf 'Completed Q%s at %s\n' "$query_length" "$(date -Is)"
done
printf 'Completed requested Q sweeps at %s\n' "$(date -Is)"
