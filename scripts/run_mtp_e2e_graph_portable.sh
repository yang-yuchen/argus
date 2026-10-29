#!/usr/bin/env bash
# Portable MTP graph-e2e sweep: Q=16, all nine paper KV lengths.
set -euo pipefail

output="${1:?Usage: $0 OUTPUT_DIRECTORY [tileops|nautilus|flex]}"
implementation="${2:-tileops}"
case "$implementation" in tileops|nautilus|flex) ;; *)
    echo "backend must be tileops, nautilus or flex" >&2; exit 2;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=${ARGUS_REPO_ROOT:-$(cd "$script_dir/.." && pwd)}
env_prefix=${ARGUS_ENV_PREFIX:-}
if [[ -n "$env_prefix" ]]; then
    if [[ -n "${CONDA_BASE:-}" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$env_prefix"
    elif command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$env_prefix"
    fi
    [[ -n "${CONDA_PREFIX:-}" ]] || export PATH="$env_prefix/bin:$PATH"
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0
export PYTORCH_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_CUDAGRAPHS=0
tileops_src=${TILEOPS_ROOT:-$repo_root/.tileops_work/src}
hf_overlay=${HF_OVERLAY_ROOT:-$repo_root/.paper_e2e_hf}
export PYTHONPATH="$hf_overlay:$repo_root:$tileops_src${PYTHONPATH:+:$PYTHONPATH}"
unset TVM_LIBRARY_PATH
if [[ "$implementation" == nautilus ]]; then
    : "${NAUTILUS_RUNTIME_ROOT:?Set NAUTILUS_RUNTIME_ROOT for the Nautilus backend}"
    : "${NAUTILUS_TVM_LIBRARY:?Set NAUTILUS_TVM_LIBRARY for the Nautilus backend}"
    export PYTHONPATH="$NAUTILUS_RUNTIME_ROOT:$PYTHONPATH"
    export TVM_LIBRARY_PATH="$NAUTILUS_TVM_LIBRARY"
fi
cd "$repo_root"
exec python -u -m scripts.bench_mtpc_e2e_graph_sweep "$implementation" "$output" \
    --repeat "${REPEAT:-30}" --warmup "${WARMUP:-5}"
