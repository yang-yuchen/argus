#!/usr/bin/env bash
# Run inside a GHx4 allocation using the same Torch/Transformers/Triton for all backends.
set -euo pipefail
cd /u/yucheny8/argus

output="${1:?Usage: bash scripts/run_paper_e2e_decode_ghx4.sh OUTPUT_DIRECTORY}"
if [[ -e "$output" ]]; then
    echo "Refusing to reuse existing output directory: $output" >&2
    exit 2
fi
mkdir -p "$output"
output=$(realpath "$output")
export PATH="/u/yucheny8/nsight-systems-2025.5/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/tmp/argus-paper-e2e-triton
export CUDA_CACHE_PATH=/tmp/argus-paper-e2e-cuda
export PYTHONNOUSERSITE=1 USE_TF=0 USE_FLAX=0
python_bin=/u/yucheny8/miniconda3/envs/nt/bin/python
nt_lib=/u/yucheny8/miniconda3/envs/nt/lib
nvidia_lib="$nt_lib/python3.13/site-packages/nvidia"
export LD_LIBRARY_PATH="$nvidia_lib/cublas/lib:$nvidia_lib/cuda_runtime/lib:$nt_lib:${LD_LIBRARY_PATH:-}"
ulimit -c 0

run_one() {
    local implementation="$1"
    local python_bin="$2"
    shift 2
    echo "Running decode/$implementation"
    "$python_bin" -u -m scripts.bench_paper_e2e decode \
        --implementation "$implementation" --output "$output/decode_${implementation}" \
        --repeat 20 --warmup 3 "$@" \
        >"$output/decode_${implementation}.log" 2>&1
}

# Validate the runtime-dependent baseline first, before the other model runs.
export PYTHONPATH=/u/yucheny8/argus
unset TVM_LIBRARY_PATH
run_one tilelang "$python_bin"

# The exported module needs the Neptune TVM runtime used to build paper_kernels.
export PYTHONPATH=/work/nvme/beqg/yucheny8/argus-paper-kernels/runtime_py313:/u/yucheny8/argus
export TVM_LIBRARY_PATH=/work/nvme/beqg/yucheny8/src/neptune-internal-refactor/build/temp.linux-aarch64-cpython-311
run_one nautilus "$python_bin"

# TileLang needs its matching TVM, in a separate process from the exported module.
export PYTHONPATH=/u/yucheny8/argus
unset TVM_LIBRARY_PATH
run_one flex "$python_bin"

"$python_bin" - "$output" <<'PY'
import csv
import json
import math
from pathlib import Path
import statistics
import sys

directory = Path(sys.argv[1])
records = [json.loads((directory / f"decode_{name}.json").read_text())
           for name in ("nautilus", "tilelang", "flex")]
base = records[0]
rows = []
for result in records:
    for key in ("config", "weights_sha256", "inputs_sha256", "gpu_uuid", "torch",
                "transformers", "metric", "warmup", "repeat"):
        if result[key] != base[key]:
            raise ValueError(f"Mismatched {key}: {result['implementation']}")
    samples = result["samples_ms"]
    if (not result["measurement_complete"] or not result["correct"]
            or len(samples) != result["repeat"]
            or not all(math.isfinite(x) and x > 0 for x in samples)):
        raise ValueError(f"Incomplete/invalid result: {result['implementation']}")
    rows.append({"implementation": result["implementation"],
                 "mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples),
                 "speedup_vs_nautilus": statistics.mean(samples) / base["mean_ms"],
                 "relative_l2": result["relative_l2"], "repeat": len(samples)})
with (directory / "results.csv").open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
PY
echo "Results written to $output"
