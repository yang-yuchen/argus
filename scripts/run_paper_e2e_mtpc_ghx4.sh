#!/usr/bin/env bash
# Exact TileOPs/FlexAttention APIs and software stack from the kernel comparison.
set -euo pipefail
cd /u/yucheny8/argus
output="${1:?Usage: bash scripts/run_paper_e2e_mtpc_ghx4.sh OUTPUT_DIRECTORY}"
if [[ -e "$output" ]]; then
    echo "Refusing to reuse output directory: $output" >&2
    exit 2
fi
test "$(git -C .tileops_work rev-parse HEAD)" = 85e7e644f1d99037e937378b8d8dc70c6568113f
mkdir -p "$output"
output=$(realpath "$output")
source envs/activate_gh200_latest.sh
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/u/yucheny8/argus/.paper_e2e_hf:/u/yucheny8/argus:/u/yucheny8/argus/.tileops_work/src
unset TVM_LIBRARY_PATH
for implementation in tileops flex nautilus; do
    if [[ "$implementation" == nautilus ]]; then
        export PYTHONPATH=/work/nvme/beqg/yucheny8/argus-paper-kernels/runtime_py313:$PYTHONPATH
        export TVM_LIBRARY_PATH=/work/nvme/beqg/yucheny8/src/neptune-internal-refactor/build/temp.linux-aarch64-cpython-311
    fi
    echo "Running mtpc/$implementation (Q16/KV65536)"
    python -u -m scripts.bench_paper_e2e mtpc --kv-length 65536 \
        --implementation "$implementation" --warmup 3 --repeat 20 \
        --output "$output/mtpc_${implementation}" >"$output/mtpc_${implementation}.log" 2>&1
done
python - "$output" <<'PY'
import csv
import json
import math
from pathlib import Path
import statistics
import sys

directory = Path(sys.argv[1])
records = [json.loads((directory / f"mtpc_{name}.json").read_text())
           for name in ("nautilus", "tileops", "flex")]
base = records[0]
rows = []
for result in records:
    for key in ("config", "weights_sha256", "inputs_sha256", "gpu_uuid", "torch",
                "transformers", "metric", "warmup", "repeat", "kernel_sha256",
                "cache_implementation"):
        if result[key] != base[key]:
            raise ValueError(f"Mismatched {key}: {result['implementation']}")
    samples = result["samples_ms"]
    if (not result["measurement_complete"] or not result["correct"]
            or len(samples) != result["repeat"]
            or not all(math.isfinite(x) and x > 0 for x in samples)):
        raise ValueError(f"Incomplete/invalid result: {result['implementation']}")
    rows.append({"implementation": result["implementation"],
                 "mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples),
                 "baseline_over_nautilus": statistics.mean(samples) / base["mean_ms"],
                 "relative_l2": result["relative_l2"],
                 "compiled_relative_l2": result["compiled_relative_l2"], "repeat": len(samples)})
with (directory / "results.csv").open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
PY
echo "Results written to $output"
