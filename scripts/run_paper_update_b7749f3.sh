#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
out=profiles_paper_kernels/update_b7749f3_20261027
if [[ -e "$out" ]]; then
    echo "Output exists: $out" >&2
    exit 2
fi
# Use the original sweep runtime and input bytes for the changed exports.
source /u/yucheny8/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /u/yucheny8/miniconda3/envs/ntawa
set -u
export PYTHONPATH=/work/nvme/beqg/yucheny8/argus-paper-kernels/runtime_py313:/u/yucheny8/argus
export TVM_LIBRARY_PATH=/work/nvme/beqg/yucheny8/src/neptune-internal-refactor/build/temp.linux-aarch64-cpython-311
export PATH=/u/yucheny8/nsight-systems-2025.5/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/tmp/argus-paper-update-b7749f3-triton
export CUDA_CACHE_PATH=/tmp/argus-paper-update-b7749f3-cuda
ulimit -c 0
mkdir -p "$out"
cp profiles_paper_kernels/update_b7749f3_inputs/*_inputs.pt "$out/"
for length in 4096 8192 16384 32768; do
    python -u -m scripts.bench_paper_sweep "decode_L${length}" \
        --output "$out/decode_L${length}" --repeat 50 \
        >"$out/decode_L${length}.log" 2>&1
done
python -m scripts.summarize_paper_sweep "$out" >"$out/results.json"
