#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
root=/u/yucheny8/argus/profiles_paper_kernels/mtpc_full_kv_20261028
case=${1:-$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$root/missing_cases.txt")}
[[ "$case" =~ ^mtpc_Q(2|4|8|32|64)_L(256|512|2048|8192|32768)$ ]]
source /u/yucheny8/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /u/yucheny8/miniconda3/envs/ntawa
set -u
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9 CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9
export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/bin:/u/yucheny8/nsight-systems-2025.5/bin:$PATH"
export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14
export PYTHONPATH=$root/source/upstream/python:$root/source/tilelang:/u/yucheny8/argus
export TVM_LIBRARY_PATH=/work/nvme/beqg/mbudiu2/neptune-release-build2
export TL_CUTLASS_PATH=/work/nvme/beqg/yucheny8/nautilus/3rdparty/tilelang/3rdparty/cutlass/include
export TL_TEMPLATE_PATH=$root/source/tilelang/src
export ARGUS_PAPER_KERNEL_ROOT=$root/kernels
export TMPDIR=$root/tmp/$case TILELANG_CACHE_DIR=$root/cache/tilelang/$case
export TILELANG_TMP_DIR=$TMPDIR/tilelang TRITON_CACHE_DIR=$root/cache/triton/$case CUDA_CACHE_PATH=$root/cache/cuda/$case
mkdir -p "$TMPDIR" "$TILELANG_TMP_DIR" "$TILELANG_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$root/tuning/$case"
ulimit -c 0
tune="$root/tuning/$case/tilelang"
echo "Tuning $case with official upstream 64-trial auto-fusion search"
if [[ ! -f "$tune/result.json" ]]; then
    python -u -c 'import sys,runpy,tvm; path=sys.argv.pop(1); runpy.run_path(path,run_name="__main__")' \
        "$root/source/upstream/python/attn_tune.py" "$case" --backend tilelang \
        --fusion auto --trials 64 --population 64 --evolve-iters 1 --seed 3 \
        --out "$tune" >"$root/tuning/$case/tune.log" 2>&1
fi
test -f "$tune/best/export/kernel.so"
mkdir -p "$root/kernels/$case/tilelang/best"
if [[ ! -d "$root/kernels/$case/tilelang/best/export" ]]; then
    cp -a "$tune/best/export" "$root/kernels/$case/tilelang/best/export"
fi
cp "$tune/inputs.pt" "$root/nautilus/${case}_inputs.pt"
complete() {
    if [[ ! -f "$1.json" ]]; then return 1; fi
    python -c 'from pathlib import Path; import sys; from scripts.summarize_sparse_compilers import validated; validated(Path(sys.argv[1]))' "$1.json" >/dev/null 2>&1
}
archive_incomplete() {
    local prefix=$1
    for ext in json csv nsys-rep sqlite log; do
        if [[ -f "$prefix.$ext" ]]; then
            local attempt="$root/incomplete_attempts/$(basename "$(dirname "$prefix")")/$case"
            mkdir -p "$attempt"
            mv "$prefix.$ext" "$attempt/$(date +%s)-$ext"
        fi
    done
}
if [[ "${FORCE_REMEASURE:-0}" == 1 ]] || ! complete "$root/nautilus/$case"; then
    archive_incomplete "$root/nautilus/$case"
    echo "Measuring Nautilus $case"
    python -u -m scripts.bench_paper_sweep "$case" --output "$root/nautilus/$case" --repeat 50 \
        >"$root/nautilus/$case.log" 2>&1
fi
q=${case#mtpc_Q}; q=${q%%_L*}; length=${case##*_L}
set +u
source envs/activate_gh200_latest.sh
set -u
export PYTHONPATH=/u/yucheny8/argus:/u/yucheny8/argus/.tileops_work/src
export PATH=/u/yucheny8/nsight-systems-2025.5/bin:$PATH
unset TVM_LIBRARY_PATH TL_CUTLASS_PATH TL_TEMPLATE_PATH
for method in flex tileops; do
    if [[ "${FORCE_REMEASURE:-0}" != 1 ]] && complete "$root/$method/$case"; then continue; fi
    archive_incomplete "$root/$method/$case"
    echo "Measuring $method $case"
    if [[ "$method" == flex ]]; then
        python -u -m scripts.bench_official_flex_mtpc "$length" --q-length "$q" \
            --input-file "$root/nautilus/${case}_inputs.pt" --output "$root/flex/$case" --repeat 50 \
            >"$root/flex/$case.log" 2>&1
    else
        python -u -m scripts.bench_official_tileops_dense mtp "$length" --q-length "$q" \
            --input-file "$root/nautilus/${case}_inputs.pt" --output "$root/tileops/$case" --repeat 50 \
            >"$root/tileops/$case.log" 2>&1
    fi
done
echo "Completed all three methods for $case"
