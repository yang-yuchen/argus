"""Measure the exported paper prefill with Argus's cold-L2 NVTX/nsys infrastructure.

Run with a Neptune TVM runtime on GH200, e.g. ``python -m scripts.bench_paper_prefill``.
This loads the exported binary; it does not rebuild the Triton source.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

import torch

from argus.operators import PF_GLOBAL
from argus.runner import BenchmarkRunner
from argus.utils import profiler_ctx, relaunch_with_profiler


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT = REPO_ROOT / "paper_kernels/prefill/tawa/best/export"
if not EXPORT.exists():
    EXPORT = REPO_ROOT / "paper_kernels.previous_20261026/prefill/tawa/best/export"
SHAPE = (1, 32, 512, 128)


class PaperPrefillRunner(BenchmarkRunner):
    def __init__(self):
        super().__init__(PF_GLOBAL, "paper_neptune", "tawa", "exported")

    def setup(self, q, k, v):
        for tensor in (q, k, v):
            if tensor.shape != SHAPE or tensor.dtype != torch.float16:
                raise ValueError(f"Paper prefill requires {SHAPE} FP16 inputs")
            if tensor.device.type != "cuda" or not tensor.is_contiguous():
                raise ValueError("Paper prefill requires contiguous CUDA inputs")
        if len({tensor.device for tensor in (q, k, v)}) != 1:
            raise ValueError("Inputs must be on the same CUDA device")

        import tvm
        # The legacy TVM callback checks this module's presence without importing it.
        import triton.tools.tensor_descriptor  # noqa: F401

        self.output = torch.empty_like(q)
        self.args = [tvm.nd.from_dlpack(x) for x in (q, k, v, self.output)]
        self.module = tvm.runtime.load_module(str(EXPORT / "kernel.so"))
        self.run()
        return True

    def run(self):
        with self.mark_ctx():
            self.module(*self.args)

    def get_output(self):
        return self.output


def measure(output, repeats):
    if not torch.cuda.is_available():
        raise RuntimeError("Run on a Slurm GH200 allocation; no CUDA device is visible")
    if platform.machine() != "aarch64" or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("The exported binary requires aarch64 and an SM90 GPU")

    import triton
    import tvm
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.manual_seed(0)
    # Match the upstream inputs: generate FP16 tensors on the CPU before copying to CUDA.
    inputs = [torch.randn(SHAPE, dtype=torch.float16).cuda() for _ in range(3)]
    runner = PaperPrefillRunner()
    runner.setup(*inputs)
    torch.cuda.synchronize()
    with sdpa_kernel(SDPBackend.MATH):
        reference = torch.nn.functional.scaled_dot_product_attention(
            *(x.float() for x in inputs)
        )
    actual = runner.get_output().float()
    torch.testing.assert_close(actual, reference, rtol=1e-2, atol=3e-3)
    relative_l2 = ((actual - reference).norm() / reference.norm()).item()
    if relative_l2 >= 1e-2:
        raise AssertionError(f"relative L2 error {relative_l2} >= 1e-2")

    metadata = {
        "shape": SHAPE, "dtype": "float16", "causal": False,
        "gpu": torch.cuda.get_device_name(), "host": platform.node(),
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "tvm": tvm.__version__, "tvm_path": tvm.__file__,
        "triton": triton.__version__, "triton_path": triton.__file__,
        "kernel_sha256": hashlib.sha256((EXPORT / "kernel.so").read_bytes()).hexdigest(),
        "correct": True, "max_abs_error": (actual - reference).abs().max().item(),
        "relative_l2": relative_l2, "repeats": repeats,
        "measurement_complete": False,
        "metric": "sum of GPU kernel durations per cold-L2 NVTX repeat (us)",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    for _ in range(10):
        runner.run()
    torch.cuda.synchronize()
    with profiler_ctx():
        runner.repeat(repeats)


def summarize_profile(output, repeats, implementation="paper_neptune(exported)"):
    from argus.stats.nsys import nsys_to_dataframe

    raw = nsys_to_dataframe(output.with_suffix(".nsys-rep"))
    timed = raw[raw["implementation"] == implementation]
    timed = timed[timed["repeatIdx"] != "warmup"]
    if set(timed["repeatIdx"]) != {str(i) for i in range(repeats)}:
        raise RuntimeError("Profile does not contain every requested repeat")
    per_repeat = timed.groupby("repeatIdx")["latency_us"].sum()
    samples = per_repeat.reindex([str(i) for i in range(repeats)]).tolist()
    timed.to_csv(output.with_suffix(".csv"), index=False)
    metadata = json.loads(output.with_suffix(".json").read_text())
    metadata.update(
        samples_us=samples, mean_us=statistics.mean(samples),
        median_us=statistics.median(samples), min_us=min(samples), measurement_complete=True,
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("profiles_paper_prefill/prefill"))
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--profile-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.profile_child:
        measure(output, args.repeat)
        return

    if output.with_suffix(".nsys-rep").exists() or output.with_suffix(".json").exists():
        parser.error("Output already exists; choose a new --output prefix")
    relaunch_with_profiler(
        output, "nsys", False,
        ["-m", "scripts.bench_paper_prefill", "--output", str(output),
         "--repeat", str(args.repeat), "--profile-child"],
        wait_for_cuda_profiler=True,
    )
    metadata = summarize_profile(output, args.repeat)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
