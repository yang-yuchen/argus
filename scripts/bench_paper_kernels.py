"""Check and profile one paper variant and implementation with Argus cold-L2 nsys timing."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch

from argus.operators import BenchmarkOperator
from argus.paper_baselines.workloads import (
    VARIANTS, NautilusUnavailable, build_exported, export_path, make_inputs, metadata, reference,
)
from argus.runner import BenchmarkRunner
from argus.utils import profiler_ctx, relaunch_with_profiler


class PaperRunner(BenchmarkRunner):
    def __init__(self, variant, implementation):
        self.variant = variant
        self.implementation = implementation
        backend = "tawa" if variant == "prefill" else "tilelang"
        super().__init__(BenchmarkOperator(variant),
                         "nautilus" if implementation == "nautilus" else backend,
                         backend, variant)

    def setup(self, *inputs):
        self.source = None
        if self.implementation == "nautilus":
            self.call = build_exported(self.variant, inputs)
        elif self.variant == "prefill":
            from argus.paper_baselines.prefill import BASELINE_SOURCE, build_prefill_baseline

            self.call = build_prefill_baseline(inputs)
            self.source = BASELINE_SOURCE
        elif self.variant in ("decode", "mtp"):
            from argus.paper_baselines.dense import BASELINE_SOURCE, build_dense_baseline

            self.call = build_dense_baseline(self.variant, inputs)
            self.source = BASELINE_SOURCE
        else:
            from argus.paper_baselines.sparse import BASELINE_SOURCE, build_sparse_baseline

            self.call = build_sparse_baseline(self.variant, inputs)
            self.source = BASELINE_SOURCE
        self.run()
        return True

    def run(self):
        with self.mark_ctx():
            self.output = self.call()

    def get_output(self):
        return self.output


def measure(variant, implementation, output, repeats, skip_unavailable_nautilus=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Run on an allocated GH200 GPU node")
    if platform.machine() != "aarch64" or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("The exported paper kernels require aarch64 and SM90")
    inputs, digest = make_inputs(variant, input_file=output.parent / f"{variant}_inputs.pt")
    ref = reference(variant, inputs)
    runner = PaperRunner(variant, implementation)
    try:
        runner.setup(*inputs)
    except NautilusUnavailable as exc:
        if implementation != "nautilus" or not skip_unavailable_nautilus:
            raise
        result = {
            "variant": variant, "implementation": implementation,
            "skipped": True, "skip_reason": str(exc), "correct": None,
            "measurement_complete": False, "inputs_sha256": digest,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "host": platform.node(), "gpu": torch.cuda.get_device_name(),
            "packages": {name: {"version": getattr(sys.modules[name], "__version__", None),
                                "path": getattr(sys.modules[name], "__file__", None)}
                         for name in ("tilelang", "tvm") if name in sys.modules},
        }
        output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    torch.cuda.synchronize()
    actual = runner.get_output().float()
    sparse = variant in ("sparse", "sparse_split")
    torch.testing.assert_close(actual, ref, rtol=1e-2, atol=8e-3 if sparse else 3e-3)
    relative_l2 = ((actual - ref).norm() / ref.norm()).item()
    if relative_l2 >= (8e-3 if sparse else 1e-2):
        raise AssertionError(f"Relative L2 error too large: {relative_l2}")
    kernel = export_path(variant) / "kernel.so"
    result = {
        "variant": variant, "implementation": implementation,
        "nvtx_marker": f"{runner.provider_name}({runner.impl_name})",
        "params": metadata(variant)["params"], "seed": 0, "inputs_sha256": digest,
        "nautilus_export_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
        "baseline_source": runner.source,
        "host": platform.node(), "gpu": torch.cuda.get_device_name(), "python": sys.version,
        "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "nvidia_smi": subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem,power.limit",
            "--format=csv,noheader",
        ], text=True).strip(),
        "packages": {name: {"version": getattr(sys.modules[name], "__version__", None),
                            "path": getattr(sys.modules[name], "__file__", None)}
                     for name in ("triton", "tilelang", "tvm") if name in sys.modules},
        "correct": True, "max_abs_error": (actual - ref).abs().max().item(),
        "relative_l2": relative_l2, "repeats": repeats, "measurement_complete": False,
        "metric": "sum of GPU kernel durations per cold-L2 NVTX repeat (us)",
        "l2_flush_bytes": 200 * 1024**2, "warmup": 10,
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    for _ in range(10):
        runner.run()
    torch.cuda.synchronize()
    with profiler_ctx():
        runner.repeat(repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--implementation", choices=("nautilus", "baseline"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--skip-unavailable-nautilus", action="store_true")
    parser.add_argument("--profile-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.profile_child:
        measure(args.variant, args.implementation, output, args.repeat,
                args.skip_unavailable_nautilus)
        return
    if output.with_suffix(".json").exists() or output.with_suffix(".nsys-rep").exists():
        parser.error("Output exists; choose a new --output prefix")
    relaunch_with_profiler(
        output, "nsys", False,
        ["-m", "scripts.bench_paper_kernels", args.variant,
         "--implementation", args.implementation, "--output", str(output),
         "--repeat", str(args.repeat), "--profile-child"]
        + (["--skip-unavailable-nautilus"] if args.skip_unavailable_nautilus else []),
        wait_for_cuda_profiler=True,
    )
    if json.loads(output.with_suffix(".json").read_text()).get("skipped"):
        print(f"Skipped unavailable Nautilus {args.variant}; reason saved to {output}.json")
        return
    print(f"Profile ready: {output}.nsys-rep; "
          "reduce with scripts.summarize_paper_kernels", flush=True)


if __name__ == "__main__":
    main()
