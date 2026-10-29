"""Benchmark official TileOPs dense MHA/GQA attention on paper inputs."""

import argparse
import json
import math
from pathlib import Path
import platform
import subprocess

import torch

from argus.operators import BenchmarkOperator
from argus.runner import BenchmarkRunner
from argus.utils import profiler_ctx, relaunch_with_profiler
from scripts.bench_paper_sweep import collect_profile, digest, load_inputs, reference, summarize_profile

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "85e7e644f1d99037e937378b8d8dc70c6568113f"


class Runner(BenchmarkRunner):
    def __init__(self, kind, length):
        super().__init__(BenchmarkOperator(f"official_tileops_{kind}_L{length}"),
                         "official_tileops", "tileops", kind)
        self.kind, self.length = kind, length

    def setup(self, q, k, v):
        from tileops.attention import GQADenseFwdOp
        from tileops.backend import BUILTIN

        self.q = q.permute(0, 2, 1, 3).contiguous()
        self.k = k.permute(0, 2, 1, 3).contiguous()
        self.v = v.permute(0, 2, 1, 3).contiguous()
        self.op = GQADenseFwdOp(is_causal=True, target=BUILTIN)
        self.run()
        kernels = list(self.op.iter_kernels())
        self.selected_kernel = type(kernels[0]).__name__ if kernels else None
        self.kernel_config = self.op.kernel_config()
        return True

    def run(self):
        with self.mark_ctx():
            self.output = self.op(self.q, self.k, self.v)

    def get_output(self):
        return self.output.permute(0, 2, 1, 3)


def inputs(kind, length, q_length=16, input_file=None):
    case = f"mtpc_Q{q_length}_L{length}" if kind == "mtp" else f"decode_L{length}"
    if input_file is not None:
        path = Path(input_file)
        if not path.is_file():
            raise FileNotFoundError(path)
        return load_inputs(case, path)
    for run in ("sweep_20261026_b", "sweep_20261026_a"):
        path = ROOT / "profiles_paper_kernels" / run / f"{case}_inputs.pt"
        if path.is_file():
            return load_inputs(case, path)
    raise FileNotFoundError(case)


def measure(kind, length, out, repeats, q_length=16, input_file=None):
    torch.backends.cuda.matmul.allow_tf32 = False
    q, k, v = inputs(kind, length, q_length, input_file)
    case = f"mtpc_Q{q_length}_L{length}" if kind == "mtp" else f"decode_L{length}"
    runner = Runner(kind, length)
    runner.setup(q, k, v)
    ref = reference(case, [q, k, v])
    actual = runner.get_output().float()
    torch.testing.assert_close(actual, ref, rtol=1e-2, atol=3e-3)
    relative_l2 = ((actual - ref).norm() / ref.norm()).item()
    if not math.isfinite(relative_l2) or relative_l2 >= 1e-2:
        raise AssertionError(f"Relative L2 error too large: {relative_l2}")
    meta = {
        "case": case, "implementation": "official_tileops_gqa_dense",
        "source": {
            "repository": "https://github.com/tile-ai/TileOPs", "commit": COMMIT,
            "api": "GQADenseFwdOp(is_causal=True,target=BUILTIN)", "layout": "BSHD",
            "semantic_mapping": "Hq=Hkv gives MHA", "selected_kernel": runner.selected_kernel,
            "kernel_config": runner.kernel_config,
            "shape": {"q": list(runner.q.shape), "kv": list(runner.k.shape)},
        },
        "correct": True, "relative_l2": relative_l2, "repeats": repeats,
        "inputs_sha256": digest([q, k, v]), "host": platform.node(),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "git": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "metric": "sum of GPU kernel durations per cold-L2 NVTX repeat (us)",
        "l2_flush_bytes": 200 * 1024**2, "warmup": 10, "measurement_complete": False,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    for _ in range(10):
        runner.run()
    torch.cuda.synchronize()
    with profiler_ctx():
        runner.repeat(repeats)


def summarize(out, repeats, kind="mtp"):
    return summarize_profile(out, repeats, f"official_tileops({kind})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("mtp", "decode"))
    parser.add_argument("length", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--q-length", type=int, default=16)
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--profile-child", action="store_true")
    args = parser.parse_args()
    if (args.repeat < 1 or args.length < 1
            or (args.kind == "mtp" and not 1 <= args.q_length <= args.length)):
        parser.error("Require positive --repeat/length and 1 <= MTP --q-length <= length")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.profile_child:
        measure(args.kind, args.length, output, args.repeat, args.q_length, args.input_file)
        return
    command = ["-m", "scripts.bench_official_tileops_dense", args.kind, str(args.length),
               "--output", str(output), "--repeat", str(args.repeat),
               "--q-length", str(args.q_length), "--profile-child"]
    if args.input_file is not None:
        command.extend(["--input-file", str(args.input_file)])
    collect_profile(
        output,
        lambda: relaunch_with_profiler(output, "nsys", False, command,
                                       wait_for_cuda_profiler=True),
        lambda: summarize(output, args.repeat, args.kind),
    )


if __name__ == "__main__":
    main()
