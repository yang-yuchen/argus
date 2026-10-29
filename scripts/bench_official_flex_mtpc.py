"""Official PyTorch FlexAttention lower-right causal MTP benchmark."""

import argparse
import json
import math
import platform
from pathlib import Path
import subprocess

import torch

from argus.operators import BenchmarkOperator
from argus.runner import BenchmarkRunner
from argus.utils import profiler_ctx, relaunch_with_profiler
from scripts.bench_paper_sweep import collect_profile, digest, load_inputs, reference, summarize_profile


ROOT = Path(__file__).resolve().parents[1]


class Runner(BenchmarkRunner):
    def __init__(self, length):
        super().__init__(BenchmarkOperator(f"official_flex_mtpc_L{length}"),
                         "official_flex_mtpc", "pytorch", str(length))
        self.length = length

    def setup(self, q, k, v):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        q_len, kv_len = q.shape[-2], k.shape[-2]
        offset = kv_len - q_len

        def lower_right(_b, _h, q_idx, kv_idx):
            return q_idx + offset >= kv_idx

        self.block_mask = create_block_mask(
            lower_right, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=q.device
        )
        self.call = torch.compile(
            flex_attention, fullgraph=True, mode="max-autotune-no-cudagraphs"
        )
        self.q, self.k, self.v = q, k, v
        self.output = None
        self.run()
        return True

    def run(self):
        with self.mark_ctx():
            self.output = self.call(self.q, self.k, self.v, block_mask=self.block_mask)

    def get_output(self):
        return self.output


def inputs(length, q_length=16, input_file=None):
    case = f"mtpc_Q{q_length}_L{length}"
    path = (Path(input_file) if input_file is not None else
            ROOT / "profiles_paper_kernels/sweep_20261026_b" / f"{case}_inputs.pt")
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_inputs(case, path)


def measure(length, output, repeats, q_length=16, input_file=None):
    torch.backends.cuda.matmul.allow_tf32 = False
    q, k, v = inputs(length, q_length, input_file)
    case = f"mtpc_Q{q_length}_L{length}"
    with torch.inference_mode():
        runner = Runner(length)
        runner.setup(q, k, v)
        ref = reference(case, [q, k, v])
        actual = runner.get_output().float()
        torch.testing.assert_close(actual, ref, rtol=1e-2, atol=3e-3)
        relative_l2 = ((actual - ref).norm() / ref.norm()).item()
        if not math.isfinite(relative_l2) or relative_l2 >= 1e-2:
            raise AssertionError(f"Relative L2 error too large: {relative_l2}")
        metadata = {"case": case, "implementation": "official_flex_lower_right",
                    "source": {"repository": "https://github.com/pytorch/pytorch",
                               "api": "torch.nn.attention.flex_attention",
                               "mask": "q_idx + (KV_LEN-Q_LEN) >= kv_idx",
                               "configuration": "fullgraph=True, max-autotune-no-cudagraphs, inference_mode"},
                    "correct": True, "relative_l2": relative_l2, "repeats": repeats,
                    "inputs_sha256": digest([q, k, v]), "host": platform.node(),
                    "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                    "git": subprocess.check_output(
                        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "metric": "sum of GPU kernel durations per cold-L2 NVTX repeat (us)",
                    "l2_flush_bytes": 200 * 1024**2, "warmup": 10,
                    "measurement_complete": False}
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        for _ in range(10): runner.run()
        torch.cuda.synchronize()
        with profiler_ctx(): runner.repeat(repeats)


def summarize(output, repeats, marker):
    return summarize_profile(output, repeats, marker)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("length", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--q-length", type=int, default=16)
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--profile-child", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1 or not 1 <= args.q_length <= args.length:
        parser.error("Require positive --repeat and 1 <= --q-length <= length")
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    if args.profile_child:
        measure(args.length, output, args.repeat, args.q_length, args.input_file); return
    command = ["-m", "scripts.bench_official_flex_mtpc", str(args.length),
               "--output", str(output), "--repeat", str(args.repeat),
               "--q-length", str(args.q_length), "--profile-child"]
    if args.input_file is not None:
        command.extend(["--input-file", str(args.input_file)])
    collect_profile(
        output,
        lambda: relaunch_with_profiler(output, "nsys", False, command,
                                       wait_for_cuda_profiler=True),
        lambda: summarize(output, args.repeat, f"official_flex_mtpc({args.length})"),
    )
    print(json.dumps(json.loads(output.with_suffix(".json").read_text()), indent=2))


if __name__ == "__main__": main()
