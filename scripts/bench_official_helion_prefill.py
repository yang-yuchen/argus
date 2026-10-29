"""Benchmark the official Hugging Face HelionDSL attention on causal prefill."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

from argus.operators import PF_CAUSAL
from argus.runner import BenchmarkRunner
from argus.utils import profiler_ctx, relaunch_with_profiler


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "argus/paper_baselines/vendor/helion_attention_official/build/torch-cuda"


def official_helion():
    name = "helion_hub_attention"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, OFFICIAL / "__init__.py", submodule_search_locations=[str(OFFICIAL)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class Runner(BenchmarkRunner):
    def __init__(self, length):
        super().__init__(PF_CAUSAL, "helion_official", "helion", str(length))
        self.length = length

    def setup(self, q, k, v):
        self.q, self.k, self.v = q, k, v
        self.output = torch.empty_like(q)
        self.attention = official_helion().attention
        self.run()
        return True

    def run(self):
        with self.mark_ctx():
            self.attention(self.q, self.k, self.v, causal=True, out=self.output)

    def get_output(self):
        return self.output


def inputs(length):
    for run in ("sweep_20261026_a", "sweep_20261026_b"):
        path = ROOT / "profiles_paper_kernels" / run / f"prefillc_L{length}_inputs.pt"
        if path.exists():
            cpu = torch.load(path, map_location="cpu", weights_only=True)
            return [x.cuda().contiguous() for x in cpu]
    raise FileNotFoundError(f"Missing saved prefill input L{length}")


def measure(length, output, repeats):
    q, k, v = inputs(length)
    runner = Runner(length)
    runner.setup(q, k, v)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), is_causal=True
        )
    actual = runner.get_output().float()
    torch.testing.assert_close(actual, ref, rtol=1e-2, atol=3e-3)
    relative_l2 = ((actual - ref).norm() / ref.norm()).item()
    metadata = {"case": f"prefillc_L{length}", "implementation": "helion_official",
                "source": {"repository": "https://huggingface.co/kernels/HelionDSL/attention",
                           "commit": "bffcab17baba6d257c084ad797d1c7bb80e5877f",
                           "api": "attention(q,k,v,causal=True,out=out)",
                           "configuration": "unmodified official SM90 AOT heuristic"},
                "correct": True, "relative_l2": relative_l2, "repeats": repeats,
                "measurement_complete": False}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    for _ in range(10): runner.run()
    torch.cuda.synchronize()
    with profiler_ctx(): runner.repeat(repeats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("length", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--profile-child", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    if args.profile_child:
        measure(args.length, output, args.repeat); return
    relaunch_with_profiler(output, "nsys", False,
        ["-m", "scripts.bench_official_helion_prefill", str(args.length),
         "--output", str(output), "--repeat", str(args.repeat), "--profile-child"],
        wait_for_cuda_profiler=True)
    print(f"profile ready: {output}.nsys-rep")


if __name__ == "__main__": main()
