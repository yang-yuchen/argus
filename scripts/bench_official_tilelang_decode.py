"""Benchmark official autotuned Q=1 decode with groups=heads (ordinary MHA)."""

import argparse
import json
from pathlib import Path
import statistics

import torch

from argus.operators import BenchmarkOperator
from argus.runner import BenchmarkRunner
from argus.utils import profiler_ctx, relaunch_with_profiler


ROOT = Path(__file__).resolve().parents[1]


class OfficialTileLangRunner(BenchmarkRunner):
    def __init__(self, length):
        super().__init__(BenchmarkOperator(f"official_tilelang_L{length}"),
                         "official_tilelang", "tilelang", str(length))
        self.length = length

    def setup(self, q, k, v):
        from tilelang.autotuner import set_autotune_inputs
        from argus.paper_baselines.vendor.tilelang_official_gqa_decode import flashattn

        # Upstream decode API uses Q=[B,H,D] and K/V=[B,L,KV_heads,D].
        self.q = q.squeeze(2).contiguous()
        self.k = k.permute(0, 2, 1, 3).contiguous()
        self.v = v.permute(0, 2, 1, 3).contiguous()
        self.mask = torch.ones((1, self.length, 32), device=q.device, dtype=torch.uint8)
        with set_autotune_inputs(self.q, self.k, self.v, self.mask):
            self.kernel = flashattn(1, 32, 32, self.length, 128)
        self.tune_result = self.kernel.get_tuner_result()
        self.output = None
        self.run()
        return True

    def run(self):
        with self.mark_ctx():
            self.output = self.kernel(self.q, self.k, self.v, self.mask)

    def get_output(self):
        return self.output.unsqueeze(2)


def inputs(length):
    path = next((ROOT / "profiles_paper_kernels" / run / f"decode_L{length}_inputs.pt"
                 for run in ("sweep_20261026_a", "sweep_20261026_b")
                 if (ROOT / "profiles_paper_kernels" / run / f"decode_L{length}_inputs.pt").exists()), None)
    if path is None:
        raise FileNotFoundError(f"No saved Nautilus input for decode L{length}")
    cpu = torch.load(path, map_location="cpu", weights_only=True)
    return [x.cuda().contiguous() for x in cpu]


def measure(length, output, repeats):
    q, k, v = inputs(length)
    runner = OfficialTileLangRunner(length)
    runner.setup(q, k, v)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    actual = runner.get_output().float()
    torch.testing.assert_close(actual, ref, rtol=1e-2, atol=3e-3)
    relative_l2 = ((actual - ref).norm() / ref.norm()).item()
    metadata = {"length": length, "implementation": "official_tilelang",
                "source": {"repository": "https://github.com/tile-ai/tilelang",
                           "commit": "fdb19f26d61e71cabcc61abbc66ab60a97e3a100",
                           "path": "examples/flash_decoding/example_gqa_decode.py",
                           "semantic_mapping": "groups=heads=32 gives ordinary MHA",
                           "configuration": {"autotune": True,
                                             "selected": runner.tune_result["config"],
                                             "tuning_latency_ms": runner.tune_result["latency"],
                                             "search_space": {"block_N": [64, 128], "block_H": [64],
                                                              "num_split": [1, 2, 4, 8],
                                                              "num_stages": [1, 2, 3],
                                                              "threads": [128]}}},
                "correct": True, "relative_l2": relative_l2, "repeats": repeats,
                "measurement_complete": False}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    for _ in range(10): runner.run()
    torch.cuda.synchronize()
    with profiler_ctx(): runner.repeat(repeats)


def summarize(output, repeats, marker):
    from argus.stats.nsys import nsys_to_dataframe

    frame = nsys_to_dataframe(output.with_suffix(".nsys-rep"))
    frame = frame[(frame["implementation"] == marker) & (frame["repeatIdx"] != "warmup")]
    per_repeat = frame.groupby("repeatIdx")["latency_us"].sum()
    samples = [float(per_repeat[str(i)]) for i in range(repeats)]
    metadata = json.loads(output.with_suffix(".json").read_text())
    metadata.update(samples_us=samples, mean_us=statistics.mean(samples),
                    median_us=statistics.median(samples), min_us=min(samples),
                    measurement_complete=True)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    frame.to_csv(output.with_suffix(".csv"), index=False)


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
        ["-m", "scripts.bench_official_tilelang_decode", str(args.length),
         "--output", str(output), "--repeat", str(args.repeat), "--profile-child"],
        wait_for_cuda_profiler=True)
    summarize(output, args.repeat, f"official_tilelang({args.length})")
    print(json.dumps(json.loads(output.with_suffix(".json").read_text()), indent=2))


if __name__ == "__main__": main()
