"""Benchmark official TileLang MHA varlen lower-right causal MTP."""

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch

from argus.operators import BenchmarkOperator
from argus.runner import BenchmarkRunner
from argus.utils import profiler_ctx, relaunch_with_profiler
from scripts.bench_paper_sweep import reference

ROOT = Path(__file__).resolve().parents[1]


class Runner(BenchmarkRunner):
    def __init__(self, length):
        super().__init__(BenchmarkOperator(f"official_tilelang_mtpc_L{length}"),
                         "official_tilelang_mtpc", "tilelang", str(length))
        self.length = length

    def setup(self, q, k, v):
        sys.path.insert(0, str(ROOT / "argus/paper_baselines/vendor"))
        from tilelang.autotuner import set_autotune_inputs
        from argus.paper_baselines.vendor.tilelang_official_mha_varlen import flashattn

        self.q = q.permute(0, 2, 1, 3).reshape(16, 4, 128).contiguous()
        self.k = k.permute(0, 2, 1, 3).reshape(self.length, 4, 128).contiguous()
        self.v = v.permute(0, 2, 1, 3).reshape(self.length, 4, 128).contiguous()
        self.cu_q = torch.tensor([0, 16], device=q.device, dtype=torch.int32)
        self.cu_k = torch.tensor([0, self.length], device=q.device, dtype=torch.int32)
        with set_autotune_inputs(self.q, self.k, self.v, self.cu_q, self.cu_k, 16):
            self.kernel = flashattn(1, 16, self.length, 4, 128, True)
        self.tune_result = self.kernel.get_tuner_result()
        self.run()
        return True

    def run(self):
        with self.mark_ctx():
            self.output = self.kernel(self.q, self.k, self.v, self.cu_q, self.cu_k, 16)

    def get_output(self):
        return self.output.permute(1, 0, 2).unsqueeze(0)


def inputs(length):
    path = ROOT / "profiles_paper_kernels/sweep_20261026_b" / f"mtpc_Q16_L{length}_inputs.pt"
    return [x.cuda().contiguous() for x in torch.load(path, map_location="cpu", weights_only=True)]


def measure(length, output, repeats):
    q, k, v = inputs(length)
    case = f"mtpc_Q16_L{length}"
    runner = Runner(length); runner.setup(q, k, v)
    ref = reference(case, [q, k, v])
    actual = runner.get_output().float()
    torch.testing.assert_close(actual, ref, rtol=1e-2, atol=3e-3)
    relative_l2 = ((actual - ref).norm() / ref.norm()).item()
    meta = {"case": case, "implementation": "official_tilelang_mha_varlen",
            "source": {"repository": "https://github.com/tile-ai/tilelang",
                       "commit": "fdb19f26d61e71cabcc61abbc66ab60a97e3a100",
                       "path": "examples/flash_attention/example_mha_fwd_varlen.py",
                       "configuration": {"autotune": True,
                              "selected": runner.tune_result["config"],
                              "tuning_latency_ms": runner.tune_result["latency"],
                              "search_space": {"block_M": [64, 128], "block_N": [64, 128],
                                               "num_stages": [0, 1, 2, 3], "threads": [128, 256]}},
                       "causal_semantics": "lower-right offset"},
            "correct": True, "relative_l2": relative_l2, "repeats": repeats,
            "measurement_complete": False}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
    for _ in range(10): runner.run()
    torch.cuda.synchronize()
    with profiler_ctx(): runner.repeat(repeats)


def summarize(output, repeats):
    from argus.stats.nsys import nsys_to_dataframe
    frame=nsys_to_dataframe(output.with_suffix(".nsys-rep")); frame=frame[frame.repeatIdx!="warmup"]
    per=frame.groupby("repeatIdx").latency_us.sum(); samples=[float(per[str(i)]) for i in range(repeats)]
    meta=json.loads(output.with_suffix(".json").read_text()); meta.update(
        samples_us=samples,mean_us=statistics.mean(samples),median_us=statistics.median(samples),
        min_us=min(samples),measurement_complete=True)
    output.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n"); frame.to_csv(output.with_suffix(".csv"),index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument("length",type=int);p.add_argument("--output",type=Path,required=True);p.add_argument("--repeat",type=int,default=50);p.add_argument("--profile-child",action="store_true");a=p.parse_args();o=a.output.resolve();o.parent.mkdir(parents=True,exist_ok=True)
    if a.profile_child: measure(a.length,o,a.repeat); return
    relaunch_with_profiler(o,"nsys",False,["-m","scripts.bench_official_tilelang_mtpc",str(a.length),"--output",str(o),"--repeat",str(a.repeat),"--profile-child"],wait_for_cuda_profiler=True);summarize(o,a.repeat)


if __name__ == "__main__": main()
