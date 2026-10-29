"""GH200 baseline registry for causal prefill and decode length comparisons."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

from argus import bench_new as bench
from argus import operators as ops
from argus.flashattn import TriDaoRunner
from argus.flashinfer import FlashInferRunner
from argus.neptune_tilelang_pypi import NeptuneTileLangPrefillRunnerBase
from argus.neptune_tilelang_pypi.decode import TileLangDecodeRunner as NeptuneTileLangDecodeRunner
from argus.pytorch.flex import FlexAttentionRunner
from argus.pytorch.sdpa import AttentionAlgo, TorchSDPARunner
from argus.tawa import TawaPrefillRunner
from argus.tawa.decoder import TawaDecodeRunner
from argus.tilelang_pypi import TileLangDecodeRunner, TileLangPrefillRunner
from argus.utils import is_unrecoverable_cuda_error, profiler_ctx

from scripts.bench_paper_strong import environment
from scripts.bench_paper_sweep import digest, reference, spec


def shared_inputs(operator, length):
    case = f"{'prefillc' if operator == 'prefill_causal' else 'decode'}_L{length}"
    root = Path(__file__).resolve().parents[1] / "profiles_paper_kernels"
    for run in ("sweep_20261026_a", "sweep_20261026_b"):
        path = root / run / f"{case}_inputs.pt"
        if path.exists():
            cpu = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(cpu, list) or len(cpu) != 3:
                raise ValueError(f"Invalid shared inputs: {path}")
            for tensor, shape in zip(cpu, spec(case)[2]):
                if tuple(tensor.shape) != shape or tensor.dtype != torch.float16:
                    raise ValueError(f"Invalid shared shape/dtype: {path}")
            return case, [x.cuda().contiguous() for x in cpu]
    raise FileNotFoundError(f"Missing original Nautilus inputs for {case}")


def run_operator(operator, shape, n_repeats, low_mem_overhead, flex_max_autotune=True):
    if low_mem_overhead:
        raise ValueError("Comparison requires the standard cold-L2 profiler capture")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Run on GH200/SM90")
    case, inputs = shared_inputs(operator, shape[1])
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = reference(case, inputs)
    op, candidates = bench.build_runners_by_op_name(flex_max_autotune)[operator]
    records, ready = [], []
    metadata = {"case": case, "environment": environment(), "inputs_sha256": digest(inputs),
                "repeats": n_repeats, "metric": "cold-L2 GPU kernel duration sum (us)",
                "warmup": 10, "l2_flush_bytes": 200 * 1024**2, "candidates": records}
    result_path = RESULT_DIR / f"{operator}_L{shape[1]}_accuracy.json"
    for runner in candidates:
        name = runner.provider_name
        if runner.impl_name:
            name += f"({runner.impl_name})"
        record = {"implementation": name}
        try:
            if not runner.setup(*inputs):
                raise RuntimeError("Runner does not support this setup")
            runner.run()
            torch.cuda.synchronize()
            if hasattr(runner, "source"):
                record["source"] = runner.source
            actual = runner.get_output().float()
            torch.testing.assert_close(actual, ref, rtol=1e-2, atol=3e-3)
            relative_l2 = ((actual - ref).norm() / ref.norm()).item()
            if not relative_l2 < 1e-2:
                raise AssertionError(f"Relative L2 too large: {relative_l2}")
            record.update(status="passed", relative_l2=relative_l2,
                          max_abs_error=(actual - ref).abs().max().item())
            ready.append(runner)
            del actual
        except Exception as exc:
            record.update(status="failed", error=str(exc))
            if is_unrecoverable_cuda_error(exc):
                records.append(record)
                result_path.write_text(json.dumps(metadata, indent=2) + "\n")
                raise
        records.append(record)
        print(f"{case}/{name}: {record['status']}", flush=True)
        result_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if not ready:
        raise RuntimeError(f"No accurate baseline for {case}")
    del ref
    for runner in ready:
        for _ in range(10):
            runner.run()
    torch.cuda.synchronize()
    metadata["environment"] = environment()
    result_path.write_text(json.dumps(metadata, indent=2) + "\n")
    with profiler_ctx():
        for runner in ready:
            runner.repeat(n_repeats)


def build_runners(_flex_max_autotune: bool = True):
    return {
        ops.PF_CAUSAL: [
            FlexAttentionRunner(ops.PF_CAUSAL, mask_expr=ops.causal_mask_cond_4d,
                                max_autotune=_flex_max_autotune),
            TorchSDPARunner(ops.PF_CAUSAL, AttentionAlgo.FLASH, True),
            TorchSDPARunner(ops.PF_CAUSAL, AttentionAlgo.CUDNN, True),
            TriDaoRunner(ops.PF_CAUSAL, causal=True),
            TileLangPrefillRunner(ops.PF_CAUSAL, is_causal=True),
            NeptuneTileLangPrefillRunnerBase(ops.PF_CAUSAL, is_causal=True),
            TawaPrefillRunner(ops.PF_CAUSAL, is_causal=True),
        ],
        ops.DC_CAUSAL: [
            FlexAttentionRunner(ops.DC_CAUSAL, max_autotune=_flex_max_autotune),
            TorchSDPARunner(ops.DC_CAUSAL, AttentionAlgo.FLASH, False),
            TorchSDPARunner(ops.DC_CAUSAL, AttentionAlgo.CUDNN, False),
            TriDaoRunner(ops.DC_CAUSAL, causal=False),
            FlashInferRunner(ops.DC_CAUSAL, is_causal=False, is_decode=True),
            TileLangDecodeRunner(ops.DC_CAUSAL, is_causal=False),
            NeptuneTileLangDecodeRunner(ops.DC_CAUSAL, is_causal=False),
            TawaDecodeRunner(ops.DC_CAUSAL),
        ],
    }


bench.build_runners = build_runners
bench.RUNNERS = build_runners()
bench.RUNNERS_BY_OP_NAME = bench.build_runners_by_op_name()
bench.DEFAULT_OPERATOR_SUITE = list(bench.RUNNERS_BY_OP_NAME)
bench.__spec__ = __spec__
bench.run_operator = run_operator


if __name__ == "__main__":
    if sys.argv[1:] == ["--list-runners"]:
        for op, (_, runners) in bench.RUNNERS_BY_OP_NAME.items():
            print(op, [f"{r.provider_name}({r.impl_name})" for r in runners])
    else:
        args = bench.parse_args()
        RESULT_DIR = args.profile_dir
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        args.func(args)
