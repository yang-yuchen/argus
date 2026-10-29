"""Measure one length-specific kernel exported by the Neptune paper sweep."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import sqlite3
import sys
import tempfile

import torch

from argus.operators import BenchmarkOperator
from argus.paper_baselines.workloads import resolve_export
from argus.paper_baselines import fp8
from argus.runner import BenchmarkRunner
from argus.utils import profiler_ctx, relaunch_with_profiler


ROOT = Path(os.environ.get(
    "ARGUS_PAPER_KERNEL_ROOT", Path(__file__).resolve().parents[1] / "paper_kernels"
))
CASES = tuple(sorted(
    p.name for p in ROOT.iterdir() if p.is_dir() and (
        re.fullmatch(r"(?:prefillc?|decode|sparse)_L\d+", p.name)
        or re.fullmatch(r"mtpc?_Q\d+_L\d+", p.name)
        or fp8.is_fp8_case(p.name)
        or p.name in {"sparse", "sparse_split"}
    )
))


def spec(case):
    if fp8.is_fp8_case(case):
        return fp8.spec(case)
    if case in ("sparse", "sparse_split"):
        return case, None, None
    sparse_match = re.fullmatch(r"sparse_L(\d+)", case)
    if sparse_match:
        length = int(sparse_match.group(1))
        return "sparse", length, ((length, 16, 576), (length, 576),
                                   (length, min(length, 2048)))
    match = re.fullmatch(r"(prefillc?|decode)_L(\d+)", case)
    if match:
        kind, length = match.group(1), int(match.group(2))
        if kind == "decode":
            return kind, length, ((1, 32, 1, 128), (1, 32, length, 128),
                                  (1, 32, length, 128))
        return kind, length, ((1, 32, length, 128),) * 3
    match = re.fullmatch(r"(mtpc?)_Q(\d+)_L(\d+)", case)
    if match:
        kind, q_length, length = match.group(1), int(match.group(2)), int(match.group(3))
        if not 0 < q_length <= length:
            raise ValueError(f"Expected 0 < query length <= KV length: {case}")
        return kind, length, ((1, 4, q_length, 128), (1, 4, length, 128),
                              (1, 4, length, 128))
    raise ValueError(f"Unknown paper sweep case: {case}")


def inputs_cpu(case):
    if fp8.is_fp8_case(case):
        return fp8.inputs_cpu(case)
    kind, length, shapes = spec(case)
    torch.manual_seed(0)
    if kind == "sparse" and length is not None:
        from argus.paper_baselines.vendor.nautilus_causal_topk import causal_topk_indices

        q = torch.randn((length, 16, 576), dtype=torch.bfloat16)
        kv = torch.randn((length, 576), dtype=torch.bfloat16)
        indices = causal_topk_indices(length, min(length, 2048))
        return [q, kv, indices]
    if kind == "sparse":
        from argus.paper_baselines.workloads import _cpu_inputs

        return _cpu_inputs("sparse")
    if kind == "sparse_split":
        from argus.paper_baselines.workloads import _cpu_inputs

        return _cpu_inputs("sparse_split")
    return [torch.randn(shape, dtype=torch.float16) for shape in shapes]


def load_inputs(case, path):
    if path.exists():
        inputs = torch.load(path, map_location="cpu", weights_only=True)
    else:
        inputs = inputs_cpu(case)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(inputs, path)
    _, _, shapes = spec(case)
    if shapes is not None:
        dtypes = ([torch.bfloat16, torch.bfloat16, torch.int32]
                  if case.startswith("sparse_L") else
                  [torch.float8_e4m3fn] * 3 if fp8.is_fp8_case(case) else [torch.float16] * 3)
        if not isinstance(inputs, (list, tuple)) or len(inputs) != len(shapes):
            raise ValueError(f"Wrong input count for {case}: {path}")
        for tensor, shape, dtype in zip(inputs, shapes, dtypes):
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                    or tensor.dtype != dtype):
                raise ValueError(f"Wrong input shape/dtype for {case}: {path}")
    return [x.cuda().contiguous() for x in inputs]


def export_dir(case):
    return resolve_export(case, ROOT)


def sparse_reference(q, kv, indices, value_dim=512, chunk=32):
    from argus.paper_baselines.workloads import sparse_reference as reference

    return reference(q, kv, indices, value_dim=value_dim, chunk=chunk)


def reference(case, inputs):
    kind, _, _ = spec(case)
    if kind in ("sparse", "sparse_split"):
        if kind == "sparse":
            q, kv, indices = inputs
        else:
            qn, qp, ckv, kpe, indices = inputs
            q = torch.cat((qn, qp), dim=-1)
            kv = torch.cat((ckv.reshape(512, 512), kpe.reshape(512, 64)), dim=-1)
        return sparse_reference(q, kv, indices)
    if fp8.is_fp8_case(case):
        inputs = fp8.canonical_inputs(inputs)
        kind = kind.removesuffix("8")
    q, k, v = (x.float() for x in inputs)
    q_len, kv_len, dim = q.shape[-2], k.shape[-2], q.shape[-1]
    # A full FP32 score matrix is 128 GiB at prefill_L32768. Online softmax
    # keeps correctness checking bounded by one key chunk.
    output = torch.zeros_like(q)
    denominator = torch.zeros((*q.shape[:-1], 1), device=q.device)
    maximum = torch.full_like(denominator, -float("inf"))
    q_positions = torch.arange(q_len, device=q.device)[:, None]
    for start in range(0, kv_len, 1024):
        stop = min(start + 1024, kv_len)
        logits = torch.matmul(q, k[..., start:stop, :].transpose(-1, -2))
        if kind in ("prefillc", "mtpc"):
            k_positions = torch.arange(start, stop, device=q.device)[None, :]
            offset = 0 if q_len == kv_len else kv_len - q_len
            logits.masked_fill_(k_positions > offset + q_positions, -float("inf"))
        logits *= dim ** -0.5
        chunk_max = logits.amax(dim=-1, keepdim=True)
        next_maximum = torch.maximum(maximum, chunk_max)
        old_scale = torch.where(torch.isfinite(next_maximum),
                                torch.exp(maximum - next_maximum),
                                torch.zeros_like(maximum))
        weights = torch.where(torch.isfinite(logits), torch.exp(logits - next_maximum),
                              torch.zeros_like(logits))
        output = output * old_scale + torch.matmul(weights, v[..., start:stop, :])
        denominator = denominator * old_scale + weights.sum(dim=-1, keepdim=True)
        maximum = next_maximum
    return output / denominator


class ExportedRunner(BenchmarkRunner):
    def __init__(self, case):
        super().__init__(BenchmarkOperator(case), "nautilus", "tvm", case)
        self.case = case

    def setup(self, *inputs):
        if spec(self.case)[0] == "sparse":
            self.output = torch.empty((*inputs[0].shape[:-1], 512), dtype=inputs[0].dtype,
                                      device=inputs[0].device)
        else:
            dtype = torch.float16 if fp8.is_fp8_case(self.case) else inputs[0].dtype
            self.output = torch.empty_like(inputs[0], dtype=dtype)
        import tvm

        if self.case.startswith("prefill"):
            import triton.tools.tensor_descriptor  # noqa: F401
        module = tvm.runtime.load_module(str(export_dir(self.case) / "kernel.so"))
        from_dlpack = fp8.torch_to_tvm if fp8.is_fp8_case(self.case) else tvm.nd.from_dlpack
        self.args = [from_dlpack(x) for x in (*inputs, self.output)]
        self.entry = module["main"]
        self.run()
        return True

    def run(self):
        with self.mark_ctx():
            self.entry(*self.args)

    def get_output(self):
        return self.output


def digest(inputs):
    value = hashlib.sha256()
    for x in inputs:
        value.update(x.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return value.hexdigest()


def summarize_profile(output, repeats, marker):
    from argus.stats.nsys import nsys_to_dataframe

    raw = nsys_to_dataframe(output.with_suffix(".nsys-rep"))
    timed = raw[(raw["implementation"] == marker) & (raw["repeatIdx"] != "warmup")]
    expected = {str(i) for i in range(repeats)}
    if set(timed["repeatIdx"]) != expected:
        raise RuntimeError(f"Profile does not contain all repeats for {marker}")
    samples = timed.groupby("repeatIdx")["latency_us"].sum().reindex(sorted(expected, key=int)).tolist()
    metadata = json.loads(output.with_suffix(".json").read_text())
    metadata.update(samples_us=samples, mean_us=statistics.mean(samples),
                    median_us=statistics.median(samples), min_us=min(samples),
                    measurement_complete=True)
    timed.to_csv(output.with_suffix(".csv"), index=False)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def collect_profile(output, launch, summarize):
    """Recollect an empty Nsight CUDA capture once, retaining the failed artifacts."""
    for attempt in range(2):
        launch()
        try:
            return summarize()
        except sqlite3.OperationalError as error:
            if attempt or str(error) != "no such table: main.CUPTI_ACTIVITY_KIND_KERNEL":
                raise
            archive_root = output.parent / "empty_cuda_capture"
            archive_root.mkdir(parents=True, exist_ok=True)
            archive = Path(tempfile.mkdtemp(prefix=f"{output.name}.", dir=archive_root))
            for suffix in (".json", ".nsys-rep", ".sqlite", ".csv"):
                path = output.with_suffix(suffix)
                if path.exists():
                    path.rename(archive / path.name)
            print(f"Empty CUDA trace preserved in {archive}; retrying capture once", flush=True)


def measure(case, output, repeats):
    if not torch.cuda.is_available() or platform.machine() != "aarch64":
        raise RuntimeError("Run on an allocated aarch64 GH200 GPU")
    cpu_path = output.parent / f"{case}_inputs.pt"
    inputs = load_inputs(case, cpu_path)
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = reference(case, inputs)
    runner = ExportedRunner(case)
    runner.setup(*inputs)
    torch.cuda.synchronize()
    actual = runner.get_output().float()
    if case.startswith("sparse_L"):
        tolerance = (1e-2, 8e-3)  # Match the existing BF16 sparse baseline accuracy gate.
    else:
        tolerance = (8e-3, 3e-3) if case in ("sparse", "sparse_split") else (1e-2, 3e-3)
    if fp8.is_fp8_case(case):
        # Match the upstream FP8 accuracy contract; FP16 gates remain separate.
        for fill in (float("nan"), 1.0):
            runner.output.fill_(fill)
            runner.run()
            checks = fp8.check_output(runner.get_output(), ref)
        actual = runner.get_output().float()
        relative_l2 = checks["relative_l2"]
    else:
        torch.testing.assert_close(actual, ref, rtol=tolerance[0], atol=tolerance[1])
        relative_l2 = ((actual - ref).norm() / ref.norm()).item()
        if relative_l2 >= tolerance[0]:
            raise AssertionError(f"Relative L2 error too large: {relative_l2}")
    kernel = export_dir(case) / "kernel.so"
    metadata = {
        "case": case, "variant": spec(case)[0], "backend": export_dir(case).relative_to(ROOT / case).parts[0],
        "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
        "inputs_sha256": digest(inputs), "host": platform.node(),
        "gpu": torch.cuda.get_device_name(), "python": sys.version,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "correct": True, "max_abs_error": (actual - ref).abs().max().item(),
        "relative_l2": relative_l2, "repeats": repeats,
        "metric": "sum of GPU kernel durations per cold-L2 NVTX repeat (us)",
        "l2_flush_bytes": 200 * 1024**2, "warmup": 10,
        "measurement_complete": False,
    }
    if fp8.is_fp8_case(case):
        metadata["fp8_contract"] = fp8.contract()
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    for _ in range(10):
        runner.run()
    torch.cuda.synchronize()
    with profiler_ctx():
        runner.repeat(repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--profile-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.with_suffix(".json").exists() or output.with_suffix(".nsys-rep").exists():
        parser.error("Output exists; choose a new --output prefix")
    marker = f"nautilus({args.case})"
    if args.profile_child:
        measure(args.case, output, args.repeat)
        return
    result = collect_profile(
        output,
        lambda: relaunch_with_profiler(
            output, "nsys", False,
            ["-m", "scripts.bench_paper_sweep", args.case, "--output", str(output),
             "--repeat", str(args.repeat), "--profile-child"],
            wait_for_cuda_profiler=True,
        ),
        lambda: summarize_profile(output, args.repeat, marker),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
