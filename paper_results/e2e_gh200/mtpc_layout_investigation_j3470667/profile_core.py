"""Diagnose the two MTP adapters without changing the official H4 attention calls."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import statistics
from time import perf_counter

import torch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, help="Optional Chrome trace of two repeats per path")
    args = parser.parse_args()
    if args.output.exists() or (args.trace is not None and args.trace.exists()):
        parser.error("Choose unused output/trace paths")
    if not torch.cuda.is_available():
        parser.error("Run on an allocated GPU")

    from tileops.attention import GQADenseFwdOp
    from tileops.backend import BUILTIN
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    q_len, kv_len, heads, dim = 16, 65536, 4, 128
    # HF Q is a BSHD projection transposed to BHSD; StaticCache K/V are BHSD.
    q_projection = torch.randn((1, q_len, 32, dim), device="cuda", dtype=torch.float16)
    q = q_projection.transpose(1, 2)[:, :heads]
    k = torch.randn((1, heads, kv_len, dim), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    fresh = (q, k, v)
    old_bhsd = [torch.empty_like(x, memory_format=torch.contiguous_format) for x in fresh]
    old_bshd = [torch.empty_like(x.transpose(1, 2), memory_format=torch.contiguous_format)
                for x in fresh]
    native_bshd = [torch.empty_like(x) for x in old_bshd]
    native_bhsd_views = [x.transpose(1, 2) for x in native_bshd]
    flex_bhsd = [torch.empty_like(x, memory_format=torch.contiguous_format) for x in fresh]
    op = GQADenseFwdOp(is_causal=True, target=BUILTIN)

    def lower_right(_b, _h, qi, ki):
        return qi + kv_len - q_len >= ki

    block_mask = create_block_mask(lower_right, B=None, H=None, Q_LEN=q_len,
                                   KV_LEN=kv_len, device=q.device)
    flex_call = torch.compile(flex_attention, fullgraph=True,
                              mode="max-autotune-no-cudagraphs")

    def old_bhsd_copy():
        for destination, source in zip(old_bhsd, fresh):
            destination.copy_(source)

    def layout_copy():
        for destination, source in zip(old_bshd, old_bhsd):
            destination.copy_(source.transpose(1, 2))

    def native_copy():
        for destination, source in zip(native_bhsd_views, fresh):
            destination.copy_(source)

    def flex_bhsd_copy():
        for destination, source in zip(flex_bhsd, fresh):
            destination.copy_(source)

    def tileops_op_only():
        return op(*old_bshd).transpose(1, 2)

    def fixed_tileops_op_only():
        return op(*native_bshd).transpose(1, 2)

    def flex_op_only():
        return flex_call(*flex_bhsd, block_mask=block_mask)

    def old_total():
        old_bhsd_copy()
        layout_copy()
        return tileops_op_only()

    def fixed_total():
        native_copy()
        return fixed_tileops_op_only()

    def flex_total():
        flex_bhsd_copy()
        return flex_op_only()

    allowed = torch.arange(kv_len, device=q.device)[None, :] <= (
        kv_len - q_len + torch.arange(q_len, device=q.device)[:, None])
    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(
            *(x.float() for x in fresh), attn_mask=allowed[None, None])
    errors = {}
    for name, run in (("old", old_total), ("fixed", fixed_total), ("flex", flex_total)):
        actual = run().float()
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=3e-3)
        error = ((actual - expected).norm() / expected.norm()).item()
        assert error < 1e-2, (name, error)
        errors[name] = error
    assert all(x.is_contiguous() for x in native_bshd)
    assert all(view.data_ptr() == native.data_ptr()
               for view, native in zip(native_bhsd_views, native_bshd))
    del expected, actual

    stages = {
        "old_bhsd_copy": old_bhsd_copy, "layout_copy": layout_copy,
        "native_copy": native_copy, "flex_bhsd_copy": flex_bhsd_copy,
        "tileops_op_only": tileops_op_only, "fixed_tileops_op_only": fixed_tileops_op_only,
        "flex_op_only": flex_op_only,
        "old_total": old_total, "fixed_total": fixed_total, "flex_total": flex_total,
    }
    measurements = {}
    for name, run in stages.items():
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        event_samples, wall_samples = [], []
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(20):
            torch.cuda.synchronize()
            begin.record()
            result = run()
            end.record()
            torch.cuda.synchronize()
            event_samples.append(begin.elapsed_time(end))
            del result
        for _ in range(20):
            torch.cuda.synchronize()
            start = perf_counter()
            result = run()
            torch.cuda.synchronize()
            wall_samples.append((perf_counter() - start) * 1000)
            del result
        measurements[name] = {}
        for metric, samples in (("cuda_event_ms", event_samples), ("wall_ms", wall_samples)):
            measurements[name][metric] = {"samples": samples, "mean": statistics.mean(samples),
                                           "median": statistics.median(samples), "min": min(samples)}
        print(name, json.dumps(measurements[name]), flush=True)

    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as profile:
            for _ in range(2):
                for name, run in (("old_total", old_total), ("fixed_total", fixed_total),
                                  ("flex_total", flex_total)):
                    with torch.profiler.record_function(name):
                        if name == "old_total":
                            with torch.profiler.record_function("old_bhsd_copy"):
                                old_bhsd_copy()
                            with torch.profiler.record_function("layout_copy"):
                                layout_copy()
                            with torch.profiler.record_function("tileops_op_only"):
                                result = tileops_op_only()
                        elif name == "fixed_total":
                            with torch.profiler.record_function("native_copy"):
                                native_copy()
                            with torch.profiler.record_function("fixed_tileops_op_only"):
                                result = fixed_tileops_op_only()
                        else:
                            with torch.profiler.record_function("flex_bhsd_copy"):
                                flex_bhsd_copy()
                            with torch.profiler.record_function("flex_op_only"):
                                result = flex_op_only()
                        torch.cuda.synchronize()
                        del result
        profile.export_chrome_trace(str(args.trace))

    kernels = list(op.iter_kernels())
    total_bytes = sum(x.numel() * x.element_size() for x in fresh)
    payload = {
        "scope": "H4/Q16/KV65536 core and adapters only, steady state, no L2 flush",
        "seed": 0, "repeat": 20, "warmup": 3, "correct": True,
        "relative_l2": errors, "measurements": measurements,
        "inputs": [{"shape": list(x.shape), "stride": list(x.stride()),
                     "source_bytes": x.numel() * x.element_size(),
                     "is_contiguous": x.is_contiguous()} for x in fresh],
        "q_projection_shape": list(q_projection.shape), "source_bytes": total_bytes,
        "copy_payload_bytes": {"old_total": total_bytes * 2, "fixed_total": total_bytes,
                               "flex_total": total_bytes},
        "copy_traffic_note": "copy_payload_bytes counts destination bytes; read+write traffic is twice that",
        "api": "GQADenseFwdOp(is_causal=True,target=BUILTIN)",
        "selected_kernel": [type(kernel).__name__ for kernel in kernels],
        "kernel_config": op.kernel_config(),
        "tileops_commit": "85e7e644f1d99037e937378b8d8dc70c6568113f",
        "flex_configuration": "fullgraph=True, max-autotune-no-cudagraphs, inference_mode",
        "mask": "key_position <= KV_LEN-Q_LEN+query_index",
        "metric_note": "event and synchronized wall samples use separate 20-repeat loops; event intervals can include host dispatch gaps",
        "host": platform.node(), "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "packages": {name: metadata.version(name) for name in ("tileops", "tilelang", "triton")},
        "trace": str(args.trace) if args.trace is not None else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
