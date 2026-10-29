"""Validate bound decode against public TileOPs after changing the resident inputs."""

import json
from pathlib import Path

import torch

from argus.paper_baselines.e2e_mtp_baselines import build_mtp_baseline


torch.manual_seed(7)
records = []
with torch.inference_mode():
    for length in (256, 512, 8192, 32768):
        tensors = [torch.randn(1, size, 32, 128, device="cuda", dtype=torch.float16)
                   .transpose(1, 2) for size in (1, length, length)]
        public, source = build_mtp_baseline("tileops", tensors)
        bound, bound_source = build_mtp_baseline("tileops_bound", tensors)
        errors = []
        for _ in range(2):
            for tensor in tensors:
                tensor.normal_()
            expected = public().float()
            actual = bound().float()
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=3e-3)
            errors.append(((actual - expected).norm() / expected.norm()).item())
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
            bound()
            bound()
        counts = {event.key: event.count for event in p.key_averages()}
        assert counts.get("aten::empty", 0) == 0, counts
        assert counts.get("tileops::attention_gqa_dense_fwd", 0) == 0, counts
        records.append({"kv_length": length, "source": source, "bound_source": bound_source,
                        "relative_l2_after_input_updates": errors, "bound_cpu_events": counts})
        print(f"PASS L{length}: updated Q/K/V, public output match, no aten::empty or public op")
Path(__file__).with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n")
