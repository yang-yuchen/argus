"""Validate six reruns and update only their points in the existing throughput figure."""

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from argus.paper_baselines.workloads import export_path
from scripts.summarize_decode_e2e_graph import valid_error
from scripts.summarize_mtpc_graph_sweep import LENGTHS, IMPLEMENTATIONS, model_flops, plot


directory = Path(__file__).resolve().parent
original = directory.parent / "mtpc_graph_sweep_3478086"
rerun_lengths = (2048, 4096)
records, comparisons, rows = {}, [], []
first = None
for length in LENGTHS:
    current = export_path("mtpc", length).resolve()
    digest = hashlib.sha256((current / "kernel.so").read_bytes()).hexdigest()
    for implementation in IMPLEMENTATIONS:
        name = f"mtpc_L{length}_{implementation}.json"
        old = json.loads((original / name).read_text())
        folder = directory if length in rerun_lengths else original
        record = json.loads((folder / name).read_text())
        assert record["variant"] == "mtpc" and record["implementation"] == implementation
        assert record["execution"] == "cuda_graph_replay"
        assert record["correct"] and record["measurement_complete"]
        assert record["repeat"] == 30 and record["warmup"] == 5
        samples = record["samples_ms"]
        assert len(samples) == 30 and all(math.isfinite(x) and x > 0 for x in samples)
        assert len(record["attention_relative_l2"]) == 256
        assert all(valid_error(x) for x in [record["relative_l2"],
                   record["compiled_relative_l2"], *record["attention_relative_l2"]])
        assert record["persistent_kv_bindings"] == 256
        assert record["full_kv_staging_bytes_per_forward"] == 0
        assert Path(record["export_directory"]).resolve() == current
        assert record["kernel_sha256"] == digest
        proof = record["graph_validation"]
        expected = (768 if implementation == "flex" else
                    256 if implementation == "tileops" and length == 256 else 512)
        assert proof["attention_kernel_count"] == expected
        assert proof["kv_update_kernel_count"] == 64 and proof["all_kernels_in_graph"]
        assert proof["attention_group_count"] == 256
        assert proof["shared_buffers_poisoned_before_replay"]
        assert len(proof["changed_token_relative_l2"]) == 2
        assert all(valid_error(x) for x in proof["changed_token_relative_l2"])
        assert valid_error(proof["changed_prefix_v_relative_l2"])
        assert valid_error(proof["restored_relative_l2"])
        assert proof["prefix_v_output_change_relative_l2"] > 1e-4
        for key in ("config", "weights_sha256", "inputs_sha256", "metric", "scope", "seed",
                    "execution", "torch", "cuda", "transformers", "packages", "baseline_source",
                    "kernel_sha256", "cache_implementation", "cache_layout"):
            assert record[key] == old[key], (length, implementation, key)
        if length in rerun_lengths:
            records[(length, implementation)] = record
            if first is None:
                first = record
            assert record["gpu_uuid"] == first["gpu_uuid"]
            assert record["host"] == first["host"]
            comparisons.append({"kv_length": length, "implementation": implementation,
                                "old_mean_ms": statistics.mean(old["samples_ms"]),
                                "old_median_ms": statistics.median(old["samples_ms"]),
                                "old_max_ms": max(old["samples_ms"]),
                                "new_mean_ms": statistics.mean(samples),
                                "new_median_ms": statistics.median(samples),
                                "new_stdev_ms": statistics.stdev(samples),
                                "new_min_ms": min(samples), "new_max_ms": max(samples)})
        mean = statistics.mean(samples)
        rows.append({"kv_length": length, "implementation": implementation,
                     "execution": record["execution"], "mean_ms": mean,
                     "median_ms": statistics.median(samples), "stdev_ms": statistics.stdev(samples),
                     "model_flops": model_flops(length),
                     "throughput_tflops": model_flops(length) / mean / 1e9,
                     "repeat": 30, "warmup": 5, "run": folder.name,
                     "host": record["host"], "gpu_uuid": record["gpu_uuid"]})
assert len(records) == 6 and len(rows) == 27
for length in rerun_lengths:
    base = records[(length, "nautilus")]
    for implementation in IMPLEMENTATIONS:
        for key in ("config", "weights_sha256", "inputs_sha256", "metric", "gpu_uuid"):
            assert records[(length, implementation)][key] == base[key]
for filename, values in (("recheck.csv", comparisons), ("results.csv", rows)):
    with (directory / filename).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
plot(rows, directory)
lines = ["# MTP 2K/4K graph replay recheck", "",
         "Six new configurations; same GPU,30samples/5warmups, unchanged benchmark protocol.",
         "All original data is preserved. No samples were deleted.", "",
         "| KV | Backend | Old mean | Old median | New mean | New median | New max |",
         "|---|---|---:|---:|---:|---:|---:|"]
for row in comparisons:
    lines.append(f"| {row['kv_length']} | {row['implementation']} | "
                 f"{row['old_mean_ms']:.3f} | {row['old_median_ms']:.3f} | "
                 f"{row['new_mean_ms']:.3f} | {row['new_median_ms']:.3f} | "
                 f"{row['new_max_ms']:.3f} |")
lines += ["", "Units: ms. The updated9-length figure replaces only these6points with the rerun;",
          "the other21measurements are reused. Run/host/GPU provenance is retained inresults.csv.",
          "All new correctness, changed-input, poisoned-buffer and graph capture checks passed.", ""]
(directory / "recheck.md").write_text("\n".join(lines))
print("Validated6reruns; reused21points; wrote comparison and updated throughput figure")
