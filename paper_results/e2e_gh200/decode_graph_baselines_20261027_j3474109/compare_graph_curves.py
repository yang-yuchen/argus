"""Compare graph-only curves, retaining per-row run and GPU provenance."""

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from argus.paper_baselines.workloads import export_path
from scripts.summarize_decode_e2e_graph import LENGTHS, valid_error


directory = Path(__file__).resolve().parent
tileops = directory.parent / "decode_graph_sweep_20261027_j3473944"
rows = []
manifest = []
for length in LENGTHS:
    records = {}
    for implementation in ("nautilus", "tileops_bound", "flex"):
        run = tileops if implementation == "tileops_bound" else directory
        record = json.loads((run / f"decode_L{length}_{implementation}.json").read_text())
        assert record["variant"] == "decode" and record["implementation"] == implementation
        assert record["execution"] == "cuda_graph_replay"
        assert record["repeat"] == 30 and record["warmup"] == 5
        assert record["correct"] and record["measurement_complete"]
        assert record["persistent_kv_bindings"] == 32
        assert record["full_kv_staging_bytes_per_forward"] == 0
        assert len(record["attention_relative_l2"]) == 32
        assert all(valid_error(x) for x in [record["relative_l2"],
                   record["compiled_relative_l2"], *record["attention_relative_l2"]])
        proof = record["graph_validation"]
        assert proof["all_kernels_in_graph"] and proof["kv_update_kernel_count"] == 64
        assert proof["shared_buffers_poisoned_before_replay"]
        assert len(proof["changed_token_relative_l2"]) == 2
        assert all(valid_error(x) for x in proof["changed_token_relative_l2"])
        assert valid_error(proof["changed_prefix_v_relative_l2"])
        assert valid_error(proof["restored_relative_l2"])
        assert proof["prefix_v_output_change_relative_l2"] > 1e-4
        expected = (96 if implementation == "flex" else
                    32 if implementation == "tileops_bound" and length < 640 else 64)
        assert proof["attention_kernel_count"] == expected
        current = export_path("decode", length).resolve()
        assert Path(record["export_directory"]).resolve() == current
        current_digest = hashlib.sha256((current / "kernel.so").read_bytes()).hexdigest()
        assert record["kernel_sha256"] == current_digest
        samples = record["samples_ms"]
        assert len(samples) == 30 and all(math.isfinite(x) and x > 0 for x in samples)
        records[implementation] = record
        rows.append({"kv_length": length, "implementation": implementation,
                     "execution": record["execution"], "mean_ms": statistics.mean(samples),
                     "median_ms": statistics.median(samples), "stdev_ms": statistics.stdev(samples),
                     "repeat": 30, "warmup": 5, "host": record["host"],
                     "gpu_uuid": record["gpu_uuid"], "run": run.name})
        manifest.append({"kv_length": length, "implementation": implementation,
                         "export_directory": str(current), "kernel_sha256": record["kernel_sha256"],
                         "baseline_source": record["baseline_source"], "run": run.name})
    assert records["nautilus"]["gpu_uuid"] == records["flex"]["gpu_uuid"]
    for implementation, record in records.items():
        for key in ("config", "model", "weights_sha256", "inputs_sha256", "torch", "cuda",
                    "transformers", "metric", "scope", "seed"):
            assert record[key] == records["nautilus"][key], (length, implementation, key)

assert len(rows) == 24
with (directory / "comparison.csv").open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(directory / "kernel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
lines = ["# Decode whole-step CUDA Graph replay comparison", "",
         "All three curves use graph replay, B1/FP16, Q1, Llama-7B backbone.",
         "Nautilus/Flex: new measurements on gh052, job3474109. "
         "TileOPs: previous graph measurements on gh004, job3473944.", "",
         "| Visible KV length | Nautilus ms | TileOPs prebound ms | FlexAttention ms |",
         "|---|---:|---:|---:|"]
for start in range(0, len(rows), 3):
    n, t, f = rows[start:start + 3]
    lines.append(f"| {n['kv_length']} | {n['mean_ms']:.3f} | {t['mean_ms']:.3f} | "
                 f"{f['mean_ms']:.3f} |")
lines += ["", "30 samples and 5 replay warmups per point. "
          "All model/fresh-input/capture checks passed.",
          "Weights, inputs, model configuration and Torch/Transformers match at each length.",
          "TileOPs is a cross-run/GPU reference; small differences from Nautilus are not a "
          "same-GPU controlled comparison.", ""]
(directory / "comparison.md").write_text("\n".join(lines))
fig, ax = plt.subplots(figsize=(9, 4))
for implementation, label in (("nautilus", "Nautilus"), ("tileops_bound", "TileOPs prebound"),
                               ("flex", "FlexAttention")):
    selected = [row for row in rows if row["implementation"] == implementation]
    ax.plot([row["kv_length"] for row in selected], [row["mean_ms"] for row in selected],
            marker="o", label=label)
ax.set_xscale("log", base=2)
ax.set_xticks(LENGTHS, [str(n) if n < 1024 else f"{n // 1024}K" for n in LENGTHS])
ax.set_xlabel("Visible KV length (tokens; Q=1)")
ax.set_ylabel("Whole-step graph replay latency (ms)")
ax.set_title("Decode · Llama-7B backbone · FP16 · batch 1")
ax.set_ylim(bottom=0)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend()
fig.text(0.5, 0.015, "All curves: whole-step graph replay. Nautilus/Flex: gh052; "
         "TileOPs: prior run on gh004.", ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
for suffix in ("pdf", "png"):
    fig.savefig(directory / f"decode_e2e_graph_comparison.{suffix}", bbox_inches="tight")
plt.close(fig)
print("Validated24graph points; wrote comparison.csv/md, manifest, PDF and PNG")
