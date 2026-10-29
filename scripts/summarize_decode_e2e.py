"""Validate and plot the complete B1/FP16 decode e2e length sweep."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from argus.paper_baselines.workloads import export_path
from argus.paper_baselines.e2e_mtp_baselines import TILEOPS_COMMIT


LENGTHS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
IMPLEMENTATIONS = ("nautilus", "tileops", "flex")
SHARED = ("config", "weights_sha256", "inputs_sha256", "gpu_uuid", "torch", "cuda",
          "transformers", "cache_implementation", "metric", "warmup", "repeat", "kernel_sha256")
CROSS_LENGTH = ("weights_sha256", "gpu_uuid", "torch", "cuda", "transformers",
                "cache_implementation", "metric", "warmup", "repeat")


def summarize(directory, repeat=20, tileops_implementation="tileops"):
    """Refuse incomplete or incomparable data before writing either summary."""
    if repeat < 1:
        raise ValueError("repeat must be positive")
    if tileops_implementation not in ("tileops", "tileops_bound"):
        raise ValueError("tileops_implementation must be tileops or tileops_bound")
    implementations = ("nautilus", tileops_implementation, "flex")
    tileops_label = "TileOPs (prebound)" if tileops_implementation == "tileops_bound" else "TileOPs"
    directory = Path(directory)
    rows = []
    first = None
    for length in LENGTHS:
        records = []
        expected_export = export_path("decode", length).resolve()
        digest = hashlib.sha256((expected_export / "kernel.so").read_bytes()).hexdigest()
        for implementation in implementations:
            path = directory / f"decode_L{length}_{implementation}.json"
            record = json.loads(path.read_text())
            if record["variant"] != "decode" or record["implementation"] != implementation:
                raise ValueError(f"Wrong case identity: {path.name}")
            if implementation == "tileops_bound":
                source = record.get("baseline_source") or {}
                expected_source = {
                    "repository": "https://github.com/tile-ai/TileOPs",
                    "commit": TILEOPS_COMMIT,
                    "api": "GQADenseFwdOp(is_causal=True,target=BUILTIN)",
                    "layout": "BSHD", "selected_kernel": "GQADecodeBs1Kernel",
                    "integration": "prebound selected TileOPs program, "
                                   "caller-owned output/workspace",
                    "execution_backend": "tvm_ffi",
                    "shape": {"q": [1, 1, 32, 128], "kv": [1, length, 32, 128]},
                }
                if (record.get("cache_layout") != "bshd"
                        or any(source.get(key) != value for key, value in expected_source.items())):
                    raise ValueError(f"Invalid bound TileOPs source/integration: {path.name}")
            samples = record["samples_ms"]
            if (record["correct"] is not True or record["measurement_complete"] is not True
                    or record["repeat"] != repeat or len(samples) != repeat
                    or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                               and math.isfinite(x) and x > 0 for x in samples)):
                raise ValueError(f"Incomplete/invalid measurements: {path.name}")
            attention_errors = record["attention_relative_l2"]
            if not isinstance(attention_errors, list) or len(attention_errors) != 32:
                raise ValueError(f"Expected 32 attention correctness checks: {path.name}")
            errors = [record["relative_l2"], record["compiled_relative_l2"], *attention_errors]
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                       and math.isfinite(x) and 0 <= x < 1e-2 for x in errors):
                raise ValueError(f"Invalid relative_l2 correctness measurements: {path.name}")
            if (record["persistent_kv_bindings"] != 32
                    or record["full_kv_staging_bytes_per_forward"] != 0
                    or record["cache_implementation"] != "persistent_native"):
                raise ValueError(
                    f"Expected 32 persistent KV bindings and no full KV staging: {path.name}"
                )
            if Path(record["export_directory"]).resolve() != expected_export:
                raise ValueError(f"Wrong export_directory: {path.name}")
            if record["kernel_sha256"] != digest:
                raise ValueError(f"Wrong kernel_sha256: {path.name}")
            for key in ("weights_sha256", "inputs_sha256", "gpu_uuid", "torch",
                        "transformers", "metric"):
                if not isinstance(record[key], str) or not record[key]:
                    raise ValueError(f"Missing {key}: {path.name}")
            if not record["metric"].startswith("synchronized wall latency in ms"):
                raise ValueError(f"Wrong metric: {path.name}")
            if records:
                for key in SHARED:
                    if record[key] != records[0][key]:
                        raise ValueError(f"Mismatched {key} at L{length}: {implementation}")
            if first is not None:
                for key in CROSS_LENGTH:
                    if record[key] != first[key]:
                        raise ValueError(f"Mismatched cross-length {key}: {path.name}")
                config = {k: v for k, v in record["config"].items()
                          if k != "max_position_embeddings"}
                first_config = {k: v for k, v in first["config"].items()
                                if k != "max_position_embeddings"}
                if config != first_config:
                    raise ValueError(f"Mismatched cross-length config: {path.name}")
            else:
                first = record
            records.append(record)
        nautilus_mean = statistics.mean(records[0]["samples_ms"])
        for record in records:
            mean = statistics.mean(record["samples_ms"])
            rows.append({"kv_length": length, "implementation": record["implementation"],
                         "mean_ms": mean, "median_ms": statistics.median(record["samples_ms"]),
                         "baseline_over_nautilus": mean / nautilus_mean, "repeat": repeat})

    with (directory / "results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Decode model forward latency", "",
             "Llama-7B backbone, B1, FP16, Q1; synchronized wall latency in ms.",
             "Persistent native KV cache; no full KV staging. Lower latency is better.", "",
             f"| KV length | Nautilus ms | {tileops_label} ms | FlexAttention ms | "
             f"{tileops_label} / Nautilus | Flex / Nautilus |",
             "|---|---:|---:|---:|---:|---:|"]
    for start in range(0, len(rows), 3):
        nautilus, tileops, flex = rows[start:start + 3]
        lines.append(f"| {nautilus['kv_length']} | {nautilus['mean_ms']:.3f} | "
                     f"{tileops['mean_ms']:.3f} | {flex['mean_ms']:.3f} | "
                     f"{tileops['baseline_over_nautilus']:.3f}× | "
                     f"{flex['baseline_over_nautilus']:.3f}× |")
    lines += ["", f"All 24 cases passed validation; {repeat} samples per case.",
              "Ratios are each method's mean latency divided by Nautilus at the same KV length.",
              "Initialization, prefix preparation, tokenizer, LM head and sampling are excluded.",
              ""]
    (directory / "results.md").write_text("\n".join(lines))
    return rows


def plot(rows, output, caption=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 3.4))
    for implementation, label in (("nautilus", "Nautilus"), ("tileops", "TileOPs"),
                                   ("tileops_bound", "TileOPs (prebound)"),
                                   ("flex", "FlexAttention")):
        selected = [row for row in rows if row["implementation"] == implementation]
        if not selected:
            continue
        ax.plot([row["kv_length"] for row in selected], [row["mean_ms"] for row in selected],
                marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(LENGTHS, [str(length) if length < 1024 else f"{length // 1024}K"
                           for length in LENGTHS])
    ax.set_xlabel("KV cache length (tokens)")
    ax.set_ylabel("Synchronized model latency (ms)")
    ax.set_ylim(bottom=0)
    ax.set_title("Decode · Llama-7B backbone · FP16 · batch 1 · query 1")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    if caption:
        fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--tileops-implementation", choices=("tileops", "tileops_bound"),
                        default="tileops")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    rows = summarize(args.directory, args.repeat, args.tileops_implementation)
    plot(rows, args.directory / "decode_e2e.pdf")
    print(f"Validated {len(rows)} cases; wrote results.csv, results.md and decode_e2e.pdf")


if __name__ == "__main__":
    main()
