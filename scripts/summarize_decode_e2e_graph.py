"""Validate and plot an eight-length whole-step decode replay sweep."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from argus.paper_baselines.e2e_mtp_baselines import TILEOPS_COMMIT
from argus.paper_baselines.workloads import export_path


LENGTHS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
LABELS = {"tileops_bound": "TileOPs prebound", "nautilus": "Nautilus", "flex": "FlexAttention"}
SHARED = ("weights_sha256", "gpu_uuid", "host", "torch", "cuda", "transformers", "packages",
          "cache_implementation", "metric", "execution", "repeat", "warmup", "seed")


def valid_error(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value < 0.01)


def summarize(directory, implementation="tileops_bound"):
    """Write outputs only after all eight graph measurements pass their gates."""
    directory = Path(directory)
    if implementation not in LABELS:
        raise ValueError(f"Unsupported graph implementation: {implementation}")
    suffix = "" if implementation == "tileops_bound" else f"_{implementation}"
    rows = []
    first = None
    for length in LENGTHS:
        path = directory / f"decode_L{length}_{implementation}.json"
        record = json.loads(path.read_text())
        if (record.get("variant") != "decode" or record.get("implementation") != implementation
                or record.get("execution") != "cuda_graph_replay"):
            raise ValueError(f"Wrong case/execution: {path.name}")
        samples = record["samples_ms"]
        if (record["correct"] is not True or record["measurement_complete"] is not True
                or record["repeat"] != 30 or record["warmup"] != 5 or len(samples) != 30
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                           and math.isfinite(x) and x > 0 for x in samples)):
            raise ValueError(f"Invalid samples/protocol: {path.name}")
        errors = record["attention_relative_l2"]
        if (not isinstance(errors, list) or len(errors) != 32
                or not all(valid_error(x) for x in [record["relative_l2"],
                           record["compiled_relative_l2"], *errors])):
            raise ValueError(f"Invalid model/attention correctness: {path.name}")
        proof = record.get("graph_validation") or {}
        fresh = proof.get("changed_token_relative_l2")
        kernels = proof.get("kernel_count")
        prefix_change = proof.get("prefix_v_output_change_relative_l2")
        attention_kernels = 32 if implementation == "tileops_bound" and length < 640 else 64
        if implementation == "flex":
            attention_kernels = 96
        if (not isinstance(fresh, list) or len(fresh) != 2
                or not all(valid_error(x) for x in fresh)
                or not valid_error(proof.get("changed_prefix_v_relative_l2"))
                or not valid_error(proof.get("restored_relative_l2"))
                or proof.get("shared_buffers_poisoned_before_replay") is not True
                or not isinstance(prefix_change, (int, float)) or isinstance(prefix_change, bool)
                or not math.isfinite(prefix_change) or prefix_change <= 1e-4
                or proof.get("all_kernels_in_graph") is not True
                or proof.get("attention_kernel_count") != attention_kernels
                or proof.get("kv_update_kernel_count") != 64
                or type(kernels) is not int or kernels <= attention_kernels + 64):
            raise ValueError(f"Invalid capture/fresh-input proof: {path.name}")
        if (record["cache_implementation"] != "persistent_native"
                or record["cache_layout"] != ("bshd" if implementation == "tileops_bound"
                                              else "bhsd")
                or record["persistent_kv_bindings"] != 32
                or record["full_kv_staging_bytes_per_forward"] != 0):
            raise ValueError(f"Invalid persistent KV cache: {path.name}")
        source = record.get("baseline_source") or {}
        expected_source = {
            "repository": "https://github.com/tile-ai/TileOPs", "commit": TILEOPS_COMMIT,
            "api": "GQADenseFwdOp(is_causal=True,target=BUILTIN)", "layout": "BSHD",
            "selected_kernel": "GQADecodeBs1Kernel",
            "kernel_config": {"block_M": 64, "block_N": 128, "threads": 160},
            "integration": "prebound selected TileOPs program, caller-owned output/workspace",
            "execution_backend": "tvm_ffi", "ctx_splits": None if length < 640 else 4,
            "shape": {"q": [1, 1, 32, 128], "kv": [1, length, 32, 128]},
        }
        if (implementation == "tileops_bound"
                and any(source.get(key) != value for key, value in expected_source.items())):
            raise ValueError(f"Wrong TileOPs source/configuration: {path.name}")
        if implementation == "nautilus" and record.get("baseline_source") is not None:
            raise ValueError(f"Wrong Nautilus export source: {path.name}")
        if implementation == "flex" and (
            source.get("repository") != "https://github.com/pytorch/pytorch"
            or source.get("api") != "torch.nn.attention.flex_attention"
            or "mask" not in source or source["mask"] is not None
            or source.get("configuration") != (
                "fullgraph=True, max-autotune-no-cudagraphs, inference_mode"
            )
        ):
            raise ValueError(f"Wrong FlexAttention source/configuration: {path.name}")
        current = export_path("decode", length).resolve()
        if Path(record["export_directory"]).resolve() != current:
            raise ValueError(f"Wrong export_directory: {path.name}")
        digest = hashlib.sha256((current / "kernel.so").read_bytes()).hexdigest()
        if record["kernel_sha256"] != digest:
            raise ValueError(f"Wrong kernel_sha256: {path.name}")
        for key in ("weights_sha256", "inputs_sha256"):
            digest = record[key]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError(f"Invalid {key}: {path.name}")
        for key in ("gpu_uuid", "host", "torch", "transformers", "metric"):
            if not isinstance(record[key], str) or not record[key]:
                raise ValueError(f"Missing {key}: {path.name}")
        if record["metric"] != "synchronized whole-step CUDA Graph replay wall latency in ms":
            raise ValueError(f"Wrong metric: {path.name}")
        if first is None:
            first = record
        else:
            for key in SHARED:
                if record[key] != first[key]:
                    raise ValueError(f"Mismatched cross-length {key}: {path.name}")
            config = {k: v for k, v in record["config"].items()
                      if k != "max_position_embeddings"}
            initial = {k: v for k, v in first["config"].items()
                       if k != "max_position_embeddings"}
            if config != initial:
                raise ValueError(f"Mismatched cross-length config: {path.name}")
        rows.append({"kv_length": length, "implementation": implementation,
                     "execution": "cuda_graph_replay", "mean_ms": statistics.mean(samples),
                     "median_ms": statistics.median(samples), "stdev_ms": statistics.stdev(samples),
                     "min_ms": min(samples), "max_ms": max(samples), "repeat": 30, "warmup": 5})

    with (directory / f"results{suffix}.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Decode whole-step CUDA Graph replay", "",
             f"{LABELS[implementation]} · Llama-7B backbone · B1 · FP16 · Q1.",
             "Synchronized wall latency of whole-step replay; lower is better.",
             "Each point: 5 warmups, 30 samples. Visible KV length includes the new token.", "",
             "| Visible KV length | Mean ms | Median ms | Stddev ms |",
             "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['kv_length']} | {row['mean_ms']:.3f} | "
                     f"{row['median_ms']:.3f} | {row['stdev_ms']:.3f} |")
    lines += ["", "All 8 cases passed model, fresh-input, graph completeness and cache checks.",
              "All points use the same GPU, model weights, software and timing protocol.",
              "Prefix setup, compilation, capture, tokenizer, LM head and sampling are excluded.",
              "Fixed-position single-step decode; other backends were not rerun.",
              "Normal compiled-forward measurements are not mixed into this replay curve.", ""]
    (directory / f"results{suffix}.md").write_text("\n".join(lines))
    return rows


def plot(rows, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    implementation = rows[0]["implementation"]
    if (implementation not in LABELS
            or any(row["implementation"] != implementation for row in rows)):
        raise ValueError("Expected one graph implementation per curve")
    output_suffix = "" if implementation == "tileops_bound" else f"_{implementation}"
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot([row["kv_length"] for row in rows], [row["mean_ms"] for row in rows],
            marker="o", label=f"{LABELS[implementation]} · CUDA Graph")
    ax.set_xscale("log", base=2)
    ax.set_xticks(LENGTHS, [str(n) if n < 1024 else f"{n // 1024}K" for n in LENGTHS])
    ax.set_xlabel("Visible KV length (tokens; Q=1)")
    ax.set_ylabel("Whole-step graph replay latency (ms)")
    ax.set_ylim(bottom=0)
    ax.set_title("Decode · Llama-7B backbone · FP16 · batch 1")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(Path(directory) / f"decode_e2e_graph{output_suffix}.{suffix}",
                    bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--implementation", choices=tuple(LABELS), default="tileops_bound")
    args = parser.parse_args()
    rows = summarize(args.directory, args.implementation)
    plot(rows, args.directory)
    print(f"Validated 8 {args.implementation} graph-replay cases; wrote CSV, Markdown, PDF and PNG")


if __name__ == "__main__":
    main()
