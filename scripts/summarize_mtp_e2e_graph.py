"""Validate and plot the three-backend, whole-step causal MTP replay experiment."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from argus.paper_baselines.e2e_mtp_baselines import TILEOPS_COMMIT
from argus.paper_baselines.workloads import export_path
from scripts.summarize_decode_e2e_graph import valid_error


IMPLEMENTATIONS = ("nautilus", "tileops", "flex")
ATTENTION_KERNELS = {"nautilus": 512, "tileops": 512, "flex": 768}
SHARED = ("config", "weights_sha256", "inputs_sha256", "gpu_uuid", "host", "torch", "cuda",
          "transformers", "cache_implementation", "metric", "execution", "repeat", "warmup",
          "seed", "export_directory", "kernel_sha256", "scope")


def summarize(directory):
    """Require matched measurements and graph proof before writing either output."""
    directory = Path(directory)
    current = export_path("mtpc", 65536).resolve()
    digest = hashlib.sha256((current / "kernel.so").read_bytes()).hexdigest()
    rows = []
    first = None
    for implementation in IMPLEMENTATIONS:
        path = directory / f"mtpc_{implementation}.json"
        record = json.loads(path.read_text())
        if (record.get("variant") != "mtpc" or record.get("implementation") != implementation
                or record.get("execution") != "cuda_graph_replay"):
            raise ValueError(f"Wrong case/execution: {path.name}")
        samples = record["samples_ms"]
        if (record["correct"] is not True or record["measurement_complete"] is not True
                or record["repeat"] != 30 or record["warmup"] != 5 or len(samples) != 30
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                           and math.isfinite(x) and x > 0 for x in samples)):
            raise ValueError(f"Invalid samples/protocol: {path.name}")
        errors = record["attention_relative_l2"]
        if (not isinstance(errors, list) or len(errors) != 256
                or not all(valid_error(x) for x in [record["relative_l2"],
                           record["compiled_relative_l2"], *errors])):
            raise ValueError(f"Invalid model/attention correctness: {path.name}")
        proof = record.get("graph_validation") or {}
        fresh = proof.get("changed_token_relative_l2")
        kernels = proof.get("kernel_count")
        prefix_change = proof.get("prefix_v_output_change_relative_l2")
        if (not isinstance(fresh, list) or len(fresh) != 2
                or not all(valid_error(x) for x in fresh)
                or not valid_error(proof.get("changed_prefix_v_relative_l2"))
                or not valid_error(proof.get("restored_relative_l2"))
                or proof.get("shared_buffers_poisoned_before_replay") is not True
                or not isinstance(prefix_change, (int, float)) or isinstance(prefix_change, bool)
                or not math.isfinite(prefix_change) or prefix_change <= 1e-4
                or proof.get("all_kernels_in_graph") is not True
                or proof.get("attention_group_count") != 256
                or type(proof.get("attention_kernel_count")) is not int
                or proof.get("attention_kernel_count") < 256
                or proof.get("attention_kernel_count") % 256
                or proof.get("attention_kernel_count") > 768
                or proof.get("kv_update_kernel_count") != 64
                or type(kernels) is not int or kernels <= proof["attention_kernel_count"] + 64):
            raise ValueError(f"Invalid capture/fresh-input proof: {path.name}")
        if (record["cache_implementation"] != "persistent_native"
                or record["cache_layout"] != ("bshd" if implementation == "tileops" else "bhsd")
                or record["persistent_kv_bindings"] != 256
                or record["full_kv_staging_bytes_per_forward"] != 0):
            raise ValueError(f"Invalid persistent KV cache: {path.name}")
        source = record.get("baseline_source") or {}
        if implementation == "tileops":
            expected_source = {
                "repository": "https://github.com/tile-ai/TileOPs", "commit": TILEOPS_COMMIT,
                "api": "GQADenseFwdOp(is_causal=True,target=BUILTIN)", "layout": "BSHD",
                "selected_kernel": "GQADensePersistentWSKernel", "kernel_config": None,
                "shape": {"q": [1, 16, 4, 128], "kv": [1, 65536, 4, 128]},
            }
            if any(source.get(key) != value for key, value in expected_source.items()):
                raise ValueError(f"Wrong TileOPs source/configuration: {path.name}")
        elif implementation == "nautilus":
            if record.get("baseline_source") is not None:
                raise ValueError(f"Wrong Nautilus export source: {path.name}")
        elif (source.get("repository") != "https://github.com/pytorch/pytorch"
              or source.get("api") != "torch.nn.attention.flex_attention"
              or source.get("mask") != "q_idx + (KV_LEN-Q_LEN) >= kv_idx"
              or source.get("configuration") != (
                  "fullgraph=True, max-autotune-no-cudagraphs, inference_mode"
              )):
            raise ValueError(f"Wrong FlexAttention source/configuration: {path.name}")
        if Path(record["export_directory"]).resolve() != current:
            raise ValueError(f"Wrong export_directory: {path.name}")
        if record["kernel_sha256"] != digest:
            raise ValueError(f"Wrong kernel_sha256: {path.name}")
        config = record["config"]
        if any(config.get(key) != value for key, value in {
            "num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32,
            "num_key_value_heads": 32, "head_dim": 128, "max_position_embeddings": 65536,
        }.items()):
            raise ValueError(f"Wrong model configuration: {path.name}")
        for key in ("weights_sha256", "inputs_sha256"):
            value = record[key]
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)):
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
                    raise ValueError(f"Mismatched {key}: {path.name}")
        mean = statistics.mean(samples)
        rows.append({"implementation": implementation, "execution": "cuda_graph_replay",
                     "mean_ms": mean, "median_ms": statistics.median(samples),
                     "stdev_ms": statistics.stdev(samples), "min_ms": min(samples),
                     "max_ms": max(samples), "baseline_over_nautilus":
                     mean / statistics.mean(first["samples_ms"]), "repeat": 30, "warmup": 5})

    with (directory / "results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# MTP whole-step CUDA Graph replay", "",
             "Llama-7B backbone · B1 · FP16 · Q16 · visible KV65536 · 32 layers.",
             "Same GPU, model weights, inputs and timing protocol for all three implementations.",
             "Each point: 5 replay warmups, 30 synchronized wall samples. Lower is better.", "",
             "| Implementation | Mean ms | Median ms | Stddev ms | Latency / Nautilus |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['implementation']} | {row['mean_ms']:.3f} | "
                     f"{row['median_ms']:.3f} | {row['stdev_ms']:.3f} | "
                     f"{row['baseline_over_nautilus']:.3f}× |")
    lines += ["", "All three cases passed model, fresh-input, graph completeness "
              "and cache checks.",
              "The full step includes new-token KV writes, all 256 attention groups and merges.",
              "Prefix setup, compilation, capture, tokenizer, LM head and sampling are excluded.",
              "Fixed-position candidate verification; draft generation "
              "and acceptance are excluded.",
              "Normal compiled-forward results are not mixed into this replay comparison.", ""]
    (directory / "results.md").write_text("\n".join(lines))
    return rows


def plot(rows, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(range(3), [row["mean_ms"] for row in rows], width=0.65,
           color=["#2ca02c", "#9467bd", "#7f7f7f"], zorder=3)
    ax.set_xticks(range(3), ["Nautilus", "TileOPs", "FlexAttn"], fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel("Latency (ms)", fontsize=18)
    ax.set_title("LLAMA - FP16 Batch 1 (MTP)", fontsize=20, pad=10)
    ax.set_xlabel("Q=16 · Visible KV Length=65536", fontsize=15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(Path(directory) / f"mtp_e2e_graph.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = summarize(args.directory)
    plot(rows, args.directory)
    print("Validated three MTP graph-replay cases; wrote CSV, Markdown, PDF and PNG")


if __name__ == "__main__":
    main()
