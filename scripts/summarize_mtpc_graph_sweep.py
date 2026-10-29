"""Validate all MTP graph sizes and plot the original model-throughput bar style."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

from argus.paper_baselines.e2e_mtp_baselines import TILEOPS_COMMIT
from argus.paper_baselines.workloads import export_path
from scripts.summarize_decode_e2e_graph import valid_error


LENGTHS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
IMPLEMENTATIONS = ("nautilus", "tileops", "flex")
SHARED = ("weights_sha256", "gpu_uuid", "host", "torch", "cuda", "transformers", "metric",
          "scope", "execution", "repeat", "warmup", "seed", "cache_implementation")
SOURCE = Path(__file__).resolve().parents[2] / "neptune-internal/scripts/csv_figures.py"


def model_flops(length, query_length=16):
    """Original full-model FLOPs convention with lower-right causal cached attention."""
    if not 0 < query_length <= length:
        raise ValueError("Query length must be positive and fit the visible KV length")
    q, h, intermediate, heads, dim, layers = query_length, 4096, 11008, 32, 128, 32
    projections = 2 * q * h * (3 * heads * dim) + 2 * q * (heads * dim) * h
    mlp = 2 * q * (h * intermediate + intermediate * h)
    pairs = q * length - q * (q - 1) // 2
    return layers * (projections + mlp + 4 * heads * dim * pairs)


def summarize(directory):
    """Refuse incomplete, mismatched or incorrectly captured inputs before writing results."""
    directory = Path(directory)
    rows, first, backend_packages = [], None, {}
    for length in LENGTHS:
        current = export_path("mtpc", length).resolve()
        digest = hashlib.sha256((current / "kernel.so").read_bytes()).hexdigest()
        per_length = None
        for implementation in IMPLEMENTATIONS:
            path = directory / f"mtpc_L{length}_{implementation}.json"
            record = json.loads(path.read_text())
            if (record.get("variant") != "mtpc" or record.get("implementation") != implementation
                    or record.get("execution") != "cuda_graph_replay"):
                raise ValueError(f"Wrong case/execution: {path.name}")
            samples = record["samples_ms"]
            if (record["correct"] is not True or record["measurement_complete"] is not True
                    or record["repeat"] != 30 or record["warmup"] != 5 or len(samples) != 30
                    or not all(type(x) in (int, float) and math.isfinite(x) and x > 0
                               for x in samples)):
                raise ValueError(f"Invalid samples/protocol: {path.name}")
            errors = record["attention_relative_l2"]
            if (not isinstance(errors, list) or len(errors) != 256
                    or not all(valid_error(x) for x in [record["relative_l2"],
                               record["compiled_relative_l2"], *errors])):
                raise ValueError(f"Invalid correctness: {path.name}")
            proof = record.get("graph_validation") or {}
            fresh = proof.get("changed_token_relative_l2")
            change, kernels = (proof.get("prefix_v_output_change_relative_l2"),
                               proof.get("kernel_count"))
            attention = 768 if implementation == "flex" else (
                256 if implementation == "tileops" and length == 256 else 512
            )
            if (not isinstance(fresh, list) or len(fresh) != 2
                    or not all(valid_error(x) for x in fresh)
                    or not valid_error(proof.get("changed_prefix_v_relative_l2"))
                    or not valid_error(proof.get("restored_relative_l2"))
                    or proof.get("shared_buffers_poisoned_before_replay") is not True
                    or type(change) not in (int, float) or not math.isfinite(change)
                    or change <= 1e-4 or proof.get("all_kernels_in_graph") is not True
                    or proof.get("attention_group_count") != 256
                    or proof.get("attention_kernel_count") != attention
                    or proof.get("kv_update_kernel_count") != 64
                    or type(kernels) is not int or kernels <= attention + 64):
                raise ValueError(f"Invalid capture/fresh-input proof: {path.name}")
            if (record["cache_implementation"] != "persistent_native"
                    or record["cache_layout"] != ("bshd" if implementation == "tileops"
                                                 else "bhsd")
                    or record["persistent_kv_bindings"] != 256
                    or record["full_kv_staging_bytes_per_forward"] != 0):
                raise ValueError(f"Invalid cache: {path.name}")
            source = record.get("baseline_source") or {}
            if implementation == "tileops":
                expected = {"repository": "https://github.com/tile-ai/TileOPs",
                            "commit": TILEOPS_COMMIT,
                            "api": "GQADenseFwdOp(is_causal=True,target=BUILTIN)",
                            "layout": "BSHD", "selected_kernel": "GQADensePersistentWSKernel",
                            "kernel_config": None,
                            "shape": {"q": [1, 16, 4, 128], "kv": [1, length, 4, 128]}}
                if any(source.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Wrong TileOPs source: {path.name}")
            elif implementation == "nautilus":
                if record.get("baseline_source") is not None:
                    raise ValueError(f"Wrong Nautilus source: {path.name}")
            elif (source.get("repository") != "https://github.com/pytorch/pytorch"
                  or source.get("api") != "torch.nn.attention.flex_attention"
                  or source.get("mask") != "q_idx + (KV_LEN-Q_LEN) >= kv_idx"
                  or source.get("configuration") != (
                      "fullgraph=True, max-autotune-no-cudagraphs, inference_mode"
                  )):
                raise ValueError(f"Wrong Flex source: {path.name}")
            if Path(record["export_directory"]).resolve() != current:
                raise ValueError(f"Wrong export_directory: {path.name}")
            if record["kernel_sha256"] != digest:
                raise ValueError(f"Wrong kernel_sha256: {path.name}")
            config = record["config"]
            expected = {"num_hidden_layers": 32, "hidden_size": 4096,
                        "intermediate_size": 11008, "num_attention_heads": 32,
                        "num_key_value_heads": 32, "head_dim": 128,
                        "max_position_embeddings": max(4096, length)}
            if any(config.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Wrong model config: {path.name}")
            for key in ("weights_sha256", "inputs_sha256"):
                value = record[key]
                if (not isinstance(value, str) or len(value) != 64
                        or any(c not in "0123456789abcdef" for c in value)):
                    raise ValueError(f"Invalid {key}: {path.name}")
            if record["metric"] != "synchronized whole-step CUDA Graph replay wall latency in ms":
                raise ValueError(f"Wrong metric: {path.name}")
            for key in ("gpu_uuid", "host", "torch", "cuda", "transformers", "scope"):
                if not isinstance(record[key], str) or not record[key]:
                    raise ValueError(f"Missing {key}: {path.name}")
            if first is None:
                first = record
            for key in SHARED:
                if record[key] != first[key]:
                    raise ValueError(f"Mismatched {key}: {path.name}")
            if {k: v for k, v in config.items() if k != "max_position_embeddings"} != {
                k: v for k, v in first["config"].items() if k != "max_position_embeddings"
            }:
                raise ValueError(f"Mismatched cross-length config: {path.name}")
            if per_length is None:
                per_length = record
            if record["inputs_sha256"] != per_length["inputs_sha256"]:
                raise ValueError(f"Mismatched inputs_sha256: {path.name}")
            packages = record.get("packages")
            if not isinstance(packages, dict) or not packages:
                raise ValueError(f"Missing packages: {path.name}")
            if implementation not in backend_packages:
                backend_packages[implementation] = packages
            if packages != backend_packages[implementation]:
                raise ValueError(f"Mismatched backend packages: {path.name}")
            mean = statistics.mean(samples)
            flops = model_flops(length)
            rows.append({"kv_length": length, "implementation": implementation,
                         "execution": "cuda_graph_replay", "mean_ms": mean,
                         "median_ms": statistics.median(samples),
                         "stdev_ms": statistics.stdev(samples), "model_flops": flops,
                         "throughput_tflops": flops / mean / 1e9, "repeat": 30, "warmup": 5})

    with (directory / "results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# MTP whole-step graph replay: all KV lengths", "",
             "Llama-7B backbone, B1, FP16, Q16, same GPU and model weights.",
             "All27points passed correctness, fresh-input, capture and cache checks.", "",
             "| KV length | Nautilus TFLOPs/s | TileOPs TFLOPs/s | FlexAttn TFLOPs/s |",
             "|---|---:|---:|---:|"]
    for index in range(0, len(rows), 3):
        n, t, f = rows[index:index + 3]
        lines.append(f"| {n['kv_length']} | {n['throughput_tflops']:.3f} | "
                     f"{t['throughput_tflops']:.3f} | {f['throughput_tflops']:.3f} |")
    lines += ["", "Each point has5replay warmups and30synchronized wall samples.",
              "TFLOPs/s = theoretical model FLOPs / mean replay seconds /1e12.",
              "Original model-FLOPs convention: QKV/output projections and two MLP linears;",
              "Q=16 for token-local work, and Q*KV-Q*(Q-1)/2 visible causal attention pairs.",
              "The64K numerator is710754172928FLOPs. This adapts the original square-prefill",
              "formula to the measured rectangular MTP step; it does not count prefix work.",
              "Prefix setup, compilation, capture, LM head, draft and acceptance are excluded.", ""]
    (directory / "results.md").write_text("\n".join(lines))
    return rows


def plot(rows, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    spec = importlib.util.spec_from_file_location("original_throughput", SOURCE)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    keys = {"nautilus": "neptune_tawa", "tileops": "tilelang", "flex": "flex_attention"}
    data = pd.DataFrame(rows).pivot(index="kv_length", columns="implementation",
                                    values="throughput_tflops").rename(columns=keys)
    data = data.reset_index().rename(columns={"kv_length": "seq_len"})
    data.to_csv(Path(directory) / "throughput_plot_input.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    original.plot_subplot(ax, data, list(keys.values()), "LLAMA - FP16 Batch 1 (MTP)")
    ax.set_xlabel("Sequence Length", fontsize=18)
    ax.set_ylim(bottom=0)
    handles, _ = ax.get_legend_handles_labels()
    fig.legend(handles, ["Nautilus", "TileOPs", "FlexAttn"], loc="upper center",
               bbox_to_anchor=(0.5, 0.995), fontsize=16, frameon=True, ncol=3)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.18, top=0.79)
    for suffix in ("pdf", "png"):
        fig.savefig(Path(directory) / f"mtp_e2e_graph_throughput.{suffix}",
                    bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    plot(summarize(args.directory), args.directory)
    print("Validated27MTP graph points; wrote results, throughput plot input, PDF and PNG")


if __name__ == "__main__":
    main()
