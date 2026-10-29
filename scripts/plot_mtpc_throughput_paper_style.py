"""Recreate the original throughput bar style for the measured MTP graph replay."""

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "neptune-internal/scripts/csv_figures.py"
INPUT = ROOT / "profiles_paper_e2e/mtpc_graph_20261027_j3474393/results.csv"
OUTPUT = INPUT.parent / "mtp_e2e_graph_throughput_paper_style"


def mtp_flops():
    """Original model-FLOPs accounting adapted to MTP's rectangular Q/KV shape.

    Linear and MLP work uses Q=16. Causal attention uses the exact lower-right visible
    lengths (65521..65536), rather than the original square-prefill half-matrix shortcut.
    """
    batch, q_len, kv_len, hidden, intermediate = 1, 16, 65536, 4096, 11008
    heads, head_dim, layers = 32, 128, 32
    qkv = 2 * batch * q_len * hidden * (3 * heads * head_dim)
    output_projection = 2 * batch * q_len * (heads * head_dim) * hidden
    mlp = 2 * batch * q_len * (hidden * intermediate + intermediate * hidden)
    visible = sum(range(kv_len - q_len + 1, kv_len + 1))
    attention = 4 * batch * heads * head_dim * visible
    return layers * (qkv + output_projection + mlp + attention)


def load_data(path):
    data = pd.read_csv(path)
    expected = {"nautilus", "tileops", "flex"}
    if (len(data) != 3 or set(data["implementation"]) != expected
            or set(data["execution"]) != {"cuda_graph_replay"}
            or not data["mean_ms"].gt(0).all()):
        raise ValueError("Expected the three complete MTP graph-replay measurements")
    data = data.copy()
    data["seq_len"] = 65536
    data["neptune_tawa"] = data["mean_ms"].map(lambda ms: mtp_flops() / (ms / 1000) / 1e12)
    data["tilelang"] = data.loc[data["implementation"] == "tileops", "neptune_tawa"].iloc[0]
    data["flex_attention"] = data.loc[data["implementation"] == "flex", "neptune_tawa"].iloc[0]
    data["neptune_tawa"] = data.loc[data["implementation"] == "nautilus", "neptune_tawa"].iloc[0]
    return data[["seq_len", "neptune_tawa", "tilelang", "flex_attention"]].drop_duplicates()


def plot(data, output):
    spec = importlib.util.spec_from_file_location("original_throughput_figure", SOURCE)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    original.plot_subplot(ax, data, ["neptune_tawa", "tilelang", "flex_attention"],
                          "LLAMA - FP16 Batch 1 (MTP)")
    ax.set_ylabel("Throughput (TFLOPs/s)", fontsize=18)
    # The original script labels this axis "Sequence Length"; MTP's value is the visible KV length.
    ax.set_xlabel("Sequence Length", fontsize=18)
    ax.set_ylim(bottom=0)
    handles, _ = ax.get_legend_handles_labels()
    fig.legend(handles, ["Nautilus", "TileOPs", "FlexAttn"], loc="upper center",
               bbox_to_anchor=(0.5, 0.995), fontsize=16, frameon=True, ncol=3)
    fig.text(0.5, 0.025, "CUDA Graph replay. FLOPs use the Llama-7B MTP model accounting.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.22, top=0.79)
    output = Path(output)
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), bbox_inches="tight", dpi=300)
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    fig, _ = plot(load_data(args.input), args.output)
    plt.close(fig)
    print(f"Wrote {args.output.with_suffix('.pdf')} and PNG")


if __name__ == "__main__":
    main()
