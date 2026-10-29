"""Plot one decode panel using the original throughput figure's grouped-bar style."""

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.summarize_mtpc_graph_sweep import model_flops

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "neptune-internal/scripts/csv_figures.py"
INPUT = ROOT / "profiles_paper_e2e/decode_graph_baselines_20261027_j3474109/comparison.csv"
OUTPUT = INPUT.parent / "decode_e2e_graph_throughput_paper_style"
LENGTHS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
METHODS = {"nautilus": "neptune_tawa", "tileops_bound": "tilelang", "flex": "flex_attention"}


def load_data(path):
    data = pd.read_csv(path)
    if (len(data) != 24 or set(data["implementation"]) != set(METHODS)
            or set(data["execution"]) != {"cuda_graph_replay"}
            or data.duplicated(["kv_length", "implementation"]).any()
            or not np.isfinite(data["mean_ms"]).all() or (data["mean_ms"] <= 0).any()):
        raise ValueError("Expected24 unique, finite graph-replay measurements")
    data["model_flops"] = data["kv_length"].map(lambda length: model_flops(length, query_length=1))
    data["throughput_tflops"] = data["model_flops"] / data["mean_ms"] / 1e9
    panel = data.pivot(index="kv_length", columns="implementation", values="throughput_tflops")
    if tuple(panel.index) != LENGTHS or panel.isna().any().any():
        raise ValueError("Expected all8 KV lengths for each implementation")
    return panel.rename(columns=METHODS).reset_index().rename(columns={"kv_length": "seq_len"})


def plot(data, output):
    spec = importlib.util.spec_from_file_location("original_throughput_figure", SOURCE)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    original.plot_subplot(ax, data, list(METHODS.values()), "LLAMA - FP16 Batch 1 (Decode)")
    ax.set_xlabel("Sequence Length", fontsize=18)
    ax.set_ylim(bottom=0)
    handles, _ = ax.get_legend_handles_labels()
    fig.legend(handles, ["Nautilus", "TileOPs", "FlexAttn"], loc="upper center",
               bbox_to_anchor=(0.5, 0.995), fontsize=16, frameon=True, ncol=3)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.18, top=0.79)
    output = Path(output)
    data.to_csv(output.with_suffix(".csv"), index=False)
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), bbox_inches="tight", dpi=300)
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Output path without suffix")
    args = parser.parse_args()
    fig, _ = plot(load_data(args.input), args.output)
    plt.close(fig)
    print(f"Wrote {args.output.with_suffix('.pdf')} and PNG")


if __name__ == "__main__":
    main()
