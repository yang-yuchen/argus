"""Fail-closed reduction of GH200 nsys profiles into a latency table.

The profile loader already sums kernels within one NVTX repeat and drops the
warmup marker. This script adds the checks the historical table generator did
not have: every requested shape, batch and implementation must be present and
must contain exactly ``--repeat`` samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from argus import plot as base_plot
from argus.stats import nsys


DEFAULT_OPERATORS = ("prefill_global", "prefill_causal", "prefill_gqa")
DEFAULT_BATCHES = (1, 8)
DEFAULT_SEQS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x)


def load_profiles(profile_dir: Path, operators: tuple[str, ...], repeats: int):
    paths = []
    for operator in operators:
        paths.extend(sorted(profile_dir.glob(f"{operator}-*.nsys-rep")))
    if not paths:
        raise ValueError(f"No .nsys-rep files found in {profile_dir}")
    for path in paths:
        raw = nsys.nsys_to_dataframe(path)
        timed = raw[raw["repeatIdx"] != "warmup"]
        for implementation, group in timed.groupby("implementation"):
            indices = set(group["repeatIdx"])
            expected = {str(i) for i in range(repeats)}
            if indices != expected:
                raise ValueError(
                    f"{path.name}/{implementation}: expected repeat indices "
                    f"0..{repeats - 1}, got {sorted(indices)}"
                )
    return base_plot.load_profile(paths, "GH200", True)


def summarize(
    profile_dir: Path,
    output: Path,
    metadata: Path,
    operators: tuple[str, ...],
    batches: tuple[int, ...],
    seq_lens: tuple[int, ...],
    required: tuple[str, ...],
    repeats: int,
) -> pd.DataFrame:
    df = load_profiles(profile_dir, operators, repeats)
    df = df[df["op_name"].isin(operators)].copy()
    if df.empty:
        raise ValueError("Profiles contain none of the requested operators")

    df["batch_size"] = df["shape"].map(lambda shape: shape.shape[0])
    df["seq_len"] = df["shape"].map(lambda shape: shape.shape[4])
    for _, item in df.iterrows():
        heads = (64, 8) if item["op_name"] == "prefill_gqa" else (32, 32)
        expected = (item["batch_size"], *heads, item["seq_len"], item["seq_len"], 128)
        if item["shape"].shape != expected:
            raise ValueError(f"Unexpected full shape for {item['op_name']}: {item['shape']}")
    expected_keys = {
        (operator, batch, seq) for operator in operators for batch in batches for seq in seq_lens
    }
    actual_keys = set(zip(df["op_name"], df["batch_size"], df["seq_len"], strict=True))
    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    if missing_keys or extra_keys:
        raise ValueError(f"Shape coverage mismatch: missing={missing_keys[:8]}, extra={extra_keys[:8]}")

    counts = df.groupby(["op_name", "batch_size", "seq_len", "implementation"]).size()
    short = {str(index): int(count) for index, count in counts.items() if count != 1}
    if short:
        raise ValueError(f"Expected one reduced row per implementation/key, got {short}")
    present = set(df["implementation"])
    missing_impl = sorted(set(required) - present)
    if missing_impl:
        raise ValueError(f"Missing required implementations: {missing_impl}")
    actual_impl_keys = set(
        zip(df["op_name"], df["batch_size"], df["seq_len"], df["implementation"], strict=True)
    )
    missing_impl_keys = sorted(
        (operator, batch, seq, implementation)
        for operator, batch, seq in expected_keys
        for implementation in required
        if (operator, batch, seq, implementation) not in actual_impl_keys
    )
    if missing_impl_keys:
        raise ValueError(f"Implementation coverage mismatch: missing={missing_impl_keys[:8]}")

    rows = []
    sample_counts: dict[str, int] = {}
    for (operator, batch, seq), group in df.groupby(
        ["op_name", "batch_size", "seq_len"], sort=False
    ):
        row = {
            "operator": base_plot.ALL_OPS[operator].name,
            "batch_size": int(batch),
            "seq_len": int(seq),
        }
        for _, item in group.iterrows():
            samples = item["latency_us"].raw_data
            if len(samples) != repeats:
                raise ValueError(
                    f"{operator}/{batch}/{seq}/{item['implementation']}: "
                    f"expected {repeats} repeats, got {len(samples)}"
                )
            key = f"{operator}|{batch}|{seq}|{item['implementation']}"
            sample_counts[key] = len(samples)
            row[item["implementation"]] = item["latency_us"].mean
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(["operator", "batch_size", "seq_len"])
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, float_format="%.3f")
    metadata.write_text(json.dumps({
        "profile_dir": str(profile_dir), "operators": operators, "batches": batches,
        "seq_lens": seq_lens, "required_implementations": required, "repeat": repeats,
        "rows": len(result), "implementations": sorted(present), "sample_counts": sample_counts,
        "metric": "sum of GPU kernel durations per NVTX repeat, then arithmetic mean (us)",
        "warmup_included": False,
    }, indent=2) + "\n")
    return result


def plot_results(result: pd.DataFrame, output_dir: Path, methods: tuple[str, ...], baseline: str):
    if baseline not in result.columns:
        raise ValueError(f"Plot baseline is missing: {baseline}")
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {method: base_plot.ALL_COLORS[i % len(base_plot.ALL_COLORS)]
              for i, method in enumerate(methods)}
    operators = {base_plot.ALL_OPS[op].name: op for op in DEFAULT_OPERATORS}
    for (operator, batch), group in result.groupby(["operator", "batch_size"], sort=False):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        group = group.sort_values("seq_len")
        for method in methods:
            ax.plot(group["seq_len"], group[baseline] / group[method], marker="o",
                    label=method, color=colors[method])
        ax.set_xscale("log", base=2)
        ax.set_xticks(group["seq_len"])
        ax.set_xticklabels([str(value) for value in group["seq_len"]])
        ax.set_xlabel("Sequence length")
        ax.set_ylabel(f"Relative performance ({baseline} = 1)")
        ax.set_title(f"{operator} — GH200, batch {batch}")
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, ncol=2)
        fig.tight_layout()
        stem = output_dir / f"gh200_{operators[operator]}_b{batch}_latest"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--operators", nargs="+", default=DEFAULT_OPERATORS)
    parser.add_argument("--batches", type=parse_ints, default=DEFAULT_BATCHES)
    parser.add_argument("--seq-lens", type=parse_ints, default=DEFAULT_SEQS)
    parser.add_argument("--required-implementation", action="append", required=True)
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--baseline", default="cuDNN")
    args = parser.parse_args()
    result = summarize(
        args.profile_dir, args.output, args.metadata, tuple(args.operators), args.batches,
        args.seq_lens, tuple(args.required_implementation), args.repeat,
    )
    if args.plot_dir:
        plot_results(result, args.plot_dir, tuple(args.required_implementation), args.baseline)


if __name__ == "__main__":
    main()
