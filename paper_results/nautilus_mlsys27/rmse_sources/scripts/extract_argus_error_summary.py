#!/usr/bin/env python3
"""Extract a Table-5-style summary from Argus RMSE JSON artifacts.

This script consumes the JSON written by `/u/yucheny8/argus/scripts/error.py`
and produces a wide summary table with one row per operator and three metrics
per method:

* RMS   = mean(sample_rmse)
* 90th% = upper-tail 90th percentile of sample RMSE
* 99th% = upper-tail 99th percentile of sample RMSE

Important: the source JSON only stores per-sample RMSE values, so this script
does NOT reconstruct elementwise absolute-error percentiles from the paper.
It summarizes the RMSE distribution that `error.py` already saved.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_INPUTS = [
    Path("/u/yucheny8/argus/logs/error/qwen_rmse_kde_3curves_1000_global.json"),
    Path("/u/yucheny8/argus/logs/error/qwen_rmse_kde_3curves_1000_causal_seed1.json"),
    Path("/u/yucheny8/argus/logs/error/qwen_rmse_kde_3curves_1000_gqa.json"),
]

METHOD_ORDER = [
    ("Nautilus", "Neptune"),
    ("FlashAttention-2", "Tri-Dao Attention"),
    ("FlexAttn", "Flex Attention"),
]

OPERATOR_LABELS = {
    "prefill_global": "PF Global",
    "prefill_causal": "PF Causal",
    "prefill_gqa": "PF GQA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Argus RMSE JSON files into a Table-5-style table."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Paths to RMSE JSON files from argus/scripts/error.py",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output table format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=1,
        help="Digits after decimal in scientific notation",
    )
    return parser.parse_args()


def sci(value: float | None, precision: int) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{precision}e}"


def upper_percentile(values: np.ndarray, pct: float) -> float:
    """Return the upper-tail percentile explicitly from the sorted values."""
    if values.size == 0:
        return float("nan")
    sorted_vals = np.sort(values)
    idx = int(math.ceil((pct / 100.0) * values.size)) - 1
    idx = min(max(idx, 0), values.size - 1)
    return float(sorted_vals[idx])


def summarize_values(values: list[float], stats: dict[str, float | int]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    rms = float(stats["mean"])
    p90 = upper_percentile(arr, 90)
    p99 = upper_percentile(arr, 99)
    return rms, p90, p99


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        operator_name = payload["operator_name"]
        methods = payload["methods"]
        row: dict[str, str] = {"Variant": OPERATOR_LABELS.get(operator_name, operator_name)}

        for source_name, display_name in METHOD_ORDER:
            method_payload = methods.get(source_name)
            if method_payload is None:
                row[f"{display_name} RMS"] = "-"
                row[f"{display_name} 90th %"] = "-"
                row[f"{display_name} 99th %"] = "-"
                continue

            values = method_payload["values"]
            stats = method_payload["stats"]
            rms, p90, p99 = summarize_values(values, stats)
            row[f"{display_name} RMS"] = rms
            row[f"{display_name} 90th %"] = p90
            row[f"{display_name} 99th %"] = p99

        rows.append(row)

    rows.sort(key=lambda row: list(OPERATOR_LABELS.values()).index(row["Variant"]))
    return rows


def to_markdown(rows: list[dict[str, str]], precision: int) -> str:
    columns = ["Variant"]
    for _, display_name in METHOD_ORDER:
        columns.extend(
            [f"{display_name} RMS", f"{display_name} 90th %", f"{display_name} 99th %"]
        )

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(sci(value, precision))
            else:
                cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_csv(rows: list[dict[str, str]], precision: int) -> str:
    columns = ["Variant"]
    for _, display_name in METHOD_ORDER:
        columns.extend(
            [f"{display_name} RMS", f"{display_name} 90th %", f"{display_name} 99th %"]
        )

    out_lines = [",".join(columns)]
    for row in rows:
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(sci(value, precision))
            else:
                cells.append(value)
        out_lines.append(",".join(cells))
    return "\n".join(out_lines)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.inputs)
    text = to_markdown(rows, args.precision) if args.format == "markdown" else to_csv(rows, args.precision)

    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
