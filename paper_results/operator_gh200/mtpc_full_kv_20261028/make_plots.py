"""Redraw committed MTPC plot-input CSVs on CPU.

The default inputs are the final optimized measurements; use --original for the
older full-kv sweep kept beside this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
OPTIMIZED = HERE.parent / "mtpc_optimize_20261028" / "plots"
sys.path.insert(0, str(REPO_ROOT / "paper_results/nautilus_mlsys27/performance_sources/scripts"))
import plot_from_csv as plotting  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", action="store_true",
                        help="use this run's older full-kv CSVs")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "paper_results/figures/operator_gh200")
    args = parser.parse_args()
    input_dir = args.input_dir or (HERE if args.original else OPTIMIZED)
    suffix = "-full-kv-20261028" if args.original else "-full-kv-optimized-20261029"
    for q in (2, 4, 8, 16, 32, 64):
        csv_path = input_dir / f"plot_input_Q{q}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        plotting.OPERATOR_DISPLAY_NAMES["MTP_CAUSAL"] = f"Causal MTP (Q={q})"
        frame = plotting.normalize_by_baseline(plotting.load_data(csv_path), "Ours")
        plotting.plot_multi_operator_row(
            frame, ["MTP_CAUSAL"], "Ours", args.output_dir,
            suffix=f"-fp16-tc1-mtpc-Q{q}{suffix}")
        print(f"{csv_path} -> Q{q}")


if __name__ == "__main__":
    main()
