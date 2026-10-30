"""Redraw committed FP8 sweep CSVs on CPU."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "paper_results/nautilus_mlsys27/performance_sources/scripts"))
import plot_from_csv as plotting  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-official", action="store_true",
                        help="plot the measured best-official envelope CSVs")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "paper_results/figures/operator_gh200")
    args = parser.parse_args()
    input_dir = args.input_dir or (HERE if args.best_official else HERE / "sweep_4e06b84_3484820")
    suffix = "-best-official-4e06b84" if args.best_official else "-4e06b84"
    for name, operator, title in (
        ("decode8", "DC_CAUSAL", "FP8 Causal Attention (Decode)"),
        ("mtpc8", "MTP_CAUSAL", "FP8 Causal MTP (Q=16)"),
    ):
        stem = f"plot_input_{name}{'_best_official' if args.best_official else ''}.csv"
        csv_path = input_dir / stem
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        plotting.OPERATOR_DISPLAY_NAMES[operator] = title
        frame = plotting.normalize_by_baseline(plotting.load_data(csv_path), "Ours")
        plotting.plot_multi_operator_row(
            frame, [operator], "Ours", args.output_dir,
            suffix=f"-fp8-tc1-{name}{suffix}")
        print(f"{csv_path} -> {args.output_dir / ('gh200-fp8-tc1-' + name + suffix + '.pdf')}")


if __name__ == "__main__":
    main()
