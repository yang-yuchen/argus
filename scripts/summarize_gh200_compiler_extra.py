"""Reduce the extra Helion/Tri-Daō Triton decode profiles."""

import argparse
import csv
import json
from pathlib import Path

from scripts.summarize_gh200_compare import merge_with_nautilus, parse_profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--nautilus-csv", type=Path,
                        default=Path("profiles_paper_kernels/sweep_20261026_merged/results.csv"))
    args = parser.parse_args()
    rows = parse_profiles(args.directory, 50, args.nautilus_csv.parent.parent)
    rows = merge_with_nautilus(rows, args.nautilus_csv)
    rows.sort(key=lambda row: (int(row["case"].split("_L")[1]), row["implementation"]))
    fields = ["case", "implementation", "mean_us", "median_us", "min_us",
              "nautilus_mean_us", "baseline_over_nautilus", "repeats"]
    with (args.directory / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (args.directory / "comparison_metadata.json").write_text(json.dumps({
        "environment": "/work/nvme/beqg/yucheny8/envs/argus-gh200-latest",
        "implementations": sorted({r["implementation"] for r in rows}),
        "cases": sorted({r["case"] for r in rows}), "repeats": 50,
    }, indent=2) + "\n")
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
