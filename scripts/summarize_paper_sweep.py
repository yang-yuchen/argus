"""Summarize completed length-sweep measurements."""

import argparse
import csv
import json
from pathlib import Path


FIELDS = ("case", "variant", "backend", "correct", "mean_us", "median_us", "min_us",
          "relative_l2", "repeats")


def summarize(directory):
    rows = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "results.json" or path.stat().st_size == 0:
            continue
        result = json.loads(path.read_text())
        if not result.get("measurement_complete"):
            raise ValueError(f"Incomplete measurement: {path}")
        rows.append({key: result.get(key) for key in FIELDS})
    if not rows:
        raise ValueError(f"No measurements in {directory}")
    with (directory / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Case | Backend | Correct | Mean (us) | Median (us) | Min (us) |",
             "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['backend']} | {row['correct']} | "
                     f"{row['mean_us']:.3f} | {row['median_us']:.3f} | {row['min_us']:.3f} |")
    (directory / "results.md").write_text("\n".join(lines) + "\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))
