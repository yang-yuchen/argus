"""Reduce a complete five-variant Nautilus/backend run into a comparison table."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from argus.paper_baselines.workloads import VARIANTS


def summarize(directory, repeats=50, variants=VARIANTS, allow_skipped_nautilus=False):
    rows = []
    for variant in variants:
        results = []
        skip_reason = ""
        for implementation in ("nautilus", "baseline"):
            path = directory / f"{variant}_{implementation}.json"
            result = json.loads(path.read_text())
            if (result.get("variant"), result.get("implementation")) != (variant, implementation):
                raise ValueError(f"Mismatched case in {path}")
            if (implementation == "nautilus" and result.get("skipped")
                    and allow_skipped_nautilus):
                skip_reason = result["skip_reason"]
                results.append(None)
                continue
            samples = result.get("samples_us", [])
            if not result.get("correct") or not result.get("measurement_complete"):
                raise ValueError(f"Incomplete or incorrect measurement: {path}")
            if result["repeats"] != repeats or len(samples) != repeats:
                raise ValueError(f"Expected {repeats} samples: {path}")
            if not all(math.isfinite(x) and x > 0 for x in samples):
                raise ValueError(f"Invalid timings: {path}")
            result["mean_us"] = statistics.mean(samples)
            result["median_us"] = statistics.median(samples)
            results.append(result)
        nautilus, baseline = results
        if nautilus is not None:
            if nautilus["inputs_sha256"] != baseline["inputs_sha256"]:
                raise ValueError(f"Nautilus and baseline used different inputs: {variant}")
            for key in ("host", "gpu", "gpu_uuid", "metric", "l2_flush_bytes"):
                if nautilus.get(key) != baseline.get(key):
                    raise ValueError(f"Different {key} for Nautilus and baseline: {variant}")
        rows.append({
            "variant": variant, "baseline": "tawa" if variant == "prefill" else "tilelang",
            "nautilus_mean_us": nautilus["mean_us"] if nautilus else None,
            "baseline_mean_us": baseline["mean_us"],
            "speedup": baseline["mean_us"] / nautilus["mean_us"] if nautilus else None,
            "nautilus_median_us": nautilus["median_us"] if nautilus else None,
            "baseline_median_us": baseline["median_us"],
            "nautilus_relative_l2": nautilus["relative_l2"] if nautilus else None,
            "baseline_relative_l2": baseline["relative_l2"],
            "repeats": repeats, "nautilus_status": "skipped" if nautilus is None else "measured",
            "skip_reason": skip_reason,
        })
    with (directory / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Variant | Baseline | Nautilus (us) | Baseline (us) | Speedup |",
             "|---|---|---:|---:|---:|"]
    for r in rows:
        nautilus_time = f"{r['nautilus_mean_us']:.3f}" if r["nautilus_mean_us"] else "skipped"
        speedup = f"{r['speedup']:.3f}x" if r["speedup"] else "--"
        lines.append(f"| {r['variant']} | {r['baseline']} | {nautilus_time} | "
                     f"{r['baseline_mean_us']:.3f} | {speedup} |")
    (directory / "results.md").write_text("\n".join(lines) + "\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=VARIANTS)
    parser.add_argument("--allow-skipped-nautilus", action="store_true")
    args = parser.parse_args()
    # Keep pandas/profile analysis in its existing environment, separate from kernel runtimes.
    from scripts.bench_paper_prefill import summarize_profile

    for variant in args.variants:
        for implementation in ("nautilus", "baseline"):
            path = args.directory / f"{variant}_{implementation}.json"
            result = json.loads(path.read_text())
            if not result.get("measurement_complete") and not result.get("skipped"):
                summarize_profile(path.with_suffix(""), args.repeat, result["nvtx_marker"])
    print(json.dumps(summarize(args.directory, args.repeat, args.variants,
                               args.allow_skipped_nautilus), indent=2))
