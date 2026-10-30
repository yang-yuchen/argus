"""Summarize validated Hopper candidates without hiding accuracy or preprocessing scope."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


VARIANTS = ("decode", "mtp", "sparse", "sparse_split")


def checked_mean(record):
    samples = record.get("samples_us", [])
    if len(samples) != 50 or not all(math.isfinite(x) and x > 0 for x in samples):
        raise ValueError(f"Invalid or missing samples: {record['name']}")
    return statistics.mean(samples)


def backend(name):
    for prefix, label in (
        ("tilelang", "TileLang"), ("triton_mslk", "Triton MSLK"),
        ("triton_vllm", "Triton vLLM"), ("flashinfer", "FlashInfer"),
        ("flashattention", "FlashAttention-2"), ("torch_flash", "PyTorch Flash"),
        ("torch_cudnn", "cuDNN"),
    ):
        if name.startswith(prefix):
            return label + (" preplanned" if "preplanned" in name else "")
    raise ValueError(f"Unknown backend: {name}")


def summarize(directory, overrides=None):
    winners, matched_winners, accuracy_rows, status_rows = [], [], [], []
    nautilus_limits = {}
    for variant in VARIANTS:
        cohort = (overrides or {}).get(variant, directory)
        oracle = json.loads((cohort / f"{variant}_nautilus.json").read_text())
        if not oracle["measurement_complete"]:
            raise ValueError(f"Incomplete Nautilus profile: {variant}")
        nautilus = oracle["candidates"][0]
        nautilus["mean_us"] = checked_mean(nautilus)
        n_rmse = nautilus["validation"][0]["accuracy"]["rmse"]
        failed_cases = [x["case"] for x in nautilus["validation"] if not x["accuracy"]["passed"]]
        if failed_cases:
            nautilus_limits[variant] = failed_cases
        base = {
            "variant": variant, "nautilus_mean_us": nautilus["mean_us"],
            "nautilus_benchmark_rmse": n_rmse,
            "nautilus_failed_cases": ",".join(failed_cases),
            "measurement_run": str(cohort),
        }
        families = ("libraries", "tilelang", "triton") if variant in ("decode", "mtp") else (
            "sparse",)
        best, matched = {}, {}
        for family in families:
            path = cohort / f"{variant}_{family}.json"
            result = json.loads(path.read_text())
            if not result["measurement_complete"]:
                raise ValueError(f"Incomplete candidate profile: {path}")
            if result["environment"]["gpu_uuid"] != oracle["environment"]["gpu_uuid"]:
                raise ValueError(f"Different GPUs: {path}")
            for record in result["candidates"]:
                status_rows.append({"variant": variant, "candidate": record["name"],
                                    "status": record["status"], "error": record.get("error", "")})
                for validation in record.get("validation", []):
                    a = validation["accuracy"]
                    accuracy_rows.append({
                        "variant": variant, "candidate": record["name"],
                        "case": validation["case"], "passed": a["passed"], "rmse": a["rmse"],
                        "nautilus_rmse": validation["nautilus_accuracy"]["rmse"],
                        "vs_nautilus_rmse": validation["vs_nautilus"]["rmse"],
                        "relative_l2": a["relative_l2"], "max_abs_error": a["max_abs_error"],
                    })
                if record["status"] != "measured":
                    continue
                record["mean_us"] = checked_mean(record)
                if not all(x["accuracy"]["passed"] for x in record["validation"]):
                    raise ValueError("An inaccurate candidate entered the performance results")
                if not all(x["vs_nautilus"]["passed"] for x in record["validation"]
                           if x["nautilus_accuracy"]["passed"]):
                    raise ValueError("A nonequivalent candidate entered the performance results")
                record["all_cases_equivalent"] = all(
                    x["nautilus_accuracy"]["passed"] for x in record["validation"]
                )
                group = backend(record["name"])
                if group not in best or record["mean_us"] < best[group]["mean_us"]:
                    best[group] = record
                rmse = record["validation"][0]["accuracy"]["rmse"]
                if rmse <= n_rmse and (group not in matched
                                       or record["mean_us"] < matched[group]["mean_us"]):
                    matched[group] = record
        for selected, rows in ((best, winners), (matched, matched_winners)):
            for group, record in selected.items():
                rmse = record["validation"][0]["accuracy"]["rmse"]
                rows.append({
                **base, "backend": group, "candidate": record["name"],
                "mean_us": record["mean_us"], "rmse": rmse,
                "rmse_ratio_to_nautilus": rmse / n_rmse if n_rmse else None,
                "baseline_over_nautilus": record["mean_us"] / nautilus["mean_us"],
                "all_cases_equivalent": record["all_cases_equivalent"],
                "scope": record["source"].get("scope", "prepared inputs"),
                })
    for name, rows in (("best_by_backend", winners), ("accuracy", accuracy_rows),
                       ("candidate_status", status_rows), ("rmse_matched_best", matched_winners)):
        if not rows:
            raise ValueError(f"No {name} rows")
        with (directory / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Hopper baselines: measured candidates passing the existing accuracy gates", "",
             "RMSE uses scripts.error.compute_rmse and FP64 reference. Times are cold-L2 GPU kernel sums.",
             "These H100/SM90-compatible candidates were measured on GH200, not a physical H100.", "",
             "| Variant | Backend | Candidate | Mean us | RMSE | Nautilus us | Baseline/Nautilus |",
             "|---|---|---|---:|---:|---:|---:|"]
    lines += [f"| {r['variant']} | {r['backend']} | {r['candidate']} | {r['mean_us']:.3f} | "
              f"{r['rmse']:.3e} | {r['nautilus_mean_us']:.3f} | "
              f"{r['baseline_over_nautilus']:.3f}x |" for r in winners]
    lines += ["", "Ratios use the same-GPU Nautilus measurement from each variant's cohort.",
              "Preplanned FlashInfer excludes CSR compaction and is reported separately.",
              "All-cases equivalence and any Nautilus failed probes are recorded in best_by_backend.csv."]
    lines += ["", "## Accuracy qualifications", "",
              "Passing the existing tolerance does not mean identical RMSE.",
              "rmse_matched_best.csv additionally requires benchmark RMSE <= Nautilus RMSE."]
    for variant, failed in nautilus_limits.items():
        lines.append(f"- Nautilus {variant} failed: {', '.join(failed)}. "
                     "Its benchmark timing is valid for the nonempty-row benchmark input only; "
                     "full-domain accuracy equivalence is not established.")
    save = {"cohorts": {v: str((overrides or {}).get(v, directory)) for v in VARIANTS},
            "nautilus_failed_cases": nautilus_limits,
            "rmse_metric": "scripts.error.compute_rmse against FP64 reference",
            "rmse_matching_rule": "benchmark RMSE <= benchmark Nautilus RMSE"}
    (directory / "summary_metadata.json").write_text(json.dumps(save, indent=2) + "\n")
    (directory / "results.md").write_text("\n".join(lines) + "\n")
    return winners


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--mtp-run", type=Path, help="Complete same-GPU MTP remeasurement cohort")
    args = parser.parse_args()
    summarize(args.directory, {"mtp": args.mtp_run} if args.mtp_run else None)
