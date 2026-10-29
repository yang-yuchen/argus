"""Summarize modern GH200 baseline profiles against the exported-kernel sweep."""

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


PROFILE_RE = re.compile(
    r"(prefill_causal|decode_causal)-1,32,32,(\d+),(\d+),128\.nsys-rep"
)
DISPLAY_NAMES = {"flex": "FlexAttention", "torch(flash)": "PyTorch Flash",
                 "torch(cudnn)": "cuDNN SDPA", "tri-dao": "FlashAttention-2",
                 "tilelang": "TileLang", "neptune_tilelang": "Local Neptune TileLang",
                 "tawa(tma)": "TAWA", "flashinfer": "FlashInfer",
                 "tawa(tma_decode)": "TAWA decode"}

CAVEAT = (
    "Baselines use argus-gh200-latest; Nautilus uses the prior Neptune TVM/Triton runtime "
    "and measurements from different GH200 nodes. Ratios are cross-run comparisons. "
    "Times are mean sums of GPU kernel durations per cold-L2 NVTX repeat; compilation, "
    "setup and CPU launch time are excluded."
)


def profile_case(filename):
    match = PROFILE_RE.fullmatch(filename)
    if match is None:
        raise ValueError(f"Unrecognized comparison profile: {filename}")
    kind, q_length, kv_length = match.group(1), int(match.group(2)), int(match.group(3))
    if q_length != (1 if kind == "decode_causal" else kv_length):
        raise ValueError(f"Unexpected Q/KV lengths: {filename}")
    prefix = "decode" if kind == "decode_causal" else "prefillc"
    return f"{prefix}_L{kv_length}"


def reduce_frame(frame, repeats, case):
    rows = []
    frame = frame[frame["repeatIdx"] != "warmup"]
    for implementation, group in frame.groupby("implementation"):
        expected = {str(i) for i in range(repeats)}
        if set(group["repeatIdx"]) != expected:
            raise ValueError(f"{case}/{implementation}: missing or unexpected repeat IDs")
        if not all(math.isfinite(value) and value > 0 for value in group["latency_us"]):
            raise ValueError(f"{case}/{implementation}: invalid timings")
        per_repeat = group.groupby("repeatIdx")["latency_us"].sum()
        samples = [float(per_repeat[str(i)]) for i in range(repeats)]
        if not all(math.isfinite(value) and value > 0 for value in samples):
            raise ValueError(f"{case}/{implementation}: invalid timings")
        rows.append({"case": case, "implementation": str(implementation),
                     "mean_us": statistics.mean(samples),
                     "median_us": statistics.median(samples),
                     "min_us": min(samples), "repeats": repeats})
    if not rows:
        raise ValueError(f"{case}: no measured baseline ranges")
    return rows


def validate_accuracy(metadata, oracle, repeats, case):
    if metadata["case"] != case or metadata["inputs_sha256"] != oracle["inputs_sha256"]:
        raise ValueError(f"{case}: mismatched case or shared input hash")
    if metadata["repeats"] != repeats or metadata["l2_flush_bytes"] != 200 * 1024**2:
        raise ValueError(f"{case}: mismatched measurement configuration")
    samples = oracle.get("samples_us", [])
    if (not oracle.get("correct") or not oracle.get("measurement_complete")
            or oracle.get("repeats") != repeats or len(samples) != repeats
            or not all(math.isfinite(value) and value > 0 for value in samples)):
        raise ValueError(f"{case}: incomplete or incorrect Nautilus measurement")
    for record in metadata["candidates"]:
        if record["status"] == "passed":
            error = record.get("relative_l2", float("nan"))
            max_abs = record.get("max_abs_error", float("nan"))
            if not (math.isfinite(error) and 0 <= error < 1e-2
                    and math.isfinite(max_abs) and max_abs >= 0):
                raise ValueError(f"{case}: invalid passed accuracy record")
    if "argus-gh200-latest" not in metadata["environment"]["python"]:
        # environment.python is a version string; package paths identify the environment.
        packages = metadata["environment"].get("packages", {})
        if not any("argus-gh200-latest" in str(package.get("path", ""))
                   for package in packages.values()):
            raise ValueError(f"{case}: modern baseline environment not recorded")
    return {record["implementation"] for record in metadata["candidates"]
            if record["status"] == "passed"}


def parse_profiles(directory, repeats, nautilus_root):
    from argus.stats.nsys import nsys_to_dataframe

    rows = []
    environment = None
    for path in sorted(directory.glob("*.nsys-rep")):
        case = profile_case(path.name)
        kind, length = case.split("_L")
        operator = "prefill_causal" if kind == "prefillc" else "decode_causal"
        metadata = json.loads((directory / f"{operator}_L{length}_accuracy.json").read_text())
        oracle_path = next((nautilus_root / run / f"{case}.json"
                            for run in ("sweep_20261026_a", "sweep_20261026_b")
                            if (nautilus_root / run / f"{case}.json").exists()), None)
        if oracle_path is None:
            raise ValueError(f"No original Nautilus measurement for {case}")
        oracle = json.loads(oracle_path.read_text())
        passed = validate_accuracy(metadata, oracle, repeats, case)
        current = metadata["environment"]
        if environment is None:
            environment = current
        for key in ("host", "gpu_uuid", "python", "torch", "cuda"):
            if current.get(key) != environment.get(key) or current.get(key) is None:
                raise ValueError(f"{case}: baseline {key} differs or is missing")
        measured = reduce_frame(nsys_to_dataframe(path), repeats, case)
        if {row["implementation"] for row in measured} != passed:
            raise ValueError(f"{case}: profiled implementations differ from passed runners")
        rows.extend(measured)
    if not rows:
        raise ValueError(f"No comparison profiles in {directory}")
    return rows


def merge_with_nautilus(rows, nautilus_csv):
    with nautilus_csv.open() as stream:
        nautilus = {row["case"]: row for row in csv.DictReader(stream)}
    for row in rows:
        source = nautilus[row["case"]]
        value = float(source["mean_us"])
        if source["correct"].lower() != "true" or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid Nautilus measurement: {row['case']}")
        row["nautilus_mean_us"] = value
        row["baseline_over_nautilus"] = row["mean_us"] / value
    return rows


def plot_comparison(rows, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    names = sorted({row["implementation"] for row in rows})
    labels = {"flex": "FlexAttention", "torch(flash)": "PyTorch Flash",
              "torch(cudnn)": "cuDNN SDPA", "tri-dao": "FlashAttention-2",
              "tilelang": "TileLang", "neptune_tilelang": "Local Neptune TileLang",
              "tawa(tma)": "TAWA", "flashinfer": "FlashInfer",
              "tawa(tma_decode)": "TAWA decode"}
    colors = {name: plt.get_cmap("tab10")(i % 10) for i, name in enumerate(names)}
    for axis, prefix, title in zip(axes, ("prefillc_", "decode_"),
                                   ("Causal prefill", "Decode")):
        subset = [row for row in rows if row["case"].startswith(prefix)]
        for name in names:
            selected = sorted((row for row in subset if row["implementation"] == name),
                              key=lambda row: int(row["case"].split("_L")[1]))
            if selected:
                axis.plot([int(row["case"].split("_L")[1]) for row in selected],
                          [row["baseline_over_nautilus"] for row in selected],
                          marker="o", markersize=3, label=labels.get(name, name),
                          color=colors[name])
        axis.axhline(1, color="black", linestyle="--", linewidth=1, label="Nautilus")
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xlabel("KV sequence length")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Baseline time / Nautilus time (higher favors Nautilus)")
    handles = {}
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles[label] = handle
    figure.legend(handles.values(), handles.keys(), loc="upper center", ncol=4, fontsize=8)
    figure.text(0.5, 0.01, "GH200 FP16, B1 H32 D128; 50 cold-L2 repeats; cross-run comparison",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.05, 1, 0.81))
    figure.savefig(destination, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_decode_comparison(rows, destination):
    """Write the single-panel decode figure in the paper CSV style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "flex": "FlexAttn", "neptune_tilelang": "Local Neptune TileLang",
        "tilelang": "TileLang", "torch(cudnn)": "cuDNN",
        "torch(flash)": "PyTorch Flash", "tri-dao": "Tri-Dao Cutlass",
        "flashinfer": "FlashInfer",
    }
    colors = {
        "Nautilus": "#2ca02c", "FlexAttn": "#7f7f7f",
        "Local Neptune TileLang": "#8c564b", "TileLang": "#9467bd",
        "cuDNN": "#1f77b4", "PyTorch Flash": "#17becf",
        "Tri-Dao Cutlass": "#aec7e8", "FlashInfer": "#bcbd22",
    }
    subset = [row for row in rows if row["case"].startswith("decode_")]
    figure, axis = plt.subplots(figsize=(6, 3.5))
    axis.axhline(1.0, color=colors["Nautilus"], linewidth=2, marker="o",
                 label="Nautilus")
    for implementation in sorted({row["implementation"] for row in subset}):
        selected = sorted((row for row in subset if row["implementation"] == implementation),
                          key=lambda row: int(row["case"].split("_L")[1]))
        label = labels.get(implementation, implementation)
        axis.plot([int(row["case"].split("_L")[1]) for row in selected],
                  [row["baseline_over_nautilus"] for row in selected], marker="o",
                  linewidth=2, markersize=6, label=label, color=colors.get(label))
    axis.set_xlabel("Sequence Length", fontsize=20)
    axis.set_ylabel("Relative Performance", fontsize=20)
    axis.set_title("Causal Attention (Decode)", fontsize=22)
    lengths = sorted({int(row["case"].split("_L")[1]) for row in subset})
    axis.set_xscale("log", base=2)
    axis.set_xticks(lengths)
    axis.set_xticklabels([f"$2^{{{length.bit_length() - 1}}}$" for length in lengths], fontsize=16)
    axis.tick_params(axis="y", labelsize=16)
    axis.set_ylim(bottom=0)
    axis.grid(True, linestyle="--", alpha=0.5)
    handles, legend_labels = axis.get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.15),
                 fontsize=10, frameon=True, ncol=4)
    figure.tight_layout()
    figure.subplots_adjust(top=0.77)
    figure.savefig(destination, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


def summarize(directory, nautilus_csv, repeats=50):
    expected = {f"{kind}_L{length}" for kind in ("prefillc", "decode")
                for length in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)}
    present = {profile_case(path.name) for path in directory.glob("*.nsys-rep")}
    if present != expected:
        raise ValueError(f"Incomplete comparison profiles; missing={sorted(expected - present)}")
    rows = merge_with_nautilus(
        parse_profiles(directory, repeats, nautilus_csv.parent.parent), nautilus_csv
    )
    rows.sort(key=lambda row: (row["case"].split("_L")[0],
                              int(row["case"].split("_L")[1]), row["implementation"]))
    fields = ["case", "implementation", "mean_us", "median_us", "min_us", "nautilus_mean_us",
              "baseline_over_nautilus", "repeats"]
    with (directory / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# GH200 causal prefill and decode comparison", "", CAVEAT, "",
             "Local Neptune TileLang is the repository runner, distinct from the exported Nautilus "
             "kernels. Its decode runner inherits the same TileLang implementation.", "",
             "## Fastest measured baseline per length", "",
             "| Case | Baseline | Baseline us | Nautilus us | Baseline/Nautilus |",
             "|---|---|---:|---:|---:|"]
    best = {}
    for row in rows:
        if row["case"] not in best or row["mean_us"] < best[row["case"]]["mean_us"]:
            best[row["case"]] = row
    for row in best.values():
        lines.append(f"| {row['case']} | {DISPLAY_NAMES.get(row['implementation'], row['implementation'])} | {row['mean_us']:.3f} | "
                     f"{row['nautilus_mean_us']:.3f} | {row['baseline_over_nautilus']:.3f}x |")
    lines += ["", "## All measured baselines", "",
              "| Case | Baseline | Baseline us | Nautilus us | Baseline/Nautilus |",
              "|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['case']} | {DISPLAY_NAMES.get(row['implementation'], row['implementation'])} | {row['mean_us']:.3f} | "
                     f"{row['nautilus_mean_us']:.3f} | {row['baseline_over_nautilus']:.3f}x |")
    failures = []
    for path in sorted(directory.glob("*_accuracy.json")):
        metadata = json.loads(path.read_text())
        for record in metadata["candidates"]:
            if record["status"] != "passed":
                message = f"{record['implementation']}: {record.get('error', record['status'])}"
                failures.append({"log": path.name, "message": message})
    for path in sorted(directory.glob("*.log")):
        for line in path.read_text(errors="replace").splitlines():
            if "WARNING" in line and any(
                    text in line for text in
                    ("failed with exception", "does not support", "mismatch")):
                failures.append({"log": path.name, "message": line})
    if failures:
        lines += ["", "## Runner warnings", ""]
        lines += [f"- `{item['log']}`: {item['message']}" for item in failures]
    (directory / "comparison.md").write_text("\n".join(lines) + "\n")
    metadata = {"baseline_environment": "/work/nvme/beqg/yucheny8/envs/argus-gh200-latest",
                "nautilus_source": str(nautilus_csv), "repeats": repeats,
                "baseline_rows": len(rows), "cases": len({row['case'] for row in rows}),
                "expected_cases": 16,
                "measurement_complete": len({row['case'] for row in rows}) == 16,
                "comparison_scope": CAVEAT, "runner_warnings": failures}
    (directory / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_comparison(rows, directory / "comparison.pdf")
    plot_decode_comparison(rows, directory / "decode_comparison.pdf")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--nautilus-csv", type=Path,
                        default=Path("profiles_paper_kernels/sweep_20261026_merged/results.csv"))
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    rows = summarize(args.directory, args.nautilus_csv, args.repeat)
    print(f"wrote {len(rows)} baseline rows to {args.directory / 'comparison.csv'}")


if __name__ == "__main__":
    main()
