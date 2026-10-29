"""Validate the eight new bound results and add them to existing reference curves."""

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from argus.paper_baselines.e2e_mtp_baselines import TILEOPS_COMMIT
from argus.paper_baselines.workloads import export_path
from scripts.summarize_decode_e2e import LENGTHS, plot


directory = Path(__file__).resolve().parent
reference = directory.parent / "decode_sweep_20261027_j3472132"
rows = []
first = None
manifest = []
for length in LENGTHS:
    current = export_path("decode", length).resolve()
    digest = hashlib.sha256((current / "kernel.so").read_bytes()).hexdigest()
    bound = json.loads((directory / f"decode_L{length}_tileops_bound.json").read_text())
    source = bound["baseline_source"]
    assert bound["cache_layout"] == "bshd"
    assert source["commit"] == TILEOPS_COMMIT
    assert source["selected_kernel"] == "GQADecodeBs1Kernel"
    assert source["kernel_config"] == {"block_M": 64, "block_N": 128, "threads": 160}
    assert source["integration"] == (
        "prebound selected TileOPs program, caller-owned output/workspace"
    )
    assert source["execution_backend"] == "tvm_ffi"
    assert source["shape"] == {"q": [1, 1, 32, 128], "kv": [1, length, 32, 128]}
    assert source["ctx_splits"] == (None if length < 640 else 4)
    if first is None:
        first = bound
    for key in ("gpu_uuid", "host", "weights_sha256", "torch", "cuda", "transformers",
                "packages", "cache_implementation", "metric", "repeat", "warmup"):
        assert bound[key] == first[key], (length, key)
    assert {k: v for k, v in bound["config"].items() if k != "max_position_embeddings"} == {
        k: v for k, v in first["config"].items() if k != "max_position_embeddings"
    }
    for implementation in ("nautilus", "tileops", "tileops_bound", "flex"):
        folder = directory if implementation == "tileops_bound" else reference
        record = json.loads((folder / f"decode_L{length}_{implementation}.json").read_text())
        repeat = 30 if implementation == "tileops_bound" else 20
        warmup = 5 if implementation == "tileops_bound" else 3
        assert record["variant"] == "decode" and record["implementation"] == implementation
        assert record["correct"] and record["measurement_complete"]
        assert record["repeat"] == repeat and record["warmup"] == warmup
        samples = record["samples_ms"]
        assert len(samples) == repeat and all(math.isfinite(x) and x > 0 for x in samples)
        assert len(record["attention_relative_l2"]) == 32
        errors = [record["relative_l2"], record["compiled_relative_l2"],
                  *record["attention_relative_l2"]]
        assert all(math.isfinite(x) and 0 <= x < 0.01 for x in errors)
        assert record["persistent_kv_bindings"] == 32
        assert record["full_kv_staging_bytes_per_forward"] == 0
        assert Path(record["export_directory"]).resolve() == current
        assert record["kernel_sha256"] == digest
        for key in ("config", "weights_sha256", "inputs_sha256", "torch", "cuda",
                    "transformers", "metric", "cache_implementation"):
            assert record[key] == bound[key], (length, implementation, key)
        rows.append({"kv_length": length, "implementation": implementation,
                     "mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples),
                     "stdev_ms": statistics.stdev(samples), "repeat": repeat, "warmup": warmup,
                     "gpu_uuid": record["gpu_uuid"], "host": record["host"],
                     "run": folder.name})
    manifest.append({"kv_length": length, "export_directory": str(current),
                     "kernel_sha256": digest, "baseline_source": source})

assert len(rows) == 32
new_rows = [row for row in rows if row["implementation"] == "tileops_bound"]
assert len(new_rows) == 8
for name, selected in (("results.csv", new_rows), ("comparison.csv", rows)):
    with (directory / name).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
(directory / "kernel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
caption = ("Cross-run supplement: TileOPs prebound measured on gh052 (30 samples); "
           "other curves reused from gh084 (20 samples).")
plot(rows, directory / "decode_e2e_with_bound.pdf", caption=caption)
plot(rows, directory / "decode_e2e_with_bound.png", caption=caption)

lines = ["# TileOPs 预绑定：decode 全8长度补测", "",
         "仅本表的 TileOPs prebound 是新测量；原 Nautilus、TileOPs public API、Flex 未重跑。",
         "GHx4作业3473405，gh052，B1/FP16，Llama-7B backbone，32层，Q1。",
         "每点预热5次、采样30次，计时为同步整模型wall latency；所有样本保留。", "",
         "| KV length | TileOPs prebound ms | Median ms | Stddev ms |",
         "|---|---:|---:|---:|"]
for row in new_rows:
    lines.append(f"| {row['kv_length']} | {row['mean_ms']:.3f} | "
                 f"{row['median_ms']:.3f} | {row['stdev_ms']:.3f} |")
lines += ["", "8点均通过逐层及整模型正确性检查、常驻cache检查、当前paper kernel哈希检查；",
          "新旧结果的模型权重/输入/配置及Torch/Transformers版本逐点相同。", "",
          "[新测量CSV](results.csv) · [加入旧曲线的PDF](decode_e2e_with_bound.pdf) · "
          "[PNG](decode_e2e_with_bound.png) · [带run/GPU来源的合并CSV](comparison.csv)", "",
          "图中其余3条曲线来自旧作业3472132、gh084，每点20个样本/预热3次。",
          "图和合并CSV明确区分来源；不同GPU/运行时段可能影响wall latency，不能把跨次比值",
          "当作同GPU隔离实验。先前同GPU32K对照见 ../decode_bound_20261027_j3473082/README.md。",
          "本次没有测MTP或其他operator，也没有把新数据覆盖到旧实验目录。", ""]
(directory / "results.md").write_text("\n".join(lines))
print("Validated8new points and24reference points; wrote CSV, report, PDF, PNG and manifest.")
