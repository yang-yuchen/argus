# Nautilus figures: data and reproduction

独立复现原 `argus/nautilus_figures/` 中的全部 **5 张 PDF**。仅需 CPU，不依赖 Argus、CUDA 或模型权重。

## 一条命令重绘并校验

已在 Linux aarch64、Python 3.13.5、Matplotlib 3.10.7、Poppler 24.03.0 环境验证。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# PDF 栅格比较需要 Poppler；Ubuntu/Debian 安装命令：
# sudo apt-get install poppler-utils
python reproduce.py
```

输出在 `figures/`。脚本可从任意工作目录执行，并在生成后自动校验全部 5 张图：

- 校验 5 份原始 CSV 和 5 张参考 PDF 的 SHA-256。
- 使用同一 Poppler 在 **144 dpi** 渲染原图和重绘图，要求尺寸与每个像素完全一致。
- 结果写入 `figures/verification.json`；缺图、坏 PDF 或像素差异都会返回非零退出码。

只检查现有输出：`python verify.py`。PDF 内的创建时间可以不同，因此不要求整个 PDF 文件的哈希相同。
回归测试：`python -m unittest discover -s tests -v`，覆盖输入数据变更和错误 PDF 的拒绝处理。

## 图、数据、脚本对应关系

| 图 | 原始 CSV（`data/`） | 绘图脚本（`scripts/`） |
|---|---|---|
| `gh200-fp16-tc1.pdf` | `datasheet_new.csv` | `plot_from_csv.py` |
| `gh200-fp8-tc1.pdf` | `datasheet_fp8.csv` | `plot_from_csv.py` |
| `rtx5090-fp16-tc1.pdf` | `5090_latency_us_new.csv` | `plot_from_csv_5090.py` |
| `throughput_plot.pdf` | `e2e_throughput.csv` | `csv_figures.py` |
| `throughput_plot_5090.pdf` | `throughput_5090new.csv` | `csv_figures_5090.py` |

原始性能 CSV 的单位为微秒；性能图按 `Nautilus latency / method latency` 归一化，Nautilus 为 1。
吞吐 CSV 的单位为 TFLOPs/s；脚本直接绘制数值，同时生成两份 geometric mean speedup 摘要 CSV。
原始 CSV 中的缺测值保留为空，没有补造数据。

## 来源和移植说明

`manifest.json` 记录原文件位置、原脚本 SHA-256、数据 SHA-256、参考图 SHA-256 及脚本修改清单。
数据与 `reference/` 中的目标 PDF 均为原样复制。

- 三张算子性能图的数据和脚本来自原 `argus/` repo。
- 两张模型吞吐图的数据和脚本来自同机关联工程 `/u/yucheny8/neptune-internal/`；这些文件未存在于 Argus 的 Git 历史中。
- GH200 FP16 必须用 `datasheet_new.csv`。源脚本后来默认使用的 `datasheet_new_asplos.csv` 不能复现目标图。
- 当前 RTX5090 吞吐源脚本已改为较高版面。依据目标 PDF 的矢量坐标恢复为 `figsize=(25, 3.5)`、`top=0.86`、`bottom=0.16`，使用 `throughput_5090new.csv` 后可精确匹配。
- 移植仅修改输入/输出路径、恢复上述版面、关闭已保存的 figure，并取消两张不在目标目录内的 baseline 图输出。保留源脚本的数据处理和绘图方法。

`reference/` 为原图，`figures/` 为实际重绘结果，`requirements.txt` 固定本次验证环境中的包版本。
`figures/reproduction.log` 保留完整生成日志。

## 限制

本包复现现有 CSV 对应的图，没有重新运行 GPU benchmark 或复测 CSV 的测量准确性。
其他平台、字体或 Poppler 版本可能产生栅格差异；校验会明确报错，需检查报告。
