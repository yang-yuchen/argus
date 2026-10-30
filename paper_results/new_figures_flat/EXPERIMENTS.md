# Nautilus 实验与选图清单

更新：2026-10-29。25 已从当前筛选图集中删除，原始实验数据保留。
当前目录共有 **14 张 PNG：12 张 operator 图、2 张 E2E 图**，编号保持不变。
“已放入”指当前平铺筛选目录，不等于已经加入论文正文。
下表按实验归类；同一实验的诊断图、重复 PDF、旧版本不重复计数。
5090 E2E 按要求不再调查或补充，已有的 17/18 operator 图保留。

路径以 `paper_results/` 为根目录。

## GH200：可以加入的图

| Operator / E2E | dtype、形状 | baseline / 数据范围 | 当前状态 | 建议 |
|---|---|---|---|---|
| Prefill operator（旧论文） | FP16；Global、Causal、GQA；L 256–32768 | FlexAttn、Helion、OpenAI Triton、TAWA、TVM、TileLang、Tri-Dao Triton | 已有标准图和可复现脚本 | **可加入**，直接用旧图 |
| Decode operator | FP16；B1/H32/Q1；KV 256–32768 | Nautilus、TileLang、FlexAttn、Helion | 当前图 6 | **可加入** |
| Causal MTP operator | FP16；H4；Q=2/4/8/16/32/64；KV 256–65536 | Nautilus、TileLang、FlexAttn | 当前图 7–12，共 6 张 | **可加入** |
| Sparse operator | BF16；H16；L 256–32768；top-2048 | 当前 4：Nautilus、TileLang、FlexAttn、Helion；另有 TileLang-only 版本 | 当前图 4 | **可加入**；若只留等价 compiler baseline，用 TileLang-only 源数据重画 |
| FP8 Decode operator | FP8 输入；B1/H32/Q1；KV 256–32768 | Nautilus、官方 TileOPs；最长点采用已测 paged official route | 当前图 14 | **可加入**，注明 paged 点的内部 FP16 运算 |
| FP8 Causal MTP operator | FP8 输入；H4/Q16；KV 256–65536 | Nautilus、官方 TileOPs；native 版和 best-official 版都已测 | 图 15/16 已按要求删除 | **数据完整但当前不加入** |
| Decode E2E | Llama-7B、FP16、B1/Q1；KV 256–32768 | Nautilus、TileLang、FlexAttn；CUDA Graph | 当前图 1 | **可加入** |
| Causal MTP E2E | Llama-7B、FP16、B1/Q16；KV 256–65536 | Nautilus、TileLang、FlexAttn；CUDA Graph | 当前图 2，含 2K/4K 复测 | **可加入** |
| Prefill E2E（旧论文） | 旧论文模型与长度 | 原论文 baseline；`prefill_final_repro` 已有标准图 | 已归档 | **可加入旧图**；本轮不补新测 Prefill |
| Attention transition | FP16；Decode → Q2…Q64 MTP → Prefill | 三条跨形状平均曲线 | 当前图 5 | **可加入**；Prefill 段沿用旧数据 |

## RTX 5090：可以加入的图

| Operator / E2E | dtype、形状 | baseline / 数据范围 | 当前状态 | 建议 |
|---|---|---|---|---|
| Prefill operator（旧论文） | FP16；原论文长度 | FlashAttention、FlexAttn、SDPA、TileLang 等 | 旧论文标准图已归档 | **可加入旧图** |
| Decode operator | FP16；B1/H32/Q1；KV 256–32768 | Nautilus、TileLang main GQA、GitHub FlexAttention | 当前图 18 | **可加入** |
| Causal MTP operator | FP16；Q16；KV 256–32768 | Nautilus、GitHub FlexAttention | 当前图 17 | **可加入** |
| Decode / MTP E2E | — | 没有完成的对应 5090 E2E sweep | 未测 / 已取消 | **不加入** |

这两张矩阵是“可以放入论文候选集”的实验清单；下表继续列出已做但当前没有放入平铺图集的实验及原因。

| 状态 | 当前/原编号 | 实验 | 已测范围 | 比较方法 | 选图结论与限制 | 数据或图的来源 |
|---|---|---|---|---|---|---|
| 已放入 | 1 | GH200 · Llama-7B · FP16 · B1/Q1 · Decode E2E | KV 256–32768，8 个长度 | Nautilus / TileLang / FlexAttn | 可选；原论文吞吐柱状图样式，CUDA Graph；部分方法来自不同 GH200 节点 | `e2e_gh200/decode_graph_baselines_20261027_j3474109/comparison.csv` |
| 已放入 | 2 | GH200 · Llama-7B · FP16 · B1/Q16 · Causal MTP E2E | KV 256–65536，9 个长度 | Nautilus / TileLang / FlexAttn | 可选；使用包含六个 2K/4K 复测点的修正版；其余长度复用原测量 | `e2e_gh200/mtpc_2k4k_recheck_j3481758/results.csv` |
| 已放入，版本需注意 | 4 | GH200 · BF16 · DSA 风格 top-2048 Sparse operator | L 256–32768，8 个长度 | Nautilus / TileLang / FlexAttn / Helion | 当前 4 是多 baseline 版本；此前要求只留 TileLang 的版本也已存在，但其预览 3 已被删除。L≤2048 时该 top-k 规则实际为 dense causal | `operator_gh200/sparse_shared_kv_20261027/plot_input.csv` |
| 已放入 | 5 | GH200 · FP16 · Decode → Causal MTP → Causal Prefill 总览 | Q=1,2,4,8,16,32,64,KV；每阶段平均 8 个 KV 长度 | Nautilus / TileLang / FlexAttn | 可选；实线、等距横轴、阶段标签在下方。Decode/Prefill 为 H32，MTP 为 H4；Prefill 来自历史数据，不能描述为固定 H 的连续实测 | `operator_gh200/attention_transition_20261029/plot_input.csv`、`source_manifest.json` |
| 已放入 | 6 | GH200 · FP16 · Decode operator | KV 256–32768，8 个长度 | Nautilus / TileLang / FlexAttn / Helion | 可选；Nautilus 使用 b7749f3 更新；baseline 来自已有复测 | `operator_gh200/decode_mtpc16_plot_b7749f3_20261027/decode_plot_input.csv` |
| 已放入 | 8 / 10 / 12 / 7 / 9 / 11 | GH200 · FP16 · Causal MTP operator，分别 Q=2/4/8/16/32/64 | 每个 Q：KV 256–65536，9 个长度；共 54 个形状 | Nautilus / TileLang / FlexAttn | 可选，共 6 张；完整 KV 扫描，包含四个慢点的调优后复测 | `operator_gh200/mtpc_optimize_20261028/plots/plot_input_Q*.csv`、`plot_manifest.json` |
| 已放入 | 14 | GH200 · FP8 · Decode operator | KV 256–32768，8 个长度 | Nautilus / TileLang | 可选；是已测官方路径的组合，32K TileLang 点采用 FP8 输入、FP16 内部运算的 paged kernel；不能称整条 baseline 都是纯 FP8 计算 | `operator_gh200/fp8_compilers_20261029/plot_input_decode8_best_official.csv` |
| 已放入 | 17 | RTX5090 · FP16 · Causal MTP Q16 operator | KV 256–32768，8 个长度 | Nautilus / FlexAttn | 已有标准图；没有 TileOPs 曲线，其 dispatcher 不支持 SM120 | `operator_rtx5090/rtx5090_decode_mtpc16_results.md` |
| 已放入 | 18 | RTX5090 · FP16 · Decode operator | KV 256–32768，8 个长度 | Nautilus / TileLang / FlexAttn | 已有标准图；TileLang 是主仓库 GQA decode 示例，MHA 参数下使用适配 SM120 的配置 | `operator_rtx5090/rtx5090_decode_mtpc16_results.md` |
| 已做但按要求不放 | 无 | GH200 · 新测 FP16 Causal Prefill operator | L 256–32768，8 个长度 | Nautilus / FlexAttn / TileLang / TAWA 等 | 数据完整，但你已说明 Prefill 采用旧结果；不再作为本轮缺图处理。5 号图沿用旧 Prefill 数据 | `operator_gh200/prefill_decode_compare_20261026_validated/comparison.csv` |
| 完整测量，但缺配套 baseline | 无 | GH200 · FP16 Global Prefill operator 新测 | L 256–32768，8 个长度 | 该 sweep 仅 Nautilus | 不是完整的新环境对比；不能直接把历史 baseline 当作本轮共同测量 | `operator_gh200/nautilus_sweep_20261026_merged/results.csv` 中 `prefill_L*` |
| 完整测量，但缺配套 baseline | 无 | GH200 · FP16 非 causal MTP Q16 operator | KV 256–65536，9 个长度 | 该 sweep 仅 Nautilus | 不同于现有 causal MTP；需要补匹配 baseline 才能作完整对比 | `operator_gh200/nautilus_sweep_20261026_merged/results.csv` 中 `mtp_Q16_L*` |
| 已做，按要求删除 | 3 | GH200 · BF16 Sparse，仅比较 TileLang | L 256–32768，8 个长度 | Nautilus / TileLang | 标准 PDF 和数据均已存在；这是此前要求去掉 Flex/Helion 后的两方法版本，本轮不擅自恢复已删编号 | `figures/operator_gh200/gh200-bf16-tc1-sparse-d35a206-shared-kv-tilelang.pdf`；`operator_gh200/sparse_shared_kv_20261027/plot_input_tilelang.csv` |
| 已做，重复路线删除 | 13 | GH200 · FP8 Decode，原 native baseline 版本 | KV 256–32768，8 个长度 | Nautilus / TileLang | 已由 14 的 best-measured-official 图覆盖；不需要两张同时放 | `figures/operator_gh200/gh200-fp8-tc1-decode8-4e06b84.pdf` |
| **完整实验，按要求删除** | 15 / 16 | **GH200 · FP8 Causal MTP Q16 operator** | **KV 256–65536，9 个长度** | Nautilus / TileLang | **做完了，且有标准 PDF**；并非缺数据。16 的 best-official 版仅 64K 替换为 paged 路径，该点内部用 FP16 运算；15 是原 native 版 | `figures/operator_gh200/gh200-fp8-tc1-mtpc8{,-best-official}-4e06b84.pdf`；`operator_gh200/fp8_compilers_20261029/plot_input_mtpc8_best_official.csv` |
| 已做，按要求删除 | 19 / 25 | GH200 · FP16 Decode E2E，早期普通 forward / prebound 实验 | KV 256–32768 | Nautilus / TileLang / FlexAttn；具体方法见各表 | 原始数据保留；当前只选 1 的 CUDA Graph 比较，不混用两种计时协议 | `e2e_gh200/decode_bound_only_20261027_j3473405/comparison.csv`；`e2e_gh200/decode_sweep_20261027_j3472132/results.csv` |
| 已做，已被替代 | 26 / 27 / 28 | GH200 · FP16 Causal MTP E2E，早期 64K 单点 / 原完整 sweep | 64K 单点；以及 KV 256–65536 | Nautilus / TileLang / FlexAttn | 26/27 为早期单点，28 为含 2K/4K 长尾异常的旧表；当前 2 已覆盖，旧异常表不再选图 | `e2e_gh200/mtpc_graph_20261027_j3474393/`；`e2e_gh200/mtpc_graph_sweep_3478086/` |
| 已做，视图已合并 | 20–24 | GH200 Decode E2E 单方法图 / latency 图 / 对比图 | 与 1 同一实验 | 同 1 | 不是缺失实验；统一由 1 的标准吞吐柱状图展现 | `e2e_gh200/decode_graph_baselines_20261027_j3474109/` |
| 部分测过 | 无 | GH200 · FP8 Global Prefill / Causal Prefill | 仅 L2048 smoke | Nautilus / TileLang | 不足以作全长度标准曲线；历史 FP8 Prefill 全长图是另一批数据 | `operator_gh200/fp8_compilers_20261029/smoke_summary.json` |
| 部分测过 | 无 | GH200 · FP8 非 causal MTP Q16 | 仅 KV4096 smoke | Nautilus / TileLang | 不足以作全长度对比，不是已完成但漏放的 full sweep | `operator_gh200/fp8_compilers_20261029/mtp_smoke_3482675/` |
| 部分测过 / 编译受限 | 无 | GH200 · 更强官方 Helion Prefill AOT | L256、L512；更长长度受共享内存限制 | Helion，已有对应 Nautilus 数据 | 只能补两个点，不能声称有完整官方 Helion 新曲线 | `operator_gh200/prefill_official_helion_20261026/prefillc_L{256,512}.json` |
| 旧论文已有 | 无 | GH200 历史 FP16 / FP8 Prefill operator（Global / Causal / GQA） | 各 CSV 已有长度范围 | 原论文 compiler baselines | 不是本轮新增图遗漏；可直接复用归档图、数据和脚本 | `prefill_final_repro/figures/gh200-fp{16,8}-tc1.pdf`；`prefill_final_repro/data/` |
| 旧论文已有 | 无 | GH200 历史 Prefill E2E | 原 CSV 中的模型和长度 | 原论文模型 baselines | 不是新增图遗漏；完整标准图及源 CSV 已归档 | `prefill_final_repro/figures/throughput_plot.pdf`；`prefill_final_repro/data/e2e_throughput.csv` |
| 历史归档，非新增性能图 | 无 | Ablation、精度/RMSE、旧 baseline 版本 | 各自历史实验范围 | 见原始清单 | 已归档，不属于本轮标准 operator/E2E 新图筛选；不是另一组未展示的全长性能对比 | `nautilus_mlsys27/manifest.json`、`performance_sources/performance_manifest.json`、`rmse_sources/rmse_manifest.json` |
| 未完成对比，不算漏图 | 无 | Sparse E2E；新一轮 FP8 E2E | 未找到完成的匹配方法 sweep | — | 不能从 operator 结果推造 E2E 图；Sparse E2E 在主 README 中明确未跑 | `README.md`；FP8 当前目录为 operator 实验 |

## 选图结论

- 现有新图覆盖：Decode、六种 Q 的 causal MTP、Sparse、跨形状总览、FP8 Decode，以及 Decode / causal MTP E2E。
- 新测 FP16 causal prefill 已有完整 CSV，但按你的要求不作为本轮新增图；Prefill 使用旧论文标准图和旧数据。
- **FP8 causal MTP 已做全**，只是被明确删掉；如之后恢复，直接用已存在的 best-official PDF。
- Sparse 当前 4 与此前“只保留 TileLang”的要求不一致；两方法 PDF 已有，表中保留这一版本差异供筛选。
- 不把 smoke、Nautilus-only sweep、诊断图或旧论文图算作遗漏的完整新对比实验。

本次只核对现有数据、清单和图的来源，没有重新运行 GPU 实验，也没有证明现有 kernel 已达到全局最优调参结果。
