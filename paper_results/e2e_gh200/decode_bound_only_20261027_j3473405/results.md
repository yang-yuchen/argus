# TileOPs 预绑定：decode 全8长度补测

后续[短/长同GPU交替计时审计](../decode_timing_audit_20261027_j3473757/README.md)确认：
该接入的CPU提交开销会掩盖attention GPU耗时随KV增长的趋势；平坦wall曲线不代表
attention本身不增长，也不能作为优化后decode的GPU执行时间曲线。

仅本表的 TileOPs prebound 是新测量；原 Nautilus、TileOPs public API、Flex 未重跑。
GHx4作业3473405，gh052，B1/FP16，Llama-7B backbone，32层，Q1。
每点预热5次、采样30次，计时为同步整模型wall latency；所有样本保留。

| KV length | TileOPs prebound ms | Median ms | Stddev ms |
|---|---:|---:|---:|
| 256 | 25.861 | 25.824 | 0.267 |
| 512 | 24.265 | 24.408 | 0.491 |
| 1024 | 24.708 | 24.568 | 0.842 |
| 2048 | 25.148 | 25.292 | 0.387 |
| 4096 | 23.614 | 23.580 | 0.210 |
| 8192 | 23.603 | 23.425 | 0.512 |
| 16384 | 25.140 | 25.227 | 0.533 |
| 32768 | 23.670 | 23.670 | 0.205 |

8点均通过逐层及整模型正确性检查、常驻cache检查、当前paper kernel哈希检查；
新旧结果的模型权重/输入/配置及Torch/Transformers版本逐点相同。

[新测量CSV](results.csv) · [加入旧曲线的PDF](decode_e2e_with_bound.pdf) · [PNG](decode_e2e_with_bound.png) · [带run/GPU来源的合并CSV](comparison.csv)

图中其余3条曲线来自旧作业3472132、gh084，每点20个样本/预热3次。
图和合并CSV明确区分来源；不同GPU/运行时段可能影响wall latency，不能把跨次比值
当作同GPU隔离实验。先前同GPU32K对照见 ../decode_bound_20261027_j3473082/README.md。
本次没有测MTP或其他operator，也没有把新数据覆盖到旧实验目录。
