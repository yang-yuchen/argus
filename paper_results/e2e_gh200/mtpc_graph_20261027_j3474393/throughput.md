# MTP throughput figure provenance

原版绘图脚本是：
`/u/yucheny8/neptune-internal/scripts/csv_figures.py`。
本图直接复用其 `plot_subplot`、颜色、分组柱状图宽度、字体、顶部图例和
`2^n` 横轴格式；只有输入数据改为本次 MTP CUDA Graph replay 结果，且只保留一个子图。

原脚本的纵轴是 **Throughput (TFLOPs/s)**，不是延迟。吞吐量换算沿用
`/u/yucheny8/neptune-internal/utils/export_to_csv.py` 的定义：

```text
TFLOPs/s = theoretical_model_FLOPs / latency_seconds / 1e12
```

原 helper 的 FLOPs 函数是为方形 prefill（Q=KV=sequence length）写的，不能直接把
65536 填进去，否则会把 MTP 的 QKV 投影和 MLP 错算成处理65536个 query。这里保留
原来的完整模型统计项目（QKV projection、attention matmul、output projection、MLP、
32 layers），并按 MTP 的矩形形状扩展：projection/MLP 使用 Q=16；attention 使用
lower-right causal mask 的实际可见 pairs（65521…65536），即：

```text
attention FLOPs = 4 * batch * heads * head_dim * sum(visible_KV_lengths)
```

因此本次图的理论 FLOPs 为 **710,754,172,928**，不是把 square-sequence 公式直接套
在65536上。对应吞吐量：

| 实现 | Replay latency | Throughput |
|---|---:|---:|
| Nautilus | 17.089 ms | 41.591 TFLOPs/s |
| TileOPs | 19.011 ms | 37.387 TFLOPs/s |
| FlexAttn | 19.311 ms | 36.805 TFLOPs/s |

图：[PDF](mtp_e2e_graph_throughput_paper_style.pdf) ·
[PNG](mtp_e2e_graph_throughput_paper_style.png) ·
脚本：`scripts/plot_mtpc_throughput_paper_style.py`。

本图没有重跑 GPU；使用已验证的三方 graph replay 样本。MTP 是矩形 Q/KV 工作负载，
所以吞吐量公式相对原脚本做了必要的形状扩展；这点已在图旁说明，避免把错误的
square-prefill FLOPs 当成 MTP 计算量。
