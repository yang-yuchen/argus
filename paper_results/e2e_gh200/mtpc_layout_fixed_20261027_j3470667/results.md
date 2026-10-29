# MTP e2e：去除TileOPs重复输入复制后的正式结果

**后续已有常驻原生cache实现**：[job3471084](../mtpc_native_cache_20261027_j3471084/results.md)。
本页仍包含每步完整KV staging，保留为历史接入方式的测量，不是常驻BSHD的最终性能。

| 实现 | 平均ms | 中位数ms | Nautilus相对该实现的加速 |
|---|---:|---:|---:|
| Nautilus | 39.474 | 39.470 | 1.000× |
| TileOPs | 65.774 | 65.774 | 1.666× |
| FlexAttention | 51.929 | 51.767 | 1.316× |

同一张GH200（gh061，job3470667）、相同Llama-7B权重/输入、B1/Q16/KV65536、
32层、StaticCache、每组20次同步整模型wall采样。三组均用专用最新baseline环境
Torch2.14.1/cu130、Triton3.8；TileOPs为0.1.15 TileLang上的原官方API和kernel。

本次修复让TileOPs直接写入原生BSHD输入，保留必要转换并删除多余BHSD中转复制。
旧结果中2.187×的TileOPs倍率由本表替代。

完整原因、同卡旧路径复现、阶段计时、CUDA trace和正确性说明见
[布局诊断报告](../mtpc_layout_investigation_j3470667/results.md)。
原始样本/环境为`mtpc_{nautilus,tileops,flex}.json`，日志为同名`.log`，汇总为`results.csv`。

完整模型编译后相对L2：Nautilus0.005829、TileOPs0.005831、Flex0.005800，均低于1%。
模型权重/输入hash、GPU UUID、配置和计时口径已经汇总脚本校验一致。
job3470667完成退出0，GPU已释放；未测试其他context长度。
