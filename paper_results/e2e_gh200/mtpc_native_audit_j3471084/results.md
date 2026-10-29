# 常驻KV的实际GPU行为审计

详见[正式报告](../mtpc_native_cache_20261027_j3471084/results.md)。

| Backend | 审计forward数 | ≥64MiB GPU复制 | ≥64MiB临时分配 | 最大aten copy输入 |
|---|---:|---:|---:|---:|
| TileOPs | 2 | 0 | 0 | 8192个FP16元素/16KiB |
| Nautilus | 2 | 0 | 0 | 8192个FP16元素/16KiB |

`summary.json`保留128次index-copy kernel及实际grid；`*.trace.json`保留原始CUDA
kernel、memcpy和内存分配事件。更新kernel只处理新增16token；K/V原生cache地址不变。

`tileops.json`和`nautilus.json`的sample延迟包含profiler开销，仅用于正确性及审计，
**不能用于性能比较**。正式性能只使用相邻正式目录的20次未加profiler采样。

全部审计使用同一GPU和专用latest环境。复现脚本`audit_native_e2e.py`在预热完成后才
启动profiler；运行参数与常规e2e相同，使用新output前缀并设置`--repeat 2`。
