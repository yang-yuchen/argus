# Decode：对齐 TileOPs 接入后的 32K 对照

## 结果

同一次 GHx4 allocation、同一张 GH200 上重新测三种实现；均为同步整模型 wall latency。

| 实现 | 平均延迟 ms | 中位数 ms | 样本标准差 ms |
|---|---:|---:|---:|
| TileOPs public API | 27.885 | 27.872 | 0.206 |
| TileOPs 预绑定 kernel/output/workspace | 25.262 | 25.261 | 0.212 |
| Nautilus 预绑定 export | 25.279 | 25.259 | 0.224 |

预绑定使 TileOPs 延迟降低约 **9.40%**，即 **1.104×** 加速。它与 Nautilus 的均值只差
0.017ms，远小于本次样本波动，中位数只差约0.002ms；本次应解读为基本持平。
这支持之前的诊断：原 model 级差距包含接入开销，不能直接归因于 attention GPU kernel。

这里的比较全部来自本次同 GPU 重测，没有拿旧节点上的31.342ms与新结果相减。
原 public API 路径和历史数据均保留；新路径明确命名为 `tileops_bound`。

## 实现改动

`argus/paper_baselines/e2e_tileops_bound.py` 复用官方 public API 已选中的
`GQADecodeBs1Kernel`，保留原 IR、target、passes、compiler flags、短序列分支及长序列
split 数。只把输出改成调用方显式传入，并提前分配 output、LSE/partial workspace、
建立 DLPack 参数绑定。每次 attention 直接调用 executable；不重新做 public API 调度。

常驻 BSHD KV、逐 token cache 写入、Q 更新和完整模型计时范围与原 TileOPs 对照一致。
官方 checkout 未修改，commit 为 `85e7e644f1d99037e937378b8d8dc70c6568113f`。
长序列配置仍为 `block_M=64, block_N=128, threads=160`，32K 使用4个 context splits。

## 实验与验证

- Slurm **3473082**，节点 **gh079**，GPU UUID
  `767b6dcf-6613-e1dc-608a-0bc65976076a`。
- 专用 latest 环境：Torch2.14.1+cu130、TileLang0.1.15、Triton3.8.0。
  Nautilus 在独立进程加载导出物匹配的旧 TVM runtime，Torch/Transformers保持相同。
- Llama-7B backbone，32层，FP16，B1、H32、D128、Q1，prefix32767、可见KV32768。
  随机权重、固定输入；三者 weights/input hash、模型配置和GPU均已逐项核对一致。
- 每配置预热5次、采样30次；三方均通过逐层及完整编译模型相对L2<1%检查。
  TileOPs 两条路径的完整模型误差值完全相同。
- 三者均有32个常驻KV绑定、每forward零全KV staging；Nautilus 实际加载的
  `kernel.so` SHA256已与当前最新 paper kernel 重新核对。
- `check_bound_gpu.py` 对256、512、8K、32K分别更新Q/K/V，再核对public API输出；
  全部通过。CPU profiler确认预绑定调用中没有`aten::empty`或TileOPs public-op事件。
- 短分支512和长分支32K的GPU源码SHA256在原路径与预绑定路径间完全一致，见
  [`gpu_source_identity.json`](gpu_source_identity.json)。改动只影响host接入。
- 专用latest环境运行4个相关测试文件：**39 passed、1 skipped**。跳过项是新接入明确
  不支持的MTP组合；测试日志为 [`tests.log`](tests.log)。

作业完成，GPU已释放。

原始30样本及环境、来源、正确性指标在 `decode_L32768_*.json`；
汇总在 [`results.csv`](results.csv)，元数据和数值检查在 [`validation.json`](validation.json)。
GPU小测试记录在 [`check_bound_gpu.json`](check_bound_gpu.json)。

## 复现

在已申请的 GHx4 节点，使用 `envs/activate_gh200_latest.sh` 和原
`scripts/run_decode_e2e_sweep_ghx4.sh` 中的环境变量，分别运行：

```bash
python -m scripts.bench_paper_e2e decode --implementation tileops \
  --kv-length 32768 --warmup 5 --repeat 30 --output NEW_DIRECTORY/decode_L32768_tileops
python -m scripts.bench_paper_e2e decode --implementation tileops_bound \
  --kv-length 32768 --warmup 5 --repeat 30 --output NEW_DIRECTORY/decode_L32768_tileops_bound
```

Nautilus 使用原脚本中的独立runtime overlay，再以相同参数运行
`--implementation nautilus`。每个output必须是未使用的新前缀。

本次仅重测 decode 32K 模型；没有重跑完整8长度曲线、Flex或MTP，也没有测生成服务延迟。
新接入针对固定地址、单query、无fused RoPE/sinks的官方decode kernel，不是通用动态推理接口。
