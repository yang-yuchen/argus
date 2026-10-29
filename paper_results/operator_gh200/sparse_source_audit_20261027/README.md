# Sparse attention：源码、测量与 autotuning 核查

> 后续共享 KV GPU 复测已完成：Flex/Helion 全部 16 个新测点通过，见 [复测报告](../profiles_sparse_shared_kv_20261027/README.md) 和 [新 PDF](../plots_from_csv/gh200-bf16-tc1-sparse-d35a206-shared-kv.pdf)。调优范围沿用原设置，以下搜索范围不足的结论仍适用。

## 结论

旧图的数字是正确性检查通过、重复测量稳定的**特定实现与配置的耗时**，但不能称为充分调优后的 compiler baseline 性能。三个 baseline 均可追溯到官方源码；其中 TileLang 是 indexed sparse MLA gather kernel，FlexAttention 和 Helion 使用通用 BlockMask attention。算法对随机稀疏索引的处理方式不同。

另发现 Flex/Helion 的本地输入准备曾将 V 复制为独立连续张量，未保持原 workload 的共享 KV 存储。已修复 adapter，并完成修正布局后的 GPU 复测。因此旧 PDF 仅供历史参考，原始 JSON、CSV、PDF 数值均未改写；共享布局的结果另存。

## 1. 实际执行的 kernel 来源

| 图例 | 官方来源与固定版本 | 实际执行内容 | 本地 adapter 工作 |
|---|---|---|---|
| Nautilus | [neptune-internal `d35a206`](https://github.com/uiuc-arc/neptune-internal/tree/d35a206bf55a6fed0a56ba3fd41d3aba6afb1e44/paper_kernels) | 上游导出的各长度 `kernel.so`，`main_kernel_kernel` | 加载导出物、提供保存的输入、检查结果与测量 |
| TileLang | [tile-ai/tilelang `fdb19f2` sparse_mla_fwd.py](https://github.com/tile-ai/tilelang/blob/fdb19f26d61e71cabcc61abbc66ab60a97e3a100/examples/deepseek_v32/sparse_mla_fwd.py) | 未改写的官方 indexed sparse MLA 示例，`sparse_mla_fwd_kernel` | 添加 batch/group 视图；将 padding -1 转为官方 sentinel L；调用官方 autotuner 搜索本地参数集合 |
| FlexAttn | [pytorch/pytorch `5c48869` public API](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/attention/flex_attention.py)，[实际 Triton 模板](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/kernel/flex/templates/flex_attention.py.jinja) | 官方 public API 经 Inductor 生成的 `triton_tem_fused_flex_attention_0` | supplied indices 转为精确 membership BlockMask；布局转换、scale、GQA 头映射、固定 tile 与调优选项 |
| Helion | [pytorch/helion `429c002` examples/flex_attention.py](https://github.com/pytorch/helion/blob/429c00227b9802538be459b918ee8d34ccbf636c/examples/flex_attention.py) | 未改写的官方 FlexAttention 示例，`_helion_helion_flex_attention_kernel` | 同样的 BlockMask；用户 mask callback 用官方 `hl.load` 读取 membership；配置调优预算 |

三条 baseline 的 attention 算法源码均未自行改写。mask callback、输入适配、调优空间是本地配置，不能把它们描述成官方为该 D576/V512/H16 随机 top-k workload 推荐的最优配置。Helion 此处也不是之前讨论的 tuned dense/paged attention 实现。

从 GitHub 独立下载并比对了 7 个实际相关源码文件，全部字节一致，见 [official_source_checks.json](official_source_checks.json)。再次检查上游 HEAD：TileLang `af08dae`、Helion `3c87497` 的相关示例没有变化；TileOPs `4062bf9` 的 DSA decode 文件和依赖声明也没有变化，见 [upstream_head_checks.json](upstream_head_checks.json)。这里的 HEAD 结论只覆盖列出的文件，并不代表整个仓库没有更新。

## 2. 为什么长序列的 Flex/Helion 很慢

使用本次实验保存的真实 indices 检查 128×128 BlockMask：八个长度的**所有 causal 块都非空**。随机稀疏连接散布在整个可见前缀，通用块稀疏实现几乎无法跳过 causal 块；Nautilus/TileLang 则直接 gather 选中的 KV。

| L | 非空 causal 块 / 全部 causal 块 | 块覆盖位置数 / 实际有效连接数 |
|---:|---:|---:|
| 256 | 3 / 3 | 1.49 |
| 512 | 10 / 10 | 1.25 |
| 1024 | 36 / 36 | 1.12 |
| 2048 | 136 / 136 | 1.06 |
| 4096 | 528 / 528 | 1.37 |
| 8192 | 2080 / 2080 | 2.32 |
| 16384 | 8256 / 8256 | 4.30 |
| 32768 | 32896 / 32896 | 8.29 |

32K 每个 head 有 65,012,736 个有效连接，非空块覆盖 538,968,064 个位置。8.29 是块覆盖位置数之比，不是测得的指令数、FLOPs 或全部延迟的归因比例。完整数据见 [blockmask_occupancy_audit.json](../profiles_sparse_compilers_20261027/blockmask_occupancy_audit.json)。

长序列 L 翻倍时，Nautilus/TileLang 大致增至 2 倍，Flex 大致 4 倍、Helion 约 4–5 倍，符合 indexed gather 与近乎 dense causal block 遍历的差别。Flex 的实际选中配置还将 QK D576 物理补齐到 1024；不能只用逻辑稀疏度解释其代价。

32K 保存的 profiler 数据：

| 实现 | 最终均值 ms | registers/thread | dynamic shared memory bytes |
|---|---:|---:|---:|
| Nautilus | 12.210 | 252 | 94,976 |
| TileLang | 13.621 | 240 | 95,232 |
| FlexAttn | 243.759 | 255 | 196,608 |
| Helion | 353.181 | 255 | 131,072 |

50 次 cold-L2 样本的 CV 约 0.3–1.3%，均值接近中位数；未发现单位换算错误。寄存器和共享内存使用较高，但现有数据不足以证明发生寄存器 spill，不能把所有差距归因于某一种硬件瓶颈。

## 3. Autotuning 是否足够：不能确认，现有搜索有明确不足

### FlexAttention

最终曲线使用官方 `max_autotune=True`，但 `BLOCK_M=32, BLOCK_N=64, num_warps=8` 固定。日志虽显示 7 个 choices，tile 几何相同，语义不同的配置仅为 pipeline stages 1/2/3。较早在部分长度做过 unrestricted EXHAUSTIVE 搜索，不等于最终每个长度都搜索了完整空间。

因此现有结果只能称为固定几何下的官方流水级数调优。32K 日志中 autotuner 分数约 952 ms，而最终 profiler 约 244 ms；后续官方源码核查找到了约 4 倍差异的主要机制：`common.py` 的 fake-input generator 将 partial/full 两套 block counts 都填为 256、indices 都填为 0..255；模板会分别遍历两套列表，不会去重。于是 tuner 输入遍历 `256*(256+256)=131072` 个 KV blocks，真实 causal BlockMask 只有 32896 个，比例 3.9844，接近测得的 952/243.759=3.906。两者使用的块遍历负载不同，不应把 tuner 分数当成真实输入延迟；full-block mask 开销、fake 数据和计时方式的剩余影响尚未逐项量化。修复真实输入的共享 storage，也不等于已经确认 autotuner 创建的 benchmark inputs 保留同样 alias；这项同样需要检查。

### Helion

使用官方 full effort，但每个长度设置软预算 180 秒。实际搜索与成功候选数来自保存的日志：

| L | searched | 未报告编译失败的候选 | 最终选择情况 |
|---:|---:|---:|---|
| 256 | 84 | 40 | final selection skipped |
| 512 | 86 | 29 | budget exceeded；final selection skipped |
| 1024 | 66 | 20 | budget exceeded；final selection skipped |
| 2048 | 72 | 28 | 2 finalists × 2 rounds |
| 4096 | 88 | 35 | final selection skipped |
| 8192 | 52 | 18 | 8 finalists × 2 rounds |
| 16384 | 26 | 10 | 8 finalists × 2 rounds |
| 32768 | 6 | 2 | 2 finalists × 4 rounds；选中 pinned seed |

尤其 32K 仅搜索 6 个、其中 4 个编译失败，不能称为充分调优。这里的候选计数不是逐候选数值正确性验证：官方示例关闭 tuner 内部 accuracy check；最终选中 kernel 另行通过独立 FP64 正确性检查。预算也不是硬性 wall-clock 上限。

### TileLang / TileOPs

图中的 TileLang 只搜索 4 组本地配置：`block_I=64`，`num_stages={1,2}`，`threads={128,256}`；各长度均选 `stages=1, threads=128`。官方示例默认是 `64/2/256`，这次校准确有提速，但没有测试其他 block_I、64 threads 或更多流水级数，最低测试边界获胜，不能据此称为充分搜索。官方示例本身没有内置推荐 autotune 空间。

TileOPs 的官方 `DSADenseFwdOp` 在当前环境曾尝试官方 4 个 WGMMA 配置，H16 路径因 M16/WGMMA 约束编译失败。实际环境为 Torch 2.14.1 + TileLang 0.1.15，而 TileOPs 声明 Torch >=2.13,<2.14、TileLang==0.1.12。因此该失败不能外推为 TileOPs 在受支持环境也无法运行，更不能证明当前 TileLang 示例是所有官方实现中最快的选择。官方还导出了 `DSADecodeBasicKernel`，其直接实例化属于应补查的官方内部 kernel 候选；当前未完成这条路径的编译与测量，不能把它描述为已经排除或默认推荐的路径。

## 4. 已修复的共享 KV 布局问题

原公共准备函数对 `kv[:, :512]` 调用 `.contiguous()`，把 V 复制到独立连续存储；这与 Nautilus/TileLang 原生共享 KV 的物理布局不同，尽管数值语义相同。准备时间未计入 kernel 时间，但存储大小、读缓存行为仍可能改变性能。

已删除这个复制，V 保留 `[1,1,L,512]` 视图，sequence stride=576、最后一维 stride=1，并与 K 共享 storage。Flex/Helion 的 source metadata 现在记录真实 storage alias；已有续跑和汇总入口会拒绝旧布局结果被当成修正布局结果复用。CPU 回归检查 alias、stride、修改 KV 后 V 的对应值同步变化。

验证命令：

```text
/work/nvme/beqg/yucheny8/envs/argus-gh200-latest/bin/python -m pytest tests/test_sparse_flex_official.py tests/test_sparse_helion_official.py tests/test_sparse_summary.py -q
9 passed
```

源码核查中，Flex 支持 sequence stride=576、最后一维 stride=1 的 V；Helion 也未要求 V contiguous，并通过 tensor indexing 读取其 strides。上述 CPU 测试验证 adapter 和结果接纳条件；后续 GPU 复测另行确认两种方法的全部 8 个长度均可编译、数值正确、共享 storage 并完成 50 次测量，见文首链接。旧数据保留，不能重命名为修复后的结果。

## 5. 结果使用范围与后续复测

1. 先在同一张 GPU 上做 copied-V 与 shared-view 的配对实验，确认修复后的 GPU 编译、正确性和耗时。
2. TileLang 扩展公开参数搜索；Flex 扩展每个长度的几何搜索；Helion 延长预算并保留有效候选数和最优值随搜索进度的变化，观察是否稳定。继续调用官方 kernel 与官方 tuner。
3. TileOPs 使用其声明支持的依赖组合重试官方 public dispatch；若显式选择其他官方 kernel，需要记录选择方式，不能改写 kernel 或伪装 GPU 架构。
4. 使用现有 benchmark、相同输入与 cold-L2 协议测量，再用现有 `plot_from_csv.py` 更新图。

后续已执行共享视图 GPU 复测：job3473409 在独立分配的 gh093 上完成 Flex/Helion 16 个点，未与其他测量共用同一 GPU。没有扩大调优范围，按用户要求保留 TileLang 现有配置和测量。旧实验跨 gh079/gh004，新旧布局也未做同 GPU 配对控制；不能将跨运行的延迟变化全部归因于布局，也不能宣称 Nautilus 与 TileLang 的差距已经排除了跨节点影响。
