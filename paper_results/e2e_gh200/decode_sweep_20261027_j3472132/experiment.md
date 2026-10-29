# Decode e2e：8长度、最新paper kernels

正式结果：[results.md](results.md)；原始长表：[results.csv](results.csv)；
图：[decode_e2e.pdf](decode_e2e.pdf)。24份JSON和三个backend日志保存在本目录。

## 实验

- Slurm **3472132**，`ghx4`，节点`gh084`，GH200 120GB。
- GPU UUID：`e2c32e81-b20c-857d-5433-5a22ca09faae`。
- 随机初始化Llama-7B backbone，32层、hidden4096、32个MHA heads、D128，FP16。
- batch=1，query长度始终为1；总可见KV长度扫256、512、1K、2K、4K、8K、16K、32K，
  prefix分别为L−1。每个长度使用一个固定随机输入，三种实现使用同一权重和输入。
- 每配置预热3次、采样20次；**24配置，共480次正式采样**。
- 完整模型`torch.compile(dynamic=False, fullgraph=False)`，recompile_limit=300。
  正式日志无重编译上限回退、OOM或运行错误。
- 同步wall latency：开始前同步，覆盖整个forward及结束同步。包含CPU dispatch、
  新token的KV写入、projection、attention、MLP、norm和小Q准备。
- 初始化、prefix计算/布局准备、编译在计时外；不含tokenizer、embedding lookup、
  LM head、采样或服务请求排队，不是完整生成服务延迟。

跨8长度，Nautilus相对TileOPs/Flex的几何平均加速为 **1.240× / 1.159×**。
各长度比值范围为 **1.096–1.352× / 1.061–1.214×**。这只表示本次观测的平均延迟；
保留所有样本，不做删点、平滑或强制单调。

后续[baseline一致性审计](audit/kernel_consistency.md)确认：kernel图中标作TileLang的
也是同一套TileOPs。32K时attention GPU时间仅差约0.100ms/forward，而整模型wall
时间差7.684ms；当前加速包含public TileOPs API与预绑定Nautilus调用的host路径差异，
不能解释为attention CUDA kernel本身获得了相同比例的加速。

## 原生cache和baseline环境

三方均复用已验证的常驻原生cache：TileOPs为BSHD、Nautilus/Flex为BHSD，每次仅写入
新增1token。decode H32导出物与模型H32匹配，每层一次attention，共32个常驻KV绑定，
不需要MTP那样的8组调用，也没有整份KV staging。

三组均来自专用`argus-gh200-latest`：Torch **2.14.1+cu130**、Triton **3.8.0**，
Transformers **4.57.1**通过现成HF依赖overlay提供；没有向原环境安装新依赖。
TileOPs使用TileLang **0.1.15**和其匹配TVM，Nautilus在独立进程加载匹配导出物的
Neptune TVM0.21。allocator统一为`expandable_segments:True`。

- TileOPs：未修改官方源码，commit`85e7e644f1d99037e937378b8d8dc70c6568113f`，
  `GQADenseFwdOp(is_causal=True,target=BUILTIN)`，各长度按官方API自动选kernel。
- Flex：官方`flex_attention`，`fullgraph=True, mode="max-autotune-no-cudagraphs"`，
  Q1位于最后位置、可见全部KV，因此不构造或传入block mask，自动使用FlexDecoding。
- 官方API的输出/workspace分配保留在e2e中；所有方法的新增KV写入计时口径相同。

每种实现的8个长度在一个Python进程中依次执行，避免反复加载依赖。每点结束会解除
注册hook对旧cache的引用、reset Dynamo并释放空闲GPU缓存，防止模型间显存累积。

## 最新导出物保障

kernel benchmark和e2e共用`workloads.resolve_export`；有更新的`tilelang_auto`二进制
时优先选它。decode缺少当前binary直接报错，不回退`paper_kernels.previous_*`。

| KV长度 | 实际导出目录（省略paper_kernels前缀） | SHA256前16位 |
|---|---|---|
| 256 | decode_L256/tilelang/best/export | 1e822280840b3ad7 |
| 512 | decode_L512/tilelang/best/export | 5c5537790c0ed2a9 |
| 1K | decode_L1024/tilelang/best/export | 8f938488ff51f16e |
| 2K | decode_L2048/tilelang/best/export | e6acd1ef4f97bc4c |
| 4K | decode_L4096/tilelang/tilelang_auto/best/export | bb369b2e703f8e4b |
| 8K | decode_L8192/tilelang/tilelang_auto/best/export | 2a518546102a4453 |
| 16K | decode_L16384/tilelang/tilelang_auto/best/export | 8f3666e0f2d3ced5 |
| 32K | decode_L32768/tilelang/tilelang_auto/best/export | b1bb79ca00294719 |

完整路径和完整SHA256保存在`kernel_manifest.json`及每个点的JSON中，已与实际当前文件
重新计算核对。较短4个长度保留当前release的原best binary，较长4个使用新版auto。

## 正确性及数据gate

24点均通过逐层attention allclose（rtol0.01、atol0.003）及相对L2<1%；完整模型与
编译后模型均对独立FP32 attention reference检查相对L2<1%。没有仅依靠`correct`标记。

汇总器还独立核对：24份结果完整、每点20个有限正样本、32个常驻KV绑定、零全KV
staging、原始误差值、实际最新binary路径和SHA256、每长度三方模型/输入hash一致、
跨长度模型权重/软件/GPU一致。配置只允许max_position_embeddings随长度扩展。
旧Torch2.9、DynamicCache的单点数据没有混入本次曲线。

## 独立32K GPU审计

正式采样后，在同GPU分别对TileOPs/Nautilus额外profile两次32K forward。
审计延迟不用于正式表，原始trace及结果位于`audit/`。

两者均无≥64MiB GPU复制或临时分配；最大`aten::copy_`为4096个FP16元素，最大GPU
memcpy为8192字节（8KiB）。KV更新kernel仅处理新增token，两次forward共128个
index-copy融合kernel，对应每层K/V各一次写入。每forward新增KV payload共512KiB，
没有把32K完整KV重复搬运。证据为`audit/summary.json`和原始`*.trace.json`。

## 复现和绘图

```bash
cd /u/yucheny8/argus
srun --account=beqg-dtai-gh --partition=ghx4 --time=01:30:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G --gpus-per-node=1 \
  bash scripts/run_decode_e2e_sweep_ghx4.sh profiles_paper_e2e/new_decode_sweep
```

只汇总/重画现有数据，使用latest环境：

```bash
python -m scripts.summarize_decode_e2e profiles_paper_e2e/decode_sweep_20261027_j3472132
```

横轴为 **KV cache length (tokens)**，log2等间距8刻度，Q始终为1；纵轴为同步整模型
延迟ms；一行decode，三条backend曲线。不是旧图的TFLOPs/s，也不扩展未测B8/FP8。

作业完成退出0，GPU已释放。本次仅测这个模型、B1和FP16；没有测训练权重质量、
完整生成服务或其他operators。
