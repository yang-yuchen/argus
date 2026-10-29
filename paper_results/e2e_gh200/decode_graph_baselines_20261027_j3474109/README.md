# Nautilus / FlexAttention：整步 CUDA Graph replay

已完成两个backend的全部8长度，共 **16配置、480个正式样本**。
三方对比：[表格](comparison.md) · [CSV](comparison.csv) ·
[PDF](decode_e2e_graph_comparison.pdf) · [PNG](decode_e2e_graph_comparison.png)。

原throughput图样式的吞吐量单子图：[PDF](decode_e2e_graph_throughput_paper_style.pdf) ·
[PNG](decode_e2e_graph_throughput_paper_style.png) ·
[TFLOPs/s绘图CSV](decode_e2e_graph_throughput_paper_style.csv)。复用
`/u/yucheny8/neptune-internal/scripts/csv_figures.py`的`plot_subplot`，保留分组柱状图、
配色、字体、指数刻度和顶部图例；纵轴为Throughput (TFLOPs/s)。复用既有24点graph
replay延迟，不重跑GPU。理论FLOPs沿用原完整模型统计项目，按实际Q1/KV矩形attention
扩展：projection/MLP只计1个query，attention计全部可见KV。TFLOPs/s=FLOPs/mean_ms/1e9。
原`decode_e2e_graph_paper_style.*`保留为旧延迟图，勿与吞吐量版本混淆。
重绘：`python -m scripts.plot_decode_e2e_paper_style`。

## 实验

- GHx4作业 **3474109**，节点 **gh052**，Nautilus/Flex使用同一张GH200 120GB。
- GPU UUID：`f7ae2b56-e98f-70ed-615e-4a112f25378f`；作业完成0:0，GPU已释放。
- 两者都在专用latest环境运行：Torch2.14.1+cu130、Triton3.8.0、Transformers4.57.1。
  Nautilus通过独立进程加载匹配最新paper导出物的TVM0.21；Flex使用官方Torch API。
- Llama-7B backbone、32层、B1、FP16、H32、D128、Q1；横轴为含当前token的总可见KV长度。
- 每点普通compiled预热5次、side-stream预热3次；捕获完整forward，完成正确性检查并恢复
  原始输入后，再预热5次replay，正式采样30次。prefix、编译、capture、检查均在计时外。
- 计时覆盖`graph.replay()`及device-wide结束同步；graph中包含新增KV写入、projection、
  attention、MLP、norm及residual。LM head、tokenizer、采样不在backbone benchmark中。

## 捕获与正确性

Nautilus使用旧TVM自身的`device.set_raw_stream(int_handle)`与Torch capture stream对齐，
退出时恢复外层Torch stream；没有更换或重编译paper kernel。
Flex使用普通Torch stream/capture context，内部compile保持原来的
`fullgraph=True, max-autotune-no-cudagraphs`，整步graph由benchmark统一捕获。

16点均通过独立FP32 attention/模型检查，并通过两组新token及历史V扰动检查。
比较前污染共享Q以及持久化输出（如果有），要求replay重新计算；graph-versus-normal的
相对L2在本次所有检查中均为0.0。恢复输入后的模型相对L2最大值：
Nautilus0.006474、Flex0.006098，均小于1%。

每点正式采样后，另profile两次replay，原始`*.graph.trace.json`保存在本目录：

- 所有GPU kernels均属于graph；每个forward都有64个KV index-copy kernels。
- Nautilus每forward有64个attention kernels，即32层各两个。
- Flex每forward有96个attention kernels，即主decode模板和两个reduction各32个；
  官方API的ones/zeros辅助kernels也在完整graph和总延迟中。
- 32个KV绑定均常驻BHSD，只写新token；没有全KV重复staging。

Nautilus逐点核对实际加载目录及SHA256，均为当前`paper_kernels`选择的最新binary。
来源记录见[kernel_manifest.json](kernel_manifest.json)。

## 对比数据与复现

TileOPs曲线保留自先前作业3473944、gh004，使用相同的整步replay口径、30samples/5warmups。
三方权重hash、输入hash、模型配置、Torch/CUDA/Transformers逐点一致。
CSV保留run、host、GPU UUID；图中明确标注TileOPs为跨作业/不同GPU参考，不能把与Nautilus
的小幅差异当作同GPU隔离实验。Nautilus与Flex本次为同GPU对照。

独立曲线与表格：

- [Nautilus](results_nautilus.md) · [CSV](results_nautilus.csv)
- [FlexAttention](results_flex.md) · [CSV](results_flex.csv)

在已申请的GHx4 GPU上，复用通用runner，第二参数指定backend：

```bash
bash scripts/run_decode_e2e_graph_sweep_ghx4.sh NEW_NAUTILUS_DIRECTORY nautilus
bash scripts/run_decode_e2e_graph_sweep_ghx4.sh NEW_FLEX_DIRECTORY flex
```

runner设置latest环境，并仅为Nautilus添加匹配的旧TVM overlay。
新backend路由、stream恢复及捕获解析测试19passed，graph汇总测试38passed。
本次测固定cache位置的单步decode，没有扩展动态生成循环或MTP。
