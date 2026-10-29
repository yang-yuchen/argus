# MTP 三方整步 CUDA Graph e2e

结果：[表格](results.md) · [CSV](results.csv) ·
[PDF](mtp_e2e_graph.pdf) · [PNG](mtp_e2e_graph.png)。

## 同 GPU 结果

| 实现 | 平均延迟 ms | 中位数 ms | 样本标准差 ms |
|---|---:|---:|---:|
| Nautilus | 17.089 | 17.088 | 0.009 |
| TileOPs | 19.011 | 19.009 | 0.030 |
| FlexAttention | 19.311 | 19.309 | 0.010 |

TileOPs 直接捕获原官方 API 成功，未添加 MTP 手工预绑定。其 Python 调度、
`torch.empty`及输出/workspace分配在准备和capture阶段执行，replay直接执行已捕获的GPU工作。
实际kernel仍是`GQADensePersistentWSKernel`，保留parts与merge计算及原始调度。
本次TileOPs略快于Flex；Nautilus相对TileOPs/Flex的延迟比为1.112×/1.130×。

## 工作负载和环境

- Slurm **3474393**，GHx4节点 **gh052**，三方同一张GH200 120GB。
- GPU UUID：`f7ae2b56-e98f-70ed-615e-4a112f25378f`。
- 随机权重Llama-7B backbone，32层，B1、FP16、hidden4096、H32、D128。
- 单次候选验证Q16，可见KV65536，prefix65520；每层8组H4 attention，共256组。
- 每配置compiled预热5次，capture stream预热3次；捕获、正确性检查及输入恢复后，
  replay再预热5次，正式采样30次，三方共 **90个正式样本**。
- 时钟覆盖整步`graph.replay()`及device-wide结束同步。图内包含新16token的KV写入、
  Q准备、全部attention/merge/分组结果拷贝、projection、MLP、norm和residual。
  prefix构建、编译、capture、检查、tokenizer、LM head、draft生成和acceptance在计时外。
- 固定cache位置，重复覆盖最后16个位置；本实验不是动态长度的完整生成服务。
- 专用latest环境：Torch2.14.1+cu130、Triton3.8.0、Transformers4.57.1。
  TileOPs使用TileLang0.1.15及匹配TVM；Nautilus在独立进程加载导出物对应的TVM0.21。

## 正确性与捕获证明

三方权重/输入hash、模型配置、GPU、软件主版本和协议一致；256组attention和完整
模型均通过独立FP32参考检查。每组KV地址常驻，TileOPs为BSHD，另两者为BHSD；
只写新增token，没有全量KV staging。

捕获后对两组随机新token及第一层历史V扰动进行检查。比较前污染全部256组共享Q
和可复用attention输出，要求replay重新计算；graph与正常模型的相对L2均为0.0。
历史V扰动导致约0.959的模型输出相对变化，证明replay读取了新的cache内容。
恢复原输入和prefix后，对FP32参考的相对L2分别为0.005805、0.005831、0.005842。

每组另profile两次replay，原始`*.graph.trace.json`和独立审计均确认：

| 实现 | 每forward总GPU kernels | Attention kernels | KV更新 kernels |
|---|---:|---:|---:|
| Nautilus | 1634 | 512 | 64 |
| TileOPs | 1634 | 512 | 64 |
| FlexAttention | 1890 | 768 | 64 |

所有GPU kernels均具有非零graph ID。Nautilus/TileOPs每组两个attention kernels；
Flex每组主kernel加两个reduction。分组输出拷贝包含在捕获和计时中。
Profiler样本单独保存，不参与上表30次正式wall均值。

## 来源与作业状态

实际Nautilus导出目录及SHA已与当前paper kernels重新核对；三方来源保存在各自JSON
和[kernel_manifest.json](kernel_manifest.json)。TileOPs仍是未修改官方源码commit
`85e7e644f1d99037e937378b8d8dc70c6568113f`，公共API为
`GQADenseFwdOp(is_causal=True,target=BUILTIN)`。

Slurm交互作业最终为`FAILED 0:9`，不能记为正常退出。三份完整结果的落盘时间分别是
23:38:46、23:40:02、23:41:27，作业终止于23:42:00；最后一份结果早于终止约33秒。
恢复后已独立核对所有JSON、原始trace、30样本、完整性和正确性，因此没有重新占用GPU。
现有记录无法确定终止信号来源。GPU已释放。

## 复现

在已申请的GHx4节点：

```bash
cd /u/yucheny8/argus
bash scripts/run_mtp_e2e_graph_ghx4.sh profiles_paper_e2e/NEW_MTP_GRAPH_RUN
```

汇总既有数据：

```bash
python -m scripts.summarize_mtp_e2e_graph profiles_paper_e2e/mtpc_graph_20261027_j3474393
```

MTP捕获/stream/解析回归16passed，三方汇总校验26passed。本次仅Q16/KV65536，未扫其他长度。
