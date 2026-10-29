# MTP TileOPs性能异常：布局与重复复制诊断

## 结论

上一轮使用的确实是专用`argus-gh200-latest`环境；主要问题是接入层多做了一遍KV复制。
修复后TileOPs仍比Flex的e2e慢，剩余差距主要来自HF BHSD布局到TileOPs连续BSHD输入的
转换，不能把e2e差距解释成attention kernel本身慢。

## 同卡受控复现

Slurm **3470667**，`ghx4`节点`gh061`，GPU UUID
`7da4881e-6b00-d83a-ca6c-b48e345bb5e6`。随机权重Llama-7B，32层，B1/Q16/KV65536，
StaticCache，每层按8组4-head调用；预热3次、20次同步整模型wall采样。

| TileOPs接入方式 | 平均ms | 中位数ms |
|---|---:|---:|
| 旧：模型BHSD→BHSD staging→BSHD staging→官方op | 86.564 | 86.557 |
| 修复：模型BHSD→原生BSHD staging→官方op | 65.774 | 65.774 |

同一GPU、模型、输入、软件环境和kernel，去掉中间复制实际减少 **20.790ms（24.0%）**。
旧路径数据为`old_tileops_same_gpu.json`，由`reproduce_old_e2e.py`保留所有新代码和
校验、仅恢复旧的BHSD中转buffer生成；修复路径来自相邻正式三方目录。

## 原因

原`AttentionCore.__call__`先把模型QKV复制到连续BHSD buffers；TileOPs helper随后又
把这些buffers复制成连续BSHD。Flex只做第一步。每组额外复制K/V128MiB，32层×8组
使每次forward额外复制约 **32GiB数据、产生约64GiB读写流量**。

修复仅改变输入buffer布局：直接分配连续BSHD，再以BHSD view暴露给统一core接口；
原有`copy_(fresh_model_tensor)`直接完成一次必要的布局转换。helper发现输入与原生
BSHD共用storage时不再重复copy；普通BHSD调用者仍刷新输入。没有把转换搬到计时外，
没有改变TileOPs API、官方源码、自动kernel选择、shape或head分组。

## GPU分阶段证据

`profile_core.py`在同一GPU、专用latest环境中，按HF实际stride构造H4/Q16/KV65536。
20次CUDA Event和20次独立同步wall计时保存在`core.json`；两次每路径的Chrome trace
保存在`core_trace.json`。`trace_summary.json`通过External id把CUDA kernel和D2D
memcpy关联到CPU阶段，包含两种GPU活动，未将CPU range长度当成GPU执行时间。

### Trace中的纯GPU活动（每组，微秒）

| 阶段 | GPU活动时间us | 活动 |
|---|---:|---|
| 旧路径多余的BHSD staging | 76.046 | 小Q copy kernel + 两次64MiB D2D |
| 旧路径BHSD→BSHD转换 | 181.499 | 三个copy kernels |
| 修复后直接写原生BSHD | 178.347 | 三个copy kernels |
| Flex直接BHSD staging | 74.526 | 小Q copy kernel + 两次64MiB D2D |
| TileOPs纯attention，修复后输入 | **48.959** | parts + main |
| Flex纯attention | **53.983** | 官方Flex kernels |

即TileOPs纯kernel仍快约5us，但其必要布局转换比Flex直接复制多约104us/组，足以
覆盖这个kernel优势。事件区间还可能包含host dispatch gaps，不能把CUDA Event时间
称为纯kernel时间。上表trace只有两次诊断调用，不作为大样本kernel基准替代。

### 20次同步wall计时（单组，中位数）

| 路径 | us |
|---|---:|
| 旧TileOPs双复制+op | 338.038 |
| 修复后TileOPs单次原生转换+op | 250.841 |
| Flex单次BHSD staging+op | 156.491 |

独立阶段会有缓存/重叠差异，不能线性乘256来精确重建整模型时间；上面的20.790ms
改善来自实际完整模型受控复现。

## 修复后的正式三方结果

正式目录：`../mtpc_layout_fixed_20261027_j3470667/`。

| 实现 | 平均ms | Nautilus相对该实现的加速 |
|---|---:|---:|
| Nautilus | 39.474 | 1.000× |
| TileOPs | 65.774 | **1.666×** |
| FlexAttention | 51.929 | **1.316×** |

此前2.187×的TileOPs对比受不必要的复制影响，已由本次结果替代。

## 环境、正确性和来源

- Python来自`/work/nvme/beqg/yucheny8/envs/argus-gh200-latest/bin/python`。
- Torch2.14.1/cu130、Triton3.8.0、TileLang0.1.15；Transformers4.57.1通过现有包overlay提供。
- TileOPs仓库无本次改动，commit85e7e644f1d99037e937378b8d8dc70c6568113f。
- API保持`GQADenseFwdOp(is_causal=True,target=BUILTIN)`，仍选
  `GQADensePersistentWSKernel`、kernel_config=null。
- Flex保持官方lower-right mask、fullgraph=True、max-autotune-no-cudagraphs。
- 三方和旧路径均通过逐层/整模型/编译后正确性检查；完整模型权重、输入hash一致。
- 新增CPU回归 **9 passed**，检查零重复copy、storage alias、fresh输入、cache和mask。
- 作业完成后GPU已释放。仅诊断此64K workload，没有扫其他模型或长度。

复现三方：`bash scripts/run_paper_e2e_mtpc_ghx4.sh NEW_OUTPUT_DIRECTORY`，在GHx4 allocation中运行。
复现阶段诊断：在激活latest环境后运行本目录`profile_core.py --output NEW_JSON --trace NEW_TRACE`。
