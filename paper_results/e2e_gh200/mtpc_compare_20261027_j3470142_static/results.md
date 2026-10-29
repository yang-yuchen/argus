# 64K causal MTP e2e：Nautilus / TileOPs / FlexAttention

**已被修正结果替代**：本次TileOPs接入多做了一遍不必要的BHSD复制，影响86.365ms和
2.187×倍率。修复后同卡重新测量见
[job3470667正式结果](../mtpc_layout_fixed_20261027_j3470667/results.md)，
[原因与GPU trace](../mtpc_layout_investigation_j3470667/results.md)。以下保留原始历史记录。

正式数据为本目录的三组JSON和日志。其他同job目录是显存诊断记录，不参与本表。

| 实现 | 平均 ms | 中位数 ms | Nautilus相对该实现的加速 |
|---|---:|---:|---:|
| Nautilus `mtpc_Q16_L65536` | 39.486 | 39.486 | 1.000× |
| 官方 TileOPs | 86.365 | 86.087 | 2.187× |
| 官方 FlexAttention | 56.510 | 56.950 | 1.431× |

## Workload与计时

- 同一张GH200，Slurm **3470142**，`ghx4-interactive`，节点`gh151`。
- GPU UUID：`5fec3f12-dcb5-1e2c-ff5b-713d0f6aa983`，可用显存97871MiB。
- 随机权重Llama-7B backbone：32层、hidden4096、32个MHA heads、D128，FP16。
- B1，真实prefix65520 tokens，追加16个候选，总KV65536。
- 每层32 heads按8组4-head调用，保持kernel对比中的Q/KV形状。
- 右下causal：`key_position <= 65520 + query_index`。
- 每组预热3次、采样20次；完整模型`torch.compile(dynamic=False, fullgraph=False)`，
  recompile_limit=300。三份日志均无重编译上限回退、OOM或编译失败。
- 计时为同步整模型wall latency，含CPU dispatch、当前token的KV写入、projection、
  attention输入staging、head分组/输出复制、MLP、norm和结束同步；稳态，不flush L2。
- 初始化、prefix计算、cache准备、编译在计时外；不含embedding lookup、LM head、
  draft model、接受/拒绝、采样，不是完整speculative decoding服务延迟。

## 与kernel对比相同的baseline

### TileOPs

未修改`.tileops_work`，commit **85e7e644f1d99037e937378b8d8dc70c6568113f**。

```python
op = GQADenseFwdOp(is_causal=True, target=BUILTIN)
out = op(q_bshd, k_bshd, v_bshd)
```

与`scripts/bench_official_tileops_dense.py`相同API和默认选择，实测选中的也是
**GQADensePersistentWSKernel**，`kernel_config=null`。Q为`[1,16,4,128]`、
K/V为`[1,65536,4,128]`。每次用最新模型投影刷新BSHD buffers；布局复制和官方
op自身的output/workspace分配计入e2e。因此这个加速比包括模型接入和布局成本，
不等同于纯kernel加速比。

### FlexAttention

与`scripts/bench_official_flex_mtpc.py`完全相同：

```python
mask = create_block_mask(lower_right, B=None, H=None, Q_LEN=16, KV_LEN=65536,
                         device=q.device)
call = torch.compile(flex_attention, fullgraph=True,
                     mode="max-autotune-no-cudagraphs")
out = call(q_bhsd, k_bhsd, v_bhsd, block_mask=mask)
```

`lower_right`为`q_idx + (KV_LEN-Q_LEN) >= kv_idx`，在inference mode运行；
mask创建和autotune在计时外，不使用decode路径的不同compile flags。

## 环境和cache设置

三组均使用kernel对比的`argus-gh200-latest`：Torch **2.14.1+cu130**、Triton **3.8.0**、
TileLang **0.1.15**；Transformers **4.57.1**从现成`nt`包以隔离symlink overlay提供。
未安装新包或修改原环境。Nautilus子进程单独加载Neptune TVM0.21的原始导出物；
TileOPs使用匹配的TVM0.25.dev0。两个TVM不在同一进程内混用。

64K DynamicCache在新PyTorch编译过程中保留多份大块KV，导致超出显存。本轮三组
统一用HF **StaticCache**：先复制真实prefix，再每次只覆盖末16个位置，prefix保持
不变。KV写入仍在计时内，所有backend相同。allocator统一为
`PYTORCH_ALLOC_CONF=expandable_segments:True`。

此前job3470074的59ms采用Torch2.9和DynamicCache；它与本轮设置不同，不用于本表对照。

## 正确性和一致性

| 实现 | 整模型相对L2 | 编译后相对L2 | 各层attention检查中的最大相对L2 |
|---|---:|---:|---:|
| Nautilus | 0.004423 | 0.005805 | 0.0002511 |
| TileOPs | 0.004461 | 0.005831 | 0.0002490 |
| FlexAttention | 0.004403 | 0.005800 | 0.0002444 |

每层attention还要求allclose（rtol0.01、atol0.003）；整模型及编译后输出要求相对
L2<0.01。三组全部通过，所有20个样本为有限正数。

汇总已强制检查相同模型配置、权重/输入hash、GPU UUID、Torch/Transformers版本、
cache实现、预热/采样数量、计时口径及对比kernel hash：

- 权重SHA256：`35a9b4db19947bed874240e6d9783fc2f22c11ba2d4ecd02e71b01e1c7034e7b`
- 输入SHA256：`20abb34c720e06ef28609300c8111da73a18cba533452072d151fd7d56350514`
- Nautilus kernel SHA256：`8595aebc3c6108a7520eca00107f1631aae1da473cbab3cb0c438b7408670200`

原始样本及来源：`mtpc_nautilus.json`、`mtpc_tileops.json`、`mtpc_flex.json`；
完整日志为同名`.log`；汇总为`results.csv`，环境overlay链接清单为`environment.json`。

## 复现

```bash
cd /u/yucheny8/argus
srun --account=beqg-dtai-gh --partition=ghx4-interactive --time=01:00:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G --gpus-per-node=1 \
  bash scripts/run_paper_e2e_mtpc_ghx4.sh profiles_paper_e2e/new_mtpc_compare
```

CPU针对性回归7项通过；GPU job完成后已释放。本次只测B1/Q16/KV65536，没有扫长度、
没有验证预训练模型质量；8次分组调用和布局复制的开销保留在结果中。
