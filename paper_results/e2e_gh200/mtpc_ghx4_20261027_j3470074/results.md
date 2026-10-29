# Causal MTP e2e — GHx4 job 3470074

| Workload | 平均 ms | 中位数 ms | 最小 ms | 采样次数 |
|---|---:|---:|---:|---:|
| Nautilus mtpc，B1 / Q16 / KV65536 | 58.996 | 58.872 | 58.838 | 20 |

## 模型与计时范围

- 随机初始化 Llama-7B backbone：32 层、hidden4096、32 个 MHA heads、D128，FP16。
- 真实 prefix forward 生成 65520-token KV cache；每次 timed forward 追加16个候选token。
- 使用 `paper_kernels/mtpc_Q16_L65536/tilelang/best/export/kernel.so`。
- 右下对齐 causal 条件为 `key_position <= 65520 + query_index`。
- 每层32 heads分成8组调用4-head kernel；分组调用、输入 staging、输出复制都计入延迟。
- 预热3次，采样20次，`torch.compile(dynamic=False, fullgraph=False)`；
  Dynamo recompile_limit=300，日志未出现达到重编译上限或回退。
- 口径为**同步整模型 wall latency**：包含CPU dispatch、cache append、attention、
  projection、MLP、norm，以及结束处的GPU同步；稳态，不显式flush L2。
- 初始化、prefix计算、cache重建/复制和编译在计时外；不含tokenizer、embedding lookup、
  LM head、draft model、接受/拒绝和采样，不代表完整speculative decoding服务延迟。

## 正确性

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| 32层 attention 检查中最大相对 L2 | 0.00025047（0.0250%） | <1% |
| 整模型相对 L2 | 0.00439482（0.439%） | <1% |
| 编译后整模型相对 L2 | 0.00578786（0.579%） | <1% |
| 整模型最大绝对误差 | 0.021484375 | 记录值 |

逐层检查还要求对独立FP32 reference满足allclose（rtol=0.01、atol=0.003）。
完整模型与使用相同FP16投影/MLP、FP32 attention reference的模型比较。
`correct=true`、`measurement_complete=true`；全部20个样本为有限正数，均值已从
原始样本重新计算核对。

## 环境和来源

- Slurm job **3470074**，`ghx4-interactive`，节点 `gh068`；完成状态 `COMPLETED / 0:0`。
- NVIDIA GH200 120GB，驱动报告可用显存97871MiB。
- GPU UUID：`fb84f198-9785-3f9f-fef5-777107d8c5f1`。
- Torch2.9.1/cu126、Transformers4.57.1、Triton3.5.1、Neptune TVM0.21。
- 权重 SHA256：`35a9b4db19947bed874240e6d9783fc2f22c11ba2d4ecd02e71b01e1c7034e7b`。
- 输入 SHA256：`20abb34c720e06ef28609300c8111da73a18cba533452072d151fd7d56350514`。
- Kernel SHA256：`8595aebc3c6108a7520eca00107f1631aae1da473cbab3cb0c438b7408670200`。

原始20次样本和完整配置：`mtpc_L65536.json`；初始化、正确性、编译及测量日志：
`mtpc_L65536.log`；精确启动环境与命令：`run.sh`；简表：`results.csv`。

## 复现

在新的GHx4 allocation中使用同一环境、换一个未存在的output前缀：

```bash
python -m scripts.bench_paper_e2e mtpc --kv-length 65536 \
  --implementation nautilus --warmup 3 --repeat 20 --output NEW_PREFIX
```

GPU在完成后已释放。本次只测64K causal MTP，不含baseline对照或长度扫描。
随机权重且仅扩大RoPE位置上限，这个实验验证执行正确性与延迟，不验证长文本质量。
