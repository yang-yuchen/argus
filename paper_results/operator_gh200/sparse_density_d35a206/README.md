# Sparse attention 密度核对（d35a206）

范围：最新 `sparse_L256` 到 `sparse_L32768`，不是旧的 `sparse` / `sparse_split` 固定案例。

## 输入规则与精确计数

上游 `python/attn_variants.py::causal_topk_indices` 从每个 query t 的 `[0,t]` 位置选择不同的列；top-k 不产生重复列，未来位置被排除。`T=KV=L`、`K=min(L,2048)`，有效项数为 `min(t+1,K)`，剩余位置填 `-1`。16 个 query heads 共用索引；head 数不改变下面的比例。

每个 head 的有效连接总数：

```text
E = sum_{t=0}^{L-1} min(t+1,K) = L*K - K*(K-1)/2
完整方阵密度 = E / L^2
相对 dense causal 的密度 = E / [L*(L+1)/2]
末行密度 = K/L
索引 padding 比例 = 1 - E/(L*K)
```

| L | K | 完整方阵密度 | 相对 dense causal 密度 | 末行密度 | Padding |
|---:|---:|---:|---:|---:|---:|
| 256 | 256 | 50.20% | 100.00% | 100.00% | 49.80% |
| 512 | 512 | 50.10% | 100.00% | 100.00% | 49.90% |
| 1024 | 1024 | 50.05% | 100.00% | 100.00% | 49.95% |
| 2048 | 2048 | 50.02% | 100.00% | 100.00% | 49.98% |
| 4096 | 2048 | 37.51% | 74.99% | 50.00% | 24.99% |
| 8192 | 2048 | 21.88% | 43.75% | 25.00% | 12.49% |
| 16384 | 2048 | 11.72% | 23.44% | 12.50% | 6.25% |
| 32768 | 2048 | 6.05% | 12.11% | 6.25% | 3.12% |

## 判断

- 固定 top-2048 的预算合理：DeepSeek-V3.2 官方 inference/model.py 的 ModelArgs 使用 index_topk=2048。
- L<=2048 时，连接集合完全等于 dense causal attention。虽然只占完整方阵约 50%，那是 causal mask 的三角区域，本身没有额外稀疏性。
- L4096 保留约 75% 的 causal 连接，仍然较密；L8192 约 43.75%；L16384/32768 约 23.44%/12.11%，才更能体现长上下文固定预算的稀疏性。
- 不能把 K/L（末行密度/标称槽位比例）当作整个 causal prompt 的有效连接密度。例如 L32768 的末行密度是 6.25%，但相对 dense causal 的整段密度是 12.11%。
- 存储和调度仍为 L*K 个槽位。早期 token 的 -1 padding 不增加有效连接，但会带来 padding/掩码和固定分块成本；padding 比例不等于 attention 密度。
- 随机 top-k 是 synthetic gather/attention kernel workload，不是训练得到的 indexer 输出。它不能说明真实模型的选择质量、局部性或端到端 indexer 开销。
- 短序列 dense causal 的退化是固定预算设计的自然结果；benchmark 应按上表明确标注，不能将整条长度曲线称为同一固定稀疏率。

## 来源与验证范围

- 上游提交：d35a206bf55a6fed0a56ba3fd41d3aba6afb1e44。输入生成器的原始函数保存在 upstream_generator.py，source.json 记录源文件 SHA256。
- 每个长度均核对了 export/meta.json：Q=[L,16,576] BF16，KV=[L,576] BF16，indices=[L,K] INT32，output=[L,16,512] BF16。
- 精确公式已对逐行求和做 assert 检查；本次未重新生成 CUDA 输入或测量 GPU kernel。
- 旧固定案例 T=2048、KV=512、带重复的输入与本次 sweep 不同；此前其 69.94% padding 数字不适用于本表。
- 官方 DeepSeek 预算参考：https://github.com/deepseek-ai/DeepSeek-V3.2/blob/main/inference/model.py#L90
