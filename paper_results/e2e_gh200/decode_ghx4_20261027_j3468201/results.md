# Decode e2e — GHx4 job 3468201

正式结果仅使用本目录的三组 JSON；其他目录为兼容性和数值诊断记录。

| 实现 | 平均 ms | 中位数 ms | Nautilus 相对该实现的加速 |
|---|---:|---:|---:|
| Nautilus `decode_L512` | 27.827 | 27.810 | 1.000× |
| TileLang MHA | 30.874 | 30.818 | 1.110× |
| FlexAttention | 32.242 | 32.139 | 1.159× |

## Workload 和口径

- Slurm job **3468201**，普通 `ghx4` 分区，节点 `gh029`，一张 NVIDIA GH200 120GB。
- GPU UUID：`73c05aa3-d14d-8b72-6403-2fe55c90fea9`。
- 随机初始化 Llama-7B backbone，32 层，hidden4096，32 个 MHA heads，D128，FP16。
- B=1；已预先生成 511-token prefix KV cache，每次追加 1 token，总 KV=512。
- 每组预热 3 次、采样 20 次；`torch.compile(dynamic=False, fullgraph=False)`，
  Dynamo recompile_limit=300。日志没有达到重编译上限的回退。
- **同步 wall latency**：开始前同步，计时覆盖完整 forward 和结束同步，
  包含 CPU dispatch、cache append、QKV staging、attention、MLP、norm。
- 排除初始化、prefix forward、cache 重建、编译、tokenizer、embedding lookup、
  LM head 和采样；稳态，不显式 flush L2。

## 对照与正确性

三组使用同一 `nt` Python：Torch2.9.1/cu126、Transformers4.57.1、Triton3.5.1。
只有 TVM runtime 在独立进程中切换：Nautilus 使用 Neptune TVM0.21；TileLang
使用自身匹配的 TVM0.22.dev0。未修改或安装环境依赖。

TileLang 是现有官方 MHA split-KV 示例：block_M128、block_N64、split4、128 threads。
为了旧版 runtime，临时 workspace 显式预分配传入；BHSD→BSHD 刷新复制计入 e2e。
FlexAttention 使用 Argus 现有 `max_autotune=True` 路径，选择 flex-decoding kernel。
这不是对所有 TileLang 版本/配置的最优性结论。

| 实现 | 整模型相对 L2 | 编译后相对 L2 |
|---|---:|---:|
| Nautilus | 0.005146 | 0.005578 |
| TileLang | 0.005295 | 0.005637 |
| FlexAttention | 0.004819 | 0.005540 |

每层 attention 还单独对独立 FP32 reference 检查 allclose（rtol0.01、atol0.003）
和相对 L2<0.01；整个32层模型要求相对 L2<0.01，记录最大绝对误差。
单 kernel 的绝对阈值不直接用于32层累积后的模型输出。

汇总脚本已经校验相同配置、权重/输入 hash、GPU UUID、软件版本和采样数量：

- 权重 SHA256：`35a9b4db19947bed874240e6d9783fc2f22c11ba2d4ecd02e71b01e1c7034e7b`
- 输入 SHA256：`2e87894f5435bdd6759c92eaf53b86d211db113f21ffe135f56a357f8eb21d36`
- 导出 kernel SHA256：`5c5537790c0ed2a9356157fdbaa6cdcbaebfe94edb4301b432852da5b72c2bab`

原始样本：`decode_nautilus.json`、`decode_tilelang.json`、`decode_flex.json`；
完整编译和正确性日志为同名 `.log`；机器可读汇总为 `results.csv`。

## 复现

```bash
cd /u/yucheny8/argus
srun --account=beqg-dtai-gh --partition=ghx4 --time=00:30:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gpus-per-node=1 \
  bash scripts/run_paper_e2e_decode_ghx4.sh profiles_paper_e2e/new_decode_run
```

本次仅测 KV512、batch1，没有做长度扫描或完整生成服务实验。64K MTP 本次仅完成
脚本和小型 HF CPU 验证，未生成 GPU 性能数据。实验结束后已释放 GPU，取消本任务
预留的依赖续跑申请3468266。
