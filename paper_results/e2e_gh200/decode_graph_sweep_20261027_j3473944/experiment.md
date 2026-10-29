# Decode：全部长度的整步 CUDA Graph replay

结果见 [results.md](results.md)、[results.csv](results.csv) 和
[PDF曲线](decode_e2e_graph.pdf) / [PNG曲线](decode_e2e_graph.png)。

## 范围和计时

本次仅重跑 **TileOPs预绑定**，长度为256、512、1K、2K、4K、8K、16K、32K。
未重跑 Nautilus、Flex 或原 public API，图中只画本次 replay 曲线。

- Slurm **3473944**，`ghx4-interactive`，节点 **gh004**，GH200 120GB。
- GPU UUID：`53dd4b8e-bf35-0a40-87f4-0e1f72ccf4c3`。
- 专用latest环境：Torch2.14.1+cu130、TileLang0.1.15、Triton3.8.0、Transformers4.57.1。
- 随机初始化 Llama-7B backbone，32层，B1、FP16、H32、D128、Q1。
  横轴为attention可见的KV总长度L，含新增token；prefix为L−1。
- 每点先完成普通模型正确性、编译及5次普通forward预热，再在对齐的side stream上
  预热3次并捕获整步forward。正确性检查后恢复原始输入和prefix，再预热5次replay。
- 每点正式采样30次，共 **240个样本**。计时前device-wide synchronize，计时覆盖
  `graph.replay()`及结束device-wide synchronize；包含host graph提交和GPU完成。
- 图内包含新增KV写入、Q准备、projection、attention、MLP、norm和residual。
  prefix构建、输入准备、编译、capture、检查、LM head和采样在计时外。
- 固定地址、固定cache position的单步decode；每次覆盖相同最后一个位置，不是动态生成循环。

## 正确性与完整性

每点均通过32层attention、完整模型及编译模型对独立FP32 reference的相对L2<1%检查。
每层复用常驻BSHD KV，仅写新增token，32个常驻绑定、零全KV staging。

捕获后更换两组token、扰动第一层历史V（+0.5），均与正常compiled模型输出一致。
每次比较前，将共享Q和attention输出buffer置NaN，要求replay重新写入/计算；
8点的这三类graph-versus-normal相对L2全部为0.0。恢复原始token与prefix后再次对
FP32 reference检查，误差仍小于1%。因此不能靠普通forward预填共享输出蒙混过关。

正式计时结束后，每点另外profile两次replay，保留`*.graph.trace.json`：

- 每个GPU kernel均有非零graph ID。
- 256/512每forward有32个attention kernels，其余长度有64个。
- 各点均有64个KV index-copy kernels，对应32层K/V各一次更新。
- GPU copy可能体现为memcpy node或`memcpy32_post` kernel，因此总kernel数不硬编码。

汇总器拒绝缺点、错误execution、非有限样本、错误source/config、错误capture/poisoning
证明、错误当前export路径/SHA。还核对8点同GPU、同模型权重、同软件和协议；配置只
允许max_position_embeddings随长度变化。输入hash每点独立保留。

实际TileOPs checkout未修改，commit为`85e7e644f1d99037e937378b8d8dc70c6568113f`。
官方选中的kernel仍为`GQADecodeBs1Kernel`；原IR、target、pass和编译参数保持不变。
完整source和当前paper workload export哈希保存在各点JSON及[kernel_manifest.json](kernel_manifest.json)。
该export哈希用于核对工作负载版本，不表示本次执行了Nautilus。

## 复现

```bash
cd /u/yucheny8/argus
srun --account=beqg-dtai-gh --partition=ghx4 --time=00:30:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G --gpus-per-node=1 \
  bash scripts/run_decode_e2e_graph_sweep_ghx4.sh profiles_paper_e2e/NEW_GRAPH_SWEEP
```

单点入口是`scripts.bench_paper_e2e decode --implementation tileops_bound --cudagraph`；
扫全部长度通过`scripts.bench_decode_e2e_sweep tileops_bound DIR --cudagraph --repeat 30 --warmup 5`。
汇总命令：

```bash
python -m scripts.summarize_decode_e2e_graph profiles_paper_e2e/decode_graph_sweep_20261027_j3473944
```

新增capture证据解析、graph汇总和CLI路由的相关测试共 **39 passed**。
作业完成0:0，GPU已释放。旧普通forward曲线保留，不与本次replay当作同口径backend比较。
