# MTP e2e：常驻原生KV cache

| 实现 | 平均ms | 中位数ms | Nautilus相对该实现的均值加速 |
|---|---:|---:|---:|
| Nautilus | 30.780 | 30.505 | 1.000× |
| TileOPs | 56.209 | 55.619 | 1.826× |
| FlexAttention | 52.842 | 49.163 | 1.717× |

Slurm **3471084**，普通`ghx4`，节点`gh061`，GPU UUID
`d076916c-7872-1811-dd2e-f074fe5693bf`。同一张GH200，随机权重Llama-7B backbone，
32层、hidden4096、32个MHA heads、FP16、B1/Q16/KV65536。预热3次，20次同步整模型
wall采样。原始样本和日志为`mtpc_{nautilus,tileops,flex}.{json,log}`，汇总为`results.csv`。

## 常驻cache和Nautilus检查

- TileOPs每层保存8组连续BSHD `[1,65536,4,128]` K/V。
- Nautilus和Flex每层保存8组连续BHSD `[1,4,65536,128]` K/V。
- prefix65520 tokens先用模型实际生成，再一次性写入各backend的原生布局，计时外完成。
- 每轮仅在最后16个位置`index_copy_`新K/V，不重排/拼接整份cache；KV写入计入e2e。
- 每组只准备16KiB Q输入；K/V直接被attention读取，没有全量KV staging。
- 32层×8组对应 **256组常驻绑定**。各组独立绑定实际K/V地址，不能沿用原来一组
  DLPack参数反复处理八组的方式，否则会读取旧组/旧缓冲。
- Nautilus原先也有每forward约32GiB的整KV复制。本次一并删除：原导出物不变，
  使用现有loader预绑定各组KV地址，后续原地更新自动可见，不重新包装或复制K/V。
- 分组launch和输出复制仍计入计时。没有把新增token的写入移到计时外。

实现使用HF `Cache`和`StaticLayer`协议，内部为5D grouped原生storage。
它只适配本benchmark的专用attention hook，不是通用`generate()` cache；普通HF
attention不能直接消费此5D返回值。模型投影、RoPE、输出投影、MLP和norm保持HF原实现。

## 环境与官方API

三组均用专用`argus-gh200-latest`：Torch **2.14.1+cu130**、Triton **3.8.0**、
Transformers **4.57.1**（现有包隔离overlay），allocator为`expandable_segments:True`。
TileOPs使用TileLang **0.1.15** / TVM0.25.dev0；Nautilus在独立子进程中加载旧导出物
匹配的TVM0.21，两个TVM不混在同一进程中。

TileOPs仍为未修改源码commit`85e7e644f1d99037e937378b8d8dc70c6568113f`，
`GQADenseFwdOp(is_causal=True,target=BUILTIN)`，实际选中
`GQADensePersistentWSKernel`，kernel_config=null。Flex仍为官方lower-right mask和
`torch.compile(flex_attention, fullgraph=True, mode="max-autotune-no-cudagraphs")`。
kernel shape和每层8组调用保持不变；不强制更换baseline kernel。

## 正确性和同口径检查

三组权重/输入hash、模型配置、GPU UUID、软件版本、cache策略、预热/采样数量及计时
口径均由汇总脚本核对一致。每层每组共256个attention绑定都经过FP32 reference检查；
整模型及编译后相对L2均要求<1%。日志没有重编译上限回退、OOM或运行失败。

| 实现 | 整模型相对L2 | 编译后相对L2 |
|---|---:|---:|
| Nautilus | 0.004423 | 0.005805 |
| TileOPs | 0.004461 | 0.005831 |
| FlexAttention | 0.004399 | 0.005817 |

- 权重SHA256：`35a9b4db19947bed874240e6d9783fc2f22c11ba2d4ecd02e71b01e1c7034e7b`
- 输入SHA256：`20abb34c720e06ef28609300c8111da73a18cba533452072d151fd7d56350514`
- Nautilus导出SHA256：`8595aebc3c6108a7520eca00107f1631aae1da473cbab3cb0c438b7408670200`

保留全部样本，未删除outlier；Flex的均值52.842ms受到少量长尾影响，中位数49.163ms。
此前使用全量staging的结果不能与本次不同cache策略直接拼表。

## 独立GPU审计

正式性能采样后，分别用另外两次TileOPs/Nautilus forward进行独立torch.profiler审计，
不将带profiler的延迟用于上表。原始trace及分析位于
`../mtpc_native_audit_j3471084/`：`{tileops,nautilus}.trace.json`、`.audit.json`、
`summary.json`和`audit_native_e2e.py`。

两者均确认：

- timed区间没有≥64MiB的GPU memcpy或Torch临时分配。
- 全部`aten::copy_`输入最大8192个FP16元素（16KiB），对应Q和组输出。
- 两次forward共128个`index_copy`融合kernel，即每层K/V各一个；grid为128或256 blocks，
  覆盖新增65536个元素，而非全64K cache的268435456个元素。
- 每forward实际新增KV payload为 **8MiB**，而原来整份KV staging为约32GiB。

CPU针对性回归 **19 passed**，覆盖三种backend的KV地址绑定、两种cache布局、prefix不变、
重复更新、仅写候选token及独立HF oracle。GPU作业完成后已释放。复现：

```bash
srun --account=beqg-dtai-gh --partition=ghx4 --time=01:00:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G --gpus-per-node=1 \
  bash scripts/run_paper_e2e_mtpc_ghx4.sh profiles_paper_e2e/new_native_cache_run
```

本次仅B1/Q16/KV65536，不含draft、LM head、采样或长文本质量验证；没有长度扫描。
