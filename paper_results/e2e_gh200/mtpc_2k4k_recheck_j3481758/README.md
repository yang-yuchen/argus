# MTP 2¹¹ /2¹² 重跑

六个配置已完成：KV2048、4096各测Nautilus、TileOPs、FlexAttention，均为Q16。
结果对比：[recheck.md](recheck.md) · [recheck.csv](recheck.csv)。
更新后的9长度吞吐量图：[PDF](mtp_e2e_graph_throughput.pdf) ·
[PNG](mtp_e2e_graph_throughput.png) · [完整CSV](results.csv)。

## 结果

| KV | Nautilus ms | TileOPs ms | FlexAttention ms |
|---|---:|---:|---:|
|2048|8.042|8.936|9.674|
|4096|8.778|9.908|11.102|

每点保留全部30样本，预热5次，计时和capture协议与原实验相同。
新样本mean和median接近；6点最大值均在正常范围，没有重现原来的长尾。

## 原异常和归因边界

- 原2K Nautilus mean13.402ms、median8.447ms，最后出现53.597/100.211ms；
  重跑mean8.042ms、max8.056ms。
- 原4K Flex mean25.784ms、median12.448ms，多次出现60–75ms；
  重跑mean11.102ms、max11.220ms。
- 原样本的独立profiler没有同等级的GPU时间暴涨。长尾未在这次重跑出现，支持
  瞬时host/调度/资源争用的解释，但未profile的旧异常样本不能精确归因。
- [launch_configs.json](launch_configs.json)保留新旧kernel launch信息。
  Nautilus使用同一export和launch；Flex主attention模板配置一致，但两个reduction的
  自动调优配置发生变化。它们仍来自同一官方API和软件版本，未手工更换baseline。
  因此不能把Flex全部延迟改善都归因于消除旧wall-time长尾。

## 验证与来源

- GHx4-interactive作业 **3481758**，节点 **gh150**。
- 六点同GPU：`3871f5fe-09c2-e9c6-9b8b-ed0f13af97a6`。
- 同一latest环境、模型、输入、kernel源码/API、timing协议；与原数据逐点比较配置、
  权重/输入hash、软件package记录和当前paper export SHA均一致。
- 六点均通过256组attention、完整模型、换token、历史V扰动和NaN污染后replay检查。
- 另用当前严格解析器复查全部6份原始trace：两个forward范围的attention阶段、
  64个KV更新kernels和所有GPU kernels在graph中均通过，见[trace_audit.json](trace_audit.json)。
- 未删除原数据或任何样本。更新图只替换这六点，另外21点复用原作业3478086；
  完整CSV逐行保留run、host和GPU UUID。该图不代表27点都在同一GPU重测。

作业的交互allocation最初未及时启动测量，这是执行遗漏；实际重跑step为5分1秒。
六点完成后释放GPU，作业最终COMPLETED0:0。

复现这次两长度子集：`profiles_paper_e2e/run_mtp_2k4k_recheck.sh`；
汇总/画图：`python profiles_paper_e2e/mtpc_2k4k_recheck_j3481758/summarize_recheck.py`。
上述是本次工作区的实验复现记录，不是通用portable入口。
