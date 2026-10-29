# Sparse attention compiler comparison — protocol and sources

> **[Source/tuning audit completed](../profiles_sparse_source_audit_20261027/README.md):** the saved results are numerically validated timings of the recorded configurations, not proof of sufficiently tuned official sparse baselines. The audit identified an unnecessary V packing copy in the Flex/Helion adapter and restricted/truncated tuning searches. The adapter is fixed; [corrected-layout GPU replay completed, 16/16 points passed](../profiles_sparse_shared_kv_20261027/README.md). Use the [updated PDF](../plots_from_csv/gh200-bf16-tc1-sparse-d35a206-shared-kv.pdf) for the shared-KV results. This directory and its original PDF preserve the historical copied-V measurements. Current resume/summary scripts require shared-KV replay for Flex/Helion and will reject these historical results as current-layout data.

**Historical measurement capture completed — all 32 method/length results numerically validated; baseline audit remains provisional as noted above.** This document records the historical experiment and replay scopes. A per-length result is usable within that historical scope only when its JSON has `correct=true`, `measurement_complete=true`, the expected input hash, all 50 samples, and the method-specific tuning scope below. The existence of eight JSON files alone does not establish completion: earlier-scope results are archived.

The machine-readable plan is [experiment_manifest.json](experiment_manifest.json). Final artifacts: [PDF](../plots_from_csv/gh200-bf16-tc1-sparse-d35a206.pdf), [latencies in microseconds](latencies_us.csv), [normalized comparison](comparison.csv), and [selected configurations](configuration_summary.json).

## Workload: exactly the upstream sparse length sweep

Source: [`uiuc-arc/neptune-internal@d35a206`](https://github.com/uiuc-arc/neptune-internal/tree/d35a206bf55a6fed0a56ba3fd41d3aba6afb1e44/paper_kernels). Eight lengths are measured:

```text
L = 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
```

For each length:

| Tensor or parameter | Contract |
|---|---|
| Q | `[L,16,576]`, BF16 |
| Shared KV | `[L,576]`, BF16; one shared cache for all 16 query heads |
| Value vectors | First 512 columns of the shared KV rows |
| Output | `[L,16,512]`, BF16 |
| Selected indices | `[L,K]`, INT32, `K=min(L,2048)` |
| Scale | `1/sqrt(512)`, including the 64-dimensional Q/K tail in each score |
| Valid selection | Query `t` selects `min(t+1,K)` distinct random positions from `[0,t]` |
| Padding | Remaining selected-index slots contain `-1` |

The input generator is the unchanged upstream `causal_topk_indices` from `python/attn_variants.py`, vendored as [`nautilus_causal_topk.py`](../argus/paper_baselines/vendor/nautilus_causal_topk.py). It implements causal prefix selection, not independent random indices with duplicates. The score includes all 576 Q/K columns; only the value contraction uses 512 columns. No baseline substitutes ordinary D128 MHA, expands the workload to H64, or drops the positional tail.

Nautilus creates and saves each input set once as `nautilus/sparse_L*_inputs.pt`. Every baseline loads those same bytes. Its recorded `inputs_sha256` must match the corresponding Nautilus result before any comparison is accepted.

### Density is length-dependent

With `K=min(L,2048)`, each head has

```text
E = sum_t min(t+1,K) = L*K - K*(K-1)/2
```

valid connections. In particular, **L≤2048 is semantically dense causal attention**: all visible positions are selected, although the kernel receives an index list. Longer sequences use a fixed top-2048 budget.

| L | K | Effective density versus dense causal | Effective density versus full L×L |
|---:|---:|---:|---:|
| 256 | 256 | 100.00% | 50.20% |
| 512 | 512 | 100.00% | 50.10% |
| 1024 | 1024 | 100.00% | 50.05% |
| 2048 | 2048 | 100.00% | 50.02% |
| 4096 | 2048 | 74.99% | 37.51% |
| 8192 | 2048 | 43.75% | 21.88% |
| 16384 | 2048 | 23.44% | 11.72% |
| 32768 | 2048 | 12.11% | 6.05% |

The exact counts, padding fractions, and source hashes are in the [density audit](../profiles_paper_kernels/sparse_density_d35a206/README.md) and its [CSV](../profiles_paper_kernels/sparse_density_d35a206/density.csv). `K/L` is the last query's density, not the whole prompt's density relative to the causal triangle. Random selection models the selected-KV computation and gather pattern; it does not reproduce a learned indexer's locality or quality.

## Implementations and final replay scopes

### Nautilus

- Unmodified exported `kernel.so` binaries from commit `d35a206bf55a6fed0a56ba3fd41d3aba6afb1e44`, one `sparse_L*` export per length.
- Loaded through the existing [`scripts/bench_paper_sweep.py`](../scripts/bench_paper_sweep.py) and compatible Neptune TVM runtime, rather than regenerated with another compiler.
- The JSON records the export SHA256, input SHA256, environment, correctness check, and per-repeat GPU times.
- The initial Nautilus gate uses the independent chunked FP32 sparse reference, elementwise `rtol=1e-2`, `atol=8e-3`, and relative L2 `<1e-2`. Official baseline replays use the FP64 gate described below. These validation scopes are recorded separately rather than described as a single identical oracle run.

### TileLang: unchanged official example, locally calibrated public parameters

The first attempt used the official [`tile-ai/TileOPs`](https://github.com/tile-ai/TileOPs/blob/85e7e644f1d99037e937378b8d8dc70c6568113f/src/tileops/_ops/attention/dsa.py) `DSADenseFwdOp` at commit `85e7e644f1d99037e937378b8d8dc70c6568113f`, with `target=BUILTIN` and `request_tune()`. On the exact H16 workload, all four selected WGMMA configurations failed compilation in the current Torch 2.14.1/cu130 + TileLang 0.1.15 environment: the emitted WGMMA shape has M16 where its SM90 lowering requires M64. This is a real compilation failure, not an accuracy failure or a claim that TileOPs lacks a sparse API. Original failure evidence is preserved under `tilelang_previous_attempt/`; the [source audit](source_audit_other.md) records the relevant guards and failure details. TileOPs declares older Torch/TileLang requirements, so this run does not establish failure in its supported dependency environment.

The adopted TileLang implementation is the complete, unchanged official [`examples/deepseek_v32/sparse_mla_fwd.py`](https://github.com/tile-ai/tilelang/blob/fdb19f26d61e71cabcc61abbc66ab60a97e3a100/examples/deepseek_v32/sparse_mla_fwd.py) at commit `fdb19f26d61e71cabcc61abbc66ab60a97e3a100`. The source and sibling `utils.py` are vendored byte-for-byte with a [SHA256 manifest](../argus/paper_baselines/vendor/tilelang_sparse_official/manifest.json).

The [adapter](../argus/paper_baselines/sparse_tilelang_example_official.py) adds batch/group view dimensions and converts `-1` to the example's documented invalid-index sentinel `L`. Valid selected indices, their order, Q/KV bytes, head count, scale, and causal semantics remain unchanged. Sentinel conversion occurs before timing.

For final replay, the official `tilelang.autotuner.autotune` API receives actual prepared tensors through `set_autotune_inputs`. The four candidates are:

```text
block_I = 64
num_stages ∈ {1,2}
threads ∈ {128,256}
```

**This search space is local GH200 calibration of public example parameters, not an upstream-recommended configuration space.** The official example has no built-in `@autotune` search. Its original default is `block_I=64, num_stages=2, threads=256`; that initial scope is retained separately as a default-configuration comparison. Kernel source, GEMMs, gathers, masks, and softmax remain unchanged. A final result must contain the calibration dictionary, selected configuration and tuning latency in `source.autotune`; default-only results must not be mixed into the final calibrated curve.

The label is **TileLang**. TileOPs and TileLang are not counted as separate compiler families.

### FlexAttention: exact selection BlockMask, bounded official autotuning

- Official public PyTorch API at commit `5c4886908584029761b579af026dcfb627c84070`: [`torch.nn.attention.flex_attention`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/attention/flex_attention.py).
- Q is `[1,16,L,576]`, K is `[1,1,L,576]`, and V is `[1,1,L,512]`; `enable_gqa=True` supplies the shared KV-head mapping and scale remains `512**-0.5`.
- A precomputed `create_block_mask` BlockMask represents exact membership in the supplied unique indices. Padding scatters to a sentinel column beyond `KV_LEN`, so `-1` cannot accidentally select KV row zero.
- Final public `kernel_options` are `BLOCK_M=32, BLOCK_N=64, num_warps=8`. The official **DEFAULT** autotuner searches pipeline-stage settings at this geometry; compilation is fullgraph, with `max_autotune=True` and CUDA graphs disabled.
- The geometry came from broader unrestricted searches that selected it at L1024/L4096. Earlier EXHAUSTIVE searches, rejected configurations, incomplete/obsolete-scope attempts and their logs are retained in `flex_previous_attempt/` and, where replay creates them, `flex_attempts/`. These are not the final curve.
- This is a public adapter configuration for D576/V512 register pressure, not an edited attention kernel or a proof of a global optimum. The final curve must use the same recorded scope at every length.

The [adapter](../argus/paper_baselines/sparse_flex_official.py) uses official masked attention rather than an invented gather kernel. Random token selection may populate almost every causal block: logical token sparsity does not imply equally sparse block traversal.

### Helion: unchanged official flexible attention example

- Official source: [`pytorch/helion/examples/flex_attention.py`](https://github.com/pytorch/helion/blob/429c00227b9802538be459b918ee8d34ccbf636c/examples/flex_attention.py), commit `429c00227b9802538be459b918ee8d34ccbf636c`.
- Vendored unchanged with [manifest](../argus/paper_baselines/vendor/helion_flex_attention_official_manifest.json), SHA256 `01624062392487b6e97646419e356722be1f54039372737577d9cbf7326a0631`.
- Same shared-KV tensor views, exact selected-index membership, scale, GQA mapping and output semantics as FlexAttention.
- The user's mask callback uses Helion's documented `hl.load(allowed, [q_idx, kv_idx])`. This changes the callback's captured-tensor load syntax only; the official attention kernel and BlockMask tables remain unchanged. Initial direct-index callbacks hit `HostTensorDirectUsage`; those failed logs are retained in `helion_previous_attempt/` and later attempt archives.
- Final tuning uses official `full` effort and `HELION_AUTOTUNE_BUDGET_SECONDS=180` per length. This is a **soft** search budget: initialization and in-flight candidate work can exceed 180 seconds. It is not a fixed end-to-end job timeout or exhaustive-search guarantee.
- The upstream example disables its own autotune accuracy check, so its selected kernel must pass the experiment's external FP64 reference gate before timing. The [adapter](../argus/paper_baselines/sparse_helion_official.py) records the callback and budget in each result.

### Other compiler names from the previous figures

The [official-source audit](source_audit_other.md) includes pinned source links and reasons for omissions:

| Compiler label | Audited reason it is not a runnable matching baseline here |
|---|---|
| OpenAI / official Triton | Dense/causal attention tutorial has equal-width Q/K/V and a D≤256 shape set; no matching arbitrary indexed D576→D512 sparse MLA example was found. |
| Tri-Dao Triton | Historical Triton path requires equal Q/K/V widths and D≤128. The newer sparse CuTeDSL MLA path is another backend and its inspected `qv` route requires Blackwell, not GH200. |
| TVM | Inspected official dense/paged-cache APIs do not provide this per-query arbitrary selected-token sparse MLA entry. |
| TAWA | No verifiable official matching sparse MLA repository/API was found; existing local dense/decode code does not establish such an implementation. |

These are limits of the audited available official implementations, not claims that the compilers cannot express sparse attention. No substitute GPU kernel is authored to populate a curve. The source audit retains historical notes from earlier tuning scopes; this README and the final per-result metadata specify the adopted replay scopes.

## Timing, validation and artifact rules

- Baselines run in the existing modern environment `/work/nvme/beqg/yucheny8/envs/argus-gh200-latest`: Torch 2.14.1/cu130, Triton 3.8, TileLang 0.1.15, Helion 1.5.1. Per-result metadata records actual host and package versions. Nautilus uses its compatible exported-module runtime; it is not silently recompiled in the baseline environment.
- The outer public baseline calls run under `torch.inference_mode()`. After compilation/tuning and independent validation, each method gets 10 warmups and 50 measured repeats, flushing 200 MiB of L2 before every repeat.
- Reported time is the **sum of GPU kernel durations inside that method's NVTX range per repeat**, obtained from Nsight Systems. Multiple attention/reduction kernels count together. CPU launch time, compiler/autotune work, index generation, BlockMask construction, sentinel conversion and layout preparation are outside the reported region. The L2-flush kernel is also outside the method range.
- This is an **attention-kernel comparison on precomputed indices**, not an end-to-end learned sparse-attention/indexer benchmark. It does not charge any method for selecting its top-k set.
- Official baselines are compared against `reference_fp64('sparse', inputs)` using the independent selected-KV gather/softmax/value reference. They require finite outputs, correct shape, elementwise `rtol=1e-2, atol=8e-3`, and relative L2 `<8e-3`. The JSON also records RMSE and maximum absolute error. No kernel is accepted merely because its own autotuner benchmarked it successfully.
- Before baseline compilation, the saved shape/dtype and SHA256 must match the corresponding validated Nautilus inputs. Baseline result files retain that same SHA256. A final summary must recheck input identity, source/tuning scopes, the 50-repeat set, and `measurement_complete` rather than importing any prior-attempt number by filename alone.
- Raw evidence: method subdirectories hold JSON, CSV, logs, `.nsys-rep` and SQLite files. The official source manifests contain hashes. `*_previous_attempt`, `*_attempts`, and the default TileLang archive preserve failures and superseded scopes; those are audit records, not additional competitors.
- The figure will reuse the existing [`plot_from_csv.py`](../plot_from_csv.py) style with `Nautilus latency / baseline latency`: Nautilus=1 and **higher means faster**. Only validated final-scope points belong in the CSV. Final performance values are intentionally not asserted in this ongoing-protocol document.

## Final mean latency (microseconds)

| L | Nautilus | TileLang | FlexAttn | Helion |
|---:|---:|---:|---:|---:|
| 256 | 19.409 | 20.614 | 35.131 | 34.606 |
| 512 | 59.453 | 63.723 | 110.495 | 100.913 |
| 1024 | 207.370 | 232.027 | 314.586 | 346.604 |
| 2048 | 762.692 | 875.489 | 984.428 | 1142.556 |
| 4096 | 1541.055 | 1749.583 | 4012.085 | 4101.934 |
| 8192 | 3027.750 | 3440.052 | 15474.355 | 20040.744 |
| 16384 | 6039.714 | 6800.668 | 60552.696 | 84871.592 |
| 32768 | 12209.567 | 13620.909 | 243759.227 | 353180.516 |

## Run provenance and limits

Job `3471992` ran on `gh079` and reached its two-hour limit after saving all Nautilus/FlexAttention points and five Helion points. Job `3472827` completed successfully on `gh004`, filling all calibrated TileLang points and the remaining Helion points. Every final result contains 50 samples and passes the source/scope/input-hash gates. Both nodes use GH200 GPUs; these are cross-run, cross-node comparisons rather than strictly paired same-device measurements. The optional identical-kernel node control was not executed, so small differences must not be treated as a statistically established hardware-independent advantage. No time rescaling or interpolation is used.

The source audit includes earlier phases; the canonical method JSON files and `configuration_summary.json` are the final scopes. Broad Flex searches and TileLang default-configuration trials are preserved in the attempt directories. Official kernel vendor hashes are unchanged.

## Reproduce the figure

```bash
python -m scripts.summarize_sparse_compilers profiles_sparse_compilers_20261027
python plot_from_csv.py --input-fp16 profiles_sparse_compilers_20261027/plot_input.csv --operators SPARSE --suffix=-bf16-tc1-sparse-d35a206 --output-dir plots_from_csv
```

`--input-fp16` is the legacy plot script argument name; the measured workload is BF16 and the plotter only reads latency columns.

## Verification

All 32 final method/length records passed validation; 33 targeted CPU tests passed. The final PDF legends and axis direction were visually checked.
