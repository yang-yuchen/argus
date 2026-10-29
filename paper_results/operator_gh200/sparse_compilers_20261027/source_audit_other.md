# Official sparse compiler baseline source audit

> Updated availability check: [official_sparse_implementations.md](../profiles_sparse_source_audit_20261027/official_sparse_implementations.md). Flex/Helion officially provide block-sparse attention. Triton also has an official organization CSR sparse-attention implementation in `triton-lang/kernels`, plus a historical v2.0.0 block-sparse attention composition. Neither has been established as a matching unchanged kernel for the current H16/D576/V512 token-index workload. See the updated report for exact sources and guards; the original main-repository-only search below was incomplete as an availability search.

Target: GH200/SM90, BF16 concatenated sparse MLA, Q `[L,16,576]`, shared KV `[L,576]`, output `[L,16,512]`; indices `[L,min(L,2048)]` are unique causal positions with `-1` padding, and score scale is `1/sqrt(512)`. Lengths 256 through 32768. This audit does not change workload dimensions or implement a new GPU kernel.

## TileLang / TileOPs: distinguish the failed candidate from the adopted official example

- The first GPU sweep attempted the official `tile-ai/TileOPs` `DSADenseFwdOp` at commit `85e7e644f1d99037e937378b8d8dc70c6568113f`, with `target=BUILTIN` and `request_tune()` enabled. The exact sparse shape remained H16, Dqk576, Dv512.
- The operator's default dispatch selected an eligible DSA decode implementation, but its official autotune search **failed to compile all four configurations** in the existing TileLang 0.1.15 environment on SM90a. The preserved evidence is `tilelang/sparse_L32768.log`: `T.wgmma_gemm() requires Hopper WGMMA lowering, but constraints were not satisfied`, with `M=16`, `N=64` or `128`, `K=256`, for `block_i=64/128` and `threads=384/512`. The run ends with `Auto-tuning failed: No configuration successfully compiled and passed benchmarking/validation.` This is an actual GPU compilation failure, not a claimed absence of an official sparse operator or an accuracy/performance result.
- The H16 workload is kept unchanged. The adopted fallback is the **unmodified official TileLang example** at commit `fdb19f26d61e71cabcc61abbc66ab60a97e3a100`: <https://github.com/tile-ai/tilelang/blob/fdb19f26d61e71cabcc61abbc66ab60a97e3a100/examples/deepseek_v32/sparse_mla_fwd.py>. Its public entry point is `sparse_mla_fwd_interface(q, kv, indices, sm_scale=512**-0.5, d_v=512, block_I=64, num_stages=2, threads=256)`.
- This official example does **not** ship an `@autotune` decorator or a configuration-search space. Its public default `64/2/256` is used; the results must not be described as official-autotuned TileLang results. Arbitrarily extending its search space would be a local tuning experiment and is outside the unchanged-official-source run.
- The source and sibling `utils.py` are vendored byte-for-byte in `argus/paper_baselines/vendor/tilelang_sparse_official/`, with their upstream paths and SHA256 values recorded in `manifest.json`. `sparse_mla_fwd.py` SHA256 is `3c79a33636ec3c9cc50014e2b61b365143eba5d4cc34eed4a1ef057cbe072c30`.
- The adapter only adds batch/group view dimensions and converts invalid index sentinel **`-1` to KV length `L`**, as used by the upstream example. Valid selected positions, their order, the causal relation, head count and Q/KV data are unchanged. The conversion and layout preparation are outside timed GPU regions. No gather, softmax, GEMM or mask code in the official kernel is rewritten.
- Fallback compilation, correctness and measurements are being performed by the main experiment. This source audit does not predeclare those runs successful; the per-length result JSON is the evidence for any completed point.

## FlexAttention: official exhaustive search, exact sparse selection

- The benchmark calls the public `torch.nn.attention.flex_attention.flex_attention` API on Q `[1,16,L,576]`, K `[1,1,L,576]`, V `[1,1,L,512]`, with `scale=512**-0.5` and `enable_gqa=True`.
- A `create_block_mask` mask represents membership in the exact supplied unique indices. Its preparation ignores `-1` padding via a sentinel column; neither membership construction nor layout packing is included in the kernel-duration timing.
- `torch.compile(..., fullgraph=True)` uses the official Inductor settings `max_autotune=True`, `triton.cudagraphs=False`, and **`max_autotune_flex_search_space="EXHAUSTIVE"`**. This is an official compiler configuration, not a locally authored attention kernel or a claim that all masked tiles become sparse-gather operations.
- The implementation and chosen settings are in `argus/paper_baselines/sparse_flex_official.py`; GPU success and timing must be read from the experiment's per-length JSON rather than inferred from source compatibility.

## Helion: include the official flexible attention implementation

- Repository: <https://github.com/pytorch/helion>
- Commit: `429c00227b9802538be459b918ee8d34ccbf636c`
- Source: <https://github.com/pytorch/helion/blob/429c00227b9802538be459b918ee8d34ccbf636c/examples/flex_attention.py>
- API: `helion_flex_attention(query, key, value, block_mask=..., scale=512**-0.5, enable_gqa=True)`.
- Inputs: Q `[1,16,L,576]`, K `[1,1,L,576]`, V `[1,1,L,512]`. The source explicitly permits a value width different from Q/K and supports GQA with a block-mask table broadcast over heads. The same exact token-selection BlockMask as the Torch FlexAttention baseline expresses the indexed workload; no kernel modification is necessary.
- The source's official `@helion.kernel` enables `static_shapes=True` and `fast_math=True`. No fixed `config` is pinned, so Helion retains its normal autotune path. The independent workload reference must validate the selected kernel because the upstream example sets `autotune_accuracy_check=False`.
- Vendored unchanged in `argus/paper_baselines/vendor/helion_flex_attention_official.py`, SHA256 `01624062392487b6e97646419e356722be1f54039372737577d9cbf7326a0631`, with a source manifest.
- Adapter: `argus/paper_baselines/sparse_helion_official.py::build_prepared(q,k,v,block_mask)`, which only calls the official wrapper and supplies layout/scale/mask.
- The entire upstream source imports successfully in the installed Helion 1.5.1 environment. Three CPU tests check unmodified source identity and reject invalid input boundaries. At this audit update, the Helion stage has not yet reached actual GPU compilation; GPU compilation and correctness remain pending in the ongoing benchmark.
- `examples/sparse_attn_indexer.py` is also official, but computes the indexer's selection scores rather than the selected-KV softmax/value attention; it is not a substitute kernel for this benchmark.

## OpenAI / Triton official: no matching ready-made example found

- Repository: <https://github.com/triton-lang/triton>
- Checked main commit: `d364bf19639a3f4a8aecb45bca0a533e2ecb6258`.
- Official tutorial: <https://github.com/triton-lang/triton/blob/d364bf19639a3f4a8aecb45bca0a533e2ecb6258/python/tutorials/06-fused-attention.py#L506>.
- The public tutorial requires equal Q/K/V head dimensions and head dimension in `{16,32,64,128,256}` (lines 511-512). It has dense/causal traversal, not the selected token-index input used here. D576 Q/K and D512 V are rejected.
- The official Gluon attention example has the same equal-dimension and dimension-set guards: <https://github.com/triton-lang/triton/blob/d364bf19639a3f4a8aecb45bca0a533e2ecb6258/python/examples/gluon/01-attention-forward.py#L1177>.
- A recursive repository-tree search found no standalone arbitrary token-index sparse MLA attention example. This is a statement about the inspected official implementations, not a limitation of Triton's ability to compile a newly written sparse kernel. A custom Triton implementation would violate the user's official-kernel requirement.

## Tri-Dao Triton: exclude; available implementation has incompatible dimensions

- Repository: <https://github.com/Dao-AILab/flash-attention>
- Checked main commit: `98c1cee0ed59307833841cdbf567fe336ceec13a`.
- Triton implementation: <https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/flash_attn_triton.py#L816>.
- This historical Triton kernel requires K and V to have the same head dimension as Q and asserts `d<=128`. It cannot express D576/Dv512 indexed sparse MLA as-is.
- There **is** a newer official sparse top-k MLA implementation in `flash_attn.cute.interface`, with an official benchmark that tests H16: <https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/benchmarks/benchmark_sparse_mla_fwd.py>. It is a separate CuTeDSL implementation, not the `Tri-Dao Triton` compiler baseline from the plot.
- Moreover, its MLA route (`qv` latent query plus Q/K rope) explicitly asserts `arch // 10 in [10,11]`, “only support Blackwell arch with qv”: <https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/cute/interface.py#L1177>. GH200 is SM90, so this candidate does not run on the target hardware.
- Historical `flash_blocksparse_attn_interface.py` wraps `flash_attn_cuda.fwd_block`, a CUDA implementation over packed equal-width QKV, not a Triton kernel or matching sparse MLA API.

## TVM: no matching official indexed sparse attention entry found

- Repository: <https://github.com/apache/tvm>
- Checked main commit: `e345029cedf0be76a087fcad9c89a97bed4f7204`.
- Official LLM kernels inspected under <https://github.com/apache/tvm/tree/e345029cedf0be76a087fcad9c89a97bed4f7204/python/tvm/relax/frontend/nn/llm> include `_prefill_kernels.py`, `_decode_kernels.py`, `_page_kernels.py` and `kv_cache.py`.
- These provide dense/paged KV-cache attention implementations; no public call accepting the workload's per-query arbitrary selected token indices was found. General sparse tensor operators or paged cache block tables are not a drop-in indexed MLA attention implementation.
- The compiler could lower a newly defined gather/softmax/value graph, but that would be a new local kernel rather than an official baseline and is not included.

## TAWA: no verified official matching sparse source located

- Existing project metadata says TAWA and the two historical Triton attention kernels are local source kernels, without independent PyPI packages (`envs/gh200_latest.md`). The local TAWA runner implements dense/causal attention and Q=1 decode; it does not expose this arbitrary indexed sparse MLA operator.
- Searches for an authoritative TAWA sparse-attention repository/API did not find a verifiable exact-match implementation. It is omitted rather than relabeling a local adaptation as official. This is not a claim that no unpublished or differently named TAWA implementation exists.

## Scope of the audit

This audit worker performed no GPU work or package changes. The main benchmark has now supplied a real TileOPs H16 compilation failure, documented above; other source-compatibility findings are kept separate from GPU success. The adopted unchanged TileLang example and official FlexAttention are being evaluated by the main run. Helion remains a source-compatible official candidate whose GPU stage is pending at this update. The other four compiler labels have no verified runnable official exact-match kernel in this audit.

## FlexAttention final query-tile setting

The unrestricted official EXHAUSTIVE searches at L1024, L2048 and L4096 all selected BLOCK_M=32 and num_warps=8; their original logs and measurements are preserved. For the final uniform replay, the public `kernel_options={"BLOCK_M":32,"num_warps":8}` limits the query tile while official EXHAUSTIVE tuning continues over the other settings. This addresses D576/V512 register pressure and avoids spending the search on oversized query tiles. It is an adapter configuration choice, not an upstream kernel edit or a proof of a global optimum. The final summarizer rejects results without this recorded setting.

### Final bounded replay

The long unrestricted candidate searches spent substantial GPU time on slow spilling configurations. The final uniform curve therefore uses the public `kernel_options={"BLOCK_M":32,"BLOCK_N":64,"num_warps":8}` geometry selected by unrestricted L1024/L4096 runs, with PyTorch's official DEFAULT autotuner choosing the pipeline-stage configuration. All eight final points use this exact scope, and initial broader searches are retained separately. The L16384/L32768 obsolete-scope tuning processes were stopped before final measurements and must be replayed. No original attention kernel or compiler code is edited. This scope is empirical adapter configuration, not a global-optimum or upstream-recommended-optimum claim.

## Observed Helion initial callback failure

The initial GPU attempts at L256 and later lengths hit `helion.exc.HostTensorDirectUsage` when the BlockMask callback directly indexed its captured GPU Boolean membership table. This is a callback lowering limitation in installed Helion 1.5.1, not an attention numerical failure. A follow-up adapter may use the officially suggested `hl.load` representation for the same mask; the original official attention source remains unchanged, and any successful replay must pass the FP64 gate. Initial logs are retained.
