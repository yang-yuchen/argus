# Other compiler baseline FP8 official-source audit

Date:2026-10-29. Source inspection only: no kernels changed, no environment installations and no GPU jobs.

## Result

Among the additional compiler baselines inspected here, none can be directly accepted under the strict Q/K/V=e4m3fn, probability=e4m3fn, FP32 accumulation, FP16 output, no quantization scale contract for the full decode/prefill/MTP set. This does not claim that the projects have no FP8 code. Their actual implementations are distinguished below.

| Baseline | Actual official FP8 capability | Admission result |
|---|---|---|
| OpenAI Triton | True; True | excluded_strict_contract |
| Tri-Dao Triton | False; FP16/BF16 implementation only | excluded_dtype |
| Tri-Dao FA3 | True; True | excluded_noncompiler |
| Tri-Dao FA4 CuTeDSL | True; False | excluded_architecture |
| Helion FP8 example | True; FP8 transposed-V code path; not GPU-tested in this audit | excluded_output_and_mask |
| Helion tuned attention bundle | False; Official FP16/BF16 sm90 pre-tuned configurations | excluded_unadvertised_fp8 |
| Helion flex example | No explicit FP8 public contract found in checked source; Depends on compiler/layout; not tested | excluded_output_contract |
| Helion paged attention | False; Existing historical example not current-main kernel | no_current_main_attention_entry |
| vLLM Triton unified attention (optional) | True; Triton GPU implementation; not GPU-tested in this audit | optional_requires_contract_decision |
| TVM | No matching attention example identified in current inspected tree; Generic compiler, but no audited matching FP8 attention | no_verified_equivalent_official_fp8_kernel |
| TAWA | Not established; Local implementation exists, but does not meet official-source requirement | official_provenance_unverified |

## Pinned sources and evidence

### OpenAI Triton

[Official source](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/06-fused-attention.py)
Commit: `cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1`.

- FP8 branch selects float8_e5m2 / tl.float8e5, not e4m3fn
- o=torch.empty_like(q), store casts to FP8, not FP16 output
- A single N_CTX and descriptor shape are used for Q/K/V; no independent rectangular Q/KV decode or MTP path
- causal supports square upper-left triangle, not general lower-right rectangular mask

Tuning: Official triton.autotune; configs over BLOCK_M, BLOCK_N, stages, warps and hardware pruning; not run locally.

### Tri-Dao Triton

[Official source](https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/flash_attn_triton.py)
Commit: `98c1cee0ed59307833841cdbf567fe336ceec13a`.

- Explicit q.dtype in [torch.float16, torch.bfloat16] assertion
- Official AMD Triton backend FP8 via aiter is ROCm-targeted, not GH200

Tuning: No FP8 configuration can pass the official dtype gate.

### Tri-Dao FA3

[Official source](https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/hopper/benchmark_flash_attention_fp8.py)
Commit: `98c1cee0ed59307833841cdbf567fe336ceec13a`.

- Hopper FP8 forward exists, but implementation is CUDA/CUTLASS rather than Triton/DSL compiler baseline
- Benchmark triton_fused_attention import is optional and that module is not present in this pinned repo tree

Tuning: Not considered for compiler-only comparison.

### Tri-Dao FA4 CuTeDSL

[Official source](https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/cute/interface.py)
Commit: `98c1cee0ed59307833841cdbf567fe336ceec13a`.

- Official FP8 guard asserts arch // 10 == 10 (SM100); GH200 is SM90
- FP8 output dtype fixed to bfloat16, not target float16

Tuning: Not runnable as FP8 on GH200 under official architecture guard.

### Helion FP8 example

[Official source](https://github.com/pytorch/helion/blob/85c59a31a5f003392e791ef25e3bc01ec26a7f6f/examples/fp8_attention.py)
Commit: `85c59a31a5f003392e791ef25e3bc01ec26a7f6f`.

- Correct e4m3fn Q/K/V and probability FP8, FP32 dot/accumulation, pretransposed V
- Output explicitly allocated and cast float8_e4m3fn, not FP16
- One seq_len reused for Q/K/V; no causal mask or distinct Q/KV lengths

Tuning: @helion.kernel(static_shapes=True) normal tuning; no official FP8 H100/B200 tuned bundle identified.

### Helion tuned attention bundle

[Official source](https://huggingface.co/kernels/HelionDSL/attention/blob/bffcab17baba6d257c084ad797d1c7bb80e5877f/README.md)
Commit: `bffcab17baba6d257c084ad797d1c7bb80e5877f`.

- Current official README declares only float16 and bfloat16
- The existence of an out parameter does not establish supported FP8 input or tuned FP8 schedules
- Causal kernel uses q_index>=k_index (upper-left); lacks lower-right rectangular argument

Tuning: Shipped H100/B200 schedules cover documented FP16/BF16 use; no FP8 validation claimed.

### Helion flex example

[Official source](https://github.com/pytorch/helion/blob/85c59a31a5f003392e791ef25e3bc01ec26a7f6f/examples/flex_attention.py)
Commit: `85c59a31a5f003392e791ef25e3bc01ec26a7f6f`.

- Can use official BlockMask and rectangular shapes
- Output allocation is empty_like(query) or dtype=query.dtype; with FP8 Q it returns FP8 rather than FP16
- Do not patch official kernel or append output cast and claim equal FP16-output precision

Tuning: Official Helion tuning; no supported equivalent FP8 output configuration identified.

### Helion paged attention

[Official source](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/kernels/helion/ops)
Commit: `fa3b9d1377219247060d3c8b49eae199c9e7e154`.

- Current official vLLM tree includes Helion FP8 quantization operators, but no Helion attention operator
- Prior portable-paged-attention implementation linked by PyTorch blog was closed/unmerged experimental vLLM PR27293; do not label it a new tuned official FP8 attention baseline

Tuning: No current-main Helion FP8 attention configuration to tune.

### vLLM Triton unified attention (optional)

[Official source](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/attention/ops/triton_unified_attention.py)
Commit: `fa3b9d1377219247060d3c8b49eae199c9e7e154`.

- Current official source supports FP8 Q as well as FP8 KV and caller-provided output dtype, with rectangular/paged causal attention
- It is inaccurate to dismiss all current official vLLM FP8 attention as BF16-Q-only
- Uses P*256 cast e4m3 then divided by256 after dot; Q/K/V descales may be set to1 but probability quantization scheme remains different
- Paged layout must be prepared outside timing; needs numerical/accuracy validation and explicit inclusion policy rather than silent replacement of requested strict contract

Tuning: Official heuristic launch parameters, optional caller tuning parameters; not independently tuned here.

### TVM

[Official source](https://github.com/apache/tvm/blob/2bf0532998d4dc7879cb342e873608b90c1d4497/src/relax/op/nn/attention.cc)
Commit: `2bf0532998d4dc7879cb342e873608b90c1d4497`.

- Relax attention type inference returns Q dtype, so FP8 Q implies FP8 output in that op
- Official contrib CuDNN SDPA runner asserts float16 only
- Official FP8 CUTLASS files found are GEMM implementations, not matching FP8 attention
- FlashInfer extern and CUDA/CUTLASS delegates must not be represented as an independently authored TVM attention compiler kernel

Tuning: No matching official FP8 attention schedule/configuration identified.

### TAWA

- This workspace documents TAWA runners as local custom kernels with no standalone package release
- GitHub official repository could not be established from source metadata and current repository searches
- Do not reuse local historical FP8 code and describe it as independently verified upstream official code

Tuning: Local configurable search space is not evidence of official repository provenance.

## Important interpretation

- A local adapter may reshape/permutate/prepack FP8 input bytes and invoke an official public entry. It must not change an official FP8 kernel from e5m2 to e4m3, alter its output allocation, add a causal mask, or convert FP16 inputs inside timing and call the result an unchanged official FP8 kernel.
- Casting an already rounded FP8 output to FP16 does not recover the target FP16-output precision.
- Current vLLM unified attention has real FP8-Q support. Unit descales make input scale arithmetic neutral, but probability scale256 changes effective quantization; it is an optional separately documented comparison, not a silently equivalent strict baseline.
- The PyTorch/Helion H100/B200 tuned bundle is documented for FP16/BF16; the separate Helion FP8 example is a real FP8 kernel with a different output and mask contract.
- These checks inspected pinned files and repository trees; negative results mean no verified matching official entry was found in this audit, not an exhaustive theorem about all historical branches.
