# Official TileLang / TileOPs FP8 audit

Checked current official heads on2026-10-29:

- TileLang: `8944978dd8761eb975a28f53d2a9be4a800be241`.
  [official examples](https://github.com/tile-ai/tilelang/tree/8944978dd8761eb975a28f53d2a9be4a800be241/examples).
  The79 attention/flash example Python/Markdown files were searched for FP8/float8;
  none supplied an FP8 attention kernel. This is a scoped source-search result,
  not a claim that TileLang cannot express FP8 attention.
- TileOPs: `2c54caece3bc298b8794a364f8ed0c126fd7289d`.
  Independent, unmodified source checkout: `.fp8_tileops_official/`.
  The official native-FP8 kernels are implemented in TileLang on SM90.

## Public APIs and exact sources

- [GQADenseFwdOp](https://github.com/tile-ai/TileOPs/blob/2c54caece3bc298b8794a364f8ed0c126fd7289d/src/tileops/_ops/attention/gqa/dense.py)
- [GQAVarlenFwdOp](https://github.com/tile-ai/TileOPs/blob/2c54caece3bc298b8794a364f8ed0c126fd7289d/src/tileops/_ops/attention/gqa/varlen.py)
- [native dense FP8](https://github.com/tile-ai/TileOPs/blob/2c54caece3bc298b8794a364f8ed0c126fd7289d/src/tileops/_kernels/attention/gqa/dense_fp8.py)
- [native decode FP8](https://github.com/tile-ai/TileOPs/blob/2c54caece3bc298b8794a364f8ed0c126fd7289d/src/tileops/_kernels/attention/gqa/decode_fp8.py)
- [native varlen FP8](https://github.com/tile-ai/TileOPs/blob/2c54caece3bc298b8794a364f8ed0c126fd7289d/src/tileops/_kernels/attention/gqa/varlen_fp8.py)
- [FP8 probability conversion](https://github.com/tile-ai/TileOPs/blob/2c54caece3bc298b8794a364f8ed0c126fd7289d/src/tileops/csrc/fp8_gqa_helper.h)

The GQA-named API supports MHA directly by setting equal query/KV head counts;
our adapter requires this and never broadcasts a smaller KV-head tensor.

## Alignment with the Nautilus paper kernels

Q, K, V remain `float8_e4m3fn`; QK and PV accumulate inFP32; output is requested
as FP16 directly. Identity FP32 `[batch, kv_heads]` descale tensors satisfy the
official API while preserving unscaled input bytes. The default attention scale
is `1/sqrt(128)`. No dropout, RoPE, sinks, softcap or sliding window is enabled.
The causal mask is bottom-right aligned.

Dense uses FA3-style register conversion of the online unnormalized exponential
weights directly to E4M3. The C++ converter uses no additional P multiplier.
Decode copies FP32 online exponential weights to an FP8 shared tensor before PV.
Thus this is native FP8 QK/PV with the same FP8 I/O/attention semantics, not a
conversion of QKV to FP16 followed by a 16-bit attention kernel. Different online
blocking/reduction orders can produce different rounding; no bitwise equivalence
is claimed.

Dense is used for square prefill and causal rectangular MTP. For decode Q=1,
KV>=2048, the same API's official dispatch prefers the context-split FP8 decode
kernel. Noncausal rectangular MTP is refused by the official dense native-FP8
kernel, so that case uses the public packed-varlen API with one request and
unchanged attention semantics.

## Actual official tuning space

`request_tune()` is enabled on the public Op. The selected kernel and its actual
`autotune_configs` count are recorded after the first prepared call:

- Native dense FP8: fixed BN224 schedule; no configurable search space.
- Native decode FP8: fixed `block_m=16, block_n=256, threads=160`, with official
  context-split heuristic; no configurable search space.
- Preferred SM90 FP8 varlen WS: fixed schedule; no configurable search space.
- If official dispatch selects the pipelined varlen fallback, its official
  `2x3x3x2=72` configurations are eligible for request_tune.

No additional tuning space is invented or attributed to the official repo.

## Local integration

`argus/paper_baselines/fp8_tilelang.py` provides
`prepare(q,k,v,causal=...) -> prepared`, with `.run()` returning FP16 BHND and
`.metadata` recording source and configuration. Layout copies, scale tensors,
compilation and optional official tuning happen during prepare.

`verify_source()` checks the pinned checkout commit, ten critical source hashes,
and that tracked package files are unchanged. The adapter inserts the pinned
source path before imports and rejects an already-imported different TileOPs.
The source manifest is
`argus/paper_baselines/vendor/fp8_tileops_official_manifest.json`.

Validation done here: five CPU tests passed; both official APIs import in the
new `argus-gh200-latest` environment with TileLang0.1.15. No GPU job was submitted
by this audit agent, so GPU compilation, correctness and timing remain subject
to the parent's integrated smoke test. Ruff is unavailable in that environment.
