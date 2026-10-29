# Strong baseline provenance

All GPU kernels were measured on GH200 (SM90), with H100/Hopper compatible CUDA paths where the upstream source supported them. “Official” below refers to the algorithm/source or released library; the launcher, fixed-shape adapter, output preallocation, and parameter sweep are local measurement code.

| Family | Upstream source | Official? | Local adaptation |
|---|---|---|---|
| PyTorch Flash/cuDNN | PyTorch 2.14.1 built-in SDPA | yes | fixed noncausal call and output view |
| FlashAttention-2 | flash-attn 2.8.3.post2 | yes | fixed noncausal call and BHSD/BSHD views |
| FlashInfer dense | flashinfer 0.7.2.post1 | yes | fixed-shape wrapper; FA3 MTP uses NHD because installed SM90 binding misreads HND sizes |
| TileLang dense | tile-ai/tilelang HEAD efa1ffc1e3dc547b11fee5acf671daff1f0f94cb | source yes | local fixed-shape adapter and bounded parameter sweep; not an upstream “best” claim |
| Meta MSLK Triton | meta-pytorch/mslk 25cf718603d33a6d6ef61f81858b6d0c67e99cba | source yes | standalone imports, fixed launch; split128/256 are explicit/experimental overrides |
| vLLM Triton | vllm 74a1ac38b00a8cf502db085d1bbd77712cf47e41 | source yes | standalone imports, independent-query mapping, fixed launch |
| TileLang sparse | tile-ai/tilelang HEAD efa1ffc1e3dc547b11fee5acf671daff1f0f94cb | source yes | noncausal `-1` mask, empty-row zero, split packing timed, bounded configurations |
| FlashInfer sparse MLA | flashinfer 0.7.2.post1 FA2/FA3 | library yes | CSR compaction; timed variant includes mask/compaction, preplanned variant excludes it |
| Nautilus | neptune-internal `nautilus-gh200`, exported `paper_kernels` | upstream export yes | no kernel changes; old custom TVM/Triton runtime needed |

Excluded implementations: FlashMLA Hopper H16 incompatible (requires H64/H128); current FlashMLA removed Hopper support; NSA/block-sparse layouts are not equivalent to these DSA indices; FP8 kernels are not FP16/BF16 accuracy-equivalent.
