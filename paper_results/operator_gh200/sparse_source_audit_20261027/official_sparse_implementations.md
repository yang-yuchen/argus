# Official sparse attention implementations: renewed source check

## Conclusion

**FlexAttention, Helion, and Triton all have official-source sparse attention implementations.**
The relevant distinction is the supported sparsity representation and input contract, not whether the word “sparse” appears in a filename. This check makes no new performance claim and runs no GPU benchmark.

## FlexAttention

- Official PyTorch [BlockMask definition](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/attention/flex_attention.py#L855): “BlockMask is our format for representing a block-sparse attention mask.”
- Official [`create_block_mask`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/attention/flex_attention.py#L1967) documents the normal workflow: define a user mask callback, construct BlockMask, and pass it to `flex_attention`.
- Official [`flex_attention`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/attention/flex_attention.py#L2365) accepts `block_mask` to control the block-sparsity pattern. Its underlying attention kernels are generated from official Triton templates.
- This is the path used by the measured baseline. Our callback encodes the supplied token selection; the attention kernel is not locally rewritten. Empty blocks are skipped; partial blocks use the callback mask.

This version's public interface does not directly accept the workload's per-query token-index gather list. `BlockMask.kv_indices` contains KV block IDs, not arbitrary individual token IDs.

## Helion

- Latest checked official HEAD: `3c8749703b606c52b2464c63d6937bd672c2f8db`.
- Official [example and API](https://github.com/pytorch/helion/blob/3c8749703b606c52b2464c63d6937bd672c2f8db/examples/flex_attention.py), and [official rendered documentation](https://helionlang.com/examples/flex_attention.html).
- [Full-block traversal](https://github.com/pytorch/helion/blob/3c8749703b606c52b2464c63d6937bd672c2f8db/examples/flex_attention.py#L122) and [partial-block traversal](https://github.com/pytorch/helion/blob/3c8749703b606c52b2464c63d6937bd672c2f8db/examples/flex_attention.py#L165) read the supplied block lists; blocks absent from the lists are not visited. Partial blocks apply the user mask.
- File SHA256 remains `01624062392487b6e97646419e356722be1f54039372737577d9cbf7326a0631`, identical to the unchanged official source used in our measurements.
- The official [`sparse_attn_indexer.py`](https://github.com/pytorch/helion/blob/3c8749703b606c52b2464c63d6937bd672c2f8db/examples/sparse_attn_indexer.py) computes indexer selection scores, not softmax attention times V, so it cannot replace the selected-KV attention being measured.
- The official [HelionDSL/attention](https://huggingface.co/kernels/HelionDSL/attention) package (checked SHA `bffcab17baba6d257c084ad797d1c7bb80e5877f`) exposes dense/causal attention entry points without BlockMask or token-index parameters; its H100/B200 AOT configurations do not directly cover this random top-k workload.

In the checked official GitHub example, docs and kernel catalog, the BlockMask example is the relevant official sparse attention path. No matching dedicated indexed-gather DSA attention API was located in those sources; this is a bounded search result, not a claim about every possible external repository.

## Triton: newly found official organization repository

The current `triton-lang/triton` tree was checked at `96ffafdd35cbd70f0a12a81ba2b3d1acb43888a2` (2191 entries, nontruncated tree). Its attention tutorial and Gluon example are dense/causal implementations with equal Q/K/V head dimensions in `{16,32,64,128,256}`; they have no arbitrary sparse-index API.

**Searching the official organization as well as the main compiler repository found a current sparse attention implementation:**

- Repository: [`triton-lang/kernels`](https://github.com/triton-lang/kernels).
- Checked HEAD: `e16236ab73f0d4334d0752f9ee4fbc6b263431a6` (42 entries, nontruncated tree).
- Official source: [`kernels/topology_sparse_attention.py`](https://github.com/triton-lang/kernels/blob/e16236ab73f0d4334d0752f9ee4fbc6b263431a6/kernels/topology_sparse_attention.py).
- Official tests: [`test/test_topology_sparse_attention.py`](https://github.com/triton-lang/kernels/blob/e16236ab73f0d4334d0752f9ee4fbc6b263431a6/test/test_topology_sparse_attention.py).
- Entry point: `scheduled_attention(q,k,v,offsets,indices,block_size)`.
- Real online-softmax attention over CSR-listed KV blocks. The separate schedule builder selects sink/local/topological blocks; the attention entry point accepts an externally supplied CSR block schedule.
- No built-in autotune search was found in this file: the wrapper fixes `num_warps=4`, with `BLOCK_M=BLOCK_N=block_size`. Checked tests and benchmarks use FP16; BF16 is not explicitly rejected, but has not been validated here.

### Why it cannot be used unchanged for this experiment

The public entry point [at line 240](https://github.com/triton-lang/kernels/blob/e16236ab73f0d4334d0752f9ee4fbc6b263431a6/kernels/topology_sparse_attention.py#L240) explicitly requires identical Q/K/V shapes and head dimension in `{16,32,64,128}`. The current experiment has Hq16/Hkv1, QK dimension576 and V dimension512, so its inputs fail those guards. The kernel uses one head dimension for Q/K/V/output and a fixed `1/sqrt(head_dim)` scale.

Its CSR list selects entire blocks shared by the queries in a query block. Inside selected blocks it applies causal masking, with no arbitrary per-query token mask or token-index gather input. Replacing our selected-token pattern with these selected blocks would change the mathematical workload. Setting block size1 is not a demonstrated usable route around the tensor-core tile and explicit head-dimension constraints.

It is therefore an official-source sparse attention implementation, but is not an equivalent unchanged baseline for the current D576/V512 indexed MLA workload. Official organization membership also does not establish that this implementation is the recommended fastest kernel for every sparse pattern.

## Triton historical official implementation

Triton v2.0.0 (`bd5c2117f62c73a9e922d5e93353a39ab3ac269b`) also contains an official complete [`triton_attention` function](https://github.com/triton-lang/triton/blob/bd5c2117f62c73a9e922d5e93353a39ab3ac269b/python/test/unit/operators/test_blocksparse.py#L177), with forward/backward correctness tests:

```text
blocksparse.matmul(Q,K) -> blocksparse.softmax -> blocksparse.matmul(P,V)
```

This is a three-operation composition from official source, not a single fused indexed-gather kernel. It depends on the historical `triton.ops.blocksparse` package, absent from the checked current main tree. It has not been compiled or measured in the current Triton 3.8 environment; existence of the old source does not establish modern compatibility or matching MLA semantics.

## Evidence and scope

Unmodified downloaded Triton sources and their fixed URLs/SHA256 values are saved in [triton_official/manifest.json](triton_official/manifest.json). No benchmark curve or input contract was changed. Flex/Helion measured sources remain the official BlockMask paths. Triton's current CSR kernel and historical implementation have not been run on GPU in this check.
