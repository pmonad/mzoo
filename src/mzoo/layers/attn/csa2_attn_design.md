# Design: FA2-style attention for SM120 in TileLang, growing into DSV4.1 CSA2

Status: outline, step 1 in progress. Target GB10 (sm121, SM120 family), tilelang 0.1.14.
Scope: head dims `{64, 96, 128, 256}` only. `D=512` (the released V4.1-Flash size) is out of scope, so no Split-D.
Training only for now: prefill-shaped forward plus backward; no decode, no paged cache, no split-KV.

## What the kernel ultimately has to compute (from `archs/dsv4/modeling_deepseek_v41.py`)

Per layer, per query token `t` with `H=64` heads, `D=512` (last `qk_rope_head_dim=64` channels carry RoPE):

```
logits  = [ q·win_kv[t-127..t] , q·main_kv[topk(t)] , sink[h] ] * D^-0.5
p       = softmax(logits)            # sink column only feeds the denominator
o       = p[:, :-1] @ [ win_kv ; main_kv[topk(t)] ]
```

Things that make this *not* plain FA2:

| Property | Consequence for the kernel |
|---|---|
| K == V is one latent, 1 KV head shared by 64 Q heads (MQA) | pack heads into the M (row) dim; one smem KV tile serves both GEMMs |
| `D<=256` (our scope) | 64x256 bf16 tile = 32 KB, Q stays resident in smem; plain FA2 tiling works |
| Per-head learnable sink | one extra `exp2(sink - m)` added to the row sum at the end (see `examples/attention_sink`) |
| Two KV sources: SWA window (128 raw tokens, fp8 cache) + main/compressed KV (ratio `m`, fp4 cache) | one online-softmax loop that walks both sources with a single running (m, l, acc) |
| Main KV visible to `t` only for entries `< (t+1)//m` | group-causal mask, separate from the token-causal window mask |
| Top-K (512) main entries per token, indices from an indexer | gather loop over indices, exactly `examples/dsa_sparse_finetune/sparse_mla_fwd.py` |
| Cross-layer reuse (Full / Reindex / Reuse modes) | kernel takes KV / index-K / indices as pointers; reuse is a model-level scheduling choice, not a kernel feature |
| Hierarchical indexer | second indexer variant scoring a candidate list instead of the full range |

Out of kernel (elementwise, stays in torch): Q/KV RoPE, inverse RoPE on the output, `wo` projection, cache append.

## SM120 constraints to design around

- Tensor cores are `mma.sync` (no wgmma, no TMA multicast). `T.gemm` lowers to the sm80/89 path; the FA2 examples are the right template, not `flash_attention_sm100`.
- 99 KB smem per block, 64K regs per SM. At `D=256`, `acc_o[64, 256]` fp32 is 128 regs/thread at 128 threads; expect `block_M=64`, `block_N=64`, 128-256 threads.
- fp8 e4m3 MMA available; MXFP4 block-scaled MMA already works here (`src/mzoo/kernels/sm120_nvfp4_blockscaled_gemm.py`, needs the `SM120A_ENABLED` flag from `tickets/0001`). The indexer's fp4 q/k with ue8m0 per 32 channels *is* OCP MXFP4, so the indexer can use it natively. Main KV fp4 (e2m1 + e4m3 per 16, no global scale) is dequantized on load, as the paper says; Q is bf16 so no fp4 MMA there.

## Steps

Each step is its own package under `src/mzoo/layers/attn/<feature>/` (`dense_attn/`, `latent_attn/`, `swa_attn/`, `csa_attn/`, `csa2_attn/`, `indexer/`) holding `fwd.py`, `bwd.py`, `attn.py`, `ref.py`, `bench.py` and their tests. A step lands as a new package copied from the previous one, never by editing an earlier package (see Versioning), checked on GPU against a torch reference (`eager_attention_forward` from the modeling file for the later steps). Forward first per step, backward per step 7.

### 1. Dense FA2 forward baseline on GB10
- Port `examples/flash_attention/example_mha_fwd_bshd.py` (bf16, causal, `D=128`) as-is; confirm it compiles for sm121 and beats SDPA-math. Establishes the smem/reg budget numbers for this chip.
- Add the sink term: pass `Sinks[H]`, add `exp2(sink*log2e - m*scale)` to `logsum` before normalizing (`examples/attention_sink/example_mha_sink_fwd_bhsd.py`). Done in `dense_attn/` (optional `sinks=` on `fwd`/`attn`, static `has_sink` flag so the sink-free kernel is unchanged; `dsinks` is a torch reduce over `delta`/`lse` in `bwd.py`).
- Deliverable: `fwd.py`, benchmark line in the justfile.

### 2. Shared-latent MQA layout
- Switch to K==V, 1 KV head: block = `(T_q query tokens x H heads)` rows, e.g. 1 token x 64 heads. Mask per row derives from the row's token, not the block.
- Reuse the same smem KV tile for `S = Q K^T` and `O += P K`. Drop the V load entirely.
- Q resident in smem (`64 x 256` bf16 = 32 KB worst case), KV tile `64 x 256` = 32 KB, so 1-2 stages fit. No D-split needed at these dims; ffpa-attn's Split-D only matters for `D > 256`.
- Keep the RoPE tail as a plain slice of D (no separate nope/rope GEMM like MLA; V4.1 rotates in place).

### 3. Sliding-window branch (the layer type every DSV4.1 layer has)
- Restrict the KV loop to `[t0-127, t0+T_q-1]`, band mask `t-127 <= k <= t` per row. With 1 token per block that is exactly 2 tiles of 64.
- fp8 window cache: v1 dequantize in torch (ue8m0 scale per 32 channels), v2 load fp8 + scales and dequantize into the bf16 smem tile in-kernel. v2 halves window-cache bandwidth, matters mostly for decode.
- Test against the model's eager path with `compress_ratio=0` layers.

### 4. Compressed attention (DeepSeek CSA-style dense main KV, no sparsity)
- Second KV source in the same loop: after the window tiles, walk all visible main entries `k < (t+1)//m` (group-causal mask). One running max/sum across both sources, so no LSE merge is needed.
- `m=1` is the uncompressed full-attention special case and doubles as the correctness test against plain causal FA2 from step 2.
- Compressed rope at latent positions and the compressor itself stay in torch.
- fp4 main cache: v1 torch dequant, v2 in-kernel e2m1 + e4m3-per-16 dequant into smem. Layout choice for the cache (`[T, 512]` fp4 packed + `[T, 32]` scales) is fixed here and reused by everything after.
- Test: `compress_ratio=4` layer of the smoke config vs eager.

### 5. Sparse selection (CSA2 core attention)
- Replace the dense main loop of step 4 with a gather over `Indices[B, S, topk]` (per token, shared across heads, which is why the 1-token x 64-heads block from step 2 is the right layout). Template: `sparse_mla_fwd.py` (`KV_shared[i, :] = KV[Indices[s, i], :]`, `-1` / out-of-range masked to `-inf`). `topk=512` = 8 tiles of 64.
- Indices come from the model's indexer at first (torch `topk`), so step 5 is testable before any indexer kernel exists.
- Reuse mode is free: same kernel, indices tensor from an earlier layer. Full/Reindex/Reuse differ only in what the caller passes.

### 6. Indexer kernels (Full and Reindex modes)
- 6a. Score kernel: `scores[s, k] = sum_h w[s,h] * relu(q[s,h,:] . k[k,:]) * scale`, `q [S, 32, 128]`, `k [T, 128]`, group-causal mask. Template: `examples/dsa_hisa/block_sparse_mqa_fp8.py` (per-token block, heads in M, weighted head reduce). Start bf16, then MXFP4 x MXFP4 via `T.mma_gemm_blockscaled` since the ue8m0-per-32 format matches.
- 6b. Top-K: `torch.topk` on the score matrix. Only fuse (`dsa_sparse_finetune/indexer_topk_reducesum.py` style) if profiling says so.
- 6c. Hierarchical: block-max pool the Full-layer scores (block of 8 entries), take top-2048 blocks, emit a per-token candidate list (16384 positions). Reindex variant of 6a scores only the candidate list via gather, so cost is constant in context length. `select_candidate_blocks` in the modeling file is the reference.

### 7. Backward passes
- Every feature step needs a backward before it is usable for training, so each package has `bwd.py` beside `fwd.py` and an `attn.py` autograd wrapper. The forward emits `lse [B, H, S]` (`dense_attn/` does already).
- Dense + sink bwd: `examples/flash_attention/example_mha_bwd_bshd.py` and `examples/attention_sink/example_mha_sink_bwd_bhsd.py` (dsink kernel). MQA bwd: dK/dV reduce over the 64 heads sharing one latent, and since K==V, `dKV = dK + dV`.
- Sparse gather bwd: `dsa_sparse_finetune/sparse_mla_bwd.py` (scatter-add into the gathered main entries). Indexer bwd: `indexer_bwd.py` in the same dir. The indexer's detached fp4 fake-quant (README) is a model bug, not a kernel one, but the kernel path is where the straight-through estimator will have to live.
- Decode (split-KV, paged main cache) is deferred until inference matters.

## Versioning (copy-forward, freeze old)

- The version is the package: `dense_attn/`, `latent_attn/`, `swa_attn/`, ... Each holds one `fwd.py`, `bwd.py`, `attn.py` (autograd wrapper; backward recomputes S/P, saves only q, k, v, o, lse), `ref.py`, `bench.py` and `*_test.py`. No version prefixes inside a package.
- A new feature = copy the previous package to a new folder, change one thing, leave the old package untouched. Bisecting a bug is a diff between two sibling packages.
- No shared kernel helpers across packages; duplication is the point. Every package exposes the same call shapes (`fwd(q, k, v, *, causal, **extras) -> (o, lse)`, `attn(...) -> o`) in BSHD layout.
- Retire a package only by deleting it in its own commit.

## Testing and tooling

- Reference: the vendored eager forward, run layer-by-layer on the 6-layer smoke config from `archs/dsv4/model.py` (heads 4, D 64) for fast tests, plus a shape check at H 64, D 256.
- Tolerance: bf16 output, compare against fp32 eager with atol ~2e-2; sparse steps compare with identical indices fed to both.
- `just src/mzoo/layers/attn/ test` runs the tests; `just src/mzoo/layers/attn/ bench` times `dense_attn` vs SDPA.
