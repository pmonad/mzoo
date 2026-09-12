# indexer.py

DSA-style sparse selector for the compressed branch: a small indexer scores every query
against one key per compressed group and publishes a 0 / $-\infty$ bias that lets attention
see only the top `index_topk` groups, optionally restricted first to top candidate blocks.

## Shapes

| symbol | shape | meaning |
|---|---|---|
| $x$ | `[B, S, D]` | hidden states; B=8, S=512, D=256 |
| $r_s$ | `[B, S, q_lora_rank]` | attention's low-rank query residual `q_norm(wq_a(x))`; rank 128 |
| $c_j$ | `[B, G, d]` | pre-RoPE compressor latents of this call; d=64 |
| $H_i$, $d_i$ | scalars | `index_n_heads` = 4, `index_head_dim` = 32 |
| $q$ | `[B, S, H_i, d_i]` | indexer queries |
| $k$ / `shared["index_k"]` | `[B, 1, T, d_i]` | indexer keys, one per group; T = groups so far (256 at r=2, 512 at r=1) |
| $I$ | `[B, S, T]` | head-mixed index scores |
| $K$, $K_b$, $b$ | scalars | `index_topk` = 64, `candidate_topk_blocks` = 64, `candidate_block_size` = 8 |
| `shared["candidates"]` | `[B, S, T]` bool | level-one block mask |
| `shared["topk_bias"]` | `[B, 1, S, T]` | 0 for selected groups, $-\infty$ otherwise |

## `select_candidate_blocks(logits, compress_lens, topk_blocks, block_size)`

Level one of the two-level top-k. Tiles the $T$ group axis into blocks of $b$, scores a block
by its max, force-includes the (partially filled) block containing the query's newest visible
group, keeps the $K_b$ best, and expands back to a per-group bool mask.

$$\beta_{s,\ell} = \max_{j \in \text{block } \ell} I_{s,j},\qquad
\beta_{s,\ell^*} = +\infty \ \text{ for } \ell^* = \Big\lfloor \tfrac{L_s - 1}{b} \Big\rfloor,\qquad
L_s = \text{visible groups}$$

```
pad logits to a multiple of b with -inf; beta = blockwise max
beta[last visible block] = +inf                     # pin the newest, partially filled block
top = topk(beta, min(K_b, num_blocks))
keep = scatter(top.indices, top.values > -inf)      # unreachable (-inf) picks are dropped
return keep repeated b times per block, cut to T
```

Smoke config: at layer 4 (r=1, T=512) there are 64 blocks and $K_b = 64$, so every reachable
block is kept and level one is a no-op there.

## `DeepseekV41Indexer`

Lives on each `index_source_layer_ids` layer. Keys are only produced by a layer that also owns
its compressor (`owns_k`, i.e. a KV source); "Reindex" sources rescore the shared keys with
their own `wq_b` / `weights_proj`. Forward has no return value; it writes `shared["topk_bias"]`
and, on the candidate source layer, `shared["candidates"]`.

Keys (KV source only, from the pre-RoPE latent, compress-RoPE at position $p_j = p_0 + r j$):

$$k_j = \mathrm{FQ}_{\text{fp4}}\big(\mathrm{RoPE}_c(\mathrm{RMSNorm}_k(W_k\, c_j))\big)$$

Queries and scores:

$$q_{s,h} = \mathrm{FQ}_{\text{fp4}}\big(\mathrm{RoPE}_c(W_{q_b}\, r_s)_h\big),\qquad
a_{s,h,j} = \frac{\mathrm{ReLU}(q_{s,h} \cdot k_j)}{\sqrt{d_i}},\qquad
w_{s,h} = \frac{(W_w\, x_s)_h}{\sqrt{H_i}}$$

$$I_{s,j} = \sum_{h=1}^{H_i} w_{s,h}\, a_{s,h,j},\qquad
I_{s,j} = -\infty \ \text{ if } j \ge L_s = \Big\lfloor \tfrac{p_s + 1}{r} \Big\rfloor$$

($p_s$ is the absolute `position_ids` value, so a group is visible once the query has passed
its last token.) Final bias, with the candidate mask applied first if this layer
`uses_candidates`:

$$\mathrm{bias}_{s,j} = \begin{cases} 0 & j \in \mathrm{top}_{K}(I_{s,\cdot}) \text{ and } I_{s,j} > -\infty \\ -\infty & \text{otherwise}\end{cases}$$

```
top = topk(I, min(K, T), sorted=False)
safe = where(top.values > -inf, top.indices, T)     # invalid picks go to a dummy slot T
bias = full([B,1,S,T+1], -inf); bias.scatter(safe, 0); bias = bias[..., :T]
```

`_fake_quant_fp4_block(., block_size=32)` (ue8m0 power-of-two scale per 32 channels) is
applied to both $k$ before it enters `compressed_kv["indexer"]` and to $q$ before scoring;
with $d_i = 32$ it is active in the smoke config (it silently no-ops when $d_i \bmod 32 \ne 0$).

## Notes / gotchas

- **The indexer receives no gradient**, so the selector stays at random init for the whole
  run and selection is effectively random top-64. This is the biggest open issue in the arch.
  Two causes, and the second is the decisive one:
  1. `topk` emits a hard 0 / $-\infty$ mask, which is not differentiable.
  2. `_fake_quant_fp4_block(x, 32)` (the ue8m0 path applied to indexer q/k) returns a tensor
     **fully detached from the autograd graph** — verified: `requires_grad=False` on the
     output. This severs `wq_b` / `wk` / `k_norm` from the loss regardless of (1). By
     contrast the fp8 path on window KV is a proper straight-through estimator (grad 1 on
     every element), so the asymmetry looks unintentional. See [quant.md](quant.md).

  The paper does **not** describe any auxiliary or KL loss for the indexer (§2.4.4 only says
  FP4 indexer q/k use "quantization-aware training", citing Jacob et al. 2018 — the canonical
  straight-through reference). §3.1.2 confirms indexer parameters *are* optimized, with
  "gradient aggregation" across shadow replicas, and §1 says sparse attention is "trained from
  scratch ... without any dense attention warmup stages". So upstream trains this jointly by
  gradient; a working STE through the fp4 quant is the missing piece here, not a missing loss.
- In the 6-layer smoke config `candidate_source_layer_id` auto-derives to 4, which is also
  the last index source, so `uses_candidates` is never true: the candidate mask is computed
  but never consumed. The two-level path only matters with an index source after the
  candidate source (e.g. the released 40-layer schedule).
- Consumer layers 3 and 5 reuse the source's `topk_bias` through `shared`; scores are fp32.
- With a cache the running keys come from `cache_layer.compressed_kv["indexer"]`; that path is
  unreachable today (see `cache.md`, cache construction fails).
