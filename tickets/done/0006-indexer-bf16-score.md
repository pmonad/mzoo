# 0006 indexer: bf16 score kernel + top-k (forward)

- status: done (2026-09-14): 26 GPU tests pass (score kernel vs `golden_ref.indexer_scores` at the 2x-torch criterion, index-set equality vs `topk_indices`, end-to-end vs the vendored `DeepseekV41Indexer`); bench B1 S4096 Hi32 Di128: score kernel 0.98 ms at T=4096 / 1.54 ms at T=16384, and `torch.topk` costs 2-4x that -- the fuse 6b asks about is in the top-k, not the GEMM.
- depends on: 0004 only as the *consumer* of the indices; the output contract (`Indices [B, S, topk]`, -1 = empty) is already pinned by `golden_ref.py::topk_indices` and the sparse level of `golden_ref_test.py`, so the kernel, its tests and bench do not wait for 0004. The end-to-end check with `csa2_attn` lands with 0004.
- package: `src/mzoo/layers/attn/indexer/`

## Goal

First indexer package: a bf16 kernel for the lightning-indexer score matrix plus the top-k that
turns it into the `Indices` tensor `csa2_attn` consumes. Forward only and bf16 only -- the MXFP4
math path is 0008, the backward is 0007, the hierarchical candidate variant is 0009.

## Semantics

Read `DeepseekV41Indexer.forward` steps 2, 3 and 5 in `archs/dsv4/modeling_deepseek_v41.py`.

- `scores[b,s,h,t] = relu(q[b,s,h,:] . k[b,t,:]) * head_dim**-0.5`, computed in fp32, with `q [B, S,
  32, 128]` (`wq_b(q_residual)` + compress-RoPE) and shared keys `k [B, T, 128]`
  (`k_norm(wk(latent))` + compress-RoPE at latent positions).
- Head reduce: `index_scores[b,s,t] = sum_h scores[b,s,h,t] * w[b,s,h]` with `w =
  weights_proj(hidden) * n_heads**-0.5`, fp32. Note the relu is applied **before** the weighting, so
  the weights may be negative.
- Visibility: `index_scores.masked_fill(entry >= (position_ids+1)//ratio, -inf)` -- absolute
  positions, so a mid-sequence chunk sees what a one-shot prefill would.
- Top-k (`index_topk=512`): picks whose score is `-inf` are clamped into a dummy slot past the end
  and dropped; scattering them at their raw index would leak future groups. The kernel emits
  `Indices [B, S, topk]` int32 with `-1` for those, matching the `sparse_mla_fwd` convention 0004
  expects.
- v1 keeps the top-k in `torch.topk` over the score matrix; fuse only if profiling says so.
- Both q and k are `_fake_quant_fp4_block(..., block_size=32)` (ue8m0) in the model. In this ticket
  that stays a torch-side round trip and the GEMM is bf16 over dequantized values; 0008 makes the
  matmul itself fp4.

## Deliverables

`fwd.py` (score kernel), `topk.py` or a thin torch call in `attn.py`, `ref.py`, `bench.py`,
`*_test.py`, `README.md` (what/how, configs, bench, accuracy, decisions, **Known issues**, next).
`bwd.py` is ticket 0007. Tests use `dense_attn/ref.py::assert_within_2x_torch` on the score matrix;
extend `just smoke`.

## Acceptance

- Reference: the torch `einsum("bshd,btd->bsht", ...)` path of `DeepseekV41Indexer.forward` itself,
  and end-to-end index-set equality against it on the smoke config (exact set match, modulo score
  ties).
- Shapes: `index_head_dim=128` tuned; also run `D=64` for the tuned pair and `96/256` pass-only per
  the project scope. `n_heads=32`, `S` up to 4096, `T` up to 16384; `top_k > T` and `T == 0`
  degenerate cases.
- Bench vs the torch einsum + `torch.topk` baseline at `S=4096, T=4096/16384`; report score-kernel
  and top-k times separately so 6b's fuse/no-fuse call is data-driven.

## References

- tilelang `examples/dsa_hisa/block_sparse_mqa_fp8.py` (per-token block, heads in M, weighted head
  reduce), `examples/dsa_sparse_finetune/index.py` and `indexer_topk_reducesum.py`,
  `examples/deepseek_v32/`.
- `csa2_attn_design.md` step 6a/6b; `fp_attn_survey.md` §5.

## Risks / open questions

- The relu-then-weight order means no monotonicity tricks; verify the fp32 head reduce is not the
  bottleneck at 32 heads.
- Score matrix is `[B, S, T]` fp32 = 256 MB at S=T=4096, B=1: consider emitting top-k per
  query-block rather than materialising it. Decide here, it constrains 0009.
- `_fake_quant_fp4_block` **detaches the graph** (`archs/dsv4/README.md`) -- a model-side bug; do
  not replicate it in `ref.py`'s gradient path (see 0007).

## Result

Package `src/mzoo/layers/attn/indexer/`: `fwd.py` (132), `attn.py` (33), `bench.py` (61),
`fwd_test.py` (83), `attn_test.py` (115), `README.md` (192). No `ref.py` -- the reference is
`golden_ref.indexer_scores` / `topk_indices` (see the README's Decisions).

**Kernel.** Keys in the tile's M dimension, query rows (`block_M = T_s * Hi`) in N, so the
weighted head reduce is a last-axis `T.reduce_sum` over `s.reshape(block_T, T_s, Hi)` --
`examples/dsa_hisa/block_sparse_mqa_fp8.py` generalised from 1 to `T_s` query tokens per
block. fp32 accumulate / relu / weight / reduce; group-causal mask folded into the store;
the key loop cut at the visibility frontier with a serial `-inf` tail fill, so a large `T`
with a small visible prefix costs stores but no GEMMs.

**Configs (untuned).** Per the user's 2026-09-14 scope call, no tuning sweeps: one smem-safe
tile per head dim that compiles and passes at both `Hi=4` and `Hi=32`.
`Di in {32, 64, 96, 128}`: `block_M=128, block_T=64, stages=2, threads=128`;
`Di=256`: `block_M=64, block_T=64, stages=1, threads=128` (smem-capped -- `block_M=128`
fits at `Hi=32` but is 5 KB over at `Hi=4`, since `block_M` counts `T_s * Hi` rows).
`Di=32` is out of project scope and exists only for the vendored tiny model's
`index_head_dim=32`. Constraints: `block_M % Hi == 0`, `S % (block_M // Hi) == 0`; `T` free
(keys zero-padded to a multiple of `block_T`).

**Bench** (`just src/mzoo/layers/attn/ bench-indexer`; B1 S4096 Hi32 Di128 ratio=1 topk=512;
baseline = the same math in torch, bf16 matmul + fp32 reduce, chunked 512 queries at a time
so the `[B,S,Hi,T]` intermediate does not dominate):

| `T` | score kernel | torch scores | speedup | `torch.topk` (ours) | `torch.topk` (torch) | score matrix |
|---|---|---|---|---|---|---|
| 4096 | 0.977 ms | 81.6 ms | 83.5x | 1.89 ms | 1.89 ms | 64 MB |
| 16384 | 1.540 ms | 331.0 ms | 215.0x | 6.09 ms | 6.01 ms | 256 MB |

**6b's fuse/no-fuse datapoint: the top-k is 2-4x the score kernel**, and it costs the same
on either side's score matrix -- so the remaining time is in `torch.topk` over `[B, S, T]`,
not in the GEMM. Fusing is worth it on both time and memory.

**Accuracy.** Max-abs error vs the fp32 `indexer_scores` across
`Di in {32,64,96,128,256}` x `Hi in {4,32}` x `ratio in {1,2,4}`: 1e-6..4e-6 on scores of
magnitude ~1-4, i.e. far inside `assert_within_2x_torch`; the `-inf` mask matched exactly
in every case. Index-set equality vs `topk_indices` is exact for `topk in {64, T+7}`.

**Materialisation decision (the ticket's open question).** v1 **materialises** the
`[B, S, T]` fp32 score matrix and runs `torch.topk` on it. That is 64 MB at
`S = T = 4096, B = 1` and **256 MB at `S = 4096, T = 16384, B = 1`** (scaling as
`B * S * T * 4`, so 2 GB at batch 8 on the long row) for a tensor whose only consumer keeps
512 columns. **0009 must revisit it**: the hierarchical candidate variant only pays off if
the full matrix is never materialised. Fused per-query-block top-k
(`dsa_sparse_finetune/indexer_topk_reducesum.py` style) is deliberately not implemented here.

**Deviations from the ticket.**
- No `ref.py` and no `topk.py` (the top-k is three lines in `attn.py`) -- as specified by the
  implementation brief.
- The end-to-end model check compares the **score values** of the two chosen sets, not the
  index sets: ~7% of the freshly-initialised tiny model's visible scores are exactly 0.0
  (all four heads relu'd to zero) and `torch.topk(sorted=False)` breaks those ties
  arbitrarily. The ticket allows this ("exact set match, modulo score ties"); observed
  difference of the sorted score vectors is exactly 0.0. Parked in
  `docs/evolution/attn/attention-kernels-impl.md`.
- `Di=32` added to `CONFIGS` so that end-to-end check can run on the vendored tiny model
  without duplicating its config.
- No new `tickets/0001` entries: this kernel has no fp32-accumulator-plus-bf16-cast
  `T.Parallel`, so the layout-inference conflict never appeared; every compile failure was
  smem budget.
