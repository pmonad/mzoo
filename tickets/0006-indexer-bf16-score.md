# 0006 indexer: bf16 score kernel + top-k (forward)

- status: todo
- depends on: 0004 (`csa2_attn` forward, the consumer of the indices)
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
