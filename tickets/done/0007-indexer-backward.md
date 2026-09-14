# 0007 indexer: backward + autograd (bf16, straight-through)

- status: done (2026-09-14): 22 bwd tests pass (`assert_within_2x_torch` on dq/dk/dw vs autograd through `golden_ref.indexer_scores`, masked-entry exactly-zero gradient, relu'(0)=0 pinned, fp64 gradcheck, STE bit-exactness, feature-off bit-identical to 0006); bench B1 S4096 Hi32 Di128: bwd kernel 7.08 ms at T=4096 (28.9x torch fwd+autograd) / 7.15 ms at T=16384 (114.2x).
- depends on: 0006 (`indexer` forward)
- package: `src/mzoo/layers/attn/indexer/` (backward half)

## Goal

Make the indexer trainable: `bwd.py` for the score kernel plus the `attn.py` autograd wrapper. Split
from 0006 because this backward is not an attention backward at all -- no softmax, no `lse`; it is a
relu-gated weighted reduce whose gradient reaches three inputs (`q`, `k`, `weights`) and whose fp4
fake-quant needs an explicit straight-through estimator.

## Semantics

Read `DeepseekV41Indexer.forward` step 2 in `archs/dsv4/modeling_deepseek_v41.py`.

With `r = q[b,s,h,:] . k[b,t,:]`, `a = relu(r) * scale`, `L = sum_h a * w[b,s,h]`:

- `dw[b,s,h] = sum_t a[b,s,h,t] * dL[b,s,t]` (times `n_heads**-0.5`).
- `dr = w[b,s,h] * dL[b,s,t] * scale * (r > 0)` -- the relu gate is on the *unweighted* dot product;
  then `dq = dr @ k`, `dk = sum_{s,h} dr * q`, with `dk` reduced over both the 32 heads and all
  queries that saw entry `t`.
- Masked entries (`entry >= (position_ids+1)//ratio`) and, in the hierarchical case, non-candidate
  entries contribute zero.
- Top-k is discrete: no gradient through the selection. The gradient reaching the indexer in
  training comes from its own auxiliary loss, not from `csa2_attn`.
- fp4 STE: the model's `_fake_quant_fp4_block` detaches, which is why the indexer never trains
  (`archs/dsv4/README.md`). The kernel path is where the STE lives -- gradient w.r.t. q/k is the
  identity with the scale stop-gradiented; the backward itself runs bf16 over the dequantized
  values, even once 0008 makes the forward fp4. Record this as a deliberate divergence from the
  vendored model.

## Deliverables

`bwd.py` (dq/dk/dweights; dk needs an fp32 split buffer or atomic over the query axis), `attn.py`
(autograd; saves `q, k, weights` and the relu mask implicitly by recompute, not a stored `[S, H, T]`
tensor), tests including a `torch.autograd.gradcheck`-style fp64 check on a tiny shape, bench rows,
and the accuracy / **Known issues** sections of `indexer/README.md`. Extend `just smoke`.

## Acceptance

- Reference: autograd through the torch einsum path of `DeepseekV41Indexer.forward` with the detach
  removed; `assert_within_2x_torch` on `dq`, `dk`, `dw`.
- Explicit test that a masked entry receives exactly zero gradient, and that `r == 0` rows use the
  subgradient convention chosen (document which).
- Shapes as 0006: `index_head_dim=128` tuned, others pass-only; `n_heads=32`.
- Bench vs autograd-through-torch at `S=4096, T=4096/16384`.

## References

- tilelang `examples/dsa_sparse_finetune/indexer_bwd.py` (the only close template; bf16 only
  upstream), `lemyx/tilelang-dsa` (public TileLang indexer *training* operator, bf16, fp8 listed as
  future work) via `fp_attn_survey.md` §5.
- `csa2_attn_design.md` step 7.

## Risks / open questions

- `dk` reduces over `S * 32` contributions per entry -- atomic contention is worse than the
  attention backward's; consider a two-stage reduce (per query block, then a cheap second kernel).
- Recomputing `r` in the backward doubles the score GEMM; compare against saving the relu mask as a
  bitmask (`[B, S, 32, T]` bits = 16 MB at S=T=4096).
- If the model-side detach is ever fixed upstream, this ticket's STE should move or be reconciled --
  flag it in `archs/dsv4/README.md` rather than silently diverging.

## Learnings from earlier steps (2026-09-14)

- The model's own top-k is not reproducible by index set on a fresh-init model (7% exact-zero
  scores, `torch.topk(sorted=False)` breaks ties arbitrarily): end-to-end checks compare the
  score values of the chosen sets, not the sets (see `indexer/attn_test.py`).
- Packed-heads tiles count `T_s * Hi` rows, so smem depends on Hi: validate every config at
  Hi=4 and Hi=32 (`indexer` D=256 fit at Hi=32 only).
- relu-before-weight is pinned by a test where half the head weights are negative (26.7 abs
  difference from weight-then-relu); the backward must gate `dq`/`dk` on the same relu mask.
- `_fake_quant_fp4_block` detaches the graph in the model; the fp32 reference for the backward
  is `golden_ref.indexer_scores` (no quant), not the model.
- No tuning sweeps (user decision 2026-09-14): one config per dim, flagged untuned.

## Result

Package `src/mzoo/layers/attn/indexer/`: `bwd.py` (161) new; `attn.py` (81) gains the
differentiable `scores()` + `_fake_quant_fp4_ste`; `bench.py` gains the bwd table;
`bwd_test.py` (181) new. Reference stays `golden_ref.indexer_scores` (autograd, detach
removed); the bf16 baseline is a local test artifact, per the fwd_test convention.

**Kernel.** Same tiling as `fwd.py` (keys in M, packed query-head rows in N). The
backward recomputes the score GEMM to re-derive the relu gate (no `[S, Hi, T]` mask
saved -- the wrapper saves only `q, k, w`), stages one `dr` tile and gets both gradient
GEMMs off it: `dr^T @ K -> dq` (block-exclusive), `dr @ Q -> dk` (fp32 atomics over the
query axis), `dw` via a fragment reduce. Subgradient: relu'(0) = 0 (torch's).

**STE.** `_fake_quant_fp4_ste = x - x.detach() + quant` in `x`'s own dtype: forward
bit-identical to the model's `_fake_quant_fp4_block`, backward the identity; scales
stop-gradiented inside the quantizer. The model's detach stays untouched; the
divergence is flagged in `archs/dsv4/README.md`.

**Two tilelang quirks hit** (details in `docs/evolution/attn/attention-kernels-impl.md`
and `tickets/0001`): `out_idx` outputs are empty-allocated so an atomics-only output
accumulated garbage (fixed by wrapper-allocated in-out buffers), and fp32 atomics make
`dk` order-nondeterministic at ~1e-4 (tests: `dq`/`dw` bit-exact, `dk` at 1e-3).

**Bench** (GB10, B1 S4096 Hi32 Di128, ratio 1): bwd kernel 7.08 ms at T=4096 /
7.15 ms at T=16384 vs torch fwd+autograd 204.6 / 816.4 ms (28.9x / 114.2x). The
recomputed score GEMM puts the backward at ~7x the forward kernel's cost -- see the
README's Known issues for the bitmask/two-stage escape hatches.
