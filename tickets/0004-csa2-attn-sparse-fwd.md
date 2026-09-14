# 0004 csa2_attn: sparse top-k gather forward

- status: todo
- depends on: 0003 (`csa_attn`)
- package: `src/mzoo/layers/attn/csa2_attn/` (forward half)

## Goal

Copy `csa_attn/` to `csa2_attn/` and replace the dense main-KV loop with a gather over per-token
top-k indices, leaving the window branch untouched. Indices come from the model's torch indexer, so
this is testable before any indexer kernel exists (tickets 0006+). Forward only; backward is ticket
0005.

## Semantics

Read `DeepseekV41Indexer.forward` step 5 and `DeepseekV41Attention.forward` (`block_bias` /
`attention_mask` concat) in `archs/dsv4/modeling_deepseek_v41.py`.

- The model expresses selection as an additive bias: `block_bias [B, 1, S, G]` is `0.0` on the
  `index_topk` (512) selected entries and `-inf` everywhere else, and is concatenated onto the
  window mask along the KV axis. The kernel instead takes `Indices [B, S, topk]` (int32, shared
  across the 64 heads -- which is why the 1 token x 64 heads block layout is the right one) and
  gathers.
- Out-of-range / padding indices: the model clamps picks whose score is `-inf` into a dummy slot
  past the end (`safe = where(valid, idx, compressed_len)`) and slices it off. The kernel mirrors
  this: index `< 0` or `>= G` contributes `-inf` logits and zero output, per `sparse_mla_fwd.py`.
- Group-causal visibility is already baked into the indices (only entries `k < (t+1)//ratio` can be
  selected), but keep the `>= G` guard: early tokens have fewer visible groups than `topk`.
- Window branch, sink column, softmax and `lse` unchanged from 0003; one running `(m, l, acc)` walks
  window tiles then `topk/64 = 8` gathered tiles.
- Full / Reindex / Reuse modes are free: same kernel, different `Indices` tensor supplied by the
  caller. Not a kernel feature.

## Deliverables

`fwd.py` (gather loop), `ref.py` (torch gather reference + the bias-form reference for
cross-checking), `bench.py`, `*_test.py`, `README.md` (what/how, configs, bench, accuracy,
decisions, **Known issues**, next); `bwd.py`/`attn.py` land in 0005 -- until then `attn.py` may
raise on backward. Extend `just smoke`.

## Acceptance

- Reference: `attn/golden_ref.py::golden` at the `sparse` level, fed the *same* indices (bias
  form) as the kernel -- identical selection on both sides, per the Conventions note on sparse
  steps.
- `topk=512`, `G` up to 16k; `D in {64, 96, 128, 256}`, 64/128 tuned, 96/256 pass-only; H=64/D=256
  shape check; degenerate cases: `topk > visible groups`, all-masked row (sink keeps it finite),
  `topk` not a multiple of `block_N`.
- `o` within 2x torch bf16; `lse` within ~1e-6 fp32.
- Bench vs `csa_attn` dense main loop at `G = 4096, 16384`: the gather must be flat in `G` while the
  dense loop grows linearly.

## References

- tilelang `examples/dsa_sparse_finetune/sparse_mla_fwd.py` (the template: `KV_shared[i, :] =
  KV[Indices[s, i], :]`, `-1` masked to `-inf`), `examples/deepseek_v4/sparse_attn_fwd_sm90.py`.
- `csa2_attn_design.md` step 5.

## Risks / open questions

- Gather is random-access into gmem at 273 GB/s with no TMA multicast on sm120; measure whether the
  512-entry gather is latency-bound and whether sorting indices per token helps locality.
- K==V means one gathered tile feeds both GEMMs -- keep that, do not gather twice.
- Indices are shared across heads but not across the tokens in a block; a multi-token block would
  need one gather per token. Stay at 1 token x 64 heads unless the bench says otherwise.
