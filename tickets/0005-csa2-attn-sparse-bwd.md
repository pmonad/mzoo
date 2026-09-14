# 0005 csa2_attn: sparse gather backward + autograd

- status: todo
- depends on: 0004 (`csa2_attn` forward)
- package: `src/mzoo/layers/attn/csa2_attn/` (backward half)

## Goal

Add `bwd.py` and the `attn.py` autograd wrapper to `csa2_attn/`, making the sparse package
trainable. Split out from 0004 because the gather backward is a different kernel shape from the
forward: the KV-block-owning loop of `dense_attn` does not exist when KV rows are reached only
through per-token indices, so dKV becomes a scatter-add.

## Semantics

Same forward semantics as 0004 (`DeepseekV41Indexer.forward` step 5 for the index conventions,
`eager_attention_forward` for the softmax/sink).

- Recompute S/P from saved `(q, kv, indices, o, lse)` -- never materialise `S^2`. `lse` already
  includes the sink column, so the recomputed P is sink-normalised.
- Window slice: dKV as in 0002/0003 (block-owning loop, `dKV = dK + dV`, fp32).
- Main slice: each query row touches 512 arbitrary entries, so dKV is a **scatter-add** into the
  gathered rows; accumulate in fp32 (`atomic_add`), never bf16 atomics. Masked / out-of-range
  indices contribute nothing.
- Both sources write into one `dKV` buffer because the model concatenates them on the KV axis; the
  caller splits it back into window and compressed tensors.
- `dsinks[h] = -sum_{b,s} exp(sink[h] - lse[b,h,s]) * delta[b,h,s]`, torch fp32, unchanged.
- No gradient flows to `Indices` (top-k selection is discrete); the indexer's own gradient path is
  ticket 0007.

## Deliverables

`bwd.py` (preprocess `delta = rowsum(dO*O)`, main kernel, postprocess cast), `attn.py` (autograd;
saves exactly `q, kv, indices, o, lse[, sinks]`, pinned by a `test_backward_saves_only_*` test as in
`dense_attn`), bench rows for `bwd` and `fwd+bwd`, tests, and the **Known issues** / bench /
accuracy sections of `csa2_attn/README.md` filled in. Extend `just smoke`.

## Acceptance

- Reference: autograd through `attn/golden_ref.py::golden` (`sparse` level) fed the *same* indices;
  `assert_within_2x_torch` on `dq`, `dkv`, `dsinks`.
- Duplicate-index case must be exercised (two picks landing on the same entry) -- scatter-add
  correctness is invisible otherwise.
- `D in {64, 96, 128, 256}`, 64/128 tuned, 96/256 pass-only; `topk=512`.
- Determinism note in the README: fp32 `atomic_add` makes dKV run-to-run non-bitwise-reproducible;
  state it rather than fixing it.
- Bench vs `csa_attn` fwd+bwd at `G = 4096, 16384`, same shapes as 0004.

## References

- tilelang `examples/dsa_sparse_finetune/sparse_mla_bwd.py` (scatter-add template; `assert dtype ==
  bfloat16` upstream), `examples/flash_attention/example_mha_bwd_bshd.py`,
  `examples/attention_sink/example_mha_sink_bwd_bhsd.py`.
- `csa2_attn_design.md` step 7; `fp_attn_survey.md` §3 (keep `dP = dO V^T` bf16).

## Risks / open questions

- fp32 atomic contention: with `topk=512` shared across all 64 heads of a token, every head's
  contribution hits the same 512 rows. Consider a per-head register reduce before the atomic.
- `dense_attn`'s bwd is already ~200 lines and slower than sdpa because of the atomic-dQ design;
  adding a second atomic (dKV) may make this the slowest package. Bench before tuning, and record
  the split.
- Layout-inference limits from `tickets/0001-tilelang-issues.md` bite hardest in the bwd
  (`block_M=32` unusable); budget tuning time for D=256.
