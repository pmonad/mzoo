# indexer

DSV4.1 lightning-indexer, **forward only, bf16**: the score kernel of step 6a of
`../csa2_attn_design.md` plus the top-k of step 6b that turns its output into the
`Indices` tensor `csa2_attn` consumes. Ticket:
[0006](../../../../../tickets/0006-indexer-bf16-score.md). The backward is 0007, the
MXFP4 math path is 0008, the hierarchical candidate variant is 0009.

| file | what |
|---|---|
| `fwd.py` | the score kernel (`fwd(q, k, w, *, compress_ratio) -> [B, S, T]` fp32) |
| `attn.py` | public entry (`attn(q, k, w, *, compress_ratio, topk) -> Indices [B, S, topk]` int32) |
| `bench.py` | score kernel and top-k vs a torch einsum + `torch.topk` baseline |
| `fwd_test.py`, `attn_test.py` | GPU tests |

## What it is

```
index_scores[b, s, t] = sum_h w[b, s, h] * relu(q[b, s, h, :] . k[b, t, :]) * Di**-0.5
```

- `q` bf16 `[B, S, Hi, Di]` (already rotated and fp4-round-tripped by the caller),
  `k` bf16 `[B, T, Di]` (one shared key per compressed group), `w` fp32 `[B, S, Hi]`
  (already carrying the `Hi**-0.5` factor), scores fp32 `[B, S, T]`.
- The relu runs **before** the weighting, so `w` may be negative and there is no
  monotonicity trick to exploit.
- Group-causal visibility, in absolute prefill positions `0..S-1`: entry `t` is visible
  to query `s` iff `t < (s + 1) // compress_ratio`. Invisible entries are `-inf`.
- `attn` then takes `torch.topk` and emits `[B, S, topk]` int32, `-1` wherever the pick
  was `-inf` (fewer than `topk` visible entries yet) or `topk > T` ran out of columns --
  exactly `golden_ref.topk_indices`' contract, which is what `sparse_mla_fwd`/0004 expects.
  `T == 0` returns all `-1`.

**Tile shape.** Copied from tilelang `examples/dsa_hisa/block_sparse_mqa_fp8.py` and
generalised from 1 to `T_s` query tokens per block. A block owns `block_M = T_s * Hi`
**query rows** -- `T_s` consecutive tokens x `Hi` heads, the natural BSHD order of
`q.view(B, S*Hi, Di)`, so the tile is one contiguous slice -- and walks the key axis in
`block_T`-row tiles. The GEMM is `K_shared [block_T, Di] @ Q_shared [block_M, Di]^T ->
s [block_T, block_M]`: **keys in M, query rows in N**. That orientation is the whole
point -- the weighted head reduce becomes a plain `T.reduce_sum(dim=-1)` over the last
axis of `s.reshape(block_T, T_s, Hi)`, the same trick the template uses, instead of a
partial reduce over rows of an MMA accumulator. Accumulate, relu, weight and reduce are
all fp32. `logits [block_T, T_s]` is staged through a small shared buffer so the
transposed store to `Scores[b, s, t]` stays coalesced along `t`.

**Group-causal early-out.** The key loop is cut at the frontier the same way FA2 cuts
its causal loop (`loop_range = min(ceildiv(T, block_T), ceildiv(((bx+1)*T_s) // ratio,
block_T))`), and a short serial loop fills the invisible tail of each row with `-inf`.
A large `T` with a small visible prefix therefore costs stores but no GEMMs.

**fp4.** Both q and k are `_fake_quant_fp4_block(..., block_size=32)` in the model. Per
ticket 0006 that stays a **torch-side round trip outside the kernel**: the kernel sees
bf16 values that have already been through it. Making the matmul itself MXFP4 is 0008.

## How to run

```
just src/mzoo/layers/attn/ test indexer -q      # GPU tests
just src/mzoo/layers/attn/ bench-indexer        # the two-row bench table below
```

## Tile configs

`CONFIGS` in `fwd.py`, keyed by index head dim. **All untuned** -- per the user's
2026-09-14 scope call, correctness is this ticket's deliverable and tuning is a later
pass. Each entry is simply the first smem-safe tile that compiles and passes at *both*
supported head counts (`Hi in {4, 32}`).

| `Di` | `block_M` (query rows) | `block_T` (key rows) | stages | threads |
|---|---|---|---|---|
| 32 | 128 | 64 | 2 | 128 |
| 64 | 128 | 64 | 2 | 128 |
| 96 | 128 | 64 | 2 | 128 |
| 128 | 128 | 64 | 2 | 128 |
| 256 | 64 | 64 | 1 | 128 |

`Di=32` is not in the project's `{64, 96, 128, 256}` scope; it is here only so the
vendored tiny model (`index_head_dim=32`) can be checked end-to-end.

`Di=256` is smem-capped. `block_M=128` fits at `Hi=32` but overflows at `Hi=4`
(106496 B, 5 KB over the 99 KB budget) because `block_M` is `T_s * Hi` query *rows*:
the same `block_M` means 32 tokens at `Hi=4` against 4 at `Hi=32`, and the
`[block_T, T_s]` staging buffer grows with it. See the parking note in
`docs/evolution/attn/attention-kernels-impl.md`.

`threads=128` everywhere, so the `tickets/0001` rule "with `threads=256`, `block_M`
must be a multiple of 128" never binds. This kernel has no fp32-accumulator-plus-bf16-cast
`T.Parallel` at all (nothing is cast back to bf16), so it did not reproduce that
layout-inference conflict -- every compile failure in this package was smem budget.

### Shape constraints

- `block_M % Hi == 0` and `S % (block_M // Hi) == 0` -- i.e. `S` must be a multiple of
  4 at `Hi=32` (32 at `Hi=4`, 2/16 for `Di=256`). Asserted in `fwd`.
- `T` is free: keys are zero-padded to a multiple of `block_T` inside `fwd` and the
  extra score columns are sliced off.
- `Di in {32, 64, 96, 128, 256}`, `Hi in {4, 32}` tested; `S` to 4096 and `T` to 16384
  benched.

## Bench

`just src/mzoo/layers/attn/ bench-indexer`, GB10 (sm121), B=1, S=4096, Hi=32, Di=128,
`compress_ratio=1`, `topk=512`. Score kernel and top-k are timed separately so ticket
6b's fuse/no-fuse call is data-driven. Baseline: the same math in torch (bf16 `q @ k^T`,
relu, fp32 weighted head reduce, group-causal `-inf`), chunked 512 queries at a time --
the model's unchunked `einsum("bshd,btd->bsht")` materialises a `[B, S, Hi, T]`
intermediate (8.6 GB fp32 on the `T=16384` row) and would measure the allocator.

| `T` | score kernel | torch scores | speedup | `torch.topk` (ours) | `torch.topk` (torch scores) | score matrix |
|---|---|---|---|---|---|---|
| 4096 | 0.977 ms | 81.6 ms | 83.5x | 1.89 ms | 1.89 ms | 64 MB |
| 16384 | 1.540 ms | 331.0 ms | 215.0x | 6.09 ms | 6.01 ms | 256 MB |

Two readings:

- The score kernel is ~70 TFLOPS at the `T=4096` row (137 GFLOP full, ~half of it
  skipped by the group-causal early-out) against a ~250 TFLOPS bf16 peak. Untuned, as
  stated above; the gap is the tuning pass's, not correctness'.
- **The top-k is 2-4x more expensive than the score kernel.** That is the answer 6b
  wanted: fusing the top-k into the score kernel is where the remaining time is, not
  in the GEMM. It is also the same cost for either side's score matrix, so it is a
  property of `torch.topk` over a `[B, S, T]` fp32 tensor, not of our output.

## Accuracy

`fwd_test.py`, `dense_attn/ref.py::assert_within_2x_torch` on the score matrix. The
fp32 reference is `golden_ref.indexer_scores` on fp32 inputs; the bf16 baseline is the
same einsum with a bf16 matmul and an fp32 head reduce (what the kernel does), so the
comparison isolates tiling rather than dtype. `-inf` entries are excluded from the
numeric comparison and checked instead as an **exact mask equality**, which is stronger.

Observed max-abs error against the fp32 reference across the shape sweep
(`Di in {32, 64, 96, 128, 256}` x `Hi in {4, 32}` x `ratio in {1, 2, 4}`): 1e-6 to
4e-6, on scores of magnitude ~1-4. Mask equality held in every case.

Selection: `attn_test.py` asserts **exact index-set equality** per query row against
`golden_ref.topk_indices` on the fp32 reference scores (random gaussian inputs, ties
improbable), for `ratio in {1, 2, 4}` and `topk in {64, T+7}`.

End-to-end against the vendored model (`test_matches_vendored_model_indexer`): the tiny
2-layer config of `golden_ref_test._build_tiny_model` is run once and the indexer's own
post-fp4 `q`/`k`, its fp32 `weights` and its published `shared["topk_bias"]` are
captured (a spy on the module-level `_fake_quant_fp4_block` -- its `block_size=32` calls
are exactly the indexer's k then q -- a forward hook on `weights_proj`, and a wrapper
around `DeepseekV41Indexer.forward`). That one is compared by the **score values** of
the two chosen sets rather than by index set: ~7% of the freshly-initialised model's
visible scores are exactly 0.0 (all four heads relu'd to zero) and `torch.topk` breaks
those ties arbitrarily. Observed difference of the sorted score vectors: exactly 0.0.
See the parking note in `docs/evolution/attn/attention-kernels-impl.md`.

## Decisions

- **No `ref.py`.** The reference is `../golden_ref.py`: `indexer_scores` (steps 2-3 of
  `DeepseekV41Indexer.forward`) and `topk_indices` (step 5). They already pin this
  package's output contract, so re-deriving the math here would only create a second
  thing to keep in sync. The bf16 *baseline* for `assert_within_2x_torch` lives in
  `fwd_test.py`, since it is a test artifact, not a reference.
- **Keys in M, query heads in N** (not queries in M). It makes the weighted head reduce
  a last-axis `reduce_sum`, which is what the template does and what tilelang handles
  natively; a reduce over sub-blocks of an MMA accumulator's row dimension is not.
- **Materialise the `[B, S, T]` score matrix** and run `torch.topk` on it, per ticket
  0006's "v1 keeps the top-k in `torch.topk`". Simplest thing that is correct, and it
  makes the fuse/no-fuse question measurable rather than assumed. See Known issues.
- **fp4 stays outside the kernel** (ticket semantics): the kernel is bf16-in.
- **Forward only, `compress_ratio` a kernel constant** (it specialises the mask and the
  loop bound, and it is fixed per layer).
- Configs are **untuned**, one per head dim; see Tile configs.

## Known issues

- **The materialised score matrix is the memory ceiling.** `[B, S, T]` fp32 is 64 MB at
  `S = T = 4096, B = 1` and **256 MB at `S = 4096, T = 16384, B = 1`** -- it scales as
  `B * S * T * 4`, so a batch of 8 at the long row is 2 GB, for a tensor whose only
  consumer is a `topk` that keeps 512 columns. **Ticket 0009 must revisit this**: the
  hierarchical variant scores a per-token candidate list instead of the full range, and
  that only pays off if the score matrix is not materialised in the first place. The
  fused alternative (emit top-k per query block, `dsa_sparse_finetune/
  indexer_topk_reducesum.py` style) is not implemented here on purpose.
- The bench says the unfused `torch.topk` costs 2-4x the score kernel, so the fuse is
  also the performance answer, not just the memory one.
- Configs are untuned; ~70 TFLOPS against a ~250 TFLOPS peak.
- `Di=256` is shape support only (smem-capped to `block_M=64`, 1 stage).
- No backward (0007), no MXFP4 matmul (0008), no decode/varlen, no cross-batch varying
  `position_ids` -- prefill positions are assumed to be `0..S-1`.

## Next

0007 (backward: the relu-gated three-input reduce, plus the straight-through estimator
the model's detached `_fake_quant_fp4_block` needs), then 0008 (MXFP4 matmul, which the
`ue8m0`-per-32 format makes native on this hardware) and 0009 (hierarchical candidates,
which is where the score-matrix materialisation has to go away). A tuning pass over
`CONFIGS` is orthogonal and can happen any time.
