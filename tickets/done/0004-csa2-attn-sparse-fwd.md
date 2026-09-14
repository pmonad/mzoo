# 0004 csa2_attn: sparse top-k gather forward

- status: done (2026-09-14): gather forward passes 28 GPU tests against `golden(level="sparse")`, is bit-for-bit `csa_attn` when handed every visible group, and is flat in `G` (2.13-2.22 ms, `G` 512..16384) where the dense loop grows 1.11 -> 3.47 ms; configs untuned
- depends on: 0003 (`csa_attn`, done and independently verified 2026-09-14)
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

## Result

Package `src/mzoo/layers/attn/csa2_attn/`: `fwd.py` (285), `attn.py` (44, forward-only --
`backward` raises, 0005 brings `bwd.py` forward), `ref.py` (50, no math), `bench.py` (68),
`fwd_test.py` (195), `model_test.py` (59), `README.md`. `just src/mzoo/layers/attn/ bench-csa2`
added; `just smoke` green.

**Configs (untuned).** `csa_attn`'s `block_N` / `num_stages` verbatim (64/64/32/32,
3/3/3/2 for D 64/96/128/256). Two forced changes, both compile-time limits, not measurements:
`block_M = 64` rows (the gather is per token, and a 16-row tile fails layout inference --
new entry in `tickets/0001`), so `T_q = ceil(64/H)` tokens per block and the gathered section
repeats once per token when `T_q > 1` (`T_q = 1` at the model's `H = 64`); and `threads = 128`
(`threads=256` needs `block_M % 128 == 0`). `G` no longer needs tile padding at all -- rows are
addressed one at a time -- but `topk` is `-1`-padded to `block_N`.

**Bench** (B1 S4096 W128 D128 H64 topk=512, fwd only, vs `csa_attn`'s dense main loop on the
same `main_kv`, `ratio=1`):

| G | csa_attn | main entries/token | csa2_attn | speedup | gather alone | GB/s gathered |
|---|---|---|---|---|---|---|
| 512 | 1.107 ms | 480 | 2.171 ms | 0.51x | 1.480 ms | 363 |
| 1024 | 1.749 ms | 896 | 2.185 ms | 0.80x | 1.482 ms | 362 |
| 2048 | 2.670 ms | 1536 | 2.130 ms | 1.25x | 1.435 ms | 374 |
| 4096 | 3.441 ms | 2048 | 2.219 ms | 1.55x | 1.522 ms | 353 |
| 16384 | 3.470 ms | 2048 | 2.205 ms | 1.57x | 1.504 ms | 357 |

Gather flat in `G`, dense linear in the visible set; break-even just under `G = 2048`.
`csa_attn` stops growing past `G = S` only because group-causality caps it at `t+1` visible
entries -- at `G = 16384` it cannot reach most of the cache at all. The gather is **not**
latency-bound: 353-374 GB/s effective on the gathered rows, above the 273 GB/s the design doc
quotes, so L2 is absorbing part of it; per attended entry it still costs ~2.1x a dense tile
read. No sorting-indices experiment (optional in this ticket).

**Accuracy.** `o` vs fp32 `golden(level="sparse")` with the *same* index list, over the bf16
`golden` baseline on the same list: **0.40-0.81x** across D {64,96,128,256} x H {4,16,64} x
topk {64,100,512} (criterion 2x); `lse` <= 1.4e-06 absolute. Bit-for-bit equalities:
all-visible indices == `csa_attn.fwd`, all-`-1` indices / `topk=0` / `G=0` == `swa_attn.fwd`
(with and without a sink). Indices exercised from both `golden_ref.topk_indices` (int64) and
the real `indexer.attn` package (int32). Model replay: the unmodified tiny DSV4.1 config's
sparse layer 1 (`index_topk=64 < compressed_len=80`), indices recovered from the captured
`topk_bias`, `o` within the 2x criterion of eager and `lse` within 5e-2 of eager's logsumexp.

**Deviations from the ticket.**
- `block_M`/`threads` are not `csa_attn`'s (see above) -- forced, and flagged untuned.
- The gathered loop is `T.Pipelined(num_stages=2)`, not `T.serial`: the trip count is the
  constant `ceil(topk/block_N)`, so `csa_attn`'s miscompile trigger (a division nested in the
  bound) is absent. Verified over the whole 28-test matrix, including the non-power-of-two
  `topk = 100`, and worth ~9% of the gathered source. Fallback is one line.
- The long case runs at H=4 (B1 S4096 G16384 topk512 D128): the fp32 `golden` logits tensor
  for that shape is 1.3 GB at H=4 and 21 GB at H=64.
- `compress_ratio` is kept in the signature but never used by the kernel (documented): the
  index list owns group-causality, exactly like `golden(level="sparse")`.

Problems parked: `docs/evolution/attn/attention-kernels-impl.md` (the per-token block shape /
16-row tile, and the pipelining decision), `tickets/0001-tilelang-issues.md` (`block_M=16`
layout-inference failure at 128 threads; `T.Pipelined` correct over a constant-trip gather).
