---
title: "Attention kernels: implementation notes"
---

# Attention kernels: implementation notes

Parking lot for problems met while implementing the kernels of this chapter
(`src/mzoo/layers/attn/`). One short section per problem: what went wrong,
what the measurement showed, how it was fixed. No expansion here; the full
write-up lands in the chapter later.

## latent_attn backward: 2x slower than SDPA

**Problem.** `latent_attn` (shared K==V latent, heads packed into the tile
rows) matches or beats SDPA in the forward (D128 H64 S4096 causal: 2.90 ms vs
3.35 ms), but fwd+bwd runs 33.2 ms against SDPA's 15.0 ms with the latent
expanded to 64 heads (D64: 14.9 vs 7.3 ms). `dense_attn`'s backward on the
same template is only ~1.2x behind SDPA, so the gap is specific to the MQA
layout. First suspects: the fp32 `atomic_add` scatter of dQ (about a third of
the main kernel in a quick ablation), a KV-owner grid that is tiny because all
H heads are rows of one query tile (32 CTAs on 48 SMs before the query-loop
split was added), and a bwd config table keyed on head dim only and tuned at
H=64.

**Fix.** Measurement first: putting the fp32 `atomic_add` back into a kernel
that touches `dQ` exactly once costs only +0.19 ms at D128 H64 S4096 (4.11 ->
4.30) and +0.06 ms at D64 (2.17 -> 2.23), so the atomic instruction was never
the problem -- doing it once per KV block (`ceil(S/block_M)` = 32 read-modify-
writes of the whole 134 MB `dQ`) was. The CTA count was the other half: the
KV-owner grid runs 32 CTAs on 48 SMs, and splitting its query loop 8 ways takes
it from 15.38 to 7.79 ms. So `dQ` moved into a third, query-owning kernel that
recomputes S and dP (+40% FLOPs) and stores `dQ` bf16 once, `dKV` kept its fp32
split buffers, and the tile tables were re-swept per dim -- keyed on head dim
only, since the H=16 sweep winner is within 1.3% of the H=64 pick at every dim.
fwd+bwd at D128 H64 went 33.6 -> 15.7 ms against SDPA's 15.2 (1.03x, was 2.2x),
D64 H64 14.8 -> 7.2 vs 7.4 (0.97x), and the new D96 H64 lands at 10.7 vs 11.7;
the backward is now 12.8 ms of kernel time (dKV 7.79, dQ 4.31, preprocess 0.66)
instead of one 28.1 ms fused kernel.

## swa_attn: the banded forward returns NaN, the unbanded one does not

**Problem.** Copying `latent_attn` forward and restricting the KV loop to the
`[t0 - window + 1, t0 + T_q - 1]` band produced `o = NaN` and `lse = NaN`
everywhere at `window` 128 and 100, while the *same* kernel at `window >= S`
(where the band never bites) was bit-for-bit identical to `latent_attn`. So the
band itself was the trigger, not the mask arithmetic: `window >= S` exercises
every line except the restricted loop bounds.

**Measurement.** The band is per *row token*, but the loop bounds come from the
block's *first* token `t0`, so the loop starts at the tile holding
`t0 - window + 1`. A row at token `t0 + T_q - 1` has its lower edge `T_q - 1`
tokens higher, and when `t0 - window + 1` lands near the top of its tile that
row sees nothing at all in the loop's first iteration. Its tile max is then
`-inf`, its running max is still the `-inf` initialiser, and FA2's rescale step
computes `exp2(-inf * scale - (-inf) * scale)` = NaN, which then poisons the
accumulator for the rest of the loop. Plain causal never hits this because tile 0
always contains column 0, which every row can see.

**Fix.** Floor the running row max at a finite, unreachably small constant
(`-1e30`) instead of `-inf`, both at init and at the per-tile `T.fill` before
`T.reduce_max(..., clear=False)`. An all-masked tile then rescales by
`exp2(0) = 1` (a no-op) while each masked logit still gives
`exp2(-inf - floor * scale) = 0`, and any real logit dominates the floor so the
normal path is untouched. Masking with a large negative *logit* instead is the
trap to avoid: `exp2(-1e30 * scale + 1e30 * scale) = 1` would add one to the row
sum for every masked column. With the floor, D in {64, 96, 128, 256} x H in
{4, 16, 64} all land at 0.4-0.7x the bf16-torch error, and the `window >= S`
identity with `latent_attn` stays exact.

## swa_attn: the inherited backward tile table and split heuristic both invert under a band

**Problem.** `swa_attn` is `latent_attn` with one change, so the obvious move is
to keep its tuned `CONFIGS`. The forward tolerates that (its winners barely
move), but the `dKV` kernel does not, and neither does `pick_splits`: carried
over unchanged they left the headline backward well short of what the band
should buy. The suspicion was that both were tuned for a loop whose length grows
with the query index, while the banded loop has a fixed length of
`ceil((block_M + window) / T_q)` query tiles per KV block.

**Measurement.** Re-sweeping at B1 H64 S4096 W128: `bwd_kv` now wants the
*smallest* latent tile, `block_M=64` at 128 threads (D=128: 0.797 ms) rather than
`latent_attn`'s `block_M=128` at 256 threads (0.896 ms, 12% worse) -- a wide KV
block drags in query tiles most of its rows cannot see, and that waste is pure
overhead once the loop no longer grows. Splits moved the other way: the inherited
`TARGET_CTAS=256 / MAX_SPLITS=8` picks 4 splits at `block_M=64`, costing 27% at
D=128 (1.098 vs 0.798 ms) and 118% at D=256 (3.703 vs 1.703). The forward and
`bwd_dq` kept preferring wide query tiles, so the inversion is specific to the
KV-owning kernel.

**Fix.** Full re-sweep of all three kernels with the band in place, a separate
splits sweep on the winning `bwd_kv` tile per dim, and `TARGET_CTAS` 256 -> 768 /
`MAX_SPLITS` 8 -> 12 (measured optimum 12-16; past ~24 the launch and fp32-reduce
overheads win back). `pick_splits` also gained a cap at the number of query tiles
in the band, so small shapes stop launching empty CTAs. Result at D128 H64 S4096:
fwd 0.624 ms and fwd+bwd 3.073 ms against `latent_attn`'s full-causal 2.903 and
15.673 -- 4.66x and 5.10x. All 204 compile failures across the twelve sweeps were
smem-budget; excluding the `tickets/0001` layout-inference cells up front meant
none were layout inference.

## indexer: the model's own top-k is not reproducible by index set

- problem: the end-to-end check of the bf16 score kernel against `DeepseekV41Indexer`'s
  own selection (index-set equality, ticket 0006's acceptance) failed on 92 of 320 query
  rows -- even though the kernel's scores matched the fp32 reference to 1.5e-8.
- measurement: 935 of the 12800 group-causally visible scores of the tiny model are
  *exactly* 0.0 (all `index_n_heads=4` heads relu'd to zero on a freshly-initialised
  model), and `torch.topk(..., sorted=False)` breaks those ties arbitrarily. Gathering
  the fp32 reference score of each side's chosen set and sorting gave a max difference
  of exactly 0.0 -- the sets differ only by tied entries.
- fix: the end-to-end test compares the *score values* of the two chosen sets (ticket
  0006 allows either), not the index sets. The random-input tests keep exact set
  equality, where ties are improbable.

## indexer: `Di=256` overflows smem only at `Hi=4`

- problem: the `Di=256` tile (`block_M=128` query rows, `block_T=64` key rows, 1 stage)
  compiled and ran at `Hi=32` but died at `Hi=4` with
  `Failed to set the allowed dynamic shared memory size to 106496`.
- measurement: `block_M` is `T_s * Hi` query rows, so the same `block_M` means 32 query
  *tokens* at `Hi=4` against 4 at `Hi=32` -- and the store-staging buffer is
  `[block_T, T_s]` fp32, i.e. 8 KB at `Hi=4` vs 1 KB at `Hi=32`. 65536 (Q) + 32768 (K) +
  8192 (staging) = 106496, exactly the reported number, 5 KB over the 99 KB budget.
- fix: `Di=256` uses `block_M=64`, which halves both the Q tile and the staging buffer
  (69632 B at `Hi=4`). The head-count dependence of the smem budget is a property of the
  packed-heads layout, not of this kernel: any `block_M` config table for it has to be
  validated at the *smallest* supported head count.

## csa_attn: the second source's pipelined loop silently computes the wrong answer

**Problem.** `csa_attn` adds a second `T.Pipelined` KV loop after the window loop,
over the compressed entries, with trip count
`min(G/block_N, ceil(((t0 + T_q) // ratio) / block_N))`. The forward compiled and
ran, and was correct at `compress_ratio=1` and at most (dim, block_N, G)
combinations -- but at a few it was wrong by ~0.8 absolute on `o` (57x the bf16
torch error) with no error and no NaN. At `num_stages=3` the same loop did not
compile at all: `TypeError: Downcast from tirx.Sub to ir.IntImm failed` in the CUDA
codegen.

**Measurement.** The wrong tokens were a clean prefix: at D=64 H=16 S=512 ratio=4
G=128 block_N=64, tokens 3..255 were wrong and 256..511 were right. Tokens 0..2 see
zero compressed entries, so the pattern says "every query block that should have
walked exactly one main tile walked none". Dumping the generated CUDA confirmed it:
the main loop had been peeled into a prologue whose `cp_async` reads
`MainKV[... - 1024]` -- a **negative** global offset -- followed by
`for (int kg = 0; kg < 1; ++kg)`, and the group-causal mask inside it had the tile
offset folded in as if `kg == 2`. A standalone kernel that just *stores* the same
bound expression returns the right values for every block index, so the arithmetic
is fine and the software pipeline is what mis-transforms it. It only bites when the
bound nests a division by `ratio` (`floordiv((bx+1)*T_q, ratio)`); at `ratio=1` that
collapses to `latent_attn`'s long-tested `ceildiv((bx+1)*T_q, block_N)` form and
every G is correct.

**Fix.** `T.serial` instead of `T.Pipelined` for the main-source loop in `fwd.py`
and `bwd_dq.py`. Correct at every (dim, heads, ratio, G) tested, including
non-tile-aligned G and S=4096. The window loop keeps its pipeline, and so does
`bwd_main.py`'s query loop (variable *start*, `swa_attn`'s form) -- neither shows
the bug. Cost: the main source loses async prefetch, which is what the second-source
overhead in the bench table is paying for. Logged in
`tickets/0001-tilelang-issues.md`; the `num_stages=3` crash is the same bug failing
loudly instead of quietly.

## csa_attn: the group-causal mask clamped to the padded G, not the real one

**Problem.** The main source is zero-padded up to a multiple of the group tile so
the kernel can read whole tiles, and the mask `g < min(G, (t + 1) // ratio)` is what
keeps the pad rows out. Passing the *padded* count as `G` makes that `min` a no-op
for the pad rows: they are zero vectors, so their logit is exactly 0, and
`exp2(0 - m)` is a real, usually non-negligible, contribution to the denominator.

**Measurement.** Invisible while `G` happened to be a multiple of every tile
(`G = S // ratio` with S and ratio powers of two -- i.e. every "natural" test).
`test_grads_groups_not_tile_aligned` (G=100, S=512, ratio=2, padded to 128) caught
it: `dq` at 4.13x the torch bf16 error against a 2x bound, while `o` and the two KV
gradients stayed inside it -- the 28 pad columns move the softmax denominator a
little and `dS = P * (dP - delta)` amplifies that more than the output does.

**Fix.** Carry the true `G` and the padded extent as two separate compile-time
constants through all three kernels: tensor shapes, grid and loop bounds use the
padded one, every mask uses the true one. Kept a non-tile-aligned `G` in both the
forward and the gradient test matrix so this cannot regress silently.
