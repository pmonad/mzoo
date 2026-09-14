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

## csa2_attn: the per-token gather does not fit the packed-heads block, and a 16-row tile does not compile

**Problem.** Every package since `latent_attn` packs `T_q = block_M // H` *tokens* x `H`
heads into a block's rows (`block_M = 256` at `csa_attn`'s tuned configs, so 4 tokens at
`H = 64`). The top-k indices are per token, so one gathered smem tile can only serve one
token's rows -- a 4-token block would have to gather four times, or the block has to shrink
to one token. Shrinking looked free at `H = 64` (64 rows) and cheap at `H = 16` (16 rows).

**Measurement.** `block_M = 16` does not compile at all here: `Layout infer conflict between
acc_s and acc_s_cast in T.Parallel loop`, loop fragment `(16, 32) -> (16,), replicate: 4`
against accumulator fragment `(16, 32) -> (4,), replicate: 1`, at `threads=128` (the only
thread count a 16-row tile could use anyway). That is the same replicate-mismatch bug
`tickets/0001-tilelang-issues.md` already records for `block_M=32/64 x threads=256`, one step
smaller. `block_M = 64` compiles and runs everywhere.

**Fix.** `block_M = 64` fixed, `T_q = ceil(64 / H)` tokens per block, and the gathered section
repeated once per token of the block with the other tokens' rows masked to `-inf` (the
`neg_floor` rowmax already handles "this row sees nothing in this tile", so the repeat needs
no new machinery). At the model's `H = 64` that is `T_q = 1` and no repeat; `H = 16` gathers
4x and `H = 4` 16x, at head counts the model does not use. `threads` drops from 256 to 128
for the same layout-inference reason. Both are compile-time limits, not measurements -- the
`block_N` / `num_stages` half of the config table is still `csa_attn`'s, untuned.

## csa2_attn: pipelining the gathered loop is safe, unlike csa_attn's dense main loop

**Problem.** `csa_attn`'s second KV source had to run on `T.serial` because `T.Pipelined`
silently mis-peeled a loop whose trip count nested a division (section above). `csa2_attn`
inherits that loop's position in the kernel, so the cheap thing is to inherit `T.serial` too
-- at the cost of the gathered source's async prefetch, which is most of its cost.

**Measurement.** The gathered loop's trip count is `ceil(topk / block_N)`, a compile-time
constant: the documented trigger is absent. Turning `T.Pipelined(num_stages=2)` on and
re-running the whole matrix (28 tests: D 64/96/128/256, H 4/16/64, topk 64/100/512, G
0/100/256/1024/16384, plus the bit-for-bit equalities against `csa_attn` and `swa_attn`) gave
identical results to the serial version, and the bench moved from 1.68 ms to 1.52 ms for the
gathered source alone (316-339 -> 353-374 GB/s effective on the gathered rows).

**Fix.** Keep `T.Pipelined` for the gather, with the evidence recorded in the package README
and `tickets/0001`; the `T.serial` workaround stays scoped to the bound shape that actually
miscompiles. Fallback is one line if a future shape ever disagrees.

## csa2_attn bwd: duplicate indices double-count, and the golden reference cannot express that

**Problem.** The ticket asked for a duplicate-index test -- the same main entry listed twice
in one token's top-k list -- because a scatter-add's correctness is invisible without one.
Writing it exposed a semantic disagreement that has nothing to do with the backward:
`golden(level="sparse")` renders an index list by `sparse_bias.scatter_(-1, indices, 0.0)`,
and `scatter_` of a constant is **idempotent**, so a duplicated entry gets exactly one
softmax column. The kernel (forward, since ticket 0004, following
`examples/dsa_sparse_finetune/sparse_mla_fwd.py`) builds one gathered column per *slot*, so
it gives the entry two columns: `2 * exp(s)` in the denominator and twice the weight in the
numerator.

**Measurement.** At B2 S512 H16 D128 G256 with entry 0 forced into two slots of every
token's list, the kernel's `o` is 6.4e-1 from the deduplicated `golden` -- 44x the bf16 noise
floor of 1.5e-2 -- and within the normal 2x criterion (0.5-0.7x) of a reference built by
*cloning* entry 0 into a fresh entry `G` and pointing the second slot at it. So the kernel is
unambiguously computing the two-column rendering, and its gradient is the consistent gradient
of that: `dmain_kv[0]` equals `dmain_ref[0] + dmain_ref[G]` to the same criterion.

**Fix.** None in the kernel -- reported, not silently changed, per the ticket. The model's
index producer is `torch.topk`, whose output is a set, so duplicates cannot occur on the
model path, and "trust `indices` completely" is the documented contract of both `golden` and
the kernel. What landed is the test (`test_grads_duplicate_indices`, which pins both the
agreement with the cloned-row reference and the disagreement with plain `golden`) and a
Known-issues entry saying that any future index producer that can emit duplicates must
either dedup or teach the kernel to mask repeats within a token's list.

## csa2_attn bwd: the dmain_kv scatter is a third of the dQ kernel, but the atomic is a tenth

**Problem.** `dmain_kv` has no owner kernel in the sparse package: a block of main entries
cannot enumerate the tokens that picked it without an inverse index. The upstream template
(`sparse_mla_bwd.py`) scatters with `T.atomic_add`, but the `latent_attn` note above says
atomics cost 5% as an instruction and 2x as a repeated read-modify-write of a large buffer,
so copying the template blind was not an option -- an owner kernel fed a torch-built CSR was
the alternative on the table.

**Measurement.** `bwd_dq.py` got a three-way `scatter` knob: `"atomic"` (real), `"store"`
(a plain store to the *same* scattered addresses, so only the read-modify-write and the
contention go away) and `"none"` (the whole `dmain_kv` half, its two GEMMs included, dies).
At B1 S4096 D128 H64 topk=512, `dQ` = 3.43 ms: `atomic - none` = 1.09-1.13 ms (32%),
`atomic - store` = 0.30-0.39 ms (9-11%). So the atomic itself is ~7% of the 5.0 ms backward
and the rest is the `dS^T Q` / `P^T dO` GEMMs, which an owner kernel would still have to run.

**Fix.** Keep the scatter. An inverse-index owner kernel is bounded above by a ~0.35 ms win
before it pays for building the CSR, which does not justify a fourth kernel; the knob stays
in the file so the number can be re-measured after the tuning pass. The cost is
non-determinism: fp32 atomic ordering varies, so `dmain_kv` is not bitwise reproducible
(`dq`/`dkv` are), which is documented in the package README rather than fixed.

## indexer bwd: tilelang `out_idx` outputs are empty-allocated, atomics accumulated garbage

**Problem.** `bwd.py` first declared `DQ`/`DW`/`DK` as `out_idx` outputs like `fwd.py`
does for `Scores`. `dq` and `dw` are written block-exclusively with `T.copy`, but
`dk` is only ever *accumulated* with `T.atomic_add` -- nothing ever writes its pad
rows or initialises it, and tilelang's returned tensors come from an empty
allocation.

**Measurement.** `dq`/`dw` inside the 2x bound, `ddk` at 245-590x the torch bf16
error (values of the order of the gradient itself, i.e. garbage memory), on every
shape in the sweep.

**Fix.** Drop `out_idx`; allocate all three gradient buffers in the wrapper
(`dk` with `torch.zeros`) and pass them as in-out tensors.

## indexer bwd: `dk` fp32 atomics are order-nondeterministic

**Problem.** Bit-exact repeat checks (`bwd` twice on identical inputs, and
`scores().backward()` vs a direct `bwd` call) failed on `dk` alone, ~2 of 128x128
elements differing by 6.1e-5.

**Measurement.** Same kernel, same inputs, two launches: `dq`/`dw` bit-identical,
`dk` differs where many query blocks' fp32 atomic adds land in different orders.
On the 256x128 test the reorder error occasionally exceeded 1e-4 (flaky test), on
gradients of magnitude ~2.

**Fix.** Treat `dk` as deterministic only up to fp32 sum reordering: the tests
assert `dq`/`dw` bit-exact and `dk` within rtol/atol 1e-3; documented in the README
with the two-stage-reduce escape hatch if training reproducibility ever needs it.
