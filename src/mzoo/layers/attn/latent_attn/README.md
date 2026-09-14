# latent_attn

Shared-latent MQA FlashAttention-2: step 2 of the CSA2 series (see
`../csa2_attn_design.md`). Copied from `dense_attn/` with exactly one change --
K and V are the *same* latent with a single KV head, shared by all H query
heads. Still no windows, no compression, no sparsity.

Source: `dense_attn/` (itself `tile-ai/tilelang@v0.1.14`
`examples/flash_attention/example_mha_{fwd,bwd}_bshd.py` + `attention_sink`);
the heads-in-rows packing follows `examples/dsa_sparse_finetune/sparse_mla_fwd.py`.

## What it is

- **Call shapes.** `fwd(q, kv, *, causal=True, sinks=None) -> (o, lse)` with
  `q` bf16 `[B, S, H, D]`, `kv` bf16 `[B, S, 1, D]` (K == V), `o` bf16
  `[B, S, H, D]`, `lse` fp32 `[B, H, S]`.
  `bwd(q, kv, o, lse, do, *, causal, sinks) -> (dq, dkv, dsinks)` with `dkv`
  bf16 `[B, S, 1, D]`. `attn(q, kv, ...) -> o` is the autograd wrapper.
  `D in {64, 96, 128, 256}`.
- **Heads in the M dimension** (`fwd.py`). A block owns `block_M` rows that are
  `T_q = block_M // H` consecutive *tokens* x `H` heads. Because BSHD memory
  order is already (token, head, dim), that tile is one contiguous
  `[block_M, D]` slice of `q.view(B, S*H, D)` -- no gather, no transpose. The
  causal mask therefore comes from the row's **token**, `bx * T_q + i // H`, not
  from the block index.
  The general rule is `T_q = block_M // H`, with `block_M % H == 0` and
  `seq_len % T_q == 0` asserted. Every `block_M` in the config tables is 64, 128
  or 256, so `H in {4, 8, 16, 32, 64}` all fit. `lse` is produced packed
  `[B, S*H]` and transposed to `[B, H, S]` in the python wrapper (a ~25 us copy
  at the headline shape).
- **One smem KV tile, two GEMMs.** `S = Q K^T` and `O += P K` both read
  `K_shared`; the V load is gone entirely. That frees `block_N x D` bf16 of the
  smem budget, which is what pays for `block_M=256` in the forward.
- **Sink** (optional, `sinks [H]` fp32): unchanged `dense_attn` semantics -- one
  extra softmax column that only enlarges the denominator, included in `lse`,
  `has_sink` a static flag. With heads in rows the per-row sink is
  `Sinks[i % H]` instead of a single block-wide scalar.
- **Backward**: `bwd.py` (driver) + `bwd_kernels.py` (`preprocess`, `bwd_kv`) +
  `bwd_dq.py` (`bwd_dq`). Three kernels and **no atomics anywhere**:
  1. `preprocess` -> `delta = rowsum(dO * O)` over the packed rows.
  2. `bwd_kv` -> a CTA owns a latent block and a contiguous chunk of the query
     tiles. `dK` and `dV` accumulate into **one** fp32 register tile
     (`dKV = dK + dV`, because K == V) and the reduction over all H heads
     sharing the latent happens inside that tile for free, since heads are rows
     of the query tile. The grid gains a **split** dimension over the query
     loop: with heads folded into rows the natural grid is only
     `B * S/block_M` CTAs (32 at B1 S4096, on 48 SMs), so each split writes its
     own fp32 slice of `dKV_partial [splits, B, S, D]`, summed in torch and cast
     to bf16. **fp32 split buffers, never bf16 atomics.** `pick_splits` targets
     ~256 CTAs, capped at 8.
  3. `bwd_dq` -> a CTA owns a packed *query* tile (same layout and causal rule as
     `fwd.py`), recomputes `S` and `dS` against the L2-resident latent, and
     stores `dQ` bf16 **exactly once**. This replaces `dense_attn`'s fp32
     `atomic_add` scatter, which ran once per KV block; see Decisions.
  `dsinks` needs no kernel: `lse` already carries the sink column, so the
  recomputed P is sink-normalised and only
  `dsinks[h] = -sum_{b,s} exp(sink[h] - lse[b,h,s]) * delta[b,h,s]` is left, a
  plain fp32 torch reduce.
- **Autograd wrapper** (`attn.py`): saves exactly `(q, kv, o, lse[, sinks])` --
  one tensor fewer than dense -- pinned by `test_backward_saves_only_q_kv_o_lse`.

## How to run

```
just src/mzoo/layers/attn/ test -q
just src/mzoo/layers/attn/ bench-latent --dim 128 --heads 64
just src/mzoo/layers/attn/ bench-latent --dim 128 --heads 64 --mode bwd
```

(The trailing slash on the path is required by `just` for this recipe form.)
`bench-latent` always runs with a random per-head sink; the `sdpa` row is the
sink-free baseline, fed `kv.expand(B, S, H, D)`.

## Tile configs

Re-tuned from scratch for the packed-heads layout at B1 **H64** S4096 causal
(the model's shape), and re-swept again after the backward was restructured
around the query-owning `dQ` kernel. D=64, 96 and 128 are tuned; **D=256 is
shape support only** -- the 99 KB smem cap leaves it three usable forward tiles
and exactly one backward tile per kernel.

fwd `CONFIGS` (head dim -> block_M = `T_q * H` rows, block_N = KV tokens, num_stages, threads):

| dim | block_M | block_N | num_stages | threads |
|---|---|---|---|---|
| 64 | 256 | 128 | 2 | 256 |
| 96 | 256 | 64 | 3 | 256 |
| 128 | 256 | 64 | 2 | 256 |
| 256 | 128 | 32 | 2 | 256 |

bwd `CONFIGS`, two tiles per dim -- `kv` (`block_M` = latent tokens, `block_N` =
packed query rows) and `dq` (the other way round, matching `fwd.py`):

| dim | kv: block_M / block_N / stages / threads | dq: block_M / block_N / stages / threads |
|---|---|---|
| 64 | 256 / 64 / 2 / 256 | 128 / 128 / 2 / 256 |
| 96 | 256 / 64 / 2 / 256 | 128 / 128 / 2 / 256 |
| 128 | 128 / 64 / 2 / 256 | 128 / 64 / 2 / 256 |
| 256 | 64 / 64 / 1 / 128 | 64 / 32 / 2 / 128 |

plus `splits` for `bwd_kv`, chosen at call time by `pick_splits` (8 at the
headline shape).

### `seq_len` constraints

Asserted in `check_shapes` (fwd) and `bwd` (both kernels). `T_q = block_M // H`
is the token count of a packed query tile; the backward additionally needs
`seq_len % kv.block_M == 0`, so the binding constraint per dim is:

| dim | fwd: `seq_len %` | bwd: `seq_len %` |
|---|---|---|
| 64 | `256 // H` | 256 |
| 96 | `256 // H` | 256 |
| 128 | `256 // H` | 128 |
| 256 | `128 // H` | 64 |

`H` must divide the packed-row tile of every kernel it touches, i.e. `H | 256`
at D<=128 and `H | 64` at D=256 -- `H in {4, 8, 16, 32, 64}` everywhere.

### Sweeps

Forward, top 5 per dim (ms, B1 H64 S4096 causal):

| dim | block_M | block_N | stages | threads | ms |
|---|---|---|---|---|---|
| 64 | **256** | **128** | **2** | **256** | **1.537** |
| 64 | 256 | 128 | 3 | 256 | 1.544 |
| 64 | 256 | 64 | 3 | 256 | 1.548 |
| 64 | 256 | 64 | 2 | 256 | 1.562 |
| 64 | 128 | 128 | 2 | 256 | 1.577 |
| 96 | **256** | **64** | **3** | **256** | **2.139** |
| 96 | 256 | 64 | 2 | 256 | 2.164 |
| 96 | 128 | 128 | 3 | 256 | 2.212 |
| 96 | 128 | 128 | 2 | 256 | 2.253 |
| 96 | 256 | 32 | 3 | 256 | 2.262 |
| 128 | **256** | **64** | **2** | **256** | **2.829** |
| 128 | 256 | 32 | 3 | 256 | 2.906 |
| 128 | 128 | 64 | 2 | 256 | 2.916 |
| 128 | 128 | 128 | 2 | 256 | 2.921 |
| 128 | 256 | 32 | 2 | 256 | 2.934 |
| 256 | 64 | 64 | 2 | 128 | 6.119 |
| 256 | **128** | **32** | **2** | **256** | **6.173** |
| 256 | 128 | 32 | 1 | 256 | 6.334 |
| 256 | 64 | 32 | 2 | 128 | 6.470 |
| 256 | 128 | 64 | 1 | 256 | 6.528 |

At D=96 the sweep also covered `block_M=192`; every `block_M=192` point at 256
threads fails layout inference (same bug as `block_M=64`, see Known issues), and
the ones that do compile at 128 threads are 2.8-12.8 ms. At D=256 those five
rows are the *entire* set that compiles and fits: every `block_M=256` tile and
every `block_M>=128` tile with `block_N>=64, stages=2` is over the 99 KB cap, and
`block_M=64 + threads=256` fails layout inference. `block_M=128, block_N=32`
wins over the `block_M=64` alternative by supporting 2 tokens/block at H=64.

Backward `bwd_kv`, top 5 per dim (ms, same shape, `sp` = splits):

| dim | block_M | block_N | stages | threads | sp | ms |
|---|---|---|---|---|---|---|
| 64 | **256** | **64** | **2** | **256** | **8** | **3.078** |
| 64 | 256 | 64 | 3 | 256 | 8 | 3.084 |
| 64 | 128 | 64 | 3 | 256 | 8 | 3.250 |
| 64 | 128 | 128 | 2 | 256 | 8 | 3.293 |
| 64 | 128 | 64 | 2 | 256 | 8 | 3.351 |
| 96 | **256** | **64** | **2** | **256** | **8** | **4.699** |
| 96 | 256 | 64 | 1 | 256 | 8 | 5.359 |
| 96 | 128 | 64 | 2 | 256 | 8 | 5.588 |
| 96 | 128 | 64 | 2 | 128 | 8 | 5.762 |
| 96 | 128 | 128 | 1 | 256 | 8 | 5.793 |
| 128 | **128** | **64** | **2** | **256** | **8** | **7.662** |
| 128 | 128 | 128 | 1 | 256 | 8 | 7.812 |
| 128 | 128 | 64 | 1 | 256 | 8 | 7.888 |
| 128 | 256 | 64 | 1 | 256 | 8 | 7.926 |
| 128 | 128 | 64 | 2 | 256 | 4 | 8.349 |
| 256 | **64** | **64** | **1** | **128** | **8** | **30.39** |
| 256 | 64 | 64 | 1 | 128 | 4 | 30.77 |

D=256 has only those two rows because everything else in the grid either
overflows smem (`block_M=128` needs 132 KB at `block_N=64, stages=1`) or hits
the 256-thread layout bug.

Backward `bwd_dq`, top 5 per dim:

| dim | block_M | block_N | stages | threads | ms |
|---|---|---|---|---|---|
| 64 | 256 | 32 | 3 | 256 | 2.182 |
| 64 | **128** | **128** | **2** | **256** | **2.193** |
| 64 | 128 | 64 | 3 | 256 | 2.193 |
| 64 | 128 | 64 | 2 | 256 | 2.208 |
| 64 | 128 | 128 | 3 | 256 | 2.223 |
| 96 | **128** | **128** | **2** | **256** | **3.293** |
| 96 | 128 | 64 | 2 | 256 | 3.328 |
| 96 | 128 | 32 | 2 | 256 | 3.468 |
| 96 | 64 | 128 | 2 | 128 | 3.476 |
| 96 | 128 | 128 | 1 | 256 | 3.568 |
| 128 | **128** | **64** | **2** | **256** | **4.272** |
| 128 | 128 | 32 | 2 | 256 | 4.370 |
| 128 | 128 | 128 | 1 | 256 | 4.490 |
| 128 | 64 | 64 | 2 | 128 | 4.606 |
| 128 | 128 | 64 | 1 | 256 | 4.608 |
| 256 | **64** | **32** | **2** | **128** | **8.821** |
| 256 | 64 | 32 | 1 | 128 | 9.741 |
| 256 | 64 | 64 | 1 | 128 | 9.988 |

At D=64 the `block_M=256` `dq` winner is 0.5% ahead of `block_M=128`, which is
inside run-to-run noise, so `block_M=128` is kept for the looser `seq_len`
constraint.

**`CONFIGS` is keyed on head dim only, verified.** The same grids were re-run at
H=16: at D=128 the H=16 winner *is* the H=64 pick in both kernels (`bwd_kv`
1.435 ms, `bwd_dq` 1.117 ms), and at D=64 the H=16 winner (`bwd_kv` 128/128/2/256
sp8, 0.756 ms) beats the H=64 pick by 1.3% (0.766 ms) while `bwd_dq` agrees
exactly. A per-(dim, H) table would therefore buy <=1.3%; not added.

**Splits still matter, CTA occupancy is why** (D=128 H=64 S=4096, `bwd_kv`
`128/64/2/256`): sp=1 -> 15.38 ms at 32 CTAs, sp=2 -> 9.60 at 64, sp=4 -> 8.30 at
128, sp=8 -> 7.79 at 256 CTAs on 48 SMs. `bwd_dq` needs no split axis at all --
its grid is `S / T_q = 2048` CTAs.

## Bench

GB10, B=1 S=4096 causal, bf16, `--mode fwd` = forward only, `--mode bwd` =
fwd+bwd through autograd. `sdpa` is torch's default backend fed
`kv.expand(B, S, H, D)`, sink-free. **D=128 H=64 is the headline (the model's
shape).** The "before" column is the same bench on the previous, fused
atomic-`dQ` backward (`dQ` scattered from the KV-owning kernel).

| dim | heads | mode | latent_attn | sdpa | ratio | before (atomic dQ) |
|---|---|---|---|---|---|---|
| 64 | 16 | fwd | 0.477 ms (72.0 TFLOPS) | 0.467 ms (73.5) | 1.02x | -- |
| 64 | 16 | fwd+bwd | 1.911 ms (45.0) | 1.905 ms (45.1) | 1.00x | 1.992 ms |
| 64 | 64 | fwd | 1.559 ms (88.2) | 1.594 ms (86.2) | 0.98x | -- |
| 64 | 64 | fwd+bwd | 7.194 ms (47.8) | 7.416 ms (46.3) | **0.97x** | 14.776 ms |
| 96 | 16 | fwd | 0.642 ms (80.3) | 0.667 ms (77.3) | 0.96x | -- |
| 96 | 16 | fwd+bwd | 2.796 ms (46.1) | 2.887 ms (44.6) | 0.97x | -- |
| 96 | 64 | fwd | 2.175 ms (94.8) | 2.328 ms (88.6) | 0.93x | -- |
| 96 | 64 | fwd+bwd | 10.747 ms (48.0) | 11.657 ms (44.2) | **0.92x** | -- |
| 128 | 16 | fwd | 0.833 ms (82.5) | 0.901 ms (76.3) | 0.92x | -- |
| 128 | 16 | fwd+bwd | 3.580 ms (48.0) | 3.725 ms (46.1) | 0.96x | 6.379 ms |
| 128 | **64** | fwd | **2.906 ms (94.6)** | 3.354 ms (82.0) | 0.87x | -- |
| 128 | **64** | fwd+bwd | **15.686 ms (43.8)** | 15.181 ms (45.3) | **1.03x** | 33.562 ms |
| 256 | 16 | fwd | 1.563 ms (87.9) | 1.737 ms (79.1) | 0.90x | -- |
| 256 | 64 | fwd | 6.675 ms (82.4) | 7.389 ms (74.4) | 0.90x | -- |

(D=96 and D=256 did not exist before, hence the empty "before" cells.) The
forward beats SDPA at every dim and both head counts; fwd+bwd is now 0.92-1.03x
of SDPA everywhere, against 1.05-2.22x before.

Backward kernel breakdown at D=128 H=64 S=4096 (splits=8): `bwd_kv` 7.79 ms,
`bwd_dq` 4.31, `preprocess` 0.66 -- 12.8 ms of kernel time, plus 0.14 ms for the
`dKV` split reduce and 0.03 ms for the `lse` transpose. The fused kernel it
replaced was recorded at 28.1 ms on its own.

Against the sibling package on the *same* shape (`dense_attn` fed the latent
replicated to H heads, B1 H64 S4096 causal, `--mode bwd`, i.e. fwd+bwd):

| dim | latent_attn | dense_attn (fused atomic dQ) |
|---|---|---|
| 64 | 7.19 ms | 18.75 ms |
| 128 | 15.69 ms | 37.79 ms |

## Accuracy

Ratio of our max-abs error to the max-abs error of the same computation run in
bf16 by torch (`golden(..., dtype=torch.bfloat16)`), both against the fp32
`golden(..., level="latent")`; the acceptance criterion is <= 2x. B2 S512
causal, with a per-head sink. `lse` is fp32 end to end so it gets an absolute
number instead.

| dim | heads | o | dq | dkv | dsinks | lse (abs) |
|---|---|---|---|---|---|---|
| 64 | 16 | 0.55 | 0.94 | 0.77 | 0.51 | 1.4e-6 |
| 64 | 64 | 0.49 | 0.96 | 0.78 | 0.84 | 1.4e-6 |
| 96 | 16 | 0.58 | 0.54 | 0.90 | 0.51 | 1.4e-6 |
| 96 | 64 | 0.65 | 0.68 | 0.57 | 0.78 | 1.4e-6 |
| 128 | 16 | 0.60 | 0.88 | 0.78 | 1.18 | 9.5e-7 |
| 128 | 64 | 0.58 | 0.68 | 0.73 | 1.43 | 1.4e-6 |
| 256 | 4 | 0.60 | 0.52 | 0.65 | 1.28 | 1.4e-6 |
| 256 | 16 | 0.44 | 0.75 | 0.57 | 1.12 | 1.4e-6 |

`dkv` sums over H heads and so has a larger absolute error than dense's `dk` or
`dv` (e.g. 9.7e-2 at H=64 D=128), but so does the torch reference doing the same
sum, hence the ratio stays below 1.

Two independent cross-checks beyond `golden`:

- `test_fwd_matches_dense_attn_with_expanded_kv`: `latent_attn.fwd(q, kv)` vs
  `dense_attn.fwd(q, k, k)` with `k = kv.expand(B, S, H, D)`, same <= 2x
  criterion with dense as the baseline -- the cheapest guard that the packed
  layout did not silently permute anything.
- `test_fwd_matches_gpt_oss`: against `golden_ref.gpt_oss_ref`, i.e.
  transformers' real gpt-oss `eager_attention_forward` (with and without sinks),
  which shares no code with `golden`'s own `_attend`.

## Decisions

- **`T_q = block_M // H`, one general rule, no per-H table.** Asserted
  `block_M % H == 0`; every supported H divides the tile at every dim. The
  alternative (fixing `T_q = 1` like `sparse_mla_fwd.py`) would cap `block_M` at
  H and break H < 64.
- **Packed `[B, S*H, D]` tensor views, not 4-D slices.** `q.view(B, S*H, D)` is
  free (BSHD is contiguous) and keeps the kernel's Q/O/dQ/lse/delta slices plain
  2-D `T.copy`s. The public API stays BSHD / `[B, H, S]`.
- **`dKV = dK + dV` in one accumulator**, not two tiles summed afterwards: K == V
  so the two GEMMs (`P^T dO` and `dS^T Q`) can target the same fp32 fragment.
- **fp32 split buffers for `dKV`**, not bf16 atomics -- 16 MB at the headline
  shape and 0.14 ms to reduce, versus non-determinism and lost precision.
- **`dQ` gets its own query-owning kernel instead of an atomic scatter.** The
  atomic *instruction* is not the problem: adding the fp32 `atomic_add` back to
  `bwd_dq`'s single output pass costs +0.19 ms at D=128 (4.11 -> 4.30) and
  +0.06 ms at D=64 (2.17 -> 2.23), ~5%. The problem was doing it once per KV
  block -- `ceil(S/block_M) = 32` read-modify-writes of the whole 134 MB `dQ`.
  Recomputing `S` and `dP` in a third kernel is +40% FLOPs and deletes all of
  that traffic plus the fp32 `dQ` buffer, its zero-fill and the `postprocess`
  cast. End to end that is 2.1x at D=128 H=64 (33.6 -> 15.7 ms fwd+bwd) and
  2.1x at D=64 (14.8 -> 7.2). So `bwd_dq.py` stays and the atomic path is not
  restored.
- **No per-(dim, H) backward config table**: measured, it is worth <=1.3% (see
  Tile configs).
- **No `ref.py` in this package.** The series-wide `../golden_ref.py`
  (`level="latent"`) replaced the per-package torch reference; only the shared
  criterion `dense_attn.ref.assert_within_2x_torch` is imported.
- `bwd.py` is split into `bwd.py` (driver), `bwd_kernels.py` (`preprocess`,
  `bwd_kv`) and `bwd_dq.py`; every file is under 200 lines.
- Fast-math (`TL_ENABLE_FAST_MATH`) kept, as in `dense_attn`.
- Fixed sequence length assumed; no varlen support.

## Known issues

- **D=256 is shape support, not a fast path.** The 99 KB smem cap plus the
  256-thread layout bug leave exactly one usable tile per backward kernel
  (`bwd_kv` 64/64/1/128 at 30.4 ms, `bwd_dq` 64/32/2/128 at 8.8 ms, B1 H64
  S4096), so the D=256 backward is ~2.5x the D=128 one for 2x the work. H=64
  *does* fit (both backward tiles are 64 packed rows = 1 token x 64 heads), so
  there is no head-count restriction beyond `H | 64`. The forward is fine
  (0.90x SDPA).
- The `tickets/0001-tilelang-issues.md` layout-inference limits reproduce
  unchanged in the packed-heads layout and in the new `bwd_dq` kernel: with
  `threads=256`, any tile whose `block_M` is not a multiple of 128 (64 *and*
  192, both measured) fails with `Layout infer conflict between <acc> and
  <acc>_cast in T.Parallel loop`; `block_M=32` never compiles in the
  KV-transposed kernel. The config tables avoid those cells and drop to 128
  threads whenever a 64-row tile is forced (D=256).
- `bwd_kv`'s `block_N` and `bwd_dq`'s `block_M` are the packed query-row tiles,
  so `H` must divide both: `H | 256` at D<=128 (via the `seq_len` table above,
  the real constraint is `H | 64` for `bwd_kv`), `H | 64` at D=256. `H > 64` is
  unsupported at every dim.
- `seq_len` must be a multiple of 256 (D=64/96), 128 (D=128) or 64 (D=256) in
  the backward, and of `block_M // H` in the forward -- see the table above.
  Asserted, not padded.
- D=96 and D=256 forward tiles are chosen from much smaller compiling sets than
  D=64/128 (five points at D=256), so their headroom is unknown, not zero.

## Next

- `swa_attn`: restrict the KV loop to the `[t0-127, t0+T_q-1]` band, per-row
  band mask. Copy this package forward; the packed-row layout, the `T_q` rule
  and the three-kernel backward carry over unchanged.
- The remaining backward gap at D=128 H=64 (1.03x SDPA) is `bwd_kv`, not `dQ`:
  7.79 of 12.8 ms of kernel time. Its tile is capped at `block_M=128` by smem,
  and splits already buy 2x; a persistent-CTA variant or `block_N=128` with a
  smaller `block_M` is the next thing to try if it matters.
- D=256 backward would need a `block_M=32` tile (blocked by the layout bug) or a
  split-D loop to get off 30 ms; out of scope while `D<=256` is shape support.
