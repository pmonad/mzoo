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
- **Heads in the M dimension** (`fwd.py`). A block owns `block_M` rows that are
  `T_q = block_M // H` consecutive *tokens* x `H` heads. Because BSHD memory
  order is already (token, head, dim), that tile is one contiguous
  `[block_M, D]` slice of `q.view(B, S*H, D)` -- no gather, no transpose. The
  causal mask therefore comes from the row's **token**, `bx * T_q + i // H`, not
  from the block index.
  The general rule is `T_q = block_M // H`, with `block_M % H == 0` and
  `seq_len % T_q == 0` asserted. Every `block_M` in the config table is 128 or
  256, so `H in {4, 8, 16, 32, 64}` all fit (H=64 -> 2 or 4 tokens/block, H=4 ->
  32 or 64 tokens/block). `lse` is produced packed `[B, S*H]` and transposed to
  `[B, H, S]` in the python wrapper (a ~25 us copy at the headline shape).
- **One smem KV tile, two GEMMs.** `S = Q K^T` and `O += P K` both read
  `K_shared`; the V load is gone entirely. That frees `block_N x D` bf16 of the
  smem budget, which is what pays for `block_M=256` in the forward.
- **Sink** (optional, `sinks [H]` fp32): unchanged `dense_attn` semantics -- one
  extra softmax column that only enlarges the denominator, included in `lse`,
  `has_sink` a static flag. With heads in rows the per-row sink is
  `Sinks[i % H]` instead of a single block-wide scalar.
- **Backward** (`bwd.py` driver + `bwd_kernels.py`): the same three kernels as
  `dense_attn` -- `preprocess` (`delta = rowsum(dO * O)`), the KV-block-owning
  main kernel, `postprocess` (fp32 `dQ` -> bf16). Two MQA changes:
  - `dK` and `dV` accumulate into **one** fp32 register tile (`dKV = dK + dV`,
    because K == V), and the reduction over all H heads sharing the latent
    happens inside that tile for free, since heads are rows of the query tile.
  - the grid gains a **split** dimension over the query loop. With heads folded
    into rows, the natural grid is only `B * S/block_M` CTAs (32 at B1 S4096 on
    48 SMs) each doing H times the work; each split takes a contiguous chunk of
    query tiles and writes its own fp32 slice of `dKV_partial [splits, B, S, D]`,
    summed in torch and cast to bf16. **fp32 split buffers, never bf16 atomics.**
    `pick_splits` targets ~256 CTAs, capped at 8.
  `dQ` is still an fp32 `atomic_add` scatter. `dsinks` is still a plain fp32
  torch reduce, no kernel.
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

Re-tuned from scratch for the packed-heads layout: a full grid at B1 **H64**
S4096 D{64,128} causal (the model's shape), 44 forward points and 72 backward
points per dim. **Only D=64 and D=128 are supported** -- D=96/256 were not
added, see Known issues.

fwd `CONFIGS` (head dim -> block_M = `T_q * H` rows, block_N = KV tokens, num_stages, threads):

| dim | block_M | block_N | num_stages | threads |
|---|---|---|---|---|
| 64 | 256 | 128 | 2 | 256 |
| 128 | 256 | 64 | 2 | 256 |

bwd `CONFIGS` (head dim -> block_M = latent tokens, block_N = packed query rows, num_stages, threads):

| dim | block_M | block_N | num_stages | threads |
|---|---|---|---|---|
| 64 | 128 | 64 | 1 | 256 |
| 128 | 128 | 64 | 1 | 256 |

plus `splits`, chosen at call time by `pick_splits` (8 at the headline shape).

Forward sweep, top 5 of each dim (ms, B1 H64 S4096 causal):

| dim | block_M | block_N | stages | threads | ms |
|---|---|---|---|---|---|
| 64 | **256** | **128** | **2** | **256** | **1.537** |
| 64 | 256 | 128 | 3 | 256 | 1.544 |
| 64 | 256 | 64 | 3 | 256 | 1.548 |
| 64 | 256 | 64 | 2 | 256 | 1.562 |
| 64 | 128 | 128 | 2 | 256 | 1.577 |
| 128 | **256** | **64** | **2** | **256** | **2.829** |
| 128 | 256 | 32 | 3 | 256 | 2.906 |
| 128 | 128 | 64 | 2 | 256 | 2.916 |
| 128 | 128 | 128 | 2 | 256 | 2.921 |
| 128 | 256 | 32 | 2 | 256 | 2.934 |

`block_M=256` wins by only ~3% over `block_M=128`, and only at 256 threads --
`block_M=256` with 128 threads is 10-30x slower (1.53 -> 21.8 ms at D=64,
block_N=128), so `threads=256` is not optional there. `block_M=64` is uniformly
worst (1.89-3.90 ms) and caps at 128 threads anyway.

Backward sweep, top 5 of each dim (ms, same shape; `sp` = splits):

| dim | block_M | block_N | stages | threads | sp | ms |
|---|---|---|---|---|---|---|
| 64 | **128** | **64** | **1** | **256** | **8** | **13.34** |
| 64 | 128 | 64 | 1 | 128 | 8 | 13.69 |
| 64 | 128 | 64 | 2 | 256 | 8 | 13.78 |
| 64 | 128 | 64 | 1 | 256 | 4 | 13.81 |
| 64 | 128 | 128 | 1 | 256 | 8 | 13.86 |
| 128 | **128** | **64** | **1** | **256** | **4** | **28.05** |
| 128 | 128 | 64 | 1 | 256 | 2 | 28.96 |
| 128 | 128 | 64 | 1 | 256 | 8 | 29.38 |
| 128 | 128 | 64 | 1 | 128 | 4 | 34.80 |
| 128 | 128 | 64 | 1 | 256 | 1 | 35.46 |

The splits axis is the big one: at D=64 / `block_M=128, block_N=64, 1 stage,
256 threads` it goes 18.36 (sp=1) -> 14.46 (2) -> 13.81 (4) -> 13.34 (8) ms.
sp=4 vs sp=8 is within noise at D=128 (28.1 vs 29.4-30.1 across reruns), so
`pick_splits` just targets ~5 waves and lands on 8. `block_M=64` is 2-4x worse
(49-59 ms at D=128); `block_N=128` at D=128 does not fit smem at all.

36 of the 144 backward points failed, all of them smem-budget failures
(`Failed to set the allowed dynamic shared memory size to ...`), not layout
inference.

## Bench

GB10, S=4096 causal, bf16, `--mode fwd` = forward only, `--mode bwd` = fwd+bwd
through autograd. `sdpa` is torch's default backend fed `kv.expand(B, S, H, D)`,
sink-free. **B=1 H=64 D=128 is the headline (the model's shape).**

| dim | heads | mode | latent_attn | sdpa |
|---|---|---|---|---|
| 128 | **64** | fwd | **2.904 ms (94.6 TFLOPS)** | 3.346 ms (82.2) |
| 128 | 64 | fwd+bwd | 33.562 ms (20.5) | 15.100 ms (45.5) |
| 64 | 64 | fwd | 1.579 ms (87.1) | 1.592 ms (86.3) |
| 64 | 64 | fwd+bwd | 14.776 ms (23.3) | 7.404 ms (46.4) |
| 128 | 16 | fwd | 0.830 ms (82.8) | 0.893 ms (77.0) |
| 128 | 16 | fwd+bwd | 6.379 ms (26.9) | 3.688 ms (46.6) |
| 64 | 16 | fwd | 0.476 ms (72.2) | 0.462 ms (74.4) |
| 64 | 16 | fwd+bwd | 1.992 ms (43.1) | 1.887 ms (45.5) |

Against the sibling package on the *same* shape (`dense_attn` fed the latent
replicated to H heads, B1 H64 S4096 causal, no sink):

| dim | latent fwd | dense fwd | latent bwd only | dense bwd only |
|---|---|---|---|---|
| 64 | 1.58 ms | 1.64 ms | ~13.4 ms | 16.32 ms |
| 128 | 2.90 ms | 3.03 ms | ~30.0 ms | 33.92 ms |

So the layout change is a 4-11% forward win and a ~12% backward win over dense
at H=64 -- modest, because at S=4096 the replicated KV mostly hits L2 anyway;
the real payoff is that a 1-head latent is what steps 3-5 need, and that the
KV tile is now half the smem so the forward can run `block_M=256`.

The forward beats SDPA at D=128 (both H=16 and H=64). The backward does not:
at H=64 it is 2.2x SDPA's fwd+bwd. The cause is the same atomic-`dQ` design
`dense_attn` flagged, and it gets worse as H grows -- replacing the
`atomic_add` scatter with a plain store (wrong answer, timing probe only) takes
the D=128 H=64 main kernel from 28.1 ms to 19.1 ms, i.e. ~32% of the backward
is fp32 atomics on `dQ`. The remaining gap is the main kernel itself.

Backward kernel breakdown at B1 H64 S4096 D128 (splits=8): main 27.8 ms,
`postprocess` 1.09, `dQ` zero-fill 0.68, `preprocess` 0.64, `dKV` split reduce
0.14, `lse` transpose 0.03 -- so the aux work is ~8% and the split buffers cost
almost nothing.

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
| 128 | 16 | 0.60 | 0.88 | 0.78 | 1.18 | 9.5e-7 |
| 128 | 64 | 0.58 | 0.68 | 0.73 | 1.43 | 1.4e-6 |

`dkv` sums over H heads and so has a larger absolute error than dense's `dk` or
`dv` (e.g. 9.7e-2 at H=64 D=128), but so does the torch reference doing the same
sum, hence the ratio stays below 1.

`fwd_test.py` also cross-checks `latent_attn.fwd(q, kv)` against
`dense_attn.fwd(q, k, k)` with `k = kv.expand(B, S, H, D)` under the same <= 2x
criterion (with dense as the baseline) -- the cheapest guard that the packed
layout did not silently permute anything.

## Decisions

- **`T_q = block_M // H`, one general rule, no per-H table.** Asserted
  `block_M % H == 0`; every supported H divides both 128 and 256. The
  alternative (fixing `T_q = 1` like `sparse_mla_fwd.py`) would cap `block_M` at
  H and break H < 64.
- **Packed `[B, S*H, D]` tensor views, not 4-D slices.** `q.view(B, S*H, D)` is
  free (BSHD is contiguous) and keeps the kernel's Q/O/dQ/lse/delta slices plain
  2-D `T.copy`s. The public API stays BSHD / `[B, H, S]`.
- **`dKV = dK + dV` in one accumulator**, not two tiles summed afterwards: K == V
  so the two GEMMs (`P^T dO` and `dS^T Q`) can target the same fp32 fragment.
- **fp32 split buffers for `dKV`**, not bf16 atomics -- 16 MB at the headline
  shape and 0.14 ms to reduce, versus non-determinism and lost precision.
- **No `ref.py` in this package.** The series-wide `../golden_ref.py`
  (`level="latent"`) replaced the per-package torch reference; only the shared
  criterion `dense_attn.ref.assert_within_2x_torch` is imported.
- `bwd.py` split into `bwd.py` (driver) + `bwd_kernels.py`, as `dense_attn`'s
  README asked for.
- Fast-math (`TL_ENABLE_FAST_MATH`) kept, as in `dense_attn`.
- Fixed sequence length assumed; no varlen support.

## Known issues

- **D=96 and D=256 are not supported here** (`dense_attn` had them as
  shape-support entries). They were left out on purpose, not because they fail:
  the task scoped this package to D=64/128, and the `block_M=256` forward that
  the sweep picked needs 256 threads, which collides with the `block_M=64 +
  threads=256` layout-inference bug that D=256 would be forced onto. Adding them
  back means a separate small sweep with `block_M <= 128`.
- Both `tickets/0001-tilelang-issues.md` layout-inference limits reproduce
  **unchanged** in the packed-heads layout, verified directly: fwd `block_M=64 +
  threads=256` -> `Layout infer conflict between acc_s and acc_s_cast`; bwd
  `block_M=32` (any threads) and `block_M=64 + threads=256` -> `Layout infer
  conflict between qkT and qkT_cast`. Folding heads into rows changes nothing
  about them, so the config tables avoid the same cells.
- Backward is ~2.2x slower than torch SDPA's fwd+bwd at H=64 (and ~1.7x at
  H=16), the atomic-`dQ` design; ~32% of the main kernel is the `dQ` atomics
  (measured, see Bench). It is still faster than `dense_attn` on the same shape.
- The bwd config's `block_N=64` must be a multiple of H, so H > 64 is
  unsupported and H must divide 64 (fine for `{4, 8, 16, 32, 64}`). At H=64 that
  means a 1-token x 64-head query tile, which is why the bwd cannot amortise the
  causal boundary across tokens the way `dense_attn` does.
- `seq_len` must be a multiple of `block_M // H` (fwd) and of `block_M=128`
  (bwd); both asserted.
- H=16 fwd+bwd is slower than `dense_attn`'s (6.38 vs 4.20 ms at D=128) because
  `CONFIGS` is keyed on head dim only and was tuned at H=64. A per-(dim, H)
  backward table would fix it; not done, H=64 is the shape that matters.

## Next

- `swa_attn`: restrict the KV loop to the `[t0-127, t0+T_q-1]` band, per-row
  band mask. Copy this package forward; the packed-row layout and the `T_q` rule
  carry over unchanged.
- If the backward becomes a training bottleneck: split-K on `dQ` (an fp32
  `[splits, B, S*H, D]` buffer like `dKV` already uses) would delete the atomic
  traffic entirely at the cost of ~16 MB x splits; worth ~30% on the main kernel.
- A per-(dim, H) backward config table, if H != 64 ever matters.
