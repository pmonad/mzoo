# swa_attn

Sliding-window shared-latent MQA FlashAttention-2: step 3 of the CSA2 series
(see `../csa2_attn_design.md`, ticket `tickets/0002-swa-attn.md`). Copied from
`latent_attn/` with exactly one change -- each query token sees only the
`window` most recent raw KV tokens, **itself included** (`t - window + 1 <= k <= t`).
Everything else is unchanged: K == V is one latent with a single KV head serving
H query heads, heads packed into the tile rows, one smem KV tile for both GEMMs,
sink semantics, `lse`, and the three-kernel backward with fp32 split `dKV` and a
query-owning `dQ` (no atomics anywhere).

This is the `compress_ratio == 0` layer type of DSV4.1, so it is the **first
package testable against the real model**: `model_test.py` replays the exact
tensors the vendored model hands to `eager_attention_forward`.

Source: `latent_attn/` (via `dense_attn/`, itself `tile-ai/tilelang@v0.1.14`
`examples/flash_attention/example_mha_{fwd,bwd}_bshd.py` + `attention_sink`).

## What it is

- **Call shapes.** `fwd(q, kv, *, window, causal=True, sinks=None) -> (o, lse)`
  with `q` bf16 `[B, S, H, D]`, `kv` bf16 `[B, S, 1, D]` (K == V), `o` bf16
  `[B, S, H, D]`, `lse` fp32 `[B, H, S]`.
  `bwd(q, kv, o, lse, do, *, window, causal, sinks) -> (dq, dkv, dsinks)` with
  `dkv` bf16 `[B, S, 1, D]`. `attn(q, kv, *, window, causal=True, sinks=None) -> o`
  is the autograd wrapper. `D in {64, 96, 128, 256}`.
- **The window.** `window` counts the query token itself, matching
  `golden(level="window", window=W)` and transformers'
  `sliding_window_overlay` (`kv_idx > q_idx - W`), both already verified against
  the model in `../golden_ref_test.py`. It is a **compile-time constant** (part of
  the jit key), clamped to `seq_len` so `window >= S` degenerates to plain causal
  and reproduces `latent_attn` to within one bf16 ulp.
- **`causal=False` asserts.** A non-causal sliding window is not a shape the model
  has, and the loop bounds assume the causal upper edge. Both `fwd` and `bwd`
  reject it; `fwd_test.py::test_fwd_rejects_non_causal` pins the message.
- **Restricted loop, not just a mask** (`fwd.py`). A block owns tokens
  `[t0, t0 + T_q - 1]` with `t0 = bx * T_q`, so it walks only the KV tiles
  overlapping `[t0 - window + 1, t0 + T_q - 1]`:
  `ceil(window / block_N) + ceil(T_q / block_N)` tiles instead of
  `ceil((t0 + T_q) / block_N)`. Work is O(S * window), not O(S^2). The mask keeps
  the per-**row token** form of `latent_attn` and gains a lower edge:
  `t - window < k <= t`, with `t = bx * T_q + i // H`.
- **Both backward kernels walk the same band.** `bwd_kv` (KV owner) loops only
  the query tiles whose tokens can see its latent block -- `k <= t <= k + window - 1`,
  i.e. `t in [k0, k0 + block_M - 1 + window - 1]`. `bwd_dq` (query owner) loops the
  same KV tiles as the forward. The band mask in both is correctness, not just
  speed: `lse` normalised the band only, so an unmasked recomputed `P` would
  include columns the forward never saw.
- **Rowmax floor instead of `-inf`** (`neg_floor = -1e30` in `fwd.py`). Unlike
  plain causal, a banded row can meet a KV tile it sees nothing in, and FA2's
  rescale then computes `-inf - (-inf)` = NaN. See Decisions.
- **fp8 window cache: nothing to do in the kernel.** v1 of ticket 0002 is a
  caller-side fake-quant round trip (`_fake_quant_fp8_block(kv, 32)` inside the
  model), so the kernel only ever receives the already-dequantized bf16 latent.
  In-kernel fp8 dequant is ticket 0010.
- Everything else -- packed `[B, S*H, D]` views, `T_q = block_M // H`, the sink as
  one denominator-only softmax column (`Sinks[i % H]` per row), `dKV = dK + dV` in
  one fp32 accumulator, fp32 split buffers, `dsinks` as a torch reduce, the
  `(q, kv, o, lse[, sinks])` save set -- is `latent_attn` verbatim.

## How to run

```
just src/mzoo/layers/attn/ test -q
just src/mzoo/layers/attn/ bench-swa --dim 128 --heads 64
just src/mzoo/layers/attn/ bench-swa --dim 128 --heads 64 --mode bwd
just src/mzoo/layers/attn/ bench-swa --dim 128 --heads 64 --window 512
```

(The trailing slash on the path is required by `just` for this recipe form.)
`bench-swa` always runs `swa`/`latent` with a random per-head sink; the `sdpa`
row is sink-free.

## Tile configs

Re-swept from scratch at B1 **H64** S4096 **window 128** (the model's shape) --
`latent_attn`'s table does **not** carry over, because the band changes what a
tile is worth. D=64, 96 and 128 are tuned; **D=256 is shape support only**.

fwd `CONFIGS` (head dim -> block_M = `T_q * H` rows, block_N = KV tokens, num_stages, threads):

| dim | block_M | block_N | num_stages | threads |
|---|---|---|---|---|
| 64 | 256 | 64 | 3 | 256 |
| 96 | 256 | 64 | 3 | 256 |
| 128 | 256 | 32 | 3 | 256 |
| 256 | 128 | 32 | 2 | 256 |

bwd `CONFIGS`, two tiles per dim -- `kv` (`block_M` = latent tokens, `block_N` =
packed query rows) and `dq` (the other way round, matching `fwd.py`):

| dim | kv: block_M / block_N / stages / threads | dq: block_M / block_N / stages / threads |
|---|---|---|
| 64 | 64 / 128 / 2 / 128 | 128 / 64 / 3 / 256 |
| 96 | 64 / 64 / 3 / 128 | 128 / 64 / 3 / 256 |
| 128 | 64 / 64 / 2 / 128 | 128 / 64 / 2 / 256 |
| 256 | 64 / 64 / 1 / 128 | 64 / 32 / 2 / 128 |

plus `splits` for `bwd_kv`, chosen at call time by `pick_splits` (12 at the
headline shape; `TARGET_CTAS = 768`, `MAX_SPLITS = 12`).

**The band inverts `bwd_kv`'s tile preference.** `latent_attn` wanted the
*largest* latent tile it could fit (`block_M` 128-256 at 256 threads) because that
kernel re-reads Q and dO once per KV block. With a window, a wide KV block instead
drags in query tiles most of its rows cannot see, so every dim now wants the
*smallest* tile, `block_M=64` at 128 threads (D=128: 0.797 ms at 64/64/2/128
against 0.896 at 128/64/2/256, the `latent_attn` pick). The forward and `bwd_dq`
keep preferring wide query tiles, unchanged.

### `seq_len` constraints

Asserted in `check_shapes` (fwd) and `bwd`. `T_q = block_M // H` is the token
count of a packed query tile; the backward additionally needs
`seq_len % kv.block_M == 0`:

| dim | fwd: `seq_len %` | bwd: `seq_len %` |
|---|---|---|
| 64 | `256 // H` | 64 |
| 96 | `256 // H` | 64 |
| 128 | `256 // H` | 64 |
| 256 | `128 // H` | 64 |

`H` must divide the packed-row tile of every kernel it touches: `H | 256` at
D<=128 (fwd) and `H | 128` / `H | 64` in the backward's `bwd_kv` / `bwd_dq`
query tiles -- `H in {4, 8, 16, 32, 64}` everywhere. `window` has **no**
alignment constraint (it is a per-row edge, not a tile count); `window=1`, 63 and
100 are all tested.

### Sweeps

All three grids excluded the cells that `tickets/0001-tilelang-issues.md` already
proves impossible (`threads=256` with `block_M` not a multiple of 128,
`block_M=32`). With those gone, **every one of the 204 failures across the 12
sweeps was a smem-budget failure** (`Failed to set the allowed dynamic shared
memory size to N`), none was layout inference.

Forward, top 5 per dim (ms, B1 H64 S4096 window 128):

| dim | block_M | block_N | stages | threads | ms |
|---|---|---|---|---|---|
| 64 | **256** | **64** | **3** | **256** | **0.319** |
| 64 | 256 | 32 | 3 | 256 | 0.324 |
| 64 | 256 | 128 | 3 | 256 | 0.337 |
| 64 | 256 | 64 | 2 | 256 | 0.337 |
| 64 | 256 | 32 | 2 | 256 | 0.342 |
| 96 | **256** | **64** | **3** | **256** | **0.465** |
| 96 | 256 | 64 | 2 | 256 | 0.475 |
| 96 | 256 | 32 | 2 | 256 | 0.477 |
| 96 | 256 | 32 | 3 | 256 | 0.483 |
| 96 | 128 | 64 | 3 | 256 | 0.484 |
| 128 | **256** | **32** | **3** | **256** | **0.601** |
| 128 | 256 | 32 | 2 | 256 | 0.602 |
| 128 | 128 | 64 | 3 | 256 | 0.608 |
| 128 | 256 | 64 | 2 | 256 | 0.608 |
| 128 | 128 | 64 | 3 | 128 | 0.616 |
| 256 | **128** | **32** | **2** | **256** | **1.214** |
| 256 | 64 | 32 | 3 | 128 | 1.239 |
| 256 | 128 | 64 | 1 | 256 | 1.244 |
| 256 | 64 | 64 | 2 | 128 | 1.269 |
| 256 | 64 | 64 | 1 | 128 | 1.329 |

Backward `bwd_kv`, top 5 per dim (`sp` = splits; the grid ran at `sp=8`, then the
top 3 per dim were re-run at `sp in {1, 2, 4}`):

| dim | block_M | block_N | stages | threads | sp | ms |
|---|---|---|---|---|---|---|
| 64 | **64** | **128** | **2** | **128** | 8 | **0.428** |
| 64 | 64 | 64 | 3 | 128 | 8 | 0.441 |
| 64 | 64 | 64 | 2 | 128 | 8 | 0.442 |
| 64 | 128 | 64 | 3 | 256 | 8 | 0.457 |
| 64 | 128 | 128 | 2 | 256 | 8 | 0.459 |
| 96 | **64** | **64** | **3** | **128** | 8 | **0.618** |
| 96 | 128 | 64 | 3 | 256 | 4 | 0.621 |
| 96 | 64 | 64 | 2 | 128 | 8 | 0.643 |
| 96 | 128 | 64 | 3 | 256 | 8 | 0.653 |
| 96 | 128 | 64 | 2 | 256 | 8 | 0.657 |
| 128 | **64** | **64** | **2** | **128** | 8 | **0.797** |
| 128 | 64 | 64 | 1 | 128 | 8 | 0.882 |
| 128 | 64 | 128 | 1 | 128 | 8 | 0.888 |
| 128 | 128 | 64 | 2 | 256 | 8 | 0.896 |
| 128 | 128 | 128 | 1 | 256 | 8 | 0.935 |
| 256 | **64** | **64** | **1** | **128** | 8 | **2.451** |
| 256 | 64 | 64 | 1 | 128 | 2 | 3.548 |

D=256 has only those rows because everything else overflows smem or hits the
256-thread layout bug.

Backward `bwd_dq`, top 5 per dim:

| dim | block_M | block_N | stages | threads | ms |
|---|---|---|---|---|---|
| 64 | 256 | 32 | 3 | 256 | 0.472 |
| 64 | 256 | 32 | 2 | 256 | 0.472 |
| 64 | **128** | **64** | **3** | **256** | **0.472** |
| 64 | 128 | 128 | 3 | 256 | 0.484 |
| 64 | 256 | 64 | 3 | 256 | 0.485 |
| 96 | **128** | **64** | **3** | **256** | **0.692** |
| 96 | 128 | 64 | 2 | 256 | 0.701 |
| 96 | 128 | 32 | 3 | 256 | 0.706 |
| 96 | 128 | 64 | 3 | 128 | 0.707 |
| 96 | 128 | 32 | 3 | 128 | 0.712 |
| 128 | **128** | **64** | **2** | **256** | **0.898** |
| 128 | 128 | 32 | 3 | 128 | 0.898 |
| 128 | 128 | 32 | 3 | 256 | 0.903 |
| 128 | 128 | 32 | 2 | 128 | 0.913 |
| 128 | 64 | 64 | 3 | 128 | 0.915 |
| 256 | **64** | **32** | **2** | **128** | **1.901** |
| 256 | 64 | 64 | 1 | 128 | 1.941 |
| 256 | 64 | 32 | 1 | 128 | 1.952 |

At D=64 the three-way tie at 0.472 is settled for `block_M=128`, the looser
`seq_len` constraint (as in `latent_attn`).

### Does a multi-token `T_q x H` block still pay under the band?

Yes, but the margin shrinks. Ticket 0002's open question, answered by the best
forward config at each `block_M` (H=64, so `T_q = block_M / 64`), B1 S4096 W128:

| dim | T_q=1 (block_M 64) | T_q=2 (128) | T_q=4 (256) |
|---|---|---|---|
| 64 | 0.376 ms | 0.350 | **0.319** |
| 96 | 0.530 | 0.484 | **0.465** |
| 128 | 0.653 | 0.608 | **0.601** |
| 256 | 1.239 | **1.214** | (smem) |

So `T_q=4` is 15% ahead of `T_q=1` at D=64 but only 8% at D=128, and the
`T_q=2 -> 4` step is worth 1% there. The reason is visible in the loop length:
at `block_N=32`, `T_q=1` walks `ceil(128/32) + 1 = 5` tiles for 1 token while
`T_q=4` walks 5 tiles for 4 tokens, so Q reuse still pays -- but the extra rows
also widen the band the block must cover, which is why the curve flattens.
It flattens further as D grows and the tile gets smem-bound (D=256 cannot hold
`block_M=256` at all). The conclusion: keep the general `T_q = block_M // H`
rule, do not special-case `T_q = 1`.

### splits

Re-swept for the chosen `bwd_kv` tile (B1 H64 S4096 W128, ms):

| dim | sp=4 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|
| 64 | 0.532 | 0.432 | **0.416** | 0.422 | 0.452 | 0.511 |
| 96 | 0.727 | 0.604 | **0.592** | 0.615 | 0.666 | 0.732 |
| 128 | 1.098 | 0.831 | 0.806 | **0.798** | 0.876 | 0.943 |
| 256 | 3.703 | 2.460 | **1.703** | 1.837 | 1.874 | 1.964 |

`latent_attn`'s `TARGET_CTAS = 256 / MAX_SPLITS = 8` would pick 4 splits here
(`block_M=64` makes the base grid 64 CTAs), which costs 27-118%. Raised to
`TARGET_CTAS = 768, MAX_SPLITS = 12`: the band shrinks per-CTA work, so the grid
needs many more waves before the tail dominates, and past ~16 splits the launch
and reduce overheads win back.

## Bench

GB10, B=1 S=4096 **window 128**, bf16, `--mode fwd` = forward only, `--mode bwd`
= fwd+bwd through autograd. `latent` is `latent_attn` on the **same inputs** doing
full causal attention (what the window replaces); `sdpa` is
`F.scaled_dot_product_attention` fed `kv.expand(B, S, H, D)` and an explicit
boolean band mask, i.e. torch's own windowed path, sink-free. TFLOPS are against
the *banded* work (`sum_t min(t+1, window)` = 516160 score entries at this shape),
so the two baselines are doing more than that by construction.
**D=128 H=64 is the headline (the model's shape).**

| dim | heads | mode | swa_attn | latent_attn (full causal) | sdpa (band mask) | vs latent | vs sdpa |
|---|---|---|---|---|---|---|---|
| 64 | 16 | fwd | 0.118 ms (17.96 TFLOPS) | 0.476 ms | 2.247 ms | 4.05x | 19.1x |
| 64 | 16 | fwd+bwd | 0.530 ms (9.98) | 1.915 ms | 9.103 ms | 3.61x | 17.2x |
| 64 | 64 | fwd | 0.345 ms (24.48) | 1.580 ms | 8.174 ms | 4.57x | 23.7x |
| 64 | 64 | fwd+bwd | 1.597 ms (13.24) | 7.205 ms | 35.056 ms | 4.51x | 22.0x |
| 96 | 16 | fwd | 0.155 ms (20.42) | 0.642 ms | 5.128 ms | 4.13x | 33.0x |
| 96 | 16 | fwd+bwd | 0.789 ms (10.05) | 2.790 ms | 22.174 ms | 3.54x | 28.1x |
| 96 | 64 | fwd | 0.488 ms (25.99) | 2.177 ms | 20.033 ms | 4.46x | 41.0x |
| 96 | 64 | fwd+bwd | 2.371 ms (13.38) | 10.436 ms | 98.021 ms | 4.40x | 41.3x |
| 128 | 16 | fwd | 0.192 ms (22.02) | 0.836 ms | 5.484 ms | 4.36x | 28.6x |
| 128 | 16 | fwd+bwd | 1.035 ms (10.22) | 3.580 ms | 23.798 ms | 3.46x | 23.0x |
| 128 | **64** | fwd | **0.624 ms (27.12)** | 2.903 ms | 21.265 ms | **4.66x** | **34.1x** |
| 128 | **64** | fwd+bwd | **3.073 ms (13.76)** | 15.673 ms | 105.648 ms | **5.10x** | **34.4x** |
| 256 | 16 | fwd | 0.332 ms (25.43) | 1.580 ms | 11.185 ms | 4.75x | 33.6x |
| 256 | 64 | fwd | 1.238 ms (27.32) | 5.830 ms | 46.547 ms | 4.71x | 37.6x |

So the window buys **4.1-4.8x forward and 3.5-5.1x fwd+bwd over full causal** at
S=4096, and 17-42x over torch's masked path. The ideal ratio against full causal
is `S(S+1)/2 / band = 16.3x`; we get 4-5x because the band makes the kernel
memory- and launch-bound rather than FLOP-bound (peak drops from ~95 TFLOPS at
full causal to ~27 banded).

The `sdpa` gap is structural, not a tuning artefact: an explicit `attn_mask`
disables torch's flash backend outright (`can_use_flash_attention` returns False
for a bool mask; `can_use_efficient_attention` returns True), so SDPA drops to the
memory-efficient kernel -- which still walks every KV block and reads the full
`S x S` mask, whatever the mask says. Hence 17-42x. That is the whole argument for
an in-kernel band.

Backward kernel breakdown at D=128 H=64 S=4096 W=128 (splits=12): `bwd_dq` 0.918
ms, `bwd_kv` 0.785, `preprocess` 0.655, plus 0.173 ms for the `dKV` split reduce
and 0.028 ms for the `lse` transpose -- 2.56 ms. `preprocess` is now **26% of the
backward**: it is `O(B*S*H*D)` and completely untouched by the window, so it did
not shrink while the two GEMM kernels shrank ~10x (`latent_attn`: 7.79 / 4.31 /
0.66 ms for the same three).

## Accuracy

Ratio of our max-abs error to the max-abs error of the same computation run in
bf16 by torch (`golden(level="window", window=128, dtype=torch.bfloat16)`), both
against the fp32 `golden(level="window", window=128)`; the acceptance criterion is
<= 2x. B2 S512, with a per-head sink. `lse` is fp32 end to end so it gets an
absolute number instead.

| dim | heads | o | dq | dkv | dsinks | lse (abs) |
|---|---|---|---|---|---|---|
| 64 | 16 | 0.55 | 0.84 | 0.45 | 0.52 | 9.5e-7 |
| 64 | 64 | 0.56 | 0.93 | 0.77 | 0.79 | 9.5e-7 |
| 96 | 16 | 0.58 | 0.54 | 0.69 | 0.50 | 9.5e-7 |
| 96 | 64 | 0.56 | 0.70 | 0.72 | 0.83 | 1.4e-6 |
| 128 | 16 | 0.60 | 0.88 | 0.58 | 0.71 | 9.5e-7 |
| 128 | 64 | 0.58 | 0.67 | 0.57 | 1.42 | 9.5e-7 |
| 256 | 4 | 0.60 | 0.52 | 0.60 | 2.07 | 1.4e-6 |
| 256 | 16 | 0.44 | 0.75 | 0.60 | 1.11 | 1.4e-6 |

`o`, `dq` and `dkv` never exceed 0.93x anywhere. `dsinks` is the noisy column --
see Known issues.

Three independent cross-checks beyond `golden`:

- `test_fwd_matches_gpt_oss`: against `golden_ref.gpt_oss_ref(window=128)`, i.e.
  transformers' real gpt-oss `eager_attention_forward` with the library's own
  `sliding_window_causal_mask_function`, which shares no code with `golden`'s
  `_attend`.
- `test_fwd_window_ge_seqlen_matches_latent`: at `window in {512, 4096}` with
  S=512 the result matches `latent_attn.fwd` at bf16-ulp scale -- max abs diff
  `0.00390625 = 2^-8` (one bf16 ulp at `|o| ~ 1`, the test's bound) and `lse`
  within 9.5e-7. Not bit-for-bit only because the two packages pick different
  tiles, so the online softmax accumulates in a different order; the band mask
  itself is provably vacuous there.
- `model_test.py::test_attn_matches_model_window_layer`: the tiny 2-layer DSV4.1
  smoke config (H=4, D=64, `sliding_window=128`) is run once; layer 0
  (`compress_ratio == 0`) is a pure sliding-window layer, so the exact
  `(query, key, value, attention_mask, scaling, attn_sink)` it hands to
  `eager_attention_forward` is captured, cast to bf16 and replayed through
  `swa_attn.attn` at B=2 S=192. Output passes the <= 2x criterion against eager's
  own fp32 result. The capture helpers are imported explicitly from
  `../golden_ref_test.py` rather than duplicated.

## Decisions

- **Floor the running rowmax at `-1e30`, do not mask with a large negative
  logit.** A banded row can meet a KV tile it sees nothing in (the loop bounds
  come from the block's *first* token, the band edge is per row), leaving both the
  running max and the tile max at `-inf` and making FA2's rescale
  `exp2(-inf * scale - (-inf) * scale)` = NaN. Flooring the *max* makes that step
  `exp2(0) = 1`, a no-op, while masked logits still give `exp2(-inf - floor) = 0`.
  Flooring the *logits* instead (mask with `-1e30` and keep an `-inf` max) is the
  trap: `exp2(-1e30 * scale + 1e30 * scale) = 1` would add 1 to the row sum per
  masked column. Only the forward needs this; both backward kernels divide by a
  known finite `lse`.
- **`window` is a compile-time constant**, not a runtime argument: it enters the
  jit key, so the loop bounds fold to constants and the band mask needs no extra
  loads. The model has exactly one value (128); a per-layer sweep would recompile
  once per distinct value, which is cheap and cached.
- **`window` is clamped to `seq_len` in `fwd`/`bwd`**, so `window=10**9` and
  `window=seq_len` share one compiled kernel and the `>= S` path is exactly the
  `latent_attn` kernel.
- **`causal=False` asserts rather than falling back.** The band is defined by a
  causal upper edge, `create_sliding_window_causal_mask` is the only mask the
  model builds, and a two-sided window would need different loop bounds in three
  kernels. Documented in every docstring and pinned by a test.
- **The band mask stays in both backward kernels.** It is not an optimisation:
  `lse` was computed over the band, so an unmasked recomputed `P` would be
  renormalised against columns the forward never saw.
- **`bwd_kv` re-tuned to the *smallest* latent tile.** See Tile configs -- the
  window inverts `latent_attn`'s preference, worth 12% at D=128.
- **`TARGET_CTAS` 256 -> 768, `MAX_SPLITS` 8 -> 12.** Measured; the inherited
  heuristic picks 4 splits here and loses 27-118%.
- **No `ref.py` in this package**, as in `latent_attn`: the series-wide
  `../golden_ref.py` at `level="window"` is the reference, and only
  `dense_attn.ref.assert_within_2x_torch` is imported.
- **The fp8 window cache is a torch-side round trip in v1**, per ticket 0002, so
  the kernel is unchanged by it. Ticket 0010 moves the dequant in-kernel.
- Fast-math (`TL_ENABLE_FAST_MATH`) kept; fixed sequence length assumed; no
  varlen.

## Known issues

- `window=1` at D=64 H=4 does not compile (TVM analyzer contradiction on the collapsed band
  loop bound; see `tickets/0001-tilelang-issues.md`). Degenerate shape, not a model one; found
  by independent verification, not chased.
- **`dsinks` straddles the 2x criterion at D=256, and always did.** At D=256 H=4
  the ratio is 2.07 at seed 0 (abs 2.83e-2 vs torch's 1.37e-2) -- over the bound.
  Across seeds 0-4 it is 2.07 / 1.20 / 0.48 / 0.82 / 1.99, and `latent_attn` on
  the same shape gives 1.28 / 2.24 / 0.51 / 1.29 / 1.99, so this is a pre-existing
  property of the `dsinks` reduce, **not** something the window introduced.
  `dsinks[h]` is a max over a single H-length vector, each element a sum of
  `B*S` terms of alternating sign, so the "2x torch" bound is statistically thin
  there. The test matrix checks `dsinks` at D in {64, 128}, H=16, where it is
  0.5-0.8. Fixing it properly means an fp64 or Kahan reduce in the reference, not
  a kernel change.
- **D=256 is shape support, not a fast path** (inherited). The 99 KB smem cap plus
  the 256-thread layout bug leave it one usable tile per backward kernel. The
  window helps it as much as any other dim (4.7x over full causal forward), but
  the absolute numbers stay ~2x D=128.
- **`preprocess` is now the third of the backward that did not get faster.** 0.655
  of 2.56 ms at the headline shape, because `delta = rowsum(dO * O)` is
  `O(B*S*H*D)` and window-independent. See Next.
- The `tickets/0001-tilelang-issues.md` layout-inference limits are unchanged
  here (`threads=256` needs `block_M % 128 == 0`; `block_M=32` never compiles in
  the KV-transposed kernel); the sweep grids simply exclude those cells, which is
  why all 204 sweep failures were smem-budget.
- `seq_len` must be a multiple of 64 in the backward and of `block_M // H` in the
  forward; `H` must divide the packed-row tiles. Asserted, not padded.
- `window` itself is unconstrained, but very small windows waste a whole tile:
  at `window=1` the loop still walks `ceil(1/block_N) + ceil(T_q/block_N)` tiles.
  Correct, just not tuned for.

## Next

- `csa_attn` (ticket 0003): add the compressed/main KV source as a second leg of
  the same online-softmax loop. Copy this package forward; the band, the packed-row
  layout and the three-kernel backward all carry over.
- **`preprocess` is the cheapest remaining win in this package**: 26% of the
  backward and untouched by the window. Fusing `delta` into `bwd_dq` (which
  already reads `O` and `dO`'s rows) or widening its `blk=32` tile should take
  most of 0.65 ms off the headline backward.
- The two GEMM kernels are now within 15% of each other (0.79 / 0.92 ms) and both
  are launch/memory-bound at 27 TFLOPS effective, vs ~95 at full causal. A
  persistent-CTA variant that keeps the latent resident across several query
  blocks is the next structural idea if the window path becomes the bottleneck.
- D=256 still needs a `block_M=32` tile (blocked by the layout bug) or a split-D
  loop; out of scope while `D<=256` is shape support.
