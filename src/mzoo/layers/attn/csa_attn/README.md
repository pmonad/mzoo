# csa_attn

Dense compressed attention: step 4 of the CSA2 series (see `../csa2_attn_design.md`,
ticket `tickets/done/0003-csa-attn.md`). Copied from `swa_attn/` with exactly one change --
after the window tiles of the raw latent, the **same online-softmax loop** walks the
visible entries of a second KV source, the compressor's latents `main_kv [B, G, 1, D]`.
Dense: every group-causally visible entry, no top-k yet (that is 0004/0005).

One running `(m, l, acc)` spans both sources, so there is **no LSE merge**: the main
tiles simply continue the softmax the window tiles started, exactly as if the KV axis
were the model's `torch.cat([kv, compressed_kv], dim=2)`. Everything else is
`swa_attn`: K == V is one latent with a single KV head serving H query heads, heads
packed into the tile rows, one smem tile per source serving both GEMMs, sink
semantics, `lse`, fp32 split `dKV` buffers and a query-owning `dQ` -- no atomics.

Source: `swa_attn/` (via `latent_attn/`, `dense_attn/`, itself `tile-ai/tilelang@v0.1.14`
`examples/flash_attention/example_mha_{fwd,bwd}_bshd.py` + `attention_sink`); the
two-source loop shape follows `examples/deepseek_v4/sparse_attn_fwd_sm90.py`.

## What it is

- **Call shapes.** `fwd(q, kv, main_kv, *, window, compress_ratio, causal=True, sinks=None)
  -> (o, lse)` with `q` bf16 `[B, S, H, D]`, `kv` bf16 `[B, S, 1, D]` (raw latent,
  K == V), `main_kv` bf16 `[B, G, 1, D]` (compressed latents, K == V), `o` bf16
  `[B, S, H, D]`, `lse` fp32 `[B, H, S]`.
  `bwd(q, kv, main_kv, o, lse, do, *, window, compress_ratio, causal, sinks)
  -> (dq, dkv, dmain_kv, dsinks)`.
  `attn(q, kv, main_kv, *, window, compress_ratio, causal=True, sinks=None) -> o`
  is the autograd wrapper. `D in {64, 96, 128, 256}`.
- **Group-causal visibility.** Main entry `g` is visible to token `t` iff
  `g < (t + 1) // compress_ratio` -- `compress_lens` in `DeepseekV41Indexer.forward`
  step 3, the same rule `golden(level="compressed")` implements. It is a *separate*
  mask from the window band, not a variant of it: the window edge is per token, this
  one is per group.
- **Restricted loop, not just a mask.** A block owning tokens `[t0, t0 + T_q - 1]` can
  see at most `(t0 + T_q) // ratio` groups, so the main loop runs
  `min(ceil(G / block_N), ceil(((t0 + T_q) // ratio) / block_N))` tiles and the tail is
  masked per row with `g < min(G, (t + 1) // ratio)`. At `ratio = 4` that is a quarter
  of the tiles a causal loop would walk.
- **`G` and `compress_ratio` are compile-time constants** (jit key), like `window`,
  `heads` and `seq_len`. **`G == 0` drops the second loop at compile time**, so the
  kernel *is* `swa_attn`'s -- tested bit-for-bit, forward and backward, not "within a
  ulp".
- **`main_kv` is zero-padded** to a multiple of the group tile in the caller
  (`pack_main`), so every kernel reads whole tiles; the pad rows are removed by the
  `min(G, ...)` in the mask, not by their values, and `dmain_kv` is sliced back to `G`.
  The kernels therefore carry the true `G` and the padded extent as two separate
  constants -- conflating them is a real bug we hit, see Known issues.
- **Rowmax floor kept** (`neg_floor = -1e30`). A row can see nothing in a main tile for
  the same reason it can in a window tile (the loop bound is the block's, the mask is
  the row's), and FA2's rescale would compute `-inf - (-inf)` = NaN.
- **Backward: four kernels, no atomics.** `preprocess` (delta) and `bwd_kv` (the window
  slice's `dKV`) are `swa_attn`'s, unchanged -- `lse` is an input to them, so the fact
  that it now also normalises the main source changes nothing. `bwd_main` is a second
  KV-owner launch whose rows are group indices: a block owning `[g0, g0 + block_M - 1]`
  walks the query tiles from `floor((ratio*g0 + ratio - 1) / T_q)` to the end of the
  sequence (unlike the window, the visible set grows without bound to the right), with
  the same group-causal mask. `bwd_dq` is one query-owning kernel that walks window
  tiles then visible main tiles into a single `acc_dq`.
- **Compressor, latent RoPE and fp4 stay in torch** (ticket 0003 v1): the kernel
  consumes already-dequantized bf16 `main_kv`. See Main-cache layout below; in-kernel
  dequant is ticket 0010. `ref.py` holds no numeric reference at all, only the helper
  that splits a captured model KV axis back into `(kv, main_kv)`.
- **`causal=False` asserts**, inherited and for the same reason.

## How to run

```
just src/mzoo/layers/attn/ test -q
just src/mzoo/layers/attn/ bench-csa --dim 128 --heads 64
just src/mzoo/layers/attn/ bench-csa --dim 128 --heads 64 --mode bwd
just src/mzoo/layers/attn/ bench-csa --dim 128 --heads 64 --ratios "(2,)"
```

(The trailing slash on the path is required by `just` for this recipe form.)

## Tile configs

**Untuned: inherited from `swa_attn` verbatim, no sweep run** (user decision for this
ticket -- correctness is the deliverable, tuning is deferred). Every one of
`swa_attn`'s configs compiled and passed with the second source added, so nothing had
to be shrunk; the main source's smem tile fits next to the window's in all four dims
because `T.serial` (see Known issues) leaves it single-buffered.

fwd `CONFIGS` (head dim -> block_M = `T_q * H` rows, block_N = KV tokens **and** main
entries per tile, num_stages, threads) -- the main loop is `T.serial`, so `num_stages`
applies to the window loop only:

| dim | block_M | block_N | num_stages | threads |
|---|---|---|---|---|
| 64 | 256 | 64 | 3 | 256 |
| 96 | 256 | 64 | 3 | 256 |
| 128 | 256 | 32 | 3 | 256 |
| 256 | 128 | 32 | 2 | 256 |

bwd `CONFIGS`, two tiles per dim -- `kv` (`block_M` = latent tokens **or** main
entries, `block_N` = packed query rows; `bwd_main` reuses this tile) and `dq` (the
other way round, matching `fwd.py`):

| dim | kv: block_M / block_N / stages / threads | dq: block_M / block_N / stages / threads |
|---|---|---|
| 64 | 64 / 128 / 2 / 128 | 128 / 64 / 3 / 256 |
| 96 | 64 / 64 / 3 / 128 | 128 / 64 / 3 / 256 |
| 128 | 64 / 64 / 2 / 128 | 128 / 64 / 2 / 256 |
| 256 | 64 / 64 / 1 / 128 | 64 / 32 / 2 / 128 |

plus `splits` for both KV-owner kernels, chosen at call time by `pick_splits`
(`TARGET_CTAS = 768`, `MAX_SPLITS = 12`, inherited). `bwd_main` gets its own split
count: its grid is `G / block_M` blocks and each walks up to `S / T_q` query tiles, so
the two sources need different numbers.

### `seq_len` / `G` constraints

Asserted in `check_shapes` (fwd) and `bwd`, unchanged from `swa_attn` -- `T_q =
block_M // H` tokens per packed query tile, `seq_len % T_q == 0` in the forward and
`seq_len % 64 == 0` in the backward, `H in {4, 8, 16, 32, 64}`. **`G` has no
constraint at all**: it is zero-padded to the tile in the caller and masked back off,
so `G = 0`, `G = 100` and `G = S` are all tested. `window` is likewise unconstrained.

## Bench

GB10, B=1 S=4096 window 128, bf16, per-head sink on every row. `--mode fwd` is forward
only, `--mode bwd` is fwd+bwd through autograd. The single baseline is **`swa_attn` on
the same `q`/`kv` with the compressed source removed**, so each row's delta is exactly
what the second source costs. TFLOPS are against the entries that row actually attends
(`swa_attn`: the 516160-entry band; `csa_attn`: band + `sum_t min(G, (t+1)//ratio)`
main entries), so the two are directly comparable per unit of work.

Another worker was building `indexer/` on the same GPU during this run, so treat the
absolute milliseconds as +-a few percent.

| dim | heads | mode | swa_attn (band only) | ratio=1, G=4096 | ratio=2, G=2048 | ratio=4, G=1024 |
|---|---|---|---|---|---|---|
| 64 | 16 | fwd | 0.120 ms (17.6 TF) | 0.534 ms (68.3) +344% | 0.308 ms (62.7) +156% | 0.207 ms (51.7) +72% |
| 64 | 16 | fwd+bwd | 0.558 ms (9.5) | 2.374 ms (38.4) +326% | 1.468 ms (32.9) +163% | 0.987 ms (27.1) +77% |
| 64 | 64 | fwd | 0.339 ms (25.0) | 1.849 ms (78.9) +446% | 1.060 ms (72.8) +213% | 0.686 ms (62.4) +103% |
| 64 | 64 | fwd+bwd | 1.621 ms (13.0) | 9.228 ms (39.5) +469% | 6.036 ms (32.0) +272% | 3.964 ms (27.0) +145% |
| 96 | 16 | fwd | 0.163 ms (19.5) | 0.753 ms (72.7) +362% | 0.459 ms (63.0) +182% | 0.294 ms (54.5) +81% |
| 96 | 16 | fwd+bwd | 0.780 ms (10.2) | 3.558 ms (38.5) +356% | 2.157 ms (33.5) +177% | 1.536 ms (26.1) +97% |
| 96 | 64 | fwd | 0.469 ms (27.1) | 2.630 ms (83.2) +461% | 1.529 ms (75.7) +226% | 0.968 ms (66.3) +107% |
| 96 | 64 | fwd+bwd | 2.409 ms (13.2) | 13.937 ms (39.3) +479% | 9.593 ms (30.2) +298% | 6.098 ms (26.3) +153% |
| 128 | 16 | fwd | 0.190 ms (22.2) | 0.973 ms (75.0) +411% | 0.560 ms (68.9) +194% | 0.369 ms (58.0) +94% |
| 128 | 16 | fwd+bwd | 1.016 ms (10.4) | 4.519 ms (40.4) +345% | 2.790 ms (34.6) +175% | 1.980 ms (27.0) +95% |
| 128 | **64** | fwd | **0.628 ms (26.9)** | 3.444 ms (84.8) +448% | 1.956 ms (78.9) +212% | **1.226 ms (69.8) +95%** |
| 128 | **64** | fwd+bwd | **3.110 ms (13.6)** | 19.074 ms (38.3) +513% | 13.468 ms (28.7) +333% | **8.397 ms (25.5) +170%** |

**Headline (D=128 H=64, the model's shape, `ratio=4`): the second source costs +95% on
the forward and +170% on fwd+bwd, for 9.4x more attended entries.** That is the number
to read, not the percentage on its own: at `ratio=4` the main source adds 4.85M visible
score entries to the band's 0.52M, so the window is 10% of the work and 34% of the
time. Per entry the compressed source is the *cheaper* of the two -- the forward runs
at 70 TFLOPS with it against 27 for the band alone, because the main loop is a long
run of fully-visible tiles with no per-row band edge, exactly the FLOP-bound regime
the band destroys. At `ratio=1` (G = S, the uncompressed special case) it reaches
85 TFLOPS at D=128 H=64, which is the highest this kernel family has measured on
GB10.

The backward scales worse than the forward (+170% vs +95% at ratio 4) for a structural
reason: the second source adds a whole extra KV-owner launch (`bwd_main`) whose query
loop has **no right edge** -- a block owning the earliest groups walks nearly the whole
sequence -- while `bwd_kv`'s band loop is a fixed `ceil((block_M + window) / T_q)` tiles.

## Accuracy

Ratio of our max-abs error to the max-abs error of the same computation run in bf16 by
torch (`golden(level="compressed", window=128, main_kv=..., compress_ratio=m,
dtype=torch.bfloat16)`), both against the fp32 `golden(...)`; the acceptance criterion
is <= 2x. B2 S512 (B1 at D=256), `G = S // ratio`, with a per-head sink. `lse` is fp32
end to end so it gets an absolute number.

| dim | heads | ratio | o | dq | dkv | dmain_kv | dsinks | lse (abs) |
|---|---|---|---|---|---|---|---|---|
| 64 | 16 | 1 | 0.63 | 0.65 | 0.61 | 1.00 | 2.16 | 1.4e-06 |
| 64 | 16 | 2 | 0.76 | 0.72 | 0.94 | 0.59 | 0.61 | 1.4e-06 |
| 64 | 16 | 4 | 0.42 | 0.38 | 0.67 | 0.56 | 1.06 | 1.4e-06 |
| 64 | 64 | 1 | 0.62 | 1.18 | 0.53 | 0.66 | 0.88 | 1.4e-06 |
| 64 | 64 | 2 | 0.58 | 0.75 | 0.75 | 0.58 | 0.80 | 1.4e-06 |
| 64 | 64 | 4 | 0.43 | 0.56 | 0.67 | 0.70 | 1.32 | 1.4e-06 |
| 96 | 16 | 1 | 0.75 | 0.78 | 0.77 | 1.00 | 1.68 | 1.4e-06 |
| 96 | 16 | 2 | 0.71 | 0.60 | 0.60 | 0.75 | 1.77 | 1.4e-06 |
| 96 | 16 | 4 | 0.67 | 0.74 | 0.60 | 0.47 | 1.23 | 1.4e-06 |
| 96 | 64 | 1 | 0.41 | 0.77 | 0.73 | 0.60 | 1.28 | 1.4e-06 |
| 96 | 64 | 2 | 0.45 | 0.65 | 0.81 | 0.54 | 1.21 | 1.4e-06 |
| 96 | 64 | 4 | 0.53 | 0.73 | 0.55 | 0.67 | 0.95 | 1.4e-06 |
| 128 | 16 | 1 | 0.54 | 0.95 | 0.56 | 0.58 | 1.12 | 1.4e-06 |
| 128 | 16 | 2 | 0.58 | 1.00 | 0.56 | 0.50 | 1.68 | 1.4e-06 |
| 128 | 16 | 4 | 0.41 | 0.73 | 0.49 | 0.50 | 0.50 | 9.5e-07 |
| 128 | 64 | 1 | 0.62 | 0.65 | 0.97 | 0.55 | 0.87 | 1.4e-06 |
| 128 | 64 | 2 | 0.57 | 0.67 | 0.64 | 0.63 | 0.86 | 1.4e-06 |
| 128 | 64 | 4 | 0.61 | 0.70 | 0.71 | 0.56 | 1.11 | 1.4e-06 |
| 256 | 4 | 1 | 0.79 | 0.96 | 0.81 | 0.61 | 0.98 | 1.4e-06 |
| 256 | 4 | 2 | 0.45 | 0.48 | 0.63 | 0.55 | 0.72 | 1.4e-06 |
| 256 | 4 | 4 | 0.55 | 0.54 | 0.64 | 0.58 | 0.64 | 1.4e-06 |
| 256 | 16 | 1 | 0.63 | 0.79 | 0.90 | 0.56 | 0.94 | 1.4e-06 |
| 256 | 16 | 2 | 0.63 | 0.49 | 0.45 | 0.61 | 0.76 | 1.4e-06 |
| 256 | 16 | 4 | 0.61 | 0.90 | 0.80 | 0.66 | 1.25 | 1.4e-06 |

`o`, `dq`, `dkv` and `dmain_kv` never exceed 1.18x anywhere in the matrix, across all
three ratios. `dmain_kv` -- the new tensor -- is the *tightest* column (0.47-1.00),
which is what you would expect: a main entry is summed over more query rows than a
window entry, so its fp32 accumulation averages more error away. `dsinks` is the noisy
column, as in `swa_attn` -- see Known issues.

Three independent cross-checks beyond the numbers above:

- `fwd_test.py::test_fwd_no_groups_is_swa` / `attn_test.py::test_grads_no_groups_is_swa`:
  at `G == 0` the output, `lse`, `dq` and `dkv` are **bit-for-bit** equal to
  `swa_attn`'s, with and without a sink. The second source is provably additive.
- `test_fwd_ratio1_main_is_kv`: the ticket's uncompressed special case -- `ratio=1`
  with `main_kv = kv` makes group-causal `g < t+1` identical to token-causal, so the
  query attends the same latent twice, once banded and once causally. `golden` was
  itself hand-checked against that closed form in `../golden_ref_test.py`.
- `model_test.py::test_attn_matches_model_compressed_layer`: the vendored DSV4.1 model,
  configured with `index_topk` past any `compressed_len` so its indexer selects every
  visible group and the eager path is *dense* group-causal. The test asserts the
  captured `topk_bias` really equals `g < (t+1)//ratio` before replaying the captured
  `(query, key, sink)` through `csa_attn.attn` at B=2 S=192 ratio=2 and checking the
  output against eager's own fp32 result with the 2x criterion. Setting
  `index_source_layer_ids=[]` instead does **not** give a dense layer: with no index
  source the model masks every compressed entry off.

## Main-cache layout (fixed here, implemented in 0010)

The layout every later package reuses, per the ticket. Per compressed entry, `D`
channels:

| buffer | shape | dtype | meaning |
|---|---|---|---|
| `main_kv_packed` | `[T, D/2]` bytes = `[T, D]` values | e2m1 (fp4), 2 values/byte, low nibble first | the rotated latent |
| `main_kv_scales` | `[T, D/16]` | e4m3 | one scale per 16 consecutive channels |

- e2m1 grid `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`, block size 16, **no per-tensor
  scale**; the block amax is clamped to `6 * 2**-9` before the scale is formed. That is
  `_fake_quant_fp4_block(x, block_size=16, e4m3_scales=True)` in the modeling file, and
  it is what the model applies to the compressed cache today.
- Channel-major within a block so a dequant reads 8 bytes of values + 1 byte of scale
  per 16 channels; FlashMLA's `kv_cache_format.h` is the byte layout to copy
  (`../fp_attn_survey.md` §4).
- This is **not** the indexer's format: indexer keys are `ue8m0` scales per 32 channels
  (OCP MXFP4), which is why 0008 can use `T.mma_gemm_blockscaled` natively and this one
  cannot -- PV stays bf16 here (§3), so the main cache is dequantized on load.
- **v1 (this package) has no code for it**: the round trip is a torch-side fake-quant
  inside the model, before the tensor ever reaches the kernel, so `csa_attn` sees bf16.
  Ticket 0010 replaces the bf16 `MainKV` argument with these two buffers and dequants
  into the smem tile.

## Decisions

- **The numeric reference is `golden(level="compressed")`, and `ref.py` holds no math.**
  `golden` is already verified against the vendored model in `../golden_ref_test.py`;
  re-deriving a second compressed reference here would only test it against itself.
  `ref.py` keeps exactly what is csa-specific and torch-side: the note that the
  compressor / latent RoPE / fp4 fake-quant stay out of the kernel path, and
  `split_captured_kv`, which undoes the model's `torch.cat([kv, compressed_kv], dim=2)`.
- **One loop, one `(m, l, acc)`, no LSE merge.** The alternative -- run the window and
  the main source as two passes and merge their LSEs -- costs an extra pass over `acc_o`
  and a second `O` buffer, and buys nothing here because both sources share the same
  query tile and the same smem budget. It only becomes interesting if the two sources
  ever want different tile shapes.
- **`G` is a compile-time constant.** It enters the jit key so the loop bound and the
  mask fold; a prefill has one `G` per (layer, sequence length), so the recompile count
  is the same as for `seq_len`, which is already a constant.
- **Zero-pad `main_kv` in the caller instead of predicating the tile load.** One
  `F.pad` of `[B, G, D]` (a no-op when already aligned, which is the model's case) is
  cheaper and far simpler than an element-wise predicated gather, and the mask has to
  run anyway. The price is that the true `G` and the padded extent must be threaded
  through the kernels separately -- see Known issues.
- **Two KV-owner launches, not one fused kernel.** `dKV` and `dMainKV` own disjoint
  outputs and share only read-only `lse`/`delta`, so two launches need no coordination,
  and their natural tiles differ (the window's query loop has a right edge, the main
  one does not). Fusing them would force one grid over two different row meanings.
- **`bwd_kv` and `preprocess` are copied byte-for-byte from `swa_attn`.** They take
  `lse` as an input, so the second source is invisible to them; keeping them identical
  makes the diff between the two packages exactly "the compressed source".
- **`T.serial` for the main loop** -- forced by a tilelang miscompile, not a choice.
  See Known issues.
- **`compress_ratio >= 1`**, and `ratio = 1` is a legal dense case (`G` may then equal
  `S`), matching the compressor's "ratio 1 = plain per-token projection" branch.
- Inherited unchanged: the `-1e30` rowmax floor, `window` as a compile-time constant
  clamped to `seq_len`, `causal=False` asserting, masks in the backward being
  correctness rather than speed, fast-math, fixed sequence length, no varlen.

## Known issues

- **`T.Pipelined` miscompiles the main loop; it is `T.serial`.** The second source's
  loop bound nests a division (`ceil((t0 + T_q) // ratio / block_N)`), and tilelang
  0.1.14's software pipeline mis-peels it: silently wrong results at `num_stages <= 2`
  for some `(dim, block_N, G)`, and a `Downcast from tirx.Sub to ir.IntImm` codegen
  crash at `num_stages = 3`. The bound expression itself evaluates correctly in a
  standalone kernel. Full write-up in `tickets/0001-tilelang-issues.md` and
  `docs/evolution/attn/attention-kernels-impl.md`. **Cost: the main source has no async
  prefetch**, which is a real part of the second-source overhead in the bench table and
  the first thing to revisit when the tuning pass happens.
- **Configs are untuned** (inherited from `swa_attn`, nothing shrunk). No sweep was run
  for this ticket. The `swa_attn` sweep showed the band inverts the `dKV` tile
  preference; `bwd_main`'s loop has no right edge, so it is much closer to
  `latent_attn`'s shape and probably wants a *larger* `block_M` than the 64 it
  inherited. Unmeasured.
- **`dmain_kv` is the slowest part of the backward at small `ratio`.** `bwd_main`'s
  query loop runs to the end of the sequence for every group block, so its total work
  is `O(G * S)` where `bwd_kv`'s is `O(S * window)` -- which is why fwd+bwd costs +513%
  at `ratio=1` while the forward costs +448%.
- **D=256 is shape support, not a fast path** (inherited): the 99 KB smem cap plus the
  256-thread layout-inference bug leave one usable tile per backward kernel.
- `window = 1` at D=64 H=4 still does not compile (inherited `swa_attn` issue, TVM
  analyzer; degenerate shape, not a model one).
- **`dsinks` still straddles the 2x criterion**, and still for the reason
  `swa_attn/README.md` documents: `dsinks[h]` is a max over an H-length vector whose
  entries each sum `B*S` terms of alternating sign, so "2x torch" is statistically thin
  there. In the matrix above it is 2.16 at D=64 H=16 `ratio=1` (the single value over
  the bound) and 1.1-1.8 at five other cells; `swa_attn` shows the same spread at
  D=256. Not introduced by the second source, and fixing it means an fp64/Kahan reduce
  in the *reference*, not a kernel change. The test matrix checks `dsinks` at
  `ratio=2`, where it is 0.6-0.9.
- The `tickets/0001-tilelang-issues.md` layout-inference limits are unchanged here
  (`threads=256` needs `block_M % 128 == 0`; `block_M=32` never compiles in the
  KV-transposed kernels).

## Next

- `csa2_attn` (tickets 0004/0005): replace the dense main loop with a gather over
  `Indices[B, S, topk]`. The loop bound and the group-causal mask are what become
  index-driven; the two-source structure, the padding scheme and all four backward
  kernels carry over.
- **Get the main loop pipelined again.** Either a minimal upstream repro + fix for the
  bound-with-division miscompile, or restructure the bound so the division is hoisted
  out of the loop header (e.g. pass `ceil(visible / block_N)` per query block in a small
  precomputed table). Worth a measurement before 0004 makes the loop sparse anyway.
- Tune, once correctness is locked: a sweep of `bwd_main`'s tile (it inherited the
  band-tuned one), and a `block_N` for the main source that is independent of the
  window's -- the two loops have different visibility structure and there is no reason
  they want the same tile.
- `preprocess` is still ~0.65 ms of the backward and still untouched by anything
  (inherited item from `swa_attn`).
