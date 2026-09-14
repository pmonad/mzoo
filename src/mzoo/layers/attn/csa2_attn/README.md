# csa2_attn

Sparse (top-k) compressed attention: step 5 of the CSA2 series (see `../csa2_attn_design.md`,
ticket `tickets/0004-csa2-attn-sparse-fwd.md`). Copied from `csa_attn/` with exactly one
change -- the dense main-KV loop becomes a **gather** over per-token top-k indices:

```
M_shared[i, :] = main_kv[b, indices[b, t, kt * block_N + i], :]
```

`indices [B, S, topk]` is what the indexer produces (`indexer.attn`, ticket 0006, or the
model's own `shared["topk_bias"]`); `-1` marks an empty slot. Cost per token is `topk`
entries whatever `G` is, where `csa_attn` walked every group-causally visible entry.

**Trainable** since ticket 0005: `bwd.py` + `bwd_dq.py` + `bwd_kernels.py` and the `attn.py`
autograd wrapper. The backward's one structural change from `csa_attn` mirrors the forward's:
`csa_attn`'s group-owning `dmain_kv` kernel cannot exist when main rows are reached only
through `indices`, so `dmain_kv` becomes a **scatter-add** (fp32 `T.atomic_add`) out of the
query-owning `dQ` kernel.

Source: `csa_attn/` (via `swa_attn/`, `latent_attn/`, `dense_attn/`, itself
`tile-ai/tilelang@v0.1.14` `examples/flash_attention/example_mha_fwd_bshd.py`); the gather
loop follows `examples/dsa_sparse_finetune/sparse_mla_fwd.py` and
`examples/deepseek_v4/sparse_attn_fwd_sm90.py`.

## What it is

- **Call shapes.** `fwd(q, kv, main_kv, indices, *, window, compress_ratio, causal=True,
  sinks=None) -> (o, lse)` with `q` bf16 `[B, S, H, D]`, `kv` bf16 `[B, S, 1, D]` (raw latent,
  K == V), `main_kv` bf16 `[B, G, 1, D]` (compressed latents, K == V), `indices` int32 **or**
  int64 `[B, S, topk]`, `o` bf16 `[B, S, H, D]`, `lse` fp32 `[B, H, S]`.
  `attn(q, kv, main_kv, indices, *, window, compress_ratio, causal=True, sinks=None) -> o`
  is the autograd wrapper and `bwd(q, kv, main_kv, indices, o, lse, do, *, window,
  compress_ratio, causal=True, sinks=None) -> (dq, dkv, dmain_kv, dsinks)` the raw backward.
  `D in {64, 96, 128, 256}`. Gradients come back for `q`, `kv`, `main_kv` and `sinks`;
  `indices` is integer and gets `None` (top-k selection is discrete -- the indexer's own
  gradient path is ticket 0007).
- **`-1` / out-of-range indices contribute nothing.** An index outside `[0, G)` gives its
  whole column a `-inf` logit, so it adds nothing to the softmax and nothing to `O` --
  `sparse_mla_fwd.py`'s convention, and exactly the model's "clamp the pick into a dummy slot
  and slice it off". The row actually *loaded* for such a column is clamped to `0`, so the
  gather never reads out of bounds; only the logit carries the validity.
- **The kernel does not re-apply group-causality.** `g < (t + 1) // compress_ratio` is the
  indexer's job and is already baked into `indices`; `compress_ratio` stays in the signature
  only for call-shape parity with `csa_attn` and is asserted, not used. This matches
  `golden(level="sparse")`, which "trusts `indices` completely", and the model's
  `eager_attention_forward`, which has no idea where its `attention_mask` bias came from.
- **One token per block.** Indices are per token and shared across heads, so a gathered tile
  serves exactly one token's rows. `block_M = T_q * H` is still `csa_attn`'s packed-row rule,
  but `T_q` is now the *smallest* whole number of tokens reaching 64 rows: `T_q = 1` at
  `H = 64` (the model's shape), 4 at `H = 16`, 16 at `H = 4`. When `T_q > 1` the gathered
  section runs once per token of the block with the other tokens' rows masked off -- correct,
  and `T_q` times the gather work, at head counts the model does not use. See Known issues.
- **One running `(m, l, acc)` spans window + gathered tiles**, so there is still no LSE merge,
  and the sink is still one denominator-only column at the very end over the combined logits.
- **`G` needs no padding any more.** `csa_attn` zero-padded `main_kv` to a tile multiple
  because it read whole tiles; the gather addresses rows one at a time, so `pack_main` is
  gone. What *is* padded is `topk`, to a multiple of `block_N`, with `-1` (= empty) slots.
- **`G`, `topk`, `window` and `compress_ratio` are compile-time constants** (jit key).
  `G == 0` or `topk == 0` drops the gather at compile time, so the kernel *is* `swa_attn`'s --
  tested bit-for-bit.
- **Rowmax floor kept** (`neg_floor = -1e30`): a row can see nothing in a gathered tile (all
  `-1`, or another token's tile when `T_q > 1`), and FA2's rescale would compute `-inf - (-inf)`.
- **The backward is three kernels** (`csa_attn` had four): `preprocess` (delta) and the
  window slice's `dKV = dK + dV` are `csa_attn`'s `bwd_kernels.py` byte-for-byte with
  `csa_attn`'s config -- `lse` is an input there, so the second source being gathered changes
  nothing -- and `bwd_dq.py` owns a packed query tile and produces `dQ` **and** `dmain_kv`.
  `csa_attn`'s group-owning `bwd_main.py` has no analogue: a block of main entries cannot
  enumerate the tokens that picked it without an inverse index.
- **`dmain_kv` is a scatter-add, and therefore non-deterministic.** Per gathered tile the
  query CTA forms `dKV = dS^T Q + P^T dO` (one accumulator, K == V) and `T.atomic_add`s it in
  **fp32** into a zeroed `[B, G, D]` buffer that torch casts to bf16. Never bf16 atomics.
  fp32 addition is not associative, so `dmain_kv` is not bitwise reproducible run to run;
  see Known issues.
- **`-1` / out-of-range columns scatter nothing**: their `P` is zeroed, so their `dS`, their
  `dQ` contribution and their scattered row are all zero, and the atomic is predicated off so
  they never touch memory at all.
- **`causal=False` asserts**, inherited and for the same reason.

## How to run

```
uv run --env-file .env pytest src/mzoo/layers/attn/csa2_attn -q   # this package only
just src/mzoo/layers/attn/ test -q                                # every attn package
just src/mzoo/layers/attn/ bench-csa2 --dim 128 --heads 64        # fwd + bwd tables
just src/mzoo/layers/attn/ bench-csa2 --fwd_only --groups "(512,1024,2048,4096,16384)"
```

(The trailing slash on the path is required by `just` for this recipe form.)

## Tile configs

**Untuned: `csa_attn`'s `block_N` / `num_stages` verbatim, no sweep run** (user decision for
tickets 0004 and 0005 -- correctness is the deliverable). Three things had to change, all
forced by compile-time limits or the smem cap rather than by measurement:

- `block_M` is **64 rows** (one token x 64 heads at the model's shape) instead of `csa_attn`'s
  256: the gather is per token, so a block that owns several tokens has to gather several
  times. 16-row tiles (which would give `T_q = 1` at `H = 16`) fail layout inference here --
  see Known issues.
- `threads` drops to **128**: `threads=256` needs `block_M % 128 == 0` on this stack
  (`tickets/0001-tilelang-issues.md`).
- the backward's gathered loop gets its own pipeline depth, `gather_stages`, which the smem
  cap forces to 1 at `D = 128` and `D = 256` (table below).

fwd `CONFIGS` (head dim -> block_N = window KV tokens **and** gathered entries per tile,
num_stages for the window loop, threads); `block_M = T_q * H` with `T_q = ceil(64 / H)`:

| dim | block_M | block_N | num_stages | threads |
|---|---|---|---|---|
| 64 | 64 | 64 | 3 | 128 |
| 96 | 64 | 64 | 3 | 128 |
| 128 | 64 | 32 | 3 | 128 |
| 256 | 64 | 32 | 2 | 128 |

The gathered loop is pipelined at `num_stages=2` (see Decisions).

bwd `CONFIGS` (`kv` = the window `dKV` owner, `csa_attn`'s entry verbatim, `block_M` latent
tokens x `block_N` packed query rows; `dq` = the query owner, whose `block_M` is again
`T_q * H = 64` rows and whose `threads` are again forced to 128):

| dim | kv block_M | kv block_N | kv stages | kv threads | dq block_N | dq stages | dq gather stages |
|---|---|---|---|---|---|---|---|
| 64 | 64 | 128 | 2 | 128 | 64 | 3 | 2 |
| 96 | 64 | 64 | 3 | 128 | 64 | 3 | 2 |
| 128 | 64 | 64 | 2 | 128 | 64 | 2 | **1** |
| 256 | 64 | 64 | 1 | 128 | 32 | 2 | **1** |

`gather_stages` is the one number no earlier package has. It is the software-pipeline depth
of the *gathered* loop, separate from the window loop's `num_stages` because the two loops
have different smem costs: at `D = 128` with `block_N = 64` the kernel needs 107776 B with
`gather_stages = 2` and 91392 B with 1 (the 99 KB cap is 101376), and at `D = 256` no
`gather_stages = 2` config fits at any `block_N`. So the backward's gather is *serial* at the
two dims the model cares most about, where the forward's is pipelined -- a smem budget, not a
correctness limit. `block_N = 32` + `gather_stages = 2` also fits at `D = 128` and is the
obvious thing for the tuning pass to try; it was not taken here because `block_N = 64` is
what makes the feature-off gradients bit-for-bit `csa_attn`'s.

### `seq_len` / `G` / `topk` constraints

`seq_len % T_q == 0` (`T_q = ceil(64 / H)` tokens per block) is the forward's only one,
asserted in `check_shapes`. The backward adds `H | kv block_N` (128 at `D = 64`, else 64 --
the window `dKV` kernel's query tile, inherited from `csa_attn`) and `seq_len % 64 == 0`.
**`G` and `topk` are unconstrained**: `G` is addressed row by row and `topk`
is `-1`-padded to the tile in the caller, so `G = 0`, `G = 100`, `G = 16384` and
`topk in {64, 100, 512}` are all tested.

## Bench

GB10, B=1 S=4096 window 128 D=128 H=64 topk=512, bf16, per-head sink. The
baseline is **`csa_attn` on the same `q`/`kv`/`main_kv`** with the main source walked densely
(`compress_ratio=1`, so it sees `min(G, t+1)` entries at token `t`). `window-only` is this
same kernel called with `topk=0` (the gather dropped at compile time), so `gather ms` is the
gathered source's own cost and `GB/s` is the effective bandwidth of the `B*S*topk*D` bf16 rows
it pulls.

| G | csa_attn (dense) | main entries/token | csa2_attn (gather) | gathered/token | speedup | window-only | gather ms | GB/s |
|---|---|---|---|---|---|---|---|---|
| 512 | 1.107 ms | 480 | 2.171 ms | 512 | 0.51x | 0.690 ms | 1.480 | 363 |
| 1024 | 1.749 ms | 896 | 2.185 ms | 512 | 0.80x | 0.703 ms | 1.482 | 362 |
| 2048 | 2.670 ms | 1536 | 2.130 ms | 512 | 1.25x | 0.695 ms | 1.435 | 374 |
| 4096 | 3.441 ms | 2048 | 2.219 ms | 512 | 1.55x | 0.697 ms | 1.522 | 353 |
| 16384 | 3.470 ms | 2048 | 2.205 ms | 512 | 1.57x | 0.702 ms | 1.504 | 357 |

(With the gathered loop on `T.serial` instead of `T.Pipelined` the same sweep reads 2.30-2.41
ms / 316-339 GB/s, i.e. the pipeline is worth ~9% of the gather -- see Decisions.)

**The headline: csa2 is flat in `G` (2.13-2.22 ms from `G = 512` to `G = 16384`) while
`csa_attn` grows linearly with the visible set (1.11 -> 3.47 ms).** Break-even is just under
`G = 2048`, i.e. when a token can see ~3x `topk` entries; below that the dense walk is
cheaper, which is the honest reading of "sparse attention is a large-context technique".

Two more things the table says:

- **`csa_attn` stops growing past `G = S`** -- not because the dense loop got smarter but
  because group-causality caps it: at `G = 16384, S = 4096, ratio = 1` a token can only see
  4096 of the 16384 entries, so the dense kernel *cannot attend the rest of the cache at all*
  while csa2 picks its 512 from anywhere in it. The flatness of the csa2 column is the real
  result; the dense column's flatness at the right edge is an artefact of the prefill shape.
- **The gather is not latency-bound.** 353-374 GB/s effective is *above* the 273 GB/s the
  design doc quotes for this GPU, so the 512-entry-per-token random read is being served
  partly out of L2 (at `G = 4096`, 4096 tokens x 512 picks over a 4096-row cache is heavy
  reuse). Per attended entry the gather still costs ~2.1x a dense tile read (1.52 ms / 512
  entries vs 2.74 ms / 2048), which is the price of the random access plus the smaller 64-row
  tile.
  **No sorting-indices experiment was run** (the ticket made it optional); the L2 numbers
  suggest locality is already decent at these `G`.

### Backward

Same shape, `dO` random, `bwd` called on the forward's own `(o, lse)`; the baseline is
`csa_attn.bwd` on the same inputs (its main source dense). `preprocess`/`dKV`/`dQ` are the
three kernel launches timed on their own; `dmain` is what the last one loses when the whole
`dmain_kv` half is compiled out (its two GEMMs *and* the scatter), and `atomic` is what it
loses when only the `T.atomic_add` is replaced by a plain store to the same addresses.

| G | csa_attn bwd | csa2_attn bwd | speedup | preprocess | dKV | dQ + dmain | dmain of that | atomic of that |
|---|---|---|---|---|---|---|---|---|
| 4096 | 16.147 ms | 4.959 ms | 3.26x | 0.674 | 0.790 | 3.431 | 1.125 (32.8%) | 0.302 (8.8%) |
| 16384 | 18.214 ms | 5.086 ms | 3.58x | 0.682 | 0.783 | 3.421 | 1.086 (31.7%) | 0.385 (11.3%) |

| G | csa_attn fwd+bwd | csa2_attn fwd+bwd | speedup |
|---|---|---|---|
| 4096 | 19.571 ms | 7.132 ms | **2.74x** |
| 16384 | 21.674 ms | 7.289 ms | **2.97x** |

The backward is where the gather pays off hardest: 3.3-3.6x against the dense main loop,
versus 1.6x in the forward. The reason is that `csa_attn` needs a *third* GEMM kernel
(`bwd_main.py`, the group-owning `dmain_kv`) that walks every visible query tile of every
group block, and that kernel alone is most of its 16-18 ms; csa2 gets the same output for
1.1 ms of extra work inside a kernel it was running anyway.

**The atomic is not the problem** (matching the `latent_attn` finding, `docs/evolution`):
replacing the fp32 `T.atomic_add` with a plain store to the *same* addresses saves only
0.30-0.39 ms, ~9-11% of the `dQ` kernel and ~7% of the whole backward. The other ~0.75 ms of
the `dmain` column is the two extra GEMMs (`dS^T Q` and `P^T dO`), which an owner-based
`dmain_kv` would still have to do -- so an inverse-index/CSR owner kernel is bounded above by
a ~0.35 ms win on a 5 ms backward and was **not** built. See Decisions.

## Accuracy

Ratio of our max-abs error to the max-abs error of the same computation run in bf16 by torch
(`golden(level="sparse", window=128, main_kv=..., compress_ratio=2, indices=<same list>,
dtype=torch.bfloat16)`), both against the fp32 `golden(...)` **fed the identical index list**;
the acceptance criterion is <= 2x. B2 S512 (B1 at D=256 and at H=64), `G = S // 2 = 256`,
random group-causal top-k sets from `indexer_scores` + `topk_indices`, with a per-head sink.
`lse` is fp32 end to end so it gets an absolute number.

| dim | heads | topk=64 | topk=100 | topk=512 | lse (abs) |
|---|---|---|---|---|---|
| 64 | 4 | 0.53 | 0.81 | 0.55 | 9.5e-07 |
| 64 | 16 | 0.49 | 0.63 | 0.56 | 9.5e-07 |
| 64 | 64 | 0.54 | 0.59 | 0.63 | 1.4e-06 |
| 96 | 4 | 0.65 | 0.47 | 0.58 | 1.4e-06 |
| 96 | 16 | 0.65 | 0.67 | 0.61 | 1.4e-06 |
| 96 | 64 | 0.59 | 0.60 | 0.50 | 1.4e-06 |
| 128 | 4 | 0.80 | 0.42 | 0.57 | 9.5e-07 |
| 128 | 16 | 0.55 | 0.40 | 0.50 | 9.5e-07 |
| 128 | 64 | 0.51 | 0.53 | 0.54 | 1.4e-06 |
| 256 | 4 | 0.54 | 0.60 | 0.55 | 1.4e-06 |
| 256 | 16 | 0.54 | 0.64 | 0.57 | 1.4e-06 |
| 256 | 64 | 0.52 | 0.56 | 0.51 | 1.4e-06 |

Every cell is below 1x -- the kernel is closer to fp32 than torch's own bf16 pass everywhere
in the matrix (max 0.81 at D=64 H=4 topk=100), and `lse` never exceeds 1.4e-06 absolute.
`topk = 100` (a multiple of no `block_N`, so the last gathered tile is part `-1` padding) is
as accurate as the aligned sizes, which is the padding check the numbers are there for.

### Gradients

Same criterion and same construction, now through autograd on both sides: the fp32 reference
is `torch.autograd` through `golden(level="sparse", ...)` fed the **identical** index list,
and the baseline is the same `golden` at `dtype=torch.bfloat16`. B2 S512 (B1 at D=256 and at
H=64), `G = S // 2 = 256`, `topk = 100`, per-head sink.

| dim | heads | o | dq | dkv | dmain_kv | dsinks |
|---|---|---|---|---|---|---|
| 64 | 4 | 0.71 | 0.52 | 0.66 | 0.58 | 0.86 |
| 64 | 16 | 0.76 | 0.72 | 0.94 | 0.55 | 0.61 |
| 64 | 64 | 0.48 | 0.81 | 0.67 | 0.53 | 0.82 |
| 96 | 4 | 0.50 | 0.56 | 0.63 | 0.52 | 0.73 |
| 96 | 16 | 0.71 | 0.60 | 0.60 | 0.67 | *1.88* |
| 96 | 64 | 0.56 | 0.77 | 0.79 | 0.48 | 1.03 |
| 128 | 4 | 0.39 | 0.57 | 0.83 | 0.54 | 1.14 |
| 128 | 16 | 0.58 | 1.00 | 0.56 | 0.68 | *1.67* |
| 128 | 64 | 0.49 | 0.73 | 0.72 | 0.54 | 0.88 |
| 256 | 4 | 0.45 | 0.49 | 0.63 | 0.53 | 0.68 |
| 256 | 16 | 0.63 | 0.71 | 0.45 | 0.66 | 0.77 |
| 256 | 64 | 0.47 | 0.61 | 0.45 | 0.56 | 0.81 |

Every `dq`/`dkv`/`dmain_kv` cell is <= 1.00x -- the kernel's gradients are at least as close
to fp32 as torch's own bf16 pass is, everywhere in the matrix. `dmain_kv` is the *most*
accurate column (0.48-0.68) despite the atomics, because it accumulates in fp32 while the
bf16 baseline reduces in bf16.

`dsinks` is the exception and behaves exactly as the design doc's test matrix predicts: a
thin `-sum exp(sink - lse) * delta` reduce whose ratio straddles 2x at small `B*S` in every
package (1.88 at D=96 H=16 here). The tests therefore assert `dsinks` at **D 64/128, H 16**
only, and the other tensors keep the unmodified 2x criterion; the two italicised cells are
reported, not hidden.

Cross-checks beyond the numbers:

- `test_fwd_all_visible_is_csa_attn`: hand the kernel *every* group-causally visible entry in
  group order and it reproduces `csa_attn.fwd` **bit-for-bit** (`o` and `lse`), sink included.
  Same entries in the same order means the per-row online softmax is literally the same
  sequence of operations, so the gather is provably just a re-addressing of the dense loop.
- `test_fwd_all_empty_is_swa` / `test_fwd_no_groups_is_swa`: a row whose indices are all `-1`
  (and the whole-kernel cases `topk = 0` / `G = 0`) is **bit-for-bit** `swa_attn.fwd`, with and
  without a sink, and stays finite -- the window always contains the token itself.
- `test_fwd_from_indexer_package`: the same check driven by the real `indexer.attn` kernel
  (int32 indices) instead of the torch `topk_indices` definition.
- `test_grads_all_visible_is_csa_attn`: the gradient half of the same feature-off case --
  every group-causally visible entry in group order -- gives `dq` and `dkv` **bit-for-bit**
  `csa_attn.bwd`'s (and `dsinks` too, it is the same torch reduce). `dmain_kv` is *not*
  bit-for-bit and cannot be: `csa_attn` reduces it in fp32 split buffers, csa2 with fp32
  atomics, so the two differ by summation order. Measured drift at B2 S512 H16 D128 G256:
  max-abs 7.8e-3 against a tensor of max-abs 1.06e+1, i.e. 7.3e-4 relative -- about one bf16
  ulp of the final cast. The test bounds it at 5e-3 relative.
- `test_grads_all_empty_is_swa`: all indices `-1` gives `dq`/`dkv` **bit-for-bit**
  `swa_attn.bwd`'s (with and without a sink) and `dmain_kv` identically zero. Note this holds
  even though csa2's `dQ` kernel uses a 64-row / 128-thread tile where `swa_attn`'s uses
  128 rows / 256 threads: `block_M` only groups rows, and each row's tile sequence and
  contraction order are set by `block_N` (64 in both), so the arithmetic is the same program.
- `test_grads_duplicate_indices`: see Known issues -- the one place where the kernel and
  `golden` genuinely disagree.
- `test_backward_saves_only_q_kv_main_indices_o_lse`: the forward saves exactly
  `(q, kv, main_kv, indices, o, lse)` (+ `sinks`), never an `S x (S + topk)` matrix, and
  `indices` is pinned non-differentiable.
- `model_test.py::test_attn_matches_model_sparse_layer`: the vendored DSV4.1 tiny model,
  unmodified -- its layer 1 has `index_topk = 64 < compressed_len = 80`, so the indexer really
  drops visible groups. The captured `shared["topk_bias"]` is turned back into an index list
  with `golden_ref_test._mask_to_indices`, replayed through `csa2_attn.attn` at B=2 S=160
  ratio=2, and checked against eager's own fp32 output with the 2x criterion (`lse` against
  the logsumexp of eager's combined logits at bf16 scale).

## Main-cache layout

Unchanged from `csa_attn` -- see `../csa_attn/README.md` -> "Main-cache layout". The gather
changes *which* rows are read, not their format; ticket 0010 (in-kernel e2m1 + e4m3-per-16
dequant) will have to dequantize a gathered row instead of a contiguous tile, which is the
same code with a different source address.

## Decisions

- **The numeric reference is `golden(level="sparse")` fed the same indices**, and `ref.py`
  holds no math (inherited). Feeding both sides the identical index list is what makes the 2x
  criterion meaningful for a sparse kernel: any disagreement is arithmetic, never selection.
- **Validity on the logit, clamp on the load.** The template loads `KV[Indices[...]]`
  unconditionally and would read `KV[-1]` for an empty slot; we clamp the loaded row to `0`
  and let the `-inf` logit do the work. One `if_then_else` in the address, no correctness
  difference, no out-of-bounds read.
- **One gathered tile feeds both GEMMs** (K == V), as in every package since `latent_attn`.
- **`T_q > 1` gathers once per token instead of dropping to a 16-row tile.** The alternative
  (16 rows, `T_q = 1` at `H = 16`) does not compile here, and padding the rows out to 64 with
  masked-off junk would cost exactly what the repeat costs. The model's shape is `H = 64`,
  where `T_q = 1` and there is no repeat at all.
- **The gathered loop is `T.Pipelined(num_stages=2)`, unlike `csa_attn`'s `T.serial` main
  loop.** `csa_attn`'s miscompile needs a division nested in the trip count; here the trip
  count is the compile-time constant `ceil(topk / block_N)`, so the trigger is absent. This
  was verified, not assumed: the whole test matrix (28 tests, including `topk = 100` -> a
  `-1`-padded tail tile, `T_q = 16`, and the bit-for-bit `csa_attn` equality) passes with the
  pipeline on, and it is worth ~9% of the gather (1.68 -> 1.52 ms). The fallback is one line:
  `for kt in T.serial(0, gather_tiles)`.
- **`compress_ratio` kept in the signature though the kernel ignores it.** Call-shape parity
  with `csa_attn` (the conventions require every package to expose the same shape) and a place
  to document who owns group-causality; it is asserted `>= 1` and never used in the math.
- **`dmain_kv` is produced by the query-owning kernel, not by an owner kernel of its own.**
  `csa_attn` had a group-owning `bwd_main.py`; here a group block cannot know which tokens
  picked it without an inverse index (a CSR built in torch every step). The template
  (`sparse_mla_bwd.py`) scatters, and the ablation says scattering is the right call: the
  atomic's *own* cost over a plain store to the same addresses is 0.30-0.39 ms on a 5 ms
  backward, so the best an owner kernel could win is ~7%, before paying for the inverse
  index. Measured, not assumed -- `bwd_dq.py`'s `scatter` knob is kept for re-measuring.
- **fp32 atomics, never bf16.** The accumulator is fp32 and the buffer is fp32 `[B, G, D]`;
  bf16 atomics would lose ~3 digits per accumulation over the ~`B*S*topk/G` tokens that hit
  a row. The cast to bf16 happens once, in torch.
- **One bf16 *shared* staging tile for the two transposed GEMMs.** `dS` feeds `dQ += dS M`
  as a normal A operand and `dKV += dS^T Q` as a transposed one, and tilelang gives a
  fragment one layout per buffer -- reusing a single `cast` fragment for both fails layout
  inference with `Get different layout for cast`. `sparse_mla_bwd.py` stages its `P`/`dP`
  casts through shared for the same reason; we do the same with one tile (`P` first, then
  `dS`). See `tickets/0001-tilelang-issues.md`.
- **The `dQ` kernel's `block_M` is the forward's, not `csa_attn`'s.** The gather is per
  token in the backward too, so the query tile is `T_q * H = 64` rows at 128 threads, not
  `csa_attn`'s 128 rows at 256 threads. This costs nothing at the model's `H = 64` and is
  what the layout-inference limits allow; it does **not** cost bit-for-bit agreement with
  the earlier packages, because `block_M` only groups independent rows.
- **`block_N = 64` at `D = 128` even though it forces the gathered loop serial.** The
  alternative (`block_N = 32`, gather pipelined) fits smem and may well be faster, but
  `block_N` sets each row's tile sequence, so changing it would break the bit-for-bit
  feature-off equalities that are this package's cheapest regression check. Untuned either
  way; the tuning pass should measure both and decide with numbers.
- Inherited unchanged: the `-1e30` rowmax floor, one running `(m, l, acc)` with no LSE merge,
  `window` as a compile-time constant clamped to `seq_len`, `causal=False` asserting,
  fast-math, fixed sequence length, no varlen.

## Known issues

- **A 16-row tile does not compile**, so small head counts pay for it. `block_M = 16` with
  `threads = 128` fails layout inference exactly like the `block_M = 32/64 x threads = 256`
  cells already in `tickets/0001-tilelang-issues.md` (`Layout infer conflict between acc_s and
  acc_s_cast`, loop fragment `replicate: 4` vs accumulator `replicate: 1`). Consequence:
  `block_M = 64`, so at `H = 16` a block owns 4 tokens and gathers 4 times, at `H = 4` 16
  tokens and 16 times. Correctness is unaffected (each pass masks the other tokens' rows off);
  the cost is only paid at head counts the model does not use.
- **Configs are untuned.** `block_N` and `num_stages` are `csa_attn`'s, which were
  `swa_attn`'s; `block_M` and `threads` were forced by the two limits above, not chosen. In
  particular `block_N = 32` at `D = 128` means the model's `topk = 512` is 16 gathered tiles
  of 32 rows, and nothing has measured whether 64 or 128 rows per gathered tile is better --
  the gathered loop has no per-row edge at all, so it probably wants a *larger* tile than the
  band does. First thing to try in the tuning pass.
- **Break-even is at `G ~ 2048`** at the model's shape (see Bench): below that the dense
  `csa_attn` loop is faster, so the sparse kernel is not a drop-in win at short context.
- **D=256 is shape support, not a fast path** (inherited): the 99 KB smem cap plus the
  256-thread layout-inference bug leave one usable tile.
- **`dmain_kv` is not bitwise reproducible.** fp32 `atomic_add` ordering varies run to run
  and fp32 addition is not associative, so two identical backward calls give `dmain_kv`
  values that differ in the last bits (verified: `torch.equal` is `False` across two runs,
  while `dq` and `dkv` are bitwise identical). This is stated, not fixed -- making it
  deterministic means an owner kernel or a deterministic reduction tree, and the ablation
  above says that is not worth 7%.
- **Duplicate indices double-count, and that disagrees with `golden`.** If a token lists the
  same entry twice, the kernel builds two gathered columns for it, so the softmax sees
  `2 * exp(s)` in its denominator and the entry gets twice the weight; the gradient is the
  consistent gradient of *that* forward (the scatter-add adds both contributions).
  `golden(level="sparse")` renders the index list by `scatter_`-ing a `0.0` bias, which is
  idempotent, so it gives a duplicated entry **one** column. The two therefore disagree by a
  lot on duplicates (max-abs 6.4e-1 vs a bf16 noise floor of 1.5e-2 at B2 S512 H16 D128).
  This is a **forward** property, inherited from ticket 0004 and from the upstream
  `sparse_mla_fwd.py` convention, and it was *not* changed here: the model's indexer emits
  `torch.topk` output, which is a set, so duplicates never occur on the model path.
  `test_grads_duplicate_indices` pins both sides of it -- that the kernel matches a
  reference built by *cloning* the duplicated row into a fresh entry (so the reference
  really does have two columns), and that plain `golden` is far away. If a future index
  producer can emit duplicates, either it must dedup or the kernel must mask repeats within
  a token's list; do not assume `golden` agreement.
- **The backward's gathered loop is serial at `D = 128` and `D = 256`** (`gather_stages = 1`),
  where the forward's is pipelined: `gather_stages = 2` needs 107776 B of smem at
  `D = 128, block_N = 64` against the 99 KB cap. Costs the backward's gather its async
  prefetch. See Tile configs.
- `window = 1` at D=64 H=4 still does not compile (inherited `swa_attn` issue, TVM analyzer;
  degenerate shape, not a model one).

## Next

- Tune (deferred, nothing here was swept): a `block_N` for the gathered source independent of
  the window's -- in the backward that is the difference between a pipelined and a serial
  gather at `D = 128` -- and a look at whether a 128-row gathered tile amortises the random
  access better. Validate any change against the two bit-for-bit feature-off tests, which a
  `block_N` change will break by construction.
- The `dQ`/`dmain_kv` kernel is 3.4 of the backward's 5.0 ms and recomputes `S` and `dP` for
  both sources; the `+40%` FLOPs that bought `swa_attn` its atomic-free `dQ` are now paid on
  the gathered source too. Worth re-checking once the tuning pass exists.
- Sorting each token's indices before the gather (the ticket's optional experiment) -- not
  tried; the 353-374 GB/s effective bandwidth says locality is not the current bottleneck.
- Ticket 0010 (in-kernel fp4 dequant) meets this package at the gathered row load.
