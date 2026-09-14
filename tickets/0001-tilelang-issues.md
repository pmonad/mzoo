# tilelang issues

Running log of TileLang issues hit on this machine (GB10, sm121). Every new
tilelang quirk hit by any kernel work gets appended here with version,
symptom, cause, workaround, status.

## sm121 (GB10) blocked from SM120 NVFP4 block-scaled MMA

- version: tilelang 0.1.14 (also unfixed on `main`; header last touched in 8f34abf, #2364)
- symptom: `T.mma_gemm_blockscaled` compiles for `sm_121a` but asserts at runtime:
  `tl::sm120_mma_sync_blockscaled requires sm_120a and CUDA 12.8 or later`
- cause: `src/tl_templates/cuda/instruction/mma_block_scale.h:42` requires `CUTLASS_ARCH_MMA_SM120A_ENABLED`.
  Everything else already allows sm121: `TargetIsSM120` (arch 120..129), and CUTLASS `cute/arch/config.hpp:168` enables
  `CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED` for SM121/SM121A (CUDA >= 12.9).
- proposed upstream fix: gate on `CUTE_ARCH_MMA_SM120_ENABLED` (covers sm120/120a/121/121a) instead of `CUTLASS_ARCH_MMA_SM120A_ENABLED`.
- workaround (in `src/mzoo/kernels/sm120_nvfp4_blockscaled_gemm.py`):
  `@tilelang.jit(compile_flags=["-DCUTLASS_ARCH_MMA_SM120A_ENABLED=1"])`.
  Blunt: also turns on other SM120A-gated CUTLASS features. Remove once fixed upstream.
- not an option: forcing target `sm_120a` (`TILELANG_DEFAULT_TARGET`) -> `CUDA_ERROR_NO_BINARY_FOR_GPU` on sm121.
- status: PR not raised yet.

## FA2 fwd: `block_M=64` + `threads=256` fails layout inference

- version: tilelang 0.1.14, GB10 (sm121), `src/mzoo/layers/attn/dense_attn/fwd.py`
- symptom: `Layout infer conflict between acc_s and acc_s_cast in T.Parallel loop` at compile time,
  for any `block_M=64` with `threads=256` (all of N=64/128, stages=1/2). `threads=128` is fine.
- impact: `D=256` must use `block_M=block_N=64, num_stages=1, threads=128` (the only config that fits
  99 KB smem), so it cannot use 256 threads. Not blocking, just caps the `D=256` config space.
- workaround: per-dim config table; 128 threads whenever `block_M=64`.

## misc

- default `/usr/bin/nvcc` is CUDA 12.0 and rejects `sm_121a`; needs `CUDA_HOME=/usr/local/cuda-13.0`.
- no `T.all` / `T.any` in 0.1.14 (`AttributeError: module 'tilelang.language' has no attribute 'all'`),
  and python `and` between two `PrimExpr`s is a truth-value test, not a TIR conjunction. The
  two-sided band mask of `swa_attn` needs `T.And(a, b)` / `T.Or(a, b)`, which do exist.

## FA2 bwd: `block_M=32` (any threads) and `block_M=64` + `threads=256` fail layout inference

- version: tilelang 0.1.14, GB10 (sm121), `src/mzoo/layers/attn/dense_attn/bwd.py`
  (port of `examples/flash_attention/example_mha_bwd_bshd.py`; `block_M` = KV rows, `block_N` = query rows)
- symptom: `Layout infer conflict between qkT and qkT_cast in T.Parallel loop` at compile time for
  every `block_M=32` config, and for `block_M=64` with `threads=256`. Same shape of bug as the fwd
  entry above (fp32 accumulator + its bf16 cast in one `T.Parallel`), just on the transposed tile.
- impact: search space is effectively `block_M in {64 @ 128 threads, 128 @ 128/256 threads}`.
  At `D=256` that leaves only `block_M=64, block_N=16, num_stages=1, threads=128` inside the 99 KB
  smem budget (`block_M=64, block_N=32` needs 103552 B, ~2 KB over), which costs ~13 ms at
  B1 H16 S4096 causal vs ~2.9 ms for `D=128`. `D=256` is a shape-support path here, not a fast one.
- workaround: per-dim config table (`CONFIGS`), as in the fwd.
- also seen: `block_N=16` + `threads=256` at `D=96` -> `No valid warp partition for T.gemm: M=16, N=96
  cannot be evenly covered` (8 warps cannot tile a 16x96 output); not a bug, just a constraint.

## Both layout-inference limits above reproduce verbatim in the packed-heads (MQA) layout

- version: tilelang 0.1.14, GB10 (sm121), `src/mzoo/layers/attn/latent_attn/{fwd.py,bwd_kernels.py}`
- context: `latent_attn` folds the H query heads into the tile's M dimension (a block is
  `T_q = block_M // H` tokens x H heads) and reuses one smem KV tile for both GEMMs.
  Checked whether the two entries above are sensitive to that layout. They are not:
  - fwd `block_M=64` + `threads=256` (block_N 64 and 128) ->
    `Layout infer conflict between acc_s and acc_s_cast in T.Parallel loop`. `threads=128` is fine.
  - bwd `block_M=32` (threads 128 *and* 256) and `block_M=64` + `threads=256` ->
    `Layout infer conflict between qkT and qkT_cast in T.Parallel loop`. `block_M=64` + `threads=128` is fine.
- so it is purely `block_M` x `threads` x (fp32 accumulator + its bf16 cast in one `T.Parallel`),
  independent of what the rows mean. Same workaround: per-dim config table avoiding those cells.
- new datapoint, not a bug: `block_M=256` forward is ~3% faster than `block_M=128` at 256 threads,
  but 10-30x *slower* at 128 threads (1.53 -> 21.8 ms, D=64 block_N=128 B1 H64 S4096) -- 8 warps are
  needed to tile a 256-row M. All 36 failures in the `latent_attn` bwd sweep were smem-budget
  (`Failed to set the allowed dynamic shared memory size to N`), none were layout inference.

## The same layout-inference conflict extends to `block_M=192`, and to a third kernel shape

- version: tilelang 0.1.14, GB10 (sm121), `src/mzoo/layers/attn/latent_attn/{fwd.py,bwd_dq.py}`
- context: sweeping `latent_attn` for `D in {96, 256}` and re-tuning the restructured backward
  (which gained a query-owning `dQ` kernel, `bwd_dq.py`).
- new datapoints, both the same bug as the two entries above:
  - `block_M=192` + `threads=256` fails exactly like `block_M=64` + `threads=256`:
    `Layout infer conflict between acc_s and acc_s_cast in T.Parallel loop`, with
    `loop Fragment((192, 64) -> (96,), replicate: 2, ...)` vs `fragment Fragment((192, 64) -> (48,), replicate: 1, ...)`.
    All 9 `block_M=192` x `threads=256` points of the D=96 forward sweep failed; `threads=128` compiles.
  - `bwd_dq.py` is a third tile shape (query rows in M, fp32 `acc_s` cast into a separate bf16
    `ds_cast` fragment rather than into `acc_s_cast`) and it fails identically:
    `Layout infer conflict between acc_s and ds_cast in T.Parallel loop` for every
    `block_M=64` + `threads=256` point.
- so the working rule on this stack is: **with `threads=256`, `block_M` must be a multiple of 128.**
  It is the replicate-2 loop fragment vs the replicate-1 accumulator fragment that conflicts,
  independent of the buffer names, of what the rows mean, and of the head dim.
- workaround: unchanged -- per-dim config table, drop to `threads=128` whenever a 64-row tile is
  forced (`latent_attn` D=256 uses 128 threads in both backward kernels for this reason).

## smem accounting: tilelang aliases shared buffers by liveness

- version: tilelang 0.1.14, GB10 (sm121), `src/mzoo/layers/attn/latent_attn/fwd.py`
- not a bug, but it invalidates hand-computed smem budgets: the forward's `Q_shared` and
  `O_shared` are both `[block_M, dim]` bf16 and naively sum to 96 KB at `block_M=256, D=96`,
  before the `K_shared` pipeline -- yet the kernel compiles and runs. `O_shared` is only live
  after the KV loop, so tilelang reuses `Q_shared`'s allocation for it.
- consequence: predict "fits / does not fit" by compiling, not by adding up tile sizes; the
  runtime error to grep for is `Failed to set the allowed dynamic shared memory size to N`
  (N > 101376 = 99 KB).

## swa_attn: `window=1` at D=64 H=4 fails in the TVM arith analyzer (degenerate band loop)

- version: tilelang 0.1.14, GB10 (sm121), `src/mzoo/layers/attn/swa_attn/fwd.py`
- symptom: compile-time `tvm.error.InternalError: Check failed: ... Trying to update var 'k' with a
  different maximum value: original=T.min(bx, 3), new=T.min(T.min(bx, 3), 3)` for `window=1`, D=64,
  H=4, S in {256, 512}. Same window compiles and runs finite at D=64 H=16, and at D=128 H=4/H=16;
  window=2 and window=63 compile at the failing shape.
- cause (probable): with `window=1` and this tile shape the band loop's start and end bounds collapse
  to the same nested `T.min` expression and the analyzer sees two different "maximum" bindings for
  the pipelined loop var. Degenerate case only; the model's window is 128.
- impact: none for the model; the swa tests cover window=1 only at D=128 H=16.
- workaround: none applied. Found by the independent verifier on 2026-09-14, not chased (user rule:
  correctness on model shapes first). Revisit with the tuning pass.

## `T.Pipelined` miscompiles a loop whose trip count nests a division (csa_attn's second KV source)

- version: tilelang 0.1.14, GB10 (sm121), `src/mzoo/layers/attn/csa_attn/{fwd.py,bwd_dq.py}`
- context: `csa_attn` walks a second KV source after the window tiles, with trip count
  `T.min(G // block_N, T.ceildiv(T.floordiv((bx + 1) * T_q, ratio), block_N))` -- the inner
  `floordiv` by the compression ratio is the new thing; every earlier package's bound was a
  plain `ceildiv` of a linear expression.
- symptom A (silent, the dangerous one): at `num_stages in {1, 2}` the kernel compiles, runs,
  and returns **wrong results** for some `(dim, block_N, G)` -- e.g. D=64 H=16 S=512 `ratio=4`
  `G=128` `block_N=64`: `o` off by 7.9e-1 (57x the bf16 torch error), tokens 3..255 wrong and
  256..511 right, i.e. every query block that should have walked exactly one main tile walked
  none. Other shapes on the same code are correct, so it looks like a passing kernel.
- symptom B (loud): the same loop at `num_stages=3` fails codegen with
  `TypeError: Downcast from tirx.Sub to ir.IntImm failed` under
  `CodeGenTileLangCUDA::VisitExpr_(CallNode)`.
- cause (from the generated CUDA): the software pipeline peels the loop into a prologue whose
  `cp_async` reads `MainKV[... - 1024]` (a negative global offset) followed by
  `for (int kg = 0; kg < 1; ++kg)`, with the mask's tile offset folded in as if `kg == 2`.
  The bound expression itself is fine: a standalone kernel that only *stores*
  `T.min(G // block_N, T.ceildiv(T.floordiv((bx + 1) * T_q, ratio), block_N))` returns the
  correct value for every block index. At `ratio == 1` the `floordiv` collapses and the bound
  becomes `latent_attn`'s long-tested form -- correct for every `G` there.
- not the trigger: the loop being the second pipelined loop in the kernel (making the window
  loop `T.serial` and pipelining only the main loop reproduces it), the loop starting at a
  literal `0`, the `T.min` (dropping it reproduces it), or the loop-variable name.
- workaround (applied): `T.serial` for that loop. Correct at every dim/heads/ratio/G tested,
  including non-tile-aligned `G` and S=4096. Costs the main source its async prefetch.
  `bwd_main.py`'s query loop keeps `T.Pipelined` -- it uses `swa_attn`'s variable-*start* form
  and does not show the bug.
- status: not reported upstream yet; needs a minimal repro outside the attention kernel.

## `block_M=16` fails layout inference too, even at `threads=128` (the per-token gather tile)

- version: tilelang 0.1.14, GB10 (sm121), `src/mzoo/layers/attn/csa2_attn/fwd.py`
- context: `csa2_attn` gathers per token, so the natural block is *one token x H heads*.
  At `H = 16` that is a 16-row tile; the earlier entries only ever tested `block_M >= 32`.
- symptom: `Layout infer conflict between acc_s and acc_s_cast in T.Parallel loop` with
  `loop Fragment((16, 32) -> (16,), replicate: 4, thread: 128, ...)` vs
  `fragment Fragment((16, 32) -> (4,), replicate: 1, thread: 128, ...)` -- the same
  replicate-N loop fragment vs replicate-1 accumulator conflict as every entry above, now at
  `block_M=16` with only 128 threads.
- so the working rule widens to: **`block_M * block_N` must give every thread its own
  accumulator elements without replication** -- in practice `block_M >= 64` at `threads=128`
  and `block_M % 128 == 0` at `threads=256`. `block_M=16/32` never compiles in this kernel
  family at any thread count we have tried.
- workaround: `block_M = 64` rows, i.e. `ceil(64 / H)` tokens per block, and gather once per
  token of the block (`csa2_attn/README.md` -> Known issues). Costs nothing at `H = 64`.

## `T.Pipelined` *is* correct when the trip count is a compile-time constant (csa2_attn's gather)

- version: tilelang 0.1.14, GB10 (sm121), `src/mzoo/layers/attn/csa2_attn/fwd.py`
- the positive counterpart to the `csa_attn` entry above: `csa2_attn`'s gathered loop runs
  `ceil(topk / block_N)` times -- a constant, with no division of a block-dependent expression
  in the bound -- and `T.Pipelined(0, gather_tiles, num_stages=2)` over a *data-dependent
  gather* (`M_shared[i, :] = MainKV[Indices[...], :]`) produces correct results at every shape
  tested (D 64/96/128/256, H 4/16/64, topk 64/100/512, G 0..16384, non-tile-aligned topk),
  including two bit-for-bit equalities against `csa_attn` and `swa_attn`.
- it is worth ~9% of the gathered source's time (1.68 -> 1.52 ms at B1 S4096 D128 H64
  topk=512), so the `T.serial` workaround should stay scoped to the bound shape that
  actually miscompiles, not applied to every second-source loop.

## A bf16 fragment cannot feed one `T.gemm` as `A` and another as `A^T`

- version: tilelang 0.1.14, GB10 (sm121), `src/mzoo/layers/attn/csa2_attn/bwd_dq.py`
- context: the sparse backward's query-owning kernel needs `dS` twice per gathered tile --
  `dQ += dS M` (normal `A`) and `dMainKV += dS^T Q` (`transpose_A=True`) -- and the natural
  thing is one `T.alloc_fragment([block_M, block_N], bfloat16)` feeding both.
- symptom: compile-time
  `tvm.error.InternalError: Get different layout for cast`, printing the two fragments,
  `forward_thread: _j // 16 * 32 + _j % 8 * 4 + _i % 8 // 2` against
  `forward_thread: _i // 16 * 32 + _i % 8 * 4 + _j % 8 // 2` -- i.e. the same tile with `_i`
  and `_j` swapped, which is exactly what the transposed operand wants.
- not a bug, a constraint: layout inference assigns **one** layout per buffer, and the two
  GEMM roles want transposed layouts. It is worth recording because the error message names
  the buffer, not the two GEMMs, so it reads like a codegen failure.
- workaround (applied, and what the upstream template does): stage the transposed operand
  through **shared** memory. `examples/dsa_sparse_finetune/sparse_mla_bwd.py` allocates
  `P_shared_cast` / `dP_shared_cast` as `T.alloc_shared`, not fragments, for the same reason.
  One `[block_M, block_N]` bf16 shared tile is enough if the two transposed GEMMs are
  ordered (`P` first, then `dS`); a second *fragment* does not help.
- cost: the staging tile is what pushes the `D=128, block_N=64` backward over the 99 KB smem
  cap at `gather_stages=2` (107776 B), so the backward's gathered loop runs serial there
  while the forward's is pipelined. Budget, not bug.

## `@tilelang.jit(out_idx=...)` outputs are empty-allocated -- silently fatal for atomic-only accumulation

- hit in: `indexer/bwd.py` (ticket 0007). `dq`/`dw` are written block-exclusively with
  `T.copy`, but `dk` is only ever accumulated with `T.atomic_add` -- nothing ever
  initialises it, and the tensors tilelang returns for `out_idx` slots come from an
  empty allocation, not a zeroed one.
- symptom: `dq`/`dw` inside the 2x-torch bound while `ddk` sat at 245-590x -- errors
  of the order of the gradient itself (uninitialised memory), on every shape. The
  forward never noticed because `fwd.py`'s single output is fully overwritten.
- workaround (applied): drop `out_idx`, allocate every output in the wrapper
  (`torch.zeros` for the atomic one) and pass them as in-out tensors. Harmless for
  copy-written outputs and it makes buffer ownership explicit.
- related, same ticket: fp32 `T.atomic_add` over many blocks is order-nondeterministic
  (~1e-4 abs reorder error at 256x128) -- expected fp32 behaviour, but it breaks
  bit-exact repeat tests on the accumulated tensor.

## a reduced fragment with a non-power-of-2 last dim has "no available layout"

- version: tilelang 0.1.14, GB10 (sm121), found via `norm_rope.py` (ticket 0011), 2026-09-14.
- trigger: `T.reduce_*` over a `T.alloc_fragment([M, N], ...)` with **N not a power of two**
  (96, 80, 48; either reduce dim) fails at compile with
  `Check failed: (has_best) is false: no available layout found`. Elementwise-only fragments at
  N=96 are fine, and so are GEMM accumulators (the `T.gemm` anchors an mma layout -- `indexer`
  runs Di=96); only the `T.Parallel`-filled-then-reduced case has no layout to infer.
- rule: give the reduced fragment a power-of-two last dim (pad and guard the fill with
  `if d < dim: ... else: 0.0`, as `make_fwd` does), **or** read a per-row `[M]` fragment inside the
  same fill loop, which anchors the row->thread mapping and makes N=96 infer (`make_bwd`'s `GD`
  loop reads `dot[i]`; verified by the D=96 test shape).
- caveat: rule 2 relies on an inference side effect; prefer rule 1 (pad) for new code.
- status: not reported upstream (repro below; loud failure, low priority).
- repro (fails as written, compiles if `F[i, d] = X[i, d]` becomes `F[i, d] = X[i, d] * rowv[i]`
  with `rowv = T.alloc_fragment([M], T.float32)` filled first):

```python
M, N = 128, 96
@tilelang.jit
def k():
    @T.prim_func
    def main(X: T.Tensor([M, N], T.float32), S: T.Tensor([N], T.float32)):
        with T.Kernel(1, threads=256) as bx:
            F, s = T.alloc_fragment([M, N], T.float32), T.alloc_fragment([N], T.float32)
            for i, d in T.Parallel(M, N):
                F[i, d] = X[i, d]
            T.reduce_sum(F, s, dim=0)
            for d in T.Parallel(N):
                S[d] = s[d]
    return main
```

## retracted: `T.reduce_sum(..., dim=0)` does *not* miscompile

- claimed (0011 worker): reducing a `[M, N]` fragment over `dim=0` returned wrong sums in
  `norm_rope`'s backward, so the dw partials were stored transposed and reduced `dim=1`.
- repro (2026-09-14) says otherwise: a standalone fill-and-reduce at (M, N, threads) =
  (128, 128, 256), (128, 64, 256) and (128, 96->128 padded, 256) matches torch column sums to
  1e-6 in both the direct and the transposed form, and `make_bwd` rewritten to reduce `dim=0`
  directly passes the whole of `norm_rope_test.py` (all 5 shapes, D=96 included).
- what the worker most likely hit is the entry above (non-pow2 last dim), which raises at compile
  in the transposed-free form unless the fill loop anchors a layout. `dim=0` is sound; the
  transposed workaround is gone.
