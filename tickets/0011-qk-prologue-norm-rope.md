# 0011 QK prologue: RMSNorm + RoPE as one fused pass (tilelang)

- status: done (2026-09-14; started as torch.compile, landed as a tilelang kernel -- see Results)
- depends on: nothing kernel-side; consumer is whichever model/bench path feeds `latent_attn`+
- location: `src/mzoo/layers/attn/norm_rope.py` + `norm_rope_test.py` + `norm_rope_bench.py`
  (fwd + bwd tilelang kernels, custom autograd.Function, eager reference kept in the same file)

## Problem

Every attention layer feeds the kernel a q that was normed then rotated in torch. Done as two eager
ops that is two full read+write passes over q. At the headline shape the numbers are:

| tensor | B1 S4096 H64 D128 bf16 | one extra read+write at GB10's 273 GB/s |
|---|---|---|
| q | 67 MB | ~0.49 ms (~17% of `latent_attn` fwd, 2.90 ms) |
| kv latent | 1 MB | ~0.01 ms (irrelevant) |

The norm is also the logit bound the sharing chapters require (QK-norm, `|q^.k^| <= d`): with one
latent serving 64 heads a single large-norm key inflames every head at once. In the vendored
DSV4.1 the k side is a true per-row norm (`kv_norm` on the latent), the q side norms the low-rank
residual before the per-head up-projection, so the per-head bound does not strictly hold for q.

## Decision: torch.compile, not a kernel

Fuse `RMSNorm(weight) -> apply_rotary_pos_emb` into one `torch.compile(fullgraph=True)` function.
Reasons, and the prior art considered:

- The op is a row reduction over D (64..256) plus an elementwise rotation on the rope slice: about
  the simplest fusion Inductor does. One generated Triton kernel, q read once and written once.
- Norm must precede RoPE (with a learnable per-channel gain the two do not commute), so fusing the
  norm into the attention kernel's Q load would drag RoPE into the kernel too, and the backward
  would need the norm Jacobian on dq/dkv plus a gain reduction, carried through every copy-forward
  package. Not worth it for ~0.5 ms until the CSA2 kernel shape is stable.
- **torchtitan's stance**: it shipped a fused Triton RMSNorm early (before `nn.RMSNorm` existed and
  when Inductor did not fuse the fp32-upcast norm pattern well) and later removed it in favour of
  `nn.RMSNorm` + per-block `torch.compile`. Their current position is that compile suffices here.
- **Liger's decision**: Triton RMSNorm/RoPE kernels exist because Liger targets *eager* HF Trainer
  runs where whole-model compile breaks (graph breaks in modeling code, varlen recompiles, compile
  time, DTensor/TP brittleness); its RoPE kernel rotates q and k in one launch. Its headline wins
  are memory, mostly from the chunked fused linear cross-entropy, an algorithmic change compile
  cannot make. None of those constraints apply to one small pure function over fixed shapes.
- **Memory** is the one place a hand kernel could still earn its place: Liger's RMSNorm saves only
  the per-row `rstd` (and has an in-place mode) for backward. Autograd through the compiled
  function may keep a full activation copy. Measure it (below); if it saves more than ~one fp32 per
  row, wrap in a custom `autograd.Function` that saves `rstd` and recomputes, still in torch.

## Semantics

Match exactly the composition of `DeepseekV41RMSNorm.forward` (fp32 norm, `weight * x.to(dtype)`)
and `apply_rotary_pos_emb` (interleaved pairs on the trailing rope slice, cos/sin `[B,S,rd/2]`,
leading nope channels pass through) from `archs/dsv4/modeling_deepseek_v41.py`. x is `[B,S,H,D]`
or `[B,S,1,D]`, bf16 or fp32. The module imports nothing from the modeling file; the test composes
the model's two functions as the reference.

## Guards (no silent fallback)

- `torch.compile(..., fullgraph=True)`: graph breaks raise instead of splitting into eager pieces.
- Test sets `torch._dynamo.config.error_on_recompile = True` and runs H in {1,4,16,64}, D in
  {64,128}, S in {256,512}: proves no per-shape recompiles beyond the intended first compile(s);
  document the `dynamic=` choice that makes this hold.
- Fusion has no flag: the test counts CUDA kernel launches with `torch.profiler` for one forward of
  the compiled fn (after warmup) and asserts `<= 2` and strictly fewer than the eager composition.

## Deliverables

- `norm_rope.py`: `qk_norm_rope(x, weight, cos, sin, *, eps=1e-6)` compiled, plus
  `qk_norm_rope_eager` for reference/debug. Short file, docstring says why it exists.
- `norm_rope_test.py`: fwd match (bf16 + fp32; `[2,512,4,64]`/`[2,512,1,64]` rd 16,
  `[1,256,64,128]` rd 64), grads wrt x and weight match eager, the three guards above.
- Bench (`fire` CLI in the same file if it stays short, else `norm_rope_bench.py`) +
  `just src/mzoo/layers/attn/ bench-prologue`: B1 S4096 H64 D128 rd 64 and D64 rd 16, bf16, eager
  norm-then-rope vs compiled, median ms and effective GB/s (read x + write out).
- Memory: `torch.cuda.memory_allocated` delta after a forward-with-grad, compiled vs eager, and the
  list of saved tensors (`torch.autograd.graph.saved_tensors_hooks`) for q `[1,4096,64,128]`.

## Acceptance

- Tests pass in `just smoke`; compiled fwd within ~10% of the 273 GB/s roofline (~0.5 ms for the
  67 MB q), i.e. one pass; eager shows two.
- Kernel-launch count and recompile guards hold across the shape set.
- Memory result recorded in the README/parking note; if the backward saves a full activation, a
  follow-up decision (custom autograd.Function saving `rstd`) is written down, not silently done.

## Results (2026-09-14, GB10, `just src/mzoo/layers/attn/ bench-prologue`)

| path | B1 S4096 H64 D128 rd64 | B1 S4096 H64 D64 rd16 |
|---|---|---|
| eager norm->rope | 10.1 ms (13 GB/s) | 4.0 ms (17 GB/s) |
| torch.compile (max-autotune, 3 kernels) | 0.96 ms | 0.56 ms |
| tilelang fused (1 kernel) | **0.73 ms (184 GB/s)** | **0.32 ms (207 GB/s)** |
| pure copy ceiling (`y.copy_(x)`) | 0.58 ms (232 GB/s) | -- |

- The compile decision above was measured first and **rejected**: Inductor cannot emit one
  kernel here -- the rstd row reduction is always its own kernel that re-reads q (249 us)
  and the stack->cat interleave is a second pointwise writing a rope-sized intermediate,
  so its floor is 3 launches / ~200 MB traffic. All four alternate formulations tried
  (slice-assign, complex, per-channel coefficients, full-width expansion) also gave 3+.
- The tilelang kernel keeps the D-row in shared, reduces rstd in-kernel
  (`reduce_sum(dim=1)`, 0.1.14's `dim=0` miscompiles -- tickets/0001), stages the normed
  row, rotates pairs, writes q once: 1 launch (test-asserted), 81% of the copy ceiling.
  Non-pow2 D (96) needs the fragment-pad workaround from tickets/0001.
- Backward is a custom autograd.Function with a tilelang bwd kernel (dx elementwise +
  dw via transposed-fragment reduce + atomics); correct vs eager, untuned.
- Memory (the ticket's open question): fwd-with-grad retains 68.2 MB fused vs 269.5 MB
  eager for q [1,4096,64,128] -- the Function saves only `rstd` (one fp32/row, 1.1 MB),
  no full activation copy, so no follow-up decision needed.
- Guards as specified, adapted to the kernel path: 1 launch (not <=2), per-shape
  `_CACHE` proves single compile per shape (the error_on_recompile guard was
  compile-specific and is gone with the compile path).
