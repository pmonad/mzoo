# 0010 csa2_fp_attn: in-kernel fp8 / fp4 cache dequant on load

- status: todo
- depends on: 0005 (`csa2_attn` fwd + bwd)
- package: `src/mzoo/layers/attn/csa2_fp_attn/`

## Goal

Copy `csa2_attn/` to `csa2_fp_attn/` and make the kernel load the quantized caches directly -- fp8
window (v2 of step 3) and fp4 main (v2 of step 4) -- dequantizing into the bf16 smem tile instead of
consuming a torch-side dequantized bf16 tensor. Merged into one ticket because it is one change
(dequant-on-load) applied to the two sources of the same loop, with one shared risk: the dequant
cost.

## Semantics

Read `_fake_quant_fp8_block`, `_fake_quant_fp4_block`, `_pow2_ceil_scale`, `_e2m1_codes` in
`archs/dsv4/modeling_deepseek_v41.py`. Numerics must not move: this ticket changes *where* the
dequant happens, not the values.

- Window cache: e4m3 values + one **ue8m0** scale per 32 channels, `scale = 2**ceil(log2(amax/448))`
  with `amax` clamped to 1e-4. The ceil rule never clips -- the OCP floor rule does, so a stock
  MXFP8 encoder will not match.
- Main cache: e2m1 codes + one **e4m3** scale per 16 channels, no per-tensor scale, `amax` clamped
  to `6*2**-9`; grid `{0,+-.5,1,1.5,2,3,4,6}`, RTNE with ties-to-even at 0.25/1.25/2.5/5.0. Layout
  `[T, D]` packed 2 codes/byte + `[T, D/16]` scales, as fixed in 0003 (FlashMLA's
  `kv_cache_format.h` shape).
- GEMMs stay bf16 (survey §2/§3): only the smem staging buffer changes. Sink, masks, gather and
  `lse` are untouched.
- Backward **recomputes the same dequantized tile**; do not stash a bf16 copy and do not re-derive
  scales. Gradient w.r.t. the cache is the identity (STE) with the scale stop-gradiented, matching
  the out-of-place `_fake_quant_*` semantics.

## Deliverables

`fwd.py` (fp8 and fp4 unpack into smem), `bwd.py`, `attn.py`, `quant.py` (host-side packers,
bit-faithful to the two `_fake_quant_*` functions), `ref.py`, `bench.py`, `*_test.py`, `README.md`
(what/how, configs, bench, accuracy, decisions, **Known issues**, next). Tests via
`dense_attn/ref.py::assert_within_2x_torch`; extend `just smoke`.

## Acceptance

- Bit-faithfulness: `quant.py` round trip must equal `_fake_quant_fp8_block(x, 32)` and
  `_fake_quant_fp4_block(x, 16, e4m3_scales=True)` **exactly**, including the clamp floors and the
  tie cases (unit tests on hand-picked values).
- Kernel output must match `csa2_attn` (0005) fed the torch-dequantized tensors to within fp32
  accumulation order -- same inputs, same numbers.
- `D in {64, 96, 128, 256}`, 64/128 tuned, 96/256 pass-only. D=96 is not a multiple of the 16/32
  blocks: decide and document pad-vs-reject.
- Bench vs 0005 at B1 S4096, `G = 4096, 16384`: the win is bandwidth (4x on the main cache, 2x on
  the window); report whether it materialises or the unpack eats it.

## References

- `fp_attn_survey.md` §4 (the whole recommendation; FlashMLA `kv_cache_format.h`,
  `sm100/dequant_utils.cuh`, FlashInfer `repack_kv_tile_to_16b`) and §7 Q3b.
- `csa2_attn_design.md` steps 3 (v2), 4 (v2), 7.

## Risks / open questions

- **The known blocker**: FlashMLA measures the fp8 MLA kernel as dequant-bound (50 CUDA-core
  cycles/token vs 34 MMA) and escapes with 2-CTA DSM clusters. sm120 has neither DSM nor cluster
  launch, so budget the unpack on the critical path; if it loses, keep 0005 as the training path and
  say so in **Known issues**.
- smem: the staging buffer coexists with Q, the window tile and the gathered main tile inside 99 KB;
  may force `num_stages=1`.
- A RULER-style check of the fp8/fp4 round trip is still owed (survey §7 Q3) -- out of scope here,
  but this package is what it would run on.

## Learnings from earlier steps (2026-09-14)

- The dequant-on-load lives inside loops that are now `T.serial` for the main source
  (tickets/0001 pipeline miscompile), so the main tile has no async prefetch to hide the
  dequant behind; measure the dequant cost separately from the loop cost.
- Keep true and padded extents separate in every mask (csa_attn's padded-G leak); pad rows of
  a quantized cache dequantize to exact zeros and would feed the denominator.
- Test matrix: non-tile-aligned G and window, feature-off (bf16 input) equal to `csa2_attn`
  bit-for-bit, smallest and largest H. No tuning sweeps (user decision 2026-09-14).
