# quant.py

**Fake** (simulated) block quantization: tensors are rounded onto an FP4 (E2M1) or FP8
(E4M3) grid and immediately dequantized, all in fp32, then cast to the input dtype.
Nothing is stored in low precision; this only reproduces the rounding the reference
kernels apply to the KV / indexer caches (QAT semantics, on even in unquantized runs).

## Shapes

| symbol | shape | meaning |
|---|---|---|
| `B`, `S`, `G` | `8`, `512`, `S/ratio` | batch, sequence, compressed groups |
| `n`, `bs` | last dim, `32` or `16` | channels quantized; block size, one scale per block |
| window KV | `[B, 1, S, 64]`, `bs=32` | fp8, ue8m0 scales, 2 blocks per token |
| compressed latent | `[B, G, 64]`, `bs=16` | fp4, e4m3 scales, 4 blocks per latent |
| indexer q / k | `[B, S, 4, 32]` / `[B, G, 32]`, `bs=32` | fp4, ue8m0 scales, 1 block per head |

## E2M1 value table (`_FP4_MAX`, `_FP4_TABLE`, `_fp4_lut`)

FP4 E2M1 = 1 sign, 2 exponent, 1 mantissa bit: 16 codes, 8 magnitudes, max 6.
`_FP4_TABLE` is indexed by the 4-bit code with sign in bit 3; `_fp4_lut` caches a copy
per device (`_FP4_LUT_CACHE`).

| code | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| value | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |
| code 8..15 | -0 | -0.5 | -1 | -1.5 | -2 | -3 | -4 | -6 |

## `_pow2_ceil_scale(t)`

Rounds positive fp32 **up** to the next power of two via IEEE-754 bits (exponent field
minus 127, plus one if the mantissa is nonzero):

$$s(t) = 2^{\lceil \log_2 t \rceil}$$

This is the **ue8m0** scale format (unsigned 8-bit exponent, no mantissa) of MX-style
block formats. Ceiling rather than nearest guarantees $|x|/s \le \text{MAX}$ for every
element in the block, so the clamp never actually clips; the cost is up to a 2x coarser
grid than an exact scale.

## `_e2m1_codes(q)`

Round-to-nearest-even of pre-clamped $|q| \le 6$ onto the E2M1 grid: the code is the
number of midpoints $\{0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5\}$ that $|q|$ reaches. Ties go
to the *even* code: at $0.25, 1.25, 2.5, 5$ the threshold is `nextafter(b, +inf)` so an
exact tie rounds down; at $0.75, 1.75, 3.5$ it rounds up. Checked:
`[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]` -> codes `[0, 2, 2, 4, 4, 6, 6]`. Sign from
`signbit`, so `-0.0` yields code 8.

## `_fake_quant_fp4_block(x, block_size, e4m3_scales=False)`

Per block of `block_size` channels with $a = \max_j |x_j|$:

$$s = \begin{cases}
2^{\lceil \log_2 (\max(a,\,6\cdot 2^{-126}) / 6) \rceil} & \text{ue8m0 (default)} \\[4pt]
\mathrm{e4m3}\!\left(\max(a,\,6\cdot 2^{-9}) / 6\right) & \texttt{e4m3\_scales=True}
\end{cases}
\qquad
\hat x_j = \mathrm{LUT}\big[\mathrm{e2m1}(\mathrm{clamp}(x_j / s,\,-6,\,6))\big]\cdot s$$

The e4m3 variant stores the scale as `float8_e4m3fn` (round-to-nearest, 3 mantissa bits,
so not necessarily a power of two); the floor $6\cdot 2^{-9}$ keeps $s \ge 2^{-9}$, the
smallest e4m3 subnormal. Because that rounding can go *down*, $a/s$ may slightly exceed 6
and the clamp can clip in this mode. Callers: indexer q/k `(32, ue8m0)`; compressed KV
latent `(16, e4m3)`. If `n % block_size != 0` the input is returned untouched.

## `_fake_quant_fp8_block(x, block_size=32)` (`_FP8_MAX = 448`)

Same recipe for FP8 E4M3 with ue8m0 scales, on the whole post-RoPE window KV vector:

$$s = 2^{\lceil \log_2(\max(a,\,10^{-4}) / 448) \rceil}, \qquad
\hat x_j = \mathrm{e4m3}\big(\mathrm{clamp}(x_j / s,\,-448,\,448)\big)\cdot s$$

Rounding to the e4m3 grid is `torch`'s native `float8_e4m3fn` cast.

## Notes / gotchas

- All three are out of place so the caller keeps the unquantized tensor in the graph, but
  gradient behaviour is *not* a uniform straight-through estimator (checked empirically):
  fp8 passes grad = 1 to `x` (the cast is differentiable); fp4 ue8m0 passes **no**
  gradient (LUT gather + int bit-ops detach it); fp4 e4m3 leaks gradient only through the
  scale path (`amax`, one element per block).
- `_pow2_ceil_scale` assumes finite `t > 0`; the callers' `clamp_min` provides that.
- `_FP4_LUT_CACHE` is keyed on `torch.device` and never evicted (one 16-float tensor per
  device; harmless).
