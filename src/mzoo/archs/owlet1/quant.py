# Split out of the frozen dsv4 baseline `src/mzoo/archs/dsv4/modeling_deepseek_v41.py`
# (see there for upstream provenance + Apache-2.0 license). Covers: fake FP4 (E2M1) and FP8 (E4M3) block-quantization helpers.

import torch


_FP4_MAX = 6.0  # float4_e2m1fn max
_FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_FP4_LUT_CACHE = {}  # device-keyed cache of the e2m1 value table


def _fp4_lut(device: torch.device) -> torch.Tensor:
    lut = _FP4_LUT_CACHE.get(device)
    if lut is None:
        lut = _FP4_TABLE.to(device)
        _FP4_LUT_CACHE[device] = lut
    return lut


def _pow2_ceil_scale(t: torch.Tensor) -> torch.Tensor:
    """`2^ceil(log2(t))` for fp32 `t > 0`, via the same IEEE-754 bit manipulation
    the reference kernel uses (exponent field minus 127, plus any nonzero
    mantissa) — the ue8m0 power-of-two scale rounding."""
    bits = t.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    k = exponent - 127 + (mantissa != 0).to(torch.int32)
    return torch.exp2(k.to(torch.float32))


def _e2m1_codes(q: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even cast of pre-clamped fp32 values onto the e2m1 grid;
    returns uint8 codes with the sign in bit 3. Ties at even-code boundaries
    (0.25, 1.25, 2.5, 5.0) round down, at odd-code boundaries round up."""
    magnitude = q.abs()
    negative = torch.signbit(q)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32, device=q.device)
    ties_up = torch.tensor([False, True, False, True, False, True, False], device=q.device)
    thresholds = torch.where(
        ties_up, boundaries, torch.nextafter(boundaries, torch.full_like(boundaries, float("inf")))
    )
    codes = (magnitude.unsqueeze(-1) >= thresholds).sum(-1).to(torch.uint8)
    return codes | (negative.to(torch.uint8) << 3)


def _fake_quant_fp4_block(x: torch.Tensor, block_size: int, e4m3_scales: bool = False) -> torch.Tensor:
    """Block-wise FP4 fake-quantization — the reference's indexer (block 32, ue8m0
    scales) and compressed-KV (block 16, e4m3 scales) quantizers. Returns the tensor
    quantized onto the e2m1 grid and dequantized (out of place; values identical to
    the reference's in-place call)."""
    n = x.shape[-1]
    if n % block_size:
        return x  # shapes the reference cannot block (tiny test configs) run unquantized
    blocks = x.float().view(*x.shape[:-1], n // block_size, block_size)
    amax = blocks.abs().amax(-1)
    if e4m3_scales:
        scale = (amax.clamp_min(6.0 * 2.0**-9) / _FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        scale = _pow2_ceil_scale(amax.clamp_min(6.0 * 2.0**-126) * (1.0 / _FP4_MAX))
    quantized = (blocks / scale.unsqueeze(-1)).clamp(-_FP4_MAX, _FP4_MAX)
    codes = _e2m1_codes(quantized)
    values = _fp4_lut(x.device)[codes.long()]
    return (values * scale.unsqueeze(-1)).view(*x.shape[:-1], n).to(x.dtype)


_FP8_MAX = 448.0  # float8_e4m3fn finite max


def _fake_quant_fp8_block(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Block-wise FP8 fake-quantization with ue8m0 (power-of-two) scales — the
    reference's window-KV quantizer, applied to the whole post-RoPE vector. Returns
    the quantized-then-dequantized tensor (out of place: training needs the original
    in the autograd graph; the values are identical to the reference's in-place
    call)."""
    n = x.shape[-1]
    if n % block_size:
        return x  # shapes the reference cannot block (tiny test configs) run unquantized
    blocks = x.float().view(*x.shape[:-1], n // block_size, block_size)
    amax = blocks.abs().amax(-1).clamp_min(1e-4)
    scale = _pow2_ceil_scale(amax * (1.0 / _FP8_MAX))
    quantized = (blocks / scale.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX)
    dequantized = quantized.to(torch.float8_e4m3fn).float() * scale.unsqueeze(-1)
    return dequantized.view(*x.shape[:-1], n).to(x.dtype)
