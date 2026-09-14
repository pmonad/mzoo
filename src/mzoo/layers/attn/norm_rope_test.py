"""Correctness + guards for the TileLang QK prologue (ticket 0011).

Reference is the frozen model's own composition: `DeepseekV41RMSNorm` then
`apply_rotary_pos_emb` from `archs/dsv4/modeling_deepseek_v41.py`, run on fp32
inputs; the same composition on bf16 inputs is the torch baseline, and the
kernel may be at most 2x further from fp32 than that baseline is
(`assert_within_2x_torch`, the series criterion). Guards: the fused forward is
exactly ONE CUDA kernel launch, and each shape compiles once (per-shape
`_CACHE`), so no silent per-call recompiles.
"""

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from mzoo.archs.dsv4.modeling_deepseek_v41 import DeepseekV41RMSNorm, apply_rotary_pos_emb
from mzoo.layers.attn.dense_attn.ref import assert_within_2x_torch
from mzoo.layers.attn.norm_rope import _CACHE, qk_norm_rope, qk_norm_rope_eager

SHAPES = [  # (B, S, H, D, rd) — includes non-power-of-two sizes
    (2, 512, 4, 64, 16),
    (2, 512, 1, 64, 16),    # kv-shaped [B, S, 1, D] (one latent head)
    (1, 256, 1, 128, 64),   # kv-shaped at the headline D
    (1, 256, 64, 128, 64),
    (3, 331, 5, 96, 24),
]


def _inputs(batch, seq, heads, dim, rope_dim, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(batch, seq, heads, dim, device="cuda", dtype=dtype)
    weight = torch.randn(dim, device="cuda", dtype=dtype)
    pos = torch.arange(seq, device="cuda", dtype=torch.float32)
    freqs = pos[:, None] / (10000.0 ** (torch.arange(0, rope_dim, 2, device="cuda") / rope_dim))
    cos, sin = freqs.cos()[None, :], freqs.sin()[None, :]  # [1, S, rd/2] fp32
    return x, weight, cos.expand(batch, -1, -1), sin.expand(batch, -1, -1)


def _reference(x, weight, cos, sin, eps=1e-6):
    """The model's own composition; `functional_call` keeps `weight` in the graph."""
    norm = DeepseekV41RMSNorm(x.shape[-1], eps=eps).to(x.device, x.dtype)
    normed = torch.func.functional_call(norm, {"weight": weight}, (x,))
    return apply_rotary_pos_emb(normed, cos, sin)


@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_fwd_within_2x_torch(shape):
    x, weight, cos, sin = _inputs(*shape)
    ref = _reference(x.float(), weight.float(), cos, sin)          # fp32 series reference
    torch_bf16 = _reference(x, weight, cos, sin)                   # same composition, bf16
    assert_within_2x_torch(qk_norm_rope(x, weight, cos, sin), ref, torch_bf16, "out")
    assert_within_2x_torch(qk_norm_rope_eager(x, weight, cos, sin), ref, torch_bf16, "out_eager")


@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_grads_within_2x_torch(shape):
    do = torch.randn(*shape[:4], device="cuda", dtype=torch.bfloat16)

    def grads(fn, dtype):
        x, weight, cos, sin = _inputs(*shape, dtype=dtype)
        x, weight = (t.clone().requires_grad_() for t in (x, weight))
        fn(x, weight, cos, sin).backward(do.to(dtype))
        return x.grad.float(), weight.grad.float()

    ref_dx, ref_dw = grads(_reference, torch.float32)              # fp32 series reference
    tdx, tdw = grads(_reference, torch.bfloat16)                   # torch bf16 baseline
    kdx, kdw = grads(qk_norm_rope, torch.bfloat16)
    # dw sums B*S*H bf16 rows, so the baseline's own accumulation error is the yardstick
    assert_within_2x_torch(kdx, ref_dx, tdx, "dx")
    assert_within_2x_torch(kdw, ref_dw, tdw, "dw")


def _launches(fn, args):
    fn(*args)  # compile outside the profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*args)
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)


def test_one_kernel_one_launch():
    args = _inputs(1, 512, 64, 128, 64)
    compiled = _launches(qk_norm_rope, args)
    eager = _launches(qk_norm_rope_eager, args)
    assert compiled == 1 and compiled < eager, (compiled, eager)


def test_compiles_once_per_shape():
    args = _inputs(2, 257, 4, 64, 16)
    qk_norm_rope(*args)
    n = len(_CACHE)
    qk_norm_rope(*args)
    assert len(_CACHE) == n  # second call reused the compiled kernel
