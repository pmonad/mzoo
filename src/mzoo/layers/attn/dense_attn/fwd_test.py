"""GPU tests for the dense forward (output + LSE) against fp32 SDPA.

Output tolerance: fp32 accumulation but P and O round through bf16
(eps 2^-8 ~ 3.9e-3), and summing block_N products per tile lets that grow, so
atol=rtol=2e-2 (observed max ~1e-2 at D=256).

``lse`` never leaves fp32 in the kernel -- only the logits feeding it are bf16
products -- so it is compared much tighter, atol=2e-3 (observed ~1.4e-6).

The sink cases reuse both tolerances unchanged: the sink is one fp32 term added
to the denominator, so it does not move the error (observed max o 7.8e-3,
lse 1.4e-6, same ballpark as the sink-free runs).
"""

import pytest
import torch

from mzoo.layers.attn.dense_attn.ref import sdpa_ref, sdpa_ref_lse
from mzoo.layers.attn.dense_attn.fwd import fwd

TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=2e-3, rtol=2e-3)


def _run(batch, heads, seq_len, dim, causal, sinks=None):
    torch.manual_seed(0)
    q, k, v = (torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    o, lse = fwd(q, k, v, causal=causal, sinks=sinks)
    assert o.shape == q.shape and o.dtype == torch.bfloat16
    assert lse.shape == (batch, heads, seq_len) and lse.dtype == torch.float32
    torch.testing.assert_close(o.float(), sdpa_ref(q, k, v, causal, sinks), **TOL)
    torch.testing.assert_close(lse, sdpa_ref_lse(q, k, v, causal, sinks), **LSE_TOL)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dim", [64, 96, 128, 256])
def test_fwd_matches_sdpa(dim, causal):
    _run(batch=2, heads=4, seq_len=512, dim=dim, causal=causal)


@pytest.mark.parametrize("causal", [True, False])
def test_fwd_long_seq(causal):
    _run(batch=1, heads=4, seq_len=2048, dim=128, causal=causal)


def _sinks(heads):
    return torch.randn(heads, device="cuda", dtype=torch.float32, generator=torch.Generator("cuda").manual_seed(7))


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dim", [64, 128])
def test_fwd_sink_matches_ref(dim, causal):
    _run(batch=2, heads=4, seq_len=512, dim=dim, causal=causal, sinks=_sinks(4))


def test_fwd_sink_d256():
    _run(batch=2, heads=4, seq_len=512, dim=256, causal=True, sinks=_sinks(4))


def test_fwd_large_negative_sink_is_noop():
    """exp2(-1e4 * log2e - m * scale) underflows to 0, so the sink drops out exactly."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 512, 4, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    o, lse = fwd(q, k, v, causal=True)
    o_s, lse_s = fwd(q, k, v, causal=True, sinks=torch.full((4,), -1e4, device="cuda"))
    assert torch.equal(o, o_s)  # bit-for-bit
    torch.testing.assert_close(lse_s, lse, atol=1e-6, rtol=0)
