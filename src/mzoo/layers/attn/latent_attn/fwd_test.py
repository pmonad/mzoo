"""GPU tests for the shared-latent MQA forward (output + LSE) against the golden reference.

``o`` is checked with the FlashAttention acceptance criterion (the shared
``dense_attn.ref.assert_within_2x_torch``): its max-abs error against the fp32
``golden(..., level="latent")`` must be <= 2x the error torch's own bf16 pass over
the same inputs makes, plus a 1e-5 floor.

``lse`` never leaves fp32 in the kernel -- only the logits feeding it are bf16
products -- so it gets a tight fixed fp32 tolerance instead (observed ~1.4e-6).

The last test is the cross-package check: ``latent_attn.fwd(q, kv)`` against
``dense_attn.fwd(q, kv.expand(...), kv.expand(...))``, i.e. the previous package
fed the same latent replicated to H heads.
"""

import pytest
import torch

from mzoo.layers.attn.dense_attn.ref import assert_within_2x_torch
from mzoo.layers.attn.golden_ref import golden, gpt_oss_ref
from mzoo.layers.attn.latent_attn.fwd import fwd

LSE_TOL = dict(atol=2e-3, rtol=2e-3)


def _inputs(batch, heads, seq_len, dim, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(batch, seq_len, 1, dim, device="cuda", dtype=torch.bfloat16)
    return q, kv


def _run(batch, heads, seq_len, dim, causal, sinks=None):
    q, kv = _inputs(batch, heads, seq_len, dim)
    o, lse = fwd(q, kv, causal=causal, sinks=sinks)
    assert o.shape == q.shape and o.dtype == torch.bfloat16
    assert lse.shape == (batch, heads, seq_len) and lse.dtype == torch.float32
    ref, ref_lse = golden(q, kv, level="latent", causal=causal, sinks=sinks)
    o_bf16, _ = golden(q, kv, level="latent", causal=causal, sinks=sinks, dtype=torch.bfloat16)
    assert_within_2x_torch(o, ref, o_bf16, "o")
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("heads", [4, 16, 64])
@pytest.mark.parametrize("dim", [64, 96, 128])
def test_fwd_matches_golden(dim, heads, causal):
    _run(batch=2, heads=heads, seq_len=512, dim=dim, causal=causal)


@pytest.mark.parametrize("heads", [4, 64])
def test_fwd_dim256(heads):
    """D=256 is smem-capped shape support: ``block_M=128`` is the only tile that fits, so
    ``T_q = 128 // H`` (2 tokens at H=64). Kept small on purpose -- one batch, causal only."""
    _run(batch=1, heads=heads, seq_len=512, dim=256, causal=True)


@pytest.mark.parametrize("causal", [True, False])
def test_fwd_long_seq(causal):
    _run(batch=1, heads=64, seq_len=4096, dim=128, causal=causal)


def _sinks(heads):
    return torch.randn(heads, device="cuda", dtype=torch.float32, generator=torch.Generator("cuda").manual_seed(7))


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dim", [64, 96, 128, 256])
def test_fwd_sink_matches_golden(dim, causal):
    _run(batch=2, heads=16, seq_len=512, dim=dim, causal=causal, sinks=_sinks(16))


def test_fwd_sink_many_heads():
    """H=64 with a sink: each packed row must pick up its own head's sink (row % H)."""
    _run(batch=1, heads=64, seq_len=512, dim=128, causal=True, sinks=_sinks(64))


def test_fwd_large_negative_sink_is_noop():
    """exp2(-1e4 * log2e - m * scale) underflows to 0, so the sink drops out exactly."""
    q, kv = _inputs(2, 16, 512, 128)
    o, lse = fwd(q, kv, causal=True)
    o_s, lse_s = fwd(q, kv, causal=True, sinks=torch.full((16,), -1e4, device="cuda"))
    assert torch.equal(o, o_s)  # bit-for-bit
    torch.testing.assert_close(lse_s, lse, atol=1e-6, rtol=0)


@pytest.mark.parametrize("causal", [True, False])
def test_fwd_matches_dense_attn_with_expanded_kv(causal):
    """Cross-package: the previous package fed the same latent replicated to H heads."""
    from mzoo.layers.attn.dense_attn.fwd import fwd as dense_fwd

    batch, heads, seq_len, dim = 2, 16, 512, 128
    q, kv = _inputs(batch, heads, seq_len, dim)
    k = kv.expand(batch, seq_len, heads, dim).contiguous()
    o, lse = fwd(q, kv, causal=causal)
    o_dense, lse_dense = dense_fwd(q, k, k, causal=causal)
    ref, ref_lse = golden(q, kv, level="latent", causal=causal)
    assert_within_2x_torch(o, ref, o_dense, "o vs dense_attn")  # <= 2x dense_attn's own error
    torch.testing.assert_close(lse, lse_dense, **LSE_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


@pytest.mark.parametrize("use_sinks", [False, True])
@pytest.mark.parametrize("dim", [64, 128])
def test_fwd_matches_gpt_oss(dim, use_sinks):
    """Cross-check against gpt_oss_ref (transformers' real gpt-oss eager path, independent of
    golden's own math) at the FlashAttention criterion. The bf16 baseline stays golden's
    fp32-softmax bf16 pass, since gpt-oss softmaxes in the logits dtype and would be noisier."""
    heads = 16
    sinks = _sinks(heads) if use_sinks else None
    q, kv = _inputs(batch=2, heads=heads, seq_len=512, dim=dim)
    o, _ = fwd(q, kv, causal=True, sinks=sinks)
    ref = gpt_oss_ref(q, kv, causal=True, sinks=sinks, dtype=torch.float32)
    o_bf16, _ = golden(q, kv, level="latent", causal=True, sinks=sinks, dtype=torch.bfloat16)
    assert_within_2x_torch(o, ref, o_bf16, "o")
