"""GPU tests for the sliding-window MQA forward (output + LSE) against the golden reference.

``o`` is checked with the FlashAttention acceptance criterion (the shared
``dense_attn.ref.assert_within_2x_torch``): its max-abs error against the fp32
``golden(..., level="window", window=W)`` must be <= 2x the error torch's own bf16
pass over the same inputs makes, plus a 1e-5 floor. ``lse`` never leaves fp32 in
the kernel, so it gets a tight fixed fp32 tolerance instead (observed ~1.4e-6).

Beyond `golden` there are three independent cross-checks:

- ``gpt_oss_ref(window=W)`` -- transformers' real gpt-oss ``eager_attention_forward``
  with ``sliding_window_causal_mask_function``, which shares no code with golden's
  own ``_attend``.
- ``window >= seq_len`` must reproduce ``latent_attn.fwd`` (plain causal) to
  within one bf16 ulp: the band mask is vacuous there, and the residual difference
  is only the two packages' different tile choices reordering the online softmax.
- the vendored DSV4.1 model itself: the tiny 2-layer config's layer 0 is a
  ``compress_ratio == 0`` sliding-window layer, so the exact tensors it hands to
  ``eager_attention_forward`` are captured and replayed through ``swa_attn``.
"""

import pytest
import torch

from mzoo.layers.attn.dense_attn.ref import assert_within_2x_torch
from mzoo.layers.attn.golden_ref import golden, gpt_oss_ref
from mzoo.layers.attn.swa_attn.fwd import fwd

LSE_TOL = dict(atol=2e-3, rtol=2e-3)
WINDOW = 128  # the model's sliding_window


def _inputs(batch, heads, seq_len, dim, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(batch, seq_len, 1, dim, device="cuda", dtype=torch.bfloat16)
    return q, kv


def _run(batch, heads, seq_len, dim, window=WINDOW, sinks=None):
    q, kv = _inputs(batch, heads, seq_len, dim)
    o, lse = fwd(q, kv, window=window, sinks=sinks)
    assert o.shape == q.shape and o.dtype == torch.bfloat16
    assert lse.shape == (batch, heads, seq_len) and lse.dtype == torch.float32
    ref, ref_lse = golden(q, kv, level="window", window=window, sinks=sinks)
    o_bf16, _ = golden(q, kv, level="window", window=window, sinks=sinks, dtype=torch.bfloat16)
    assert_within_2x_torch(o, ref, o_bf16, "o")
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


@pytest.mark.parametrize("heads", [4, 16, 64])
@pytest.mark.parametrize("dim", [64, 96, 128])
def test_fwd_matches_golden(dim, heads):
    _run(batch=2, heads=heads, seq_len=512, dim=dim)


@pytest.mark.parametrize("heads", [4, 64])
def test_fwd_dim256(heads):
    """D=256 is smem-capped shape support (see README -> Known issues); kept small."""
    _run(batch=1, heads=heads, seq_len=512, dim=256)


@pytest.mark.parametrize("window", [1, 63, 100])
def test_fwd_window_not_tile_aligned(window):
    """``window`` is a per-row edge, not a tile count: 100 is not a multiple of any
    ``block_N``, 63 straddles one tile, 1 is self-attention only."""
    _run(batch=2, heads=16, seq_len=512, dim=128, window=window)


@pytest.mark.parametrize("window", [512, 4096])
def test_fwd_window_ge_seqlen_matches_latent(window):
    """``window >= seq_len`` degenerates to plain causal: the band mask never fires, so
    the result must equal ``latent_attn`` up to the online softmax's summation order
    (observed 2^-8 = one bf16 ulp at |o| ~ 1; lse to 9.5e-7)."""
    from mzoo.layers.attn.latent_attn.fwd import fwd as latent_fwd

    q, kv = _inputs(1, 16, 512, 128)
    o, lse = fwd(q, kv, window=window)
    o_lat, lse_lat = latent_fwd(q, kv, causal=True)
    assert (o.float() - o_lat.float()).abs().max().item() <= 2.0**-8  # one bf16 ulp at |o| ~ 1
    torch.testing.assert_close(lse, lse_lat, atol=1e-5, rtol=0)


def test_fwd_long_seq():
    """S=4096 with W=128: the band makes this O(S*W), 16x less work than full causal."""
    _run(batch=1, heads=64, seq_len=4096, dim=128)


def _sinks(heads):
    return torch.randn(heads, device="cuda", dtype=torch.float32, generator=torch.Generator("cuda").manual_seed(7))


@pytest.mark.parametrize("dim", [64, 96, 128, 256])
def test_fwd_sink_matches_golden(dim):
    _run(batch=2, heads=16, seq_len=512, dim=dim, sinks=_sinks(16))


def test_fwd_sink_many_heads():
    """H=64 with a sink: each packed row must pick up its own head's sink (row % H)."""
    _run(batch=1, heads=64, seq_len=512, dim=128, sinks=_sinks(64))


def test_fwd_large_negative_sink_is_noop():
    """exp2(-1e4 * log2e - m * scale) underflows to 0, so the sink drops out exactly.
    Also pins that the finite ``neg_floor`` rowmax init never leaks into the sink column."""
    q, kv = _inputs(2, 16, 512, 128)
    o, lse = fwd(q, kv, window=WINDOW)
    o_s, lse_s = fwd(q, kv, window=WINDOW, sinks=torch.full((16,), -1e4, device="cuda"))
    assert torch.equal(o, o_s)  # bit-for-bit
    torch.testing.assert_close(lse_s, lse, atol=1e-6, rtol=0)


@pytest.mark.parametrize("use_sinks", [False, True])
def test_fwd_matches_gpt_oss(use_sinks):
    """Cross-check against transformers' real gpt-oss eager path with its own
    ``sliding_window_causal_mask_function``, independent of golden's math."""
    heads = 16
    sinks = _sinks(heads) if use_sinks else None
    q, kv = _inputs(batch=2, heads=heads, seq_len=512, dim=128)
    o, _ = fwd(q, kv, window=WINDOW, sinks=sinks)
    ref = gpt_oss_ref(q, kv, causal=True, sinks=sinks, window=WINDOW, dtype=torch.float32)
    o_bf16, _ = golden(q, kv, level="window", window=WINDOW, sinks=sinks, dtype=torch.bfloat16)
    assert_within_2x_torch(o, ref, o_bf16, "o")


def test_fwd_rejects_non_causal():
    """Documented restriction: the model only ever has the causal band."""
    q, kv = _inputs(1, 16, 512, 128)
    with pytest.raises(AssertionError, match="causal-only"):
        fwd(q, kv, window=WINDOW, causal=False)
