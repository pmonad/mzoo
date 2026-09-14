"""GPU tests for the two-source (window + compressed) MQA forward against the golden reference.

``o`` uses the FlashAttention acceptance criterion (``dense_attn.ref.assert_within_2x_torch``):
max-abs error against the fp32 ``golden(level="compressed", window=W, main_kv=..., compress_ratio=m)``
must be <= 2x the error torch's own bf16 pass makes on the same inputs. ``lse`` never
leaves fp32 in the kernel, so it gets a tight fixed fp32 tolerance.

Two structural checks beyond the numeric one:

- ``G == 0`` must be **bit-for-bit** ``swa_attn.fwd``: the second loop is dropped at
  compile time, so the two kernels are the same program.
- ``compress_ratio=1`` with ``main_kv = kv`` is the ticket's "uncompressed second
  source" case -- group-causal ``g < t + 1`` is then exactly token-causal, so the
  query attends the same latent twice, once banded and once causally.
"""

import pytest
import torch

from mzoo.layers.attn.csa_attn.fwd import fwd
from mzoo.layers.attn.dense_attn.ref import assert_within_2x_torch
from mzoo.layers.attn.golden_ref import golden

LSE_TOL = dict(atol=2e-3, rtol=2e-3)
WINDOW = 128  # the model's sliding_window
RATIO = 2


def _inputs(batch, heads, seq_len, dim, groups, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(batch, seq_len, 1, dim, device="cuda", dtype=torch.bfloat16)
    main_kv = torch.randn(batch, groups, 1, dim, device="cuda", dtype=torch.bfloat16)
    return q, kv, main_kv


def _run(batch, heads, seq_len, dim, ratio=RATIO, window=WINDOW, groups=None, sinks=None):
    groups = seq_len // ratio if groups is None else groups
    q, kv, main_kv = _inputs(batch, heads, seq_len, dim, groups)
    o, lse = fwd(q, kv, main_kv, window=window, compress_ratio=ratio, sinks=sinks)
    assert o.shape == q.shape and o.dtype == torch.bfloat16
    assert lse.shape == (batch, heads, seq_len) and lse.dtype == torch.float32
    kw = dict(level="compressed", window=window, main_kv=main_kv, compress_ratio=ratio, sinks=sinks)
    ref, ref_lse = golden(q, kv, **kw)
    o_bf16, _ = golden(q, kv, dtype=torch.bfloat16, **kw)
    assert_within_2x_torch(o, ref, o_bf16, "o")
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


@pytest.mark.parametrize("heads", [4, 16, 64])
@pytest.mark.parametrize("dim", [64, 96, 128])
def test_fwd_matches_golden(dim, heads):
    _run(batch=2, heads=heads, seq_len=512, dim=dim)


@pytest.mark.parametrize("ratio", [1, 2, 4])
def test_fwd_compress_ratios(ratio):
    """``ratio`` sets both the visibility rule and ``G = S // ratio``; it is a jit constant."""
    _run(batch=2, heads=16, seq_len=512, dim=128, ratio=ratio)


@pytest.mark.parametrize("heads", [4, 64])
def test_fwd_dim256(heads):
    """D=256 is smem-capped shape support (see README -> Known issues); kept small."""
    _run(batch=1, heads=heads, seq_len=512, dim=256, ratio=4)


def test_fwd_window_ge_seqlen():
    """``window >= seq_len`` is plain causal on the raw latent, still plus the main source."""
    _run(batch=2, heads=16, seq_len=512, dim=128, window=512)


def test_fwd_ratio1_main_is_kv():
    """The ticket's uncompressed special case: ``ratio=1`` with ``main_kv = kv`` means
    group-causal ``g < t + 1`` == token-causal, i.e. the same latent attended twice."""
    torch.manual_seed(0)
    b, s, h, d = 2, 512, 16, 128
    q = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(b, s, 1, d, device="cuda", dtype=torch.bfloat16)
    o, lse = fwd(q, kv, kv, window=WINDOW, compress_ratio=1)
    kw = dict(level="compressed", window=WINDOW, main_kv=kv, compress_ratio=1)
    ref, ref_lse = golden(q, kv, **kw)
    o_bf16, _ = golden(q, kv, dtype=torch.bfloat16, **kw)
    assert_within_2x_torch(o, ref, o_bf16, "o")
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


@pytest.mark.parametrize("use_sinks", [False, True])
def test_fwd_no_groups_is_swa(use_sinks):
    """``G == 0`` drops the main loop at compile time, so this must be *bit-for-bit*
    ``swa_attn`` -- not merely within a bf16 ulp."""
    from mzoo.layers.attn.swa_attn.fwd import fwd as swa_fwd

    q, kv, _ = _inputs(2, 16, 512, 128, groups=0)
    main_kv = torch.empty(2, 0, 1, 128, device="cuda", dtype=torch.bfloat16)
    sinks = torch.randn(16, device="cuda", dtype=torch.float32) if use_sinks else None
    o, lse = fwd(q, kv, main_kv, window=WINDOW, compress_ratio=4, sinks=sinks)
    o_swa, lse_swa = swa_fwd(q, kv, window=WINDOW, sinks=sinks)
    assert torch.equal(o, o_swa)
    assert torch.equal(lse, lse_swa)


def test_fwd_groups_not_tile_aligned():
    """``G`` is zero-padded to the tile in the caller and masked with ``g < min(G, ...)``;
    100 is a multiple of no ``block_N``, and here ``G < S // ratio`` so the ``min`` bites."""
    _run(batch=2, heads=16, seq_len=512, dim=128, ratio=2, groups=100)


def test_fwd_long_seq():
    """S=4096, G=1024: the band is O(S*W) and the main source O(S*G/2)."""
    _run(batch=1, heads=64, seq_len=4096, dim=128, ratio=4)


def _sinks(heads):
    return torch.randn(heads, device="cuda", dtype=torch.float32, generator=torch.Generator("cuda").manual_seed(7))


@pytest.mark.parametrize("dim", [64, 128])
def test_fwd_sink_matches_golden(dim):
    """The sink is one denominator-only column over the *combined* window+main logits."""
    _run(batch=2, heads=16, seq_len=512, dim=dim, sinks=_sinks(16))


def test_fwd_large_negative_sink_is_noop():
    """exp2(-1e4 * log2e - m * scale) underflows to 0, so the sink drops out exactly --
    and the finite ``neg_floor`` rowmax never leaks into the sink column."""
    q, kv, main_kv = _inputs(2, 16, 512, 128, groups=256)
    kw = dict(window=WINDOW, compress_ratio=RATIO)
    o, lse = fwd(q, kv, main_kv, **kw)
    o_s, lse_s = fwd(q, kv, main_kv, sinks=torch.full((16,), -1e4, device="cuda"), **kw)
    assert torch.equal(o, o_s)
    torch.testing.assert_close(lse_s, lse, atol=1e-6, rtol=0)


def test_fwd_rejects_non_causal():
    """Documented restriction, inherited: the model only ever has the causal band."""
    q, kv, main_kv = _inputs(1, 16, 512, 128, groups=256)
    with pytest.raises(AssertionError, match="causal-only"):
        fwd(q, kv, main_kv, window=WINDOW, compress_ratio=RATIO, causal=False)
