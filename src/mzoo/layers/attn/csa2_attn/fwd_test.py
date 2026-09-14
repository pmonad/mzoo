"""GPU tests for the gathered (window + top-k main) MQA forward against the golden reference.

``o`` uses the FlashAttention acceptance criterion (``dense_attn.ref.assert_within_2x_torch``):
max-abs error against the fp32 ``golden(level="sparse", window=W, main_kv=..., compress_ratio=m,
indices=...)`` must be <= 2x the error torch's own bf16 pass makes **on the same indices**.
``lse`` never leaves fp32 in the kernel, so it gets a tight fixed fp32 tolerance.

Indices are produced two ways, both honouring the ``-1``-is-empty contract:
``golden_ref.indexer_scores`` + ``topk_indices`` on random indexer inputs (the torch
definition), and the real ``indexer.attn`` kernel package (ticket 0006). Structural checks
beyond the numeric one:

- all indices ``-1`` (nothing selected) and ``G == 0`` must be **bit-for-bit** ``swa_attn.fwd``;
- indices listing *every* group-causally visible entry in group order must be **bit-for-bit**
  ``csa_attn.fwd`` (the dense main loop) -- same entries, same order, same online-softmax
  accumulation per row, so the two kernels are numerically the same program.
"""

import pytest
import torch

from mzoo.layers.attn.csa2_attn.fwd import fwd
from mzoo.layers.attn.dense_attn.ref import assert_within_2x_torch
from mzoo.layers.attn.golden_ref import golden, indexer_scores, topk_indices

LSE_TOL = dict(atol=2e-3, rtol=2e-3)
WINDOW = 128  # the model's sliding_window
RATIO = 2
HI, DI = 4, 64  # indexer heads / head dim used to synthesise the index sets


def _inputs(batch, heads, seq_len, dim, groups, seed=0):
    g = torch.Generator("cuda").manual_seed(seed)
    kw = dict(device="cuda", dtype=torch.bfloat16, generator=g)
    q = torch.randn(batch, seq_len, heads, dim, **kw)
    kv = torch.randn(batch, seq_len, 1, dim, **kw)
    main_kv = torch.randn(batch, groups, 1, dim, **kw)
    return q, kv, main_kv


def _indices(batch, seq_len, groups, ratio, topk, *, seed=1, via_kernel=False):
    """A valid group-causal top-k set per token: random indexer inputs -> scores -> top-k.

    ``via_kernel=True`` routes through the real ``indexer`` package (int32) instead of the
    torch definition in ``golden_ref`` (int64); both obey the same ``-1`` = empty contract.
    """
    g = torch.Generator("cuda").manual_seed(seed)
    kw = dict(device="cuda", generator=g)
    q = torch.randn(batch, seq_len, HI, DI, dtype=torch.bfloat16, **kw)
    k = torch.randn(batch, groups, DI, dtype=torch.bfloat16, **kw)
    w = torch.randn(batch, seq_len, HI, dtype=torch.float32, **kw) * HI**-0.5
    if via_kernel:
        from mzoo.layers.attn.indexer.attn import attn as indexer_attn
        return indexer_attn(q, k, w, compress_ratio=ratio, topk=topk)
    return topk_indices(indexer_scores(q.float(), k.float(), w, ratio), topk)


def _check(q, kv, main_kv, idx, ratio, window=WINDOW, sinks=None):
    o, lse = fwd(q, kv, main_kv, idx, window=window, compress_ratio=ratio, sinks=sinks)
    batch, seq_len, heads, _ = q.shape
    assert o.shape == q.shape and o.dtype == torch.bfloat16
    assert lse.shape == (batch, heads, seq_len) and lse.dtype == torch.float32
    kw = dict(level="sparse", window=window, main_kv=main_kv, compress_ratio=ratio,
              indices=idx.long(), sinks=sinks)  # golden scatters, so it wants int64
    ref, ref_lse = golden(q, kv, **kw)
    o_bf16, _ = golden(q, kv, dtype=torch.bfloat16, **kw)
    assert_within_2x_torch(o, ref, o_bf16, "o")
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
    return o, lse


def _run(batch, heads, seq_len, dim, ratio=RATIO, topk=64, window=WINDOW, groups=None,
         sinks=None, via_kernel=False):
    groups = seq_len // ratio if groups is None else groups
    q, kv, main_kv = _inputs(batch, heads, seq_len, dim, groups)
    idx = _indices(batch, seq_len, groups, ratio, topk, via_kernel=via_kernel)
    return _check(q, kv, main_kv, idx, ratio, window=window, sinks=sinks)


@pytest.mark.parametrize("heads", [4, 16, 64])
@pytest.mark.parametrize("dim", [64, 96, 128])
def test_fwd_matches_golden(dim, heads):
    """``T_q = 64 // heads`` tokens per block, so ``heads`` also picks the gather layout."""
    _run(batch=2, heads=heads, seq_len=512, dim=dim)


@pytest.mark.parametrize("heads", [4, 64])
def test_fwd_dim256(heads):
    """D=256 is smem-capped shape support (see README -> Known issues); kept small."""
    _run(batch=1, heads=heads, seq_len=512, dim=256, ratio=4)


@pytest.mark.parametrize("topk", [64, 100, 512])
def test_fwd_topk(topk):
    """``topk=100`` is a multiple of no ``block_N`` (padded with ``-1`` slots in the caller);
    ``topk=512`` is the model's and exceeds what early tokens can see."""
    _run(batch=1, heads=64, seq_len=512, dim=128, ratio=RATIO, topk=topk)


@pytest.mark.parametrize("ratio", [2, 4])
def test_fwd_compress_ratios(ratio):
    """``ratio`` only shapes the index sets here -- the kernel never re-applies group-causality."""
    _run(batch=2, heads=16, seq_len=512, dim=128, ratio=ratio, topk=100)


def test_fwd_groups_not_power_of_two():
    """``G=100 < topk`` and not a multiple of any tile: every row runs out of visible entries,
    so the second half of each index list is ``-1``. ``G`` needs no padding in this package."""
    idx = _indices(2, 512, 100, RATIO, 512)
    assert (idx < 0).any() and (idx.max() == 99)
    q, kv, main_kv = _inputs(2, 16, 512, 128, groups=100)
    _check(q, kv, main_kv, idx, RATIO)


def test_fwd_topk_exceeds_visible_groups():
    """Early tokens see fewer than ``topk`` groups, so their lists are ``-1``-padded; token 0
    sees none at all and is window-only."""
    idx = _indices(1, 512, 1024, 4, 512)
    assert (idx[:, 0] < 0).all(), "token 0 has no group-causally visible entry"
    q, kv, main_kv = _inputs(1, 64, 512, 128, groups=1024)
    _check(q, kv, main_kv, idx, 4)


def test_fwd_from_indexer_package():
    """The real producer of ``Indices`` (ticket 0006), int32, end to end."""
    o, _ = _run(batch=2, heads=16, seq_len=512, dim=128, topk=64, via_kernel=True)
    assert torch.isfinite(o.float()).all()


def test_fwd_long_seq():
    """S=4096 with G=16384 (4x the sequence) and topk=512: the gather is flat in G."""
    _run(batch=1, heads=4, seq_len=4096, dim=128, ratio=1, topk=512, groups=16384,
         window=WINDOW)


@pytest.mark.parametrize("dim", [64, 128])
def test_fwd_sink_matches_golden(dim):
    """The sink is one denominator-only column over the *combined* window+gathered logits."""
    sinks = torch.randn(16, device="cuda", dtype=torch.float32,
                        generator=torch.Generator("cuda").manual_seed(7))
    _run(batch=2, heads=16, seq_len=512, dim=dim, topk=100, sinks=sinks)


@pytest.mark.parametrize("use_sinks", [False, True])
def test_fwd_all_empty_is_swa(use_sinks):
    """A row whose indices are all ``-1`` still sees its window, so it stays finite -- and the
    whole kernel is then *bit-for-bit* ``swa_attn``, with and without a sink."""
    from mzoo.layers.attn.swa_attn.fwd import fwd as swa_fwd

    q, kv, main_kv = _inputs(2, 16, 512, 128, groups=256)
    sinks = torch.randn(16, device="cuda", dtype=torch.float32) if use_sinks else None
    idx = torch.full((2, 512, 64), -1, device="cuda", dtype=torch.int32)
    o, lse = fwd(q, kv, main_kv, idx, window=WINDOW, compress_ratio=RATIO, sinks=sinks)
    o_swa, lse_swa = swa_fwd(q, kv, window=WINDOW, sinks=sinks)
    assert torch.isfinite(o.float()).all() and torch.isfinite(lse).all()
    assert torch.equal(o, o_swa) and torch.equal(lse, lse_swa)


def test_fwd_no_groups_is_swa():
    """``G == 0`` drops the gather at compile time: bit-for-bit ``swa_attn`` again."""
    from mzoo.layers.attn.swa_attn.fwd import fwd as swa_fwd

    q, kv, _ = _inputs(2, 16, 512, 128, groups=0)
    main_kv = torch.empty(2, 0, 1, 128, device="cuda", dtype=torch.bfloat16)
    idx = torch.full((2, 512, 64), -1, device="cuda", dtype=torch.int64)
    o, lse = fwd(q, kv, main_kv, idx, window=WINDOW, compress_ratio=RATIO)
    o_swa, lse_swa = swa_fwd(q, kv, window=WINDOW)
    assert torch.equal(o, o_swa) and torch.equal(lse, lse_swa)


def test_fwd_all_visible_is_csa_attn():
    """Feature-off case: hand the kernel *every* group-causally visible entry, in group order
    (``-1`` for the rest), and it must reproduce ``csa_attn``'s dense main loop bit-for-bit --
    same entries, same order, so the per-row online softmax is the same sequence of operations."""
    from mzoo.layers.attn.csa_attn.fwd import fwd as csa_fwd

    batch, seq_len, heads, dim, groups = 2, 512, 16, 128, 256
    q, kv, main_kv = _inputs(batch, heads, seq_len, dim, groups)
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
    t = torch.arange(seq_len, device="cuda").view(-1, 1)
    g = torch.arange(groups, device="cuda").view(1, -1)
    visible = g < ((t + 1) // RATIO)  # the dense group-causal rule, as an index list
    idx = torch.where(visible, g.expand_as(visible), torch.full_like(g.expand_as(visible), -1))
    idx = idx.unsqueeze(0).expand(batch, -1, -1).contiguous().int()
    o, lse = fwd(q, kv, main_kv, idx, window=WINDOW, compress_ratio=RATIO, sinks=sinks)
    o_csa, lse_csa = csa_fwd(q, kv, main_kv, window=WINDOW, compress_ratio=RATIO, sinks=sinks)
    assert torch.equal(o, o_csa) and torch.equal(lse, lse_csa)


def test_fwd_rejects_non_causal():
    """Documented restriction, inherited: the model only ever has the causal band."""
    q, kv, main_kv = _inputs(1, 16, 512, 128, groups=256)
    idx = _indices(1, 512, 256, RATIO, 64)
    with pytest.raises(AssertionError, match="causal-only"):
        fwd(q, kv, main_kv, idx, window=WINDOW, compress_ratio=RATIO, causal=False)
