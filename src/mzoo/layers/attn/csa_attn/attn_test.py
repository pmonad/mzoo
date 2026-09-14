"""GPU tests for csa_attn gradients against the fp32 golden reference's autograd.

``o``, ``dq``, ``dkv``, ``dmain_kv`` and ``dsinks`` all use the FlashAttention
acceptance criterion (``dense_attn.ref.assert_within_2x_torch``) against
``golden(level="compressed", window=W, main_kv=..., compress_ratio=m)``.

``dkv`` and ``dmain_kv`` are two separate KV-owner launches over disjoint outputs
(``bwd_kernels.bwd_kv`` and ``bwd_main.bwd_main``); both are MQA reductions over all
H heads and, since K == V, both are ``dK + dV``. ``dq`` comes from one query-owning
kernel that walks both sources. The group-causal mask matters on every side: ``lse``
normalised window + main + sink together, so an unmasked recomputed ``P`` would
renormalise against columns the forward never saw.
"""

import pytest
import torch

from mzoo.layers.attn.csa_attn.attn import attn
from mzoo.layers.attn.dense_attn.ref import assert_within_2x_torch
from mzoo.layers.attn.golden_ref import golden

WINDOW = 128  # the model's sliding_window
RATIO = 2


def _leaves(batch, heads, seq_len, dim, groups, use_sink, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(batch, seq_len, 1, dim, device="cuda", dtype=torch.bfloat16)
    main_kv = torch.randn(batch, groups, 1, dim, device="cuda", dtype=torch.bfloat16)
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32) if use_sink else None
    do = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)
    return q, kv, main_kv, sinks, do


def _grads(batch, heads, seq_len, dim, ratio=RATIO, window=WINDOW, groups=None, use_sink=False):
    """Returns ``(o, o_ref, o_bf16, got, ref_grads, bf16_grads)``; grad lists are
    ``[dq, dkv, dmain_kv]`` plus ``dsinks`` when ``use_sink``."""
    groups = seq_len // ratio if groups is None else groups
    q, kv, main_kv, sinks, do = _leaves(batch, heads, seq_len, dim, groups, use_sink)

    def run(fn, dtype=None):
        ins = [q.detach().clone(), kv.detach().clone(), main_kv.detach().clone()]
        if dtype is torch.float32:
            ins = [x.float() for x in ins]
        ins = [x.requires_grad_() for x in ins]
        sk = sinks.detach().clone().requires_grad_() if use_sink else None
        o = fn(*ins, sk)
        o.backward(do.float() if dtype is torch.float32 else do)
        return o, [x.grad for x in ins] + ([sk.grad] if use_sink else [])

    gold = lambda a, b, m, s, dt: golden(  # noqa: E731
        a, b, level="compressed", window=window, main_kv=m, compress_ratio=ratio, sinks=s, dtype=dt)[0]
    o, got = run(lambda a, b, m, s: attn(a, b, m, window=window, compress_ratio=ratio, sinks=s))
    o_ref, ref_grads = run(lambda a, b, m, s: gold(a, b, m, s, torch.float32), torch.float32)
    o_bf16, bf16_grads = run(lambda a, b, m, s: gold(a, b, m, s, torch.bfloat16))
    return o, o_ref, o_bf16, got, ref_grads, bf16_grads


def _check(names, got, ref_grads, bf16_grads):
    for name, g, r, t in zip(names, got, ref_grads, bf16_grads):
        assert_within_2x_torch(g, r, t, name)


NAMES = ["dq", "dkv", "dmain_kv"]


@pytest.mark.parametrize("dim", [64, 96, 128])
def test_grads_match_golden(dim):
    o, o_ref, o_bf16, got, ref_grads, bf16_grads = _grads(batch=2, heads=16, seq_len=512, dim=dim)
    assert_within_2x_torch(o, o_ref, o_bf16, "o")
    assert all(g.dtype == torch.bfloat16 for g in got)
    assert got[1].shape == (2, 512, 1, dim) and got[2].shape == (2, 256, 1, dim)
    _check(NAMES, got, ref_grads, bf16_grads)


@pytest.mark.parametrize("ratio", [1, 4])
def test_grads_compress_ratios(ratio):
    """``ratio`` changes which query tiles each main block walks (``t >= ratio*g0 + ratio - 1``)."""
    _, _, _, got, ref_grads, bf16_grads = _grads(batch=1, heads=16, seq_len=512, dim=128, ratio=ratio)
    _check(NAMES, got, ref_grads, bf16_grads)


def test_grads_dim256():
    """D=256 grads, kept small (H=4, B=1): smem forces 64-row tiles, so H must divide 64."""
    _, _, _, got, ref_grads, bf16_grads = _grads(batch=1, heads=4, seq_len=512, dim=256, ratio=4)
    _check(NAMES, got, ref_grads, bf16_grads)


@pytest.mark.parametrize("heads", [4, 64])
def test_grads_head_counts(heads):
    """H must divide the bwd ``block_N`` (64): H=4 -> 16 tokens/tile, H=64 -> 1 token/tile."""
    _, _, _, got, ref_grads, bf16_grads = _grads(batch=1, heads=heads, seq_len=512, dim=128)
    _check(NAMES, got, ref_grads, bf16_grads)


def test_grads_groups_not_tile_aligned():
    """``G = 100`` is zero-padded to the group tile; the pad rows must contribute nothing
    and the returned ``dmain_kv`` must be sliced back to exactly G."""
    _, _, _, got, ref_grads, bf16_grads = _grads(batch=1, heads=16, seq_len=512, dim=128, groups=100)
    assert got[2].shape == (1, 100, 1, 128)
    _check(NAMES, got, ref_grads, bf16_grads)


def test_grads_no_groups_is_swa():
    """``G == 0``: no ``bwd_main`` launch at all, ``dmain_kv`` empty, and ``dq``/``dkv``
    must match ``swa_attn``'s backward bit-for-bit (same kernels, same inputs)."""
    from mzoo.layers.attn.swa_attn.attn import attn as swa_attn

    q, kv, _, _, do = _leaves(2, 16, 512, 128, 0, use_sink=False)
    main_kv = torch.empty(2, 0, 1, 128, device="cuda", dtype=torch.bfloat16)

    def run(fn, *leaves):
        ins = [x.detach().clone().requires_grad_() for x in leaves]
        fn(*ins).backward(do)
        return [x.grad for x in ins]

    dq, dkv, dmain = run(lambda a, b, m: attn(a, b, m, window=WINDOW, compress_ratio=4), q, kv, main_kv)
    dq_s, dkv_s = run(lambda a, b: swa_attn(a, b, window=WINDOW), q, kv)
    assert dmain.shape == (2, 0, 1, 128)
    assert torch.equal(dq, dq_s) and torch.equal(dkv, dkv_s)


def test_grads_long_seq():
    _, _, _, got, ref_grads, bf16_grads = _grads(batch=1, heads=64, seq_len=4096, dim=128, ratio=4)
    _check(NAMES, got, ref_grads, bf16_grads)


@pytest.mark.parametrize("dim", [64, 128])
def test_grads_with_sink_match_golden(dim):
    o, o_ref, o_bf16, got, ref_grads, bf16_grads = _grads(batch=2, heads=16, seq_len=512, dim=dim, use_sink=True)
    assert_within_2x_torch(o, o_ref, o_bf16, "o")
    assert got[3].dtype == torch.float32 and got[3].shape == (16,)
    _check(NAMES + ["dsinks"], got, ref_grads, bf16_grads)


def test_backward_saves_only_q_kv_main_o_lse():
    """FA2 recompute: the forward saves exactly (q, kv, main_kv, o, lse), never an S x (S+G)
    matrix. ``window``/``compress_ratio`` ride on ``ctx`` as plain ints."""
    batch, heads, seq_len, dim, groups = 1, 16, 1024, 64, 256
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16).requires_grad_()
    kv = torch.randn(batch, seq_len, 1, dim, device="cuda", dtype=torch.bfloat16).requires_grad_()
    main_kv = torch.randn(batch, groups, 1, dim, device="cuda", dtype=torch.bfloat16).requires_grad_()
    o = attn(q, kv, main_kv, window=WINDOW, compress_ratio=4)

    saved = o.grad_fn.saved_tensors
    assert len(saved) == 5
    assert saved[0].shape == saved[3].shape == (batch, seq_len, heads, dim)
    assert saved[1].shape == (batch, seq_len, 1, dim)
    assert saved[2].shape == (batch, groups, 1, dim)
    assert saved[4].shape == (batch, heads, seq_len)

    total = sum(t.numel() for t in saved)
    expect = (2 * heads + 1) * batch * seq_len * dim + batch * groups * dim + batch * heads * seq_len
    assert total == expect
    assert total < batch * heads * seq_len * (seq_len + groups)  # an S x (S+G) P matrix

    sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
    saved_s = attn(q, kv, main_kv, window=WINDOW, compress_ratio=4, sinks=sinks).grad_fn.saved_tensors
    assert len(saved_s) == 6 and saved_s[5].shape == (heads,)
    assert sum(t.numel() for t in saved_s) == expect + heads
