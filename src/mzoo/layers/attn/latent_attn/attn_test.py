"""GPU tests for latent_attn gradients against the fp32 golden reference's autograd.

``o``, ``dq``, ``dkv`` and ``dsinks`` are all checked with the FlashAttention
acceptance criterion (the shared ``dense_attn.ref.assert_within_2x_torch``): each
tensor's max-abs error against the fp32 ``golden(..., level="latent")`` autograd
result must be <= 2x the error torch's own bf16 pass makes on the same inputs and
the same ``do``.

``dkv`` is the MQA reduction: one ``[B, S, 1, D]`` gradient summed over all H
query heads, and since K == V it is ``dK + dV``.
"""

import pytest
import torch

from mzoo.layers.attn.dense_attn.ref import assert_within_2x_torch
from mzoo.layers.attn.golden_ref import golden
from mzoo.layers.attn.latent_attn.attn import attn


def _leaves(batch, heads, seq_len, dim, use_sink, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(batch, seq_len, 1, dim, device="cuda", dtype=torch.bfloat16)
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32) if use_sink else None
    do = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)
    return q, kv, sinks, do


def _grads(batch, heads, seq_len, dim, causal, use_sink=False):
    """Returns ``(o, o_ref, o_bf16, got, ref_grads, bf16_grads)``; grad lists are
    ``[dq, dkv]`` plus ``dsinks`` when ``use_sink``."""
    q, kv, sinks, do = _leaves(batch, heads, seq_len, dim, use_sink)

    def run(fn, dtype=None):
        ins = [q.detach().clone(), kv.detach().clone()]
        if dtype is torch.float32:
            ins = [x.float() for x in ins]
        ins = [x.requires_grad_() for x in ins]
        sk = sinks.detach().clone().requires_grad_() if use_sink else None
        o = fn(ins[0], ins[1], sk)
        o.backward(do.float() if dtype is torch.float32 else do)
        return o, [x.grad for x in ins] + ([sk.grad] if use_sink else [])

    o, got = run(lambda a, b, s: attn(a, b, causal=causal, sinks=s))
    o_ref, ref_grads = run(lambda a, b, s: golden(a, b, level="latent", causal=causal, sinks=s)[0], torch.float32)
    o_bf16, bf16_grads = run(
        lambda a, b, s: golden(a, b, level="latent", causal=causal, sinks=s, dtype=torch.bfloat16)[0])
    return o, o_ref, o_bf16, got, ref_grads, bf16_grads


def _check(names, got, ref_grads, bf16_grads):
    for name, g, r, t in zip(names, got, ref_grads, bf16_grads):
        assert_within_2x_torch(g, r, t, name)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dim", [64, 128])
def test_grads_match_golden(dim, causal):
    o, o_ref, o_bf16, got, ref_grads, bf16_grads = _grads(batch=2, heads=16, seq_len=512, dim=dim, causal=causal)
    assert_within_2x_torch(o, o_ref, o_bf16, "o")
    assert got[0].dtype == torch.bfloat16 and got[1].dtype == torch.bfloat16
    assert got[1].shape == (2, 512, 1, dim)  # dkv reduced over all H heads
    _check(["dq", "dkv"], got, ref_grads, bf16_grads)


@pytest.mark.parametrize("heads", [4, 64])
def test_grads_head_counts(heads):
    """H must divide the bwd ``block_N`` (64): H=4 -> 16 tokens/tile, H=64 -> 1 token/tile."""
    _, _, _, got, ref_grads, bf16_grads = _grads(batch=1, heads=heads, seq_len=512, dim=128, causal=True)
    _check(["dq", "dkv"], got, ref_grads, bf16_grads)


def test_grads_long_seq():
    _, _, _, got, ref_grads, bf16_grads = _grads(batch=1, heads=64, seq_len=4096, dim=128, causal=True)
    _check(["dq", "dkv"], got, ref_grads, bf16_grads)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dim", [64, 128])
def test_grads_with_sink_match_golden(dim, causal):
    o, o_ref, o_bf16, got, ref_grads, bf16_grads = _grads(batch=2, heads=16, seq_len=512, dim=dim,
                                                          causal=causal, use_sink=True)
    assert_within_2x_torch(o, o_ref, o_bf16, "o")
    assert got[2].dtype == torch.float32 and got[2].shape == (16,)
    _check(["dq", "dkv", "dsinks"], got, ref_grads, bf16_grads)


def test_backward_saves_only_q_kv_o_lse():
    """FA2 recompute: forward must save exactly (q, kv, o, lse), never an S x S matrix."""
    batch, heads, seq_len, dim = 1, 16, 1024, 64
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16).requires_grad_()
    kv = torch.randn(batch, seq_len, 1, dim, device="cuda", dtype=torch.bfloat16).requires_grad_()
    o = attn(q, kv)

    saved = o.grad_fn.saved_tensors
    assert len(saved) == 4
    assert saved[0].shape == saved[2].shape == (batch, seq_len, heads, dim)
    assert saved[1].shape == (batch, seq_len, 1, dim)
    assert saved[3].shape == (batch, heads, seq_len)

    total_elems = sum(t.numel() for t in saved)
    expect = (2 * heads + 1) * batch * seq_len * dim + batch * heads * seq_len
    ssq_elems = batch * heads * seq_len * seq_len
    assert total_elems == expect
    assert total_elems < ssq_elems  # would be off by ~1000x if an S x S matrix were saved

    # with a sink the saved set grows by exactly the [H] vector
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
    saved_s = attn(q, kv, sinks=sinks).grad_fn.saved_tensors
    assert len(saved_s) == 5
    assert saved_s[4].shape == (heads,)
    assert sum(t.numel() for t in saved_s) == expect + heads
