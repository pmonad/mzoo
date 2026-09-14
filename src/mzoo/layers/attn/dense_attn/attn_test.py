"""GPU tests for dense_attn gradients against fp32 SDPA autograd.

``o``, dq/dk/dv and dsinks are all checked with the FlashAttention acceptance
criterion (see ``ref.assert_within_2x_torch``): each tensor's max-abs error against
the fp32-autograd reference must be <= 2x the error torch's own bf16 kernel makes
against that same reference, on the same inputs and the same ``do``.
"""

import pytest
import torch

from mzoo.layers.attn.dense_attn.attn import attn
from mzoo.layers.attn.dense_attn.ref import assert_within_2x_torch, sdpa_ref, torch_bf16_ref


def _grads(batch, heads, seq_len, dim, causal):
    torch.manual_seed(0)
    qkv = [torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
    do = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)

    o = attn(*qkv, causal=causal)
    o.backward(do)
    got = [x.grad for x in qkv]

    fp32_in = [x.detach().float().requires_grad_() for x in qkv]
    o_fp32 = sdpa_ref(*fp32_in, causal)
    o_fp32.backward(do.float())

    bf16_in = [x.detach().clone().requires_grad_() for x in qkv]
    o_bf16 = torch_bf16_ref(*bf16_in, causal)
    o_bf16.backward(do)

    return o, o_fp32, o_bf16, got, [x.grad for x in fp32_in], [x.grad for x in bf16_in]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dim", [64, 96, 128, 256])
def test_grads_match_sdpa(dim, causal):
    o, o_fp32, o_bf16, got, fp32_grad, bf16_grad = _grads(batch=2, heads=4, seq_len=512, dim=dim, causal=causal)
    assert_within_2x_torch(o, o_fp32, o_bf16, "o")
    for name, g, r, t in zip("qkv", got, fp32_grad, bf16_grad):
        assert g.dtype == torch.bfloat16
        assert_within_2x_torch(g, r, t, f"d{name}")


@pytest.mark.parametrize("causal", [True, False])
def test_grads_long_seq(causal):
    _, _, _, got, fp32_grad, bf16_grad = _grads(batch=1, heads=4, seq_len=2048, dim=128, causal=causal)
    for name, g, r, t in zip("qkv", got, fp32_grad, bf16_grad):
        assert_within_2x_torch(g, r, t, f"d{name}")


def test_backward_saves_only_qkvo_lse():
    """FA2 recompute: forward must save exactly (q, k, v, o, lse), never an S x S matrix.

    ``grad_fn.saved_tensors`` on a custom ``torch.autograd.Function`` node is
    accessible in torch 2.14 (it proxies ``ctx.saved_tensors``), so we check it
    directly rather than falling back to a memory-delta heuristic.
    """
    batch, heads, seq_len, dim = 1, 2, 1024, 64
    q, k, v = (torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16).requires_grad_()
               for _ in range(3))
    o = attn(q, k, v)

    saved = o.grad_fn.saved_tensors
    assert len(saved) == 5
    qkvo_shape = (batch, seq_len, heads, dim)
    for t in saved[:4]:
        assert t.shape == qkvo_shape
    assert saved[4].shape == (batch, heads, seq_len)

    total_elems = sum(t.numel() for t in saved)
    qkvo_lse_elems = 4 * batch * seq_len * heads * dim + batch * heads * seq_len
    ssq_elems = batch * heads * seq_len * seq_len
    assert total_elems == qkvo_lse_elems
    assert total_elems < 2 * qkvo_lse_elems
    assert total_elems < ssq_elems  # would be off by ~256x if an S x S matrix were saved

    # with a sink the saved set grows by exactly the [H] vector
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
    saved_s = attn(q, k, v, sinks=sinks).grad_fn.saved_tensors
    assert len(saved_s) == 6
    assert saved_s[5].shape == (heads,)
    assert sum(t.numel() for t in saved_s) == qkvo_lse_elems + heads


def _grads_with_sink(batch, heads, seq_len, dim, causal):
    torch.manual_seed(0)
    qkv = [torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32).requires_grad_()
    do = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)

    o = attn(*qkv, causal=causal, sinks=sinks)
    o.backward(do)
    got = [x.grad for x in qkv]

    fp32_in = [x.detach().float().requires_grad_() for x in qkv]
    fp32_sinks = sinks.detach().clone().requires_grad_()
    o_fp32 = sdpa_ref(*fp32_in, causal, fp32_sinks)
    o_fp32.backward(do.float())

    bf16_in = [x.detach().clone().requires_grad_() for x in qkv]
    bf16_sinks = sinks.detach().clone().requires_grad_()
    o_bf16 = torch_bf16_ref(*bf16_in, causal, bf16_sinks)
    o_bf16.backward(do)

    return (o, o_fp32, o_bf16, got, [x.grad for x in fp32_in], [x.grad for x in bf16_in],
            sinks.grad, fp32_sinks.grad, bf16_sinks.grad)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dim", [64, 128])
def test_grads_with_sink_match_ref(dim, causal):
    (o, o_fp32, o_bf16, got, fp32_grad, bf16_grad,
     dsinks, fp32_dsinks, bf16_dsinks) = _grads_with_sink(batch=2, heads=4, seq_len=512, dim=dim, causal=causal)
    assert_within_2x_torch(o, o_fp32, o_bf16, "o")
    for name, g, r, t in zip("qkv", got, fp32_grad, bf16_grad):
        assert_within_2x_torch(g, r, t, f"d{name}")
    assert dsinks.dtype == torch.float32 and dsinks.shape == (4,)
    assert_within_2x_torch(dsinks, fp32_dsinks, bf16_dsinks, "dsinks")
