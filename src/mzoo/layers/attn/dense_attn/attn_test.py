"""GPU tests for dense_attn gradients against fp32 SDPA autograd.

Tolerance: dq/dk/dv accumulate in fp32 but P, dS and the grads themselves round
through bf16, and the backward chains three more bf16 GEMMs after the forward, so
errors are a few times the forward's. atol=rtol=3e-2 covers every case measured
here: worst observed is 1.9e-2 (dq, D=256 causal), then ~1.8e-2 (dv, D=128
causal); non-causal stays under 3e-3 everywhere, and S=2048 D=128 under 1.2e-2.
Same tolerance for all dims -- D=256 is the loosest but still has 1.6x margin.

With a per-head sink dq/dk/dv keep that tolerance (worst observed 1.45e-2, dv
D=256 causal). ``dsinks`` is a sum over all B*S rows, so it is checked relative
to the reference vector's own scale: worst measured
``max|d - ref| / max|ref|`` is 6.4e-3 (D=96 causal), and 3e-2 is used.
"""

import pytest
import torch
import torch.nn.functional as F

from mzoo.layers.attn.dense_attn.attn import attn
from mzoo.layers.attn.dense_attn.ref import sdpa_ref

TOL = dict(atol=3e-2, rtol=3e-2)
DSINK_RTOL = 3e-2  # relative to max|dsinks_ref|; worst measured 6.4e-3


def _grads(batch, heads, seq_len, dim, causal):
    torch.manual_seed(0)
    qkv = [torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
    do = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)
    o = attn(*qkv, causal=causal)
    o.backward(do)

    ref = [x.detach().float().transpose(1, 2).requires_grad_() for x in qkv]
    ro = F.scaled_dot_product_attention(*ref, is_causal=causal).transpose(1, 2)
    ro.backward(do.float())
    return o, ro, [x.grad for x in qkv], [x.grad.transpose(1, 2) for x in ref]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dim", [64, 96, 128, 256])
def test_grads_match_sdpa(dim, causal):
    o, ro, got, ref = _grads(batch=2, heads=4, seq_len=512, dim=dim, causal=causal)
    torch.testing.assert_close(o.float(), ro, atol=2e-2, rtol=2e-2)
    for name, g, r in zip("qkv", got, ref):
        assert g.dtype == torch.bfloat16
        torch.testing.assert_close(g.float(), r, msg=lambda m, n=name: f"d{n}: {m}", **TOL)


@pytest.mark.parametrize("causal", [True, False])
def test_grads_long_seq(causal):
    _, _, got, ref = _grads(batch=1, heads=4, seq_len=2048, dim=128, causal=causal)
    for name, g, r in zip("qkv", got, ref):
        torch.testing.assert_close(g.float(), r, msg=lambda m, n=name: f"d{n}: {m}", **TOL)


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
    attn(*qkv, causal=causal, sinks=sinks).backward(do)

    ref = [x.detach().float().requires_grad_() for x in qkv]
    ref_sinks = sinks.detach().clone().requires_grad_()
    sdpa_ref(*ref, causal, ref_sinks).backward(do.float())
    return [x.grad for x in qkv], [x.grad for x in ref], sinks.grad, ref_sinks.grad


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dim", [64, 128])
def test_grads_with_sink_match_ref(dim, causal):
    got, ref, dsinks, ref_dsinks = _grads_with_sink(batch=2, heads=4, seq_len=512, dim=dim, causal=causal)
    for name, g, r in zip("qkv", got, ref):
        torch.testing.assert_close(g.float(), r, msg=lambda m, n=name: f"d{n}: {m}", **TOL)
    assert dsinks.dtype == torch.float32 and dsinks.shape == (4,)
    rel = ((dsinks - ref_dsinks).abs().max() / ref_dsinks.abs().max()).item()
    assert rel < DSINK_RTOL, f"dsinks relative error {rel:.2e}"
