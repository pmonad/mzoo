"""GPU tests for the indexer backward (ticket 0007) against autograd through the
torch einsum path with the detach removed.

Three layers of checking:

- ``check``: the kernel's dq/dk/dw vs (a) the fp32 autograd reference and (b) the same
  autograd run on bf16 q/k -- the shared ``assert_within_2x_torch`` acceptance, which
  isolates tiling rather than dtype.
- Semantics pinned exactly: a masked entry receives **exactly zero** gradient
  (whatever junk the loss left in ``dL``'s masked slots is ignored), and the relu
  subgradient at ``r == 0`` is 0 (torch's convention).
- ``test_gradcheck_fp64``: a gradcheck on the pure-torch fp64 mirror of the semantics
  (mask applied multiplicatively, not via ``-inf``), so the formula itself -- not just
  the kernel -- is checked against autograd on a tiny shape.
"""

import pytest
import torch

from mzoo.layers.attn.dense_attn.ref import assert_within_2x_torch
from mzoo.layers.attn.indexer.attn import _fake_quant_fp4_ste, scores
from mzoo.layers.attn.indexer.bwd import bwd
from mzoo.layers.attn.indexer.fwd_test import inputs


def torch_bwd(q, k, w, dl, compress_ratio):
    """Autograd through the torch einsum path (the model's forward with the detach
    removed). Returns fp32 grads when the inputs are fp32, bf16-matmul grads when
    they are bf16 -- used as reference and baseline respectively."""
    q, k, w = (x.detach().clone().requires_grad_(True) for x in (q, k, w))
    raw = torch.einsum("bshd,btd->bsht", q, k).float().relu_() * q.shape[-1]**-0.5
    L = (raw * w.float().unsqueeze(-1)).sum(2)
    s, t = q.shape[1], k.shape[1]
    vis = (torch.arange(t, device=q.device).view(1, -1)
           < (torch.arange(s, device=q.device).view(-1, 1) + 1) // compress_ratio)
    L.backward(torch.where(vis, dl, torch.zeros_like(dl)))
    return q.grad, k.grad, w.grad


def check(batch, seq_len, heads, dim, seq_kv, compress_ratio, seed=0):
    q, k, w = inputs(batch, seq_len, heads, dim, seq_kv, seed)
    dl = torch.randn(batch, seq_len, seq_kv, device="cuda", dtype=torch.float32,
                     generator=torch.Generator("cuda").manual_seed(seed + 1))
    got = bwd(q, k, w, dl, compress_ratio=compress_ratio)
    ref = torch_bwd(q.float(), k.float(), w, dl, compress_ratio)
    base = torch_bwd(q, k, w, dl, compress_ratio)  # bf16 matmul autograd
    for name, g, r, b in zip(("dq", "dk", "dw"), got, ref, base):
        assert g.shape == r.shape, f"{name} shape {g.shape} vs {r.shape}"
        assert_within_2x_torch(g.float(), r, b.float(), f"d{name}")


@pytest.mark.parametrize("heads", [4, 32])
@pytest.mark.parametrize("dim", [64, 96, 128, 256])
def test_bwd_matches_autograd(dim, heads):
    check(batch=2, seq_len=256, heads=heads, dim=dim, seq_kv=128, compress_ratio=2)


@pytest.mark.parametrize("compress_ratio", [1, 2, 4])
def test_bwd_compress_ratios(compress_ratio):
    check(batch=1, seq_len=256, heads=32, dim=128, seq_kv=256 // compress_ratio,
          compress_ratio=compress_ratio)


@pytest.mark.parametrize("seq_kv", [128, 400, 7, 129])
def test_bwd_kv_lengths(seq_kv):
    """Aligned, far past the visible frontier (whole tail must be zero), and two
    non-tile-aligned lengths (the key/dL zero-padding path)."""
    check(batch=1, seq_len=256, heads=32, dim=128, seq_kv=seq_kv, compress_ratio=2)


def test_bwd_long_seq():
    check(batch=1, seq_len=1024, heads=32, dim=128, seq_kv=256, compress_ratio=4)


def test_masked_entries_get_exactly_zero_gradient():
    """Masked entries contribute exactly zero: filling their dL slots with junk (or
    nan) must leave dq/dk/dw bit-identical, and a fully-masked key row gets an
    all-zero dk row."""
    q, k, w = inputs(1, 256, 32, 128, 128)
    dl = torch.randn(1, 256, 128, device="cuda")
    s_idx = torch.arange(256, device="cuda").view(-1, 1)
    t_idx = torch.arange(128, device="cuda").view(1, -1)
    masked = t_idx >= (s_idx + 1) // 2
    junk = dl.clone()
    junk[masked.expand_as(dl)] = float("nan")
    clean = bwd(q, k, w, dl, compress_ratio=2)
    poisoned = bwd(q, k, w, junk, compress_ratio=2)
    for name, g, p in zip(("dq", "dk", "dw"), clean, poisoned):
        assert torch.isfinite(p).all(), f"{name} picked up nan from a masked dL slot"
        if name == "dk":  # fp32 atomics: order-nondeterministic at ~1e-5, not bitwise
            assert torch.allclose(g.float(), p.float(), rtol=1e-3, atol=1e-3)
        else:
            assert torch.equal(g, p), "masked dL entries leaked into the gradient"
    # and dL supported only on masked slots gives exactly zero gradients everywhere
    dl_masked_only = dl.masked_fill(~masked, 0.0)
    assert all(torch.count_nonzero(g) == 0
               for g in bwd(q, k, w, dl_masked_only, compress_ratio=2))


def test_relu_zero_subgradient_is_zero():
    """Documented convention: relu'(0) = 0 (torch's subgradient). A head whose q is
    exactly zero has r == 0 everywhere, so it must receive exactly zero dq/dw."""
    q, k, w = inputs(1, 64, 32, 128, 32)
    q = q.clone()
    q[:, :, 0, :] = 0  # head 0: r == 0 for every (s, t)
    dl = torch.randn(1, 64, 32, device="cuda")
    dq, dk, dw = bwd(q, k, w, dl, compress_ratio=1)
    assert torch.count_nonzero(dq[:, :, 0, :]) == 0, "dq at r == 0 must be exactly 0"
    assert torch.count_nonzero(dw[:, :, 0]) == 0, "dw at r == 0 must be exactly 0"
    # and torch agrees (the convention is pinned, not arbitrary): full compare
    ref = torch_bwd(q.float(), k.float(), w, dl, 1)
    for g, r in zip((dq, dk, dw), ref):
        assert torch.allclose(g.float(), r, atol=2e-2, rtol=2e-2)


def test_gradcheck_fp64():
    """gradcheck on the fp64 semantics (mask applied multiplicatively, since -inf
    slots are outside autograd's domain): autograd through the forward einsum must
    equal the analytic relu-gated formulas on a tiny shape."""
    torch.manual_seed(0)
    ratio, heads, dim = 2, 3, 5
    q = torch.randn(2, 7, heads, dim, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 4, dim, dtype=torch.float64, requires_grad=True)
    w = torch.randn(2, 7, heads, dtype=torch.float64, requires_grad=True) * heads**-0.5

    def visible_scores(q, k, w):
        raw = torch.einsum("bshd,btd->bsht", q, k).relu() * dim**-0.5
        L = (raw * w.unsqueeze(-1)).sum(2)
        s, t = q.shape[1], k.shape[1]
        vis = (torch.arange(t).view(1, -1) < (torch.arange(s).view(-1, 1) + 1) // ratio)
        return L * vis  # finite mask, gradcheck-able

    assert torch.autograd.gradcheck(visible_scores, (q, k, w), eps=1e-6, atol=1e-4)


def test_scores_forward_feature_off_is_bit_exact():
    """fake_quant_fp4=False (feature off) must equal the previous version's path
    bit-for-bit: same kernel call, identical bytes."""
    from mzoo.layers.attn.indexer.fwd import fwd
    q, k, w = inputs(1, 256, 32, 128, 128)
    assert torch.equal(scores(q, k, w, compress_ratio=2), fwd(q, k, w, compress_ratio=2))


def test_scores_backward_via_autograd():
    """The public autograd wrapper returns exactly what ``bwd`` returns."""
    q, k, w = inputs(1, 256, 32, 128, 128)
    q, k, w = (x.detach().requires_grad_(True) for x in (q, k, w))
    dl = torch.rand(1, 256, 128, device="cuda") * (torch.rand(1, 256, 128, device="cuda") > 0.5)
    L = scores(q, k, w, compress_ratio=2)
    L.backward(dl)
    direct = bwd(q.detach(), k.detach(), w.detach(), dl.contiguous(), compress_ratio=2)
    for name, g, d in zip(("dq", "dk", "dw"), (q.grad, k.grad, w.grad), direct):
        if name == "dk":
            assert torch.allclose(g.float(), d.float(), rtol=1e-3, atol=1e-3)
        else:
            assert torch.equal(g, d)


def test_fake_quant_fp4_ste():
    """STE forward is bit-identical to the model's (detached) quantizer; STE backward
    is the identity, i.e. the gradient is the kernel's own bwd on the quantized values."""
    from mzoo.archs.dsv4.modeling_deepseek_v41 import _fake_quant_fp4_block
    q, k, w = inputs(1, 256, 32, 128, 128)
    q, k, w = (x.detach().requires_grad_(True) for x in (q, k, w))
    L = scores(q, k, w, compress_ratio=2, fake_quant_fp4=True)
    quant_q, quant_k = _fake_quant_fp4_block(q.detach(), 32), _fake_quant_fp4_block(k.detach(), 32)
    assert torch.equal(L.detach(), scores(quant_q, quant_k, w.detach(), compress_ratio=2))
    dl = torch.randn(1, 256, 128, device="cuda")
    L.backward(dl)
    direct = bwd(quant_q, quant_k, w.detach(), dl.contiguous(), compress_ratio=2)
    for name, g, d in zip(("dq", "dk", "dw"), (q.grad, k.grad, w.grad), direct):
        if name == "dk":
            assert torch.allclose(g.float(), d.float(), rtol=1e-3, atol=1e-3)
        else:
            assert torch.equal(g, d)
    # the STE helper itself: forward bits equal the model call, backward identity
    x = torch.randn(64, 128, device="cuda", dtype=torch.float64, requires_grad=True)
    assert torch.equal(_fake_quant_fp4_ste(x).detach(), _fake_quant_fp4_block(x.detach(), 32))
    assert torch.autograd.grad(_fake_quant_fp4_ste(x).sum(), x)[0].eq(1).all()
