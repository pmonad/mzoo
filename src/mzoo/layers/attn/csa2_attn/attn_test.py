"""GPU tests for csa2_attn gradients against the fp32 golden reference's autograd.

``o``, ``dq``, ``dkv``, ``dmain_kv`` and ``dsinks`` all use the FlashAttention acceptance
criterion (``dense_attn.ref.assert_within_2x_torch``) against
``golden(level="sparse", window=W, main_kv=..., compress_ratio=m, indices=<the same list>)``
-- the fp32 reference and the bf16 baseline are fed the *identical* index list, so any
disagreement is arithmetic and never selection (``csa2_attn_design.md`` -> Conventions).

Three backward kernels: ``bwd_kernels.preprocess`` (delta), ``bwd_kernels.bwd_kv`` (the
window slice's ``dKV = dK + dV``, ``csa_attn``'s kernel and config untouched) and
``bwd_dq.bwd_dq``, which owns a packed query tile and produces ``dQ`` *and* the
scatter-added ``dmain_kv``. ``csa_attn``'s group-owning ``bwd_main`` has no analogue here,
which is why ``dmain_kv`` is the one tensor whose accumulation order is not reproducible.

The two checks that only exist in this package:

- ``test_grads_duplicate_indices``: the same entry listed twice by one token *and* listed
  by every token. The kernel gives a duplicate two softmax columns (the forward
  double-counts it) while ``golden`` scatters an idempotent 0.0 bias and sees it once, so
  the reference is built by *cloning* the duplicated row into a new entry. The test pins
  both halves of that: the discrepancy with plain ``golden`` is real, and the kernel is
  the exact gradient of its own forward.
- ``test_grads_all_visible_is_csa_attn`` / ``test_grads_all_empty_is_swa``: the feature-off
  cases, bit-for-bit against the previous packages' backwards.
"""

import pytest
import torch

from mzoo.layers.attn.csa2_attn.attn import attn
from mzoo.layers.attn.dense_attn.ref import assert_within_2x_torch
from mzoo.layers.attn.golden_ref import golden, indexer_scores, topk_indices

WINDOW = 128  # the model's sliding_window
RATIO = 2
HI, DI = 4, 64  # indexer heads / head dim used to synthesise the index sets
NAMES = ["dq", "dkv", "dmain_kv"]


def _leaves(batch, heads, seq_len, dim, groups, use_sink, seed=0):
    g = torch.Generator("cuda").manual_seed(seed)
    kw = dict(device="cuda", dtype=torch.bfloat16, generator=g)
    q = torch.randn(batch, seq_len, heads, dim, **kw)
    kv = torch.randn(batch, seq_len, 1, dim, **kw)
    main_kv = torch.randn(batch, groups, 1, dim, **kw)
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32, generator=g) if use_sink else None
    do = torch.randn(batch, seq_len, heads, dim, **kw)
    return q, kv, main_kv, sinks, do


def _indices(batch, seq_len, groups, ratio, topk, *, seed=1):
    """A valid group-causal top-k set per token (``golden_ref``'s contract, ``-1`` = empty)."""
    g = torch.Generator("cuda").manual_seed(seed)
    kw = dict(device="cuda", generator=g)
    qi = torch.randn(batch, seq_len, HI, DI, dtype=torch.bfloat16, **kw)
    ki = torch.randn(batch, groups, DI, dtype=torch.bfloat16, **kw)
    w = torch.randn(batch, seq_len, HI, dtype=torch.float32, **kw) * HI**-0.5
    return topk_indices(indexer_scores(qi.float(), ki.float(), w, ratio), topk)


def _run(q, kv, main_kv, idx, do, sinks, ratio, window):
    """kernel / fp32 golden / bf16 golden, each with its own leaves -> (o, grads) triples."""

    def run(fn, dtype=None):
        ins = [q.detach().clone(), kv.detach().clone(), main_kv.detach().clone()]
        if dtype is torch.float32:
            ins = [x.float() for x in ins]
        ins = [x.requires_grad_() for x in ins]
        sk = sinks.detach().clone().requires_grad_() if sinks is not None else None
        o = fn(*ins, sk)
        o.backward(do.float() if dtype is torch.float32 else do)
        return o, [x.grad for x in ins] + ([sk.grad] if sinks is not None else [])

    gold = lambda a, b, m, s, dt: golden(  # noqa: E731
        a, b, level="sparse", window=window, main_kv=m, compress_ratio=ratio,
        indices=idx.long(), sinks=s, dtype=dt)[0]
    got = run(lambda a, b, m, s: attn(a, b, m, idx, window=window, compress_ratio=ratio, sinks=s))
    ref = run(lambda a, b, m, s: gold(a, b, m, s, torch.float32), torch.float32)
    bf16 = run(lambda a, b, m, s: gold(a, b, m, s, torch.bfloat16))
    return got, ref, bf16


def _grads(batch, heads, seq_len, dim, ratio=RATIO, window=WINDOW, groups=None, topk=64,
           use_sink=False):
    groups = seq_len // ratio if groups is None else groups
    q, kv, main_kv, sinks, do = _leaves(batch, heads, seq_len, dim, groups, use_sink)
    idx = _indices(batch, seq_len, groups, ratio, topk).int()
    return _run(q, kv, main_kv, idx, do, sinks, ratio, window)


def _check(names, got, ref, bf16):
    for name, g, r, t in zip(names, got[1], ref[1], bf16[1]):
        assert_within_2x_torch(g, r, t, name)


@pytest.mark.parametrize("heads", [4, 16, 64])
@pytest.mark.parametrize("dim", [64, 96, 128])
def test_grads_match_golden(dim, heads):
    """The core matrix. ``T_q = ceil(64 / heads)`` tokens per block in *both* the forward and
    the ``dQ``/``dmain_kv`` kernel, so ``heads`` picks the gather layout on both sides."""
    got, ref, bf16 = _grads(batch=2, heads=heads, seq_len=512, dim=dim)
    assert_within_2x_torch(got[0], ref[0], bf16[0], "o")
    assert all(g.dtype == torch.bfloat16 for g in got[1])
    assert got[1][1].shape == (2, 512, 1, dim) and got[1][2].shape == (2, 256, 1, dim)
    _check(NAMES, got, ref, bf16)


@pytest.mark.parametrize("topk", [100, 512])
def test_grads_topk(topk):
    """``topk=100`` is a multiple of no ``block_N``, so the last gathered tile is part ``-1``
    padding and must scatter nothing; ``topk=512`` is the model's."""
    _check(NAMES, *_grads(batch=1, heads=64, seq_len=512, dim=128, topk=topk))


@pytest.mark.parametrize("ratio", [2, 4])
def test_grads_compress_ratios(ratio):
    """``ratio`` only shapes the index sets -- neither kernel re-applies group-causality."""
    _check(NAMES, *_grads(batch=2, heads=16, seq_len=512, dim=128, ratio=ratio, topk=100))


@pytest.mark.parametrize("heads", [4, 64])
def test_grads_dim256(heads):
    """D=256 is smem-capped shape support (one usable tile, see README); kept small, and at
    the smallest and largest head count because smem depends on H through ``T_q * H``."""
    _check(NAMES, *_grads(batch=1, heads=heads, seq_len=512, dim=256, ratio=4, topk=64))


def test_grads_groups_not_power_of_two():
    """``G = 100 < topk`` and a multiple of no tile: every token runs out of visible entries,
    so most of each index list is ``-1``. ``G`` needs no padding in this package, and
    ``dmain_kv`` comes back at exactly ``G`` rows."""
    got, ref, bf16 = _grads(batch=2, heads=16, seq_len=512, dim=128, groups=100, topk=512)
    assert got[1][2].shape == (2, 100, 1, 128)
    _check(NAMES, got, ref, bf16)


def test_grads_long_seq():
    """S=4096 with G=16384 (4x the sequence) and topk=512, the bench shape. H=4 keeps the
    golden reference's dense [B, H, S, S+G] logits affordable."""
    _check(NAMES, *_grads(batch=1, heads=4, seq_len=4096, dim=128, ratio=1, groups=16384,
                          topk=512))


@pytest.mark.parametrize("dim", [64, 128])
def test_grads_with_sink_match_golden(dim):
    """``dsinks`` is checked at D 64/128, H=16 only: it is a thin torch reduce whose 2x ratio
    straddles the bound at small B*S in every package (design doc -> Test matrix)."""
    got, ref, bf16 = _grads(batch=2, heads=16, seq_len=512, dim=dim, topk=100, use_sink=True)
    assert_within_2x_torch(got[0], ref[0], bf16[0], "o")
    assert got[1][3].dtype == torch.float32 and got[1][3].shape == (16,)
    _check(NAMES + ["dsinks"], got, ref, bf16)


def test_grads_duplicate_indices():
    """**The scatter-add test.** Every token lists entry 0 twice, so one main row receives
    two contributions from the same token and one contribution from all ``B*S`` tokens.

    ``golden`` cannot be the reference for the duplicate itself: it renders the index list
    by scattering a 0.0 bias, which is idempotent, so a duplicated entry gets *one* softmax
    column; the kernel builds one gathered column per slot and so gives it *two*. The
    reference here therefore **clones** the duplicated row into a fresh entry ``G`` and
    points the second slot at it -- same math, no duplicate -- and ``dmain_kv[0]`` must
    equal ``dmain_ref[0] + dmain_ref[G]``. The first assertion pins that the plain-``golden``
    discrepancy is real and not silently papered over (see README -> Known issues).
    """
    batch, heads, seq_len, dim, groups, topk = 2, 16, 512, 128, 256, 64
    q, kv, main_kv, sinks, do = _leaves(batch, heads, seq_len, dim, groups, use_sink=True)
    idx = _indices(batch, seq_len, groups, RATIO, topk).int()
    idx[:, :, 0] = 0
    idx[:, :, 1] = 0  # the same entry twice in one token's list, for every token
    other = idx == 0  # ...and *exactly* twice: the top-k set may already have picked it,
    other[:, :, :2] = False  # and a triple has no two-row clone to compare against
    idx[other] = -1

    kw = dict(level="sparse", window=WINDOW, compress_ratio=RATIO, sinks=sinks)
    o = attn(q, kv, main_kv, idx, window=WINDOW, compress_ratio=RATIO, sinks=sinks)
    dedup, _ = golden(q, kv, main_kv=main_kv, indices=idx.long(), **kw)
    # the clone: entry G is a copy of entry 0, and the second slot points at it
    main2 = torch.cat([main_kv, main_kv[:, :1]], dim=1)
    idx2 = idx.clone().long()
    idx2[:, :, 1] = groups
    doubled, _ = golden(q, kv, main_kv=main2, indices=idx2, **kw)
    doubled_bf16, _ = golden(q, kv, main_kv=main2, indices=idx2, dtype=torch.bfloat16, **kw)

    assert_within_2x_torch(o, doubled, doubled_bf16, "o")  # the kernel *is* the doubled rendering
    err_dedup = (o.float() - dedup.float()).abs().max().item()
    err_doubled = (o.float() - doubled.float()).abs().max().item()
    assert err_dedup > 5 * err_doubled, (
        f"the double-counting discrepancy with golden's deduplicated rendering should be "
        f"unmistakable (dedup {err_dedup:.3e} vs doubled {err_doubled:.3e})")

    def run(m, i, dtype=None):
        ins = [q.detach().clone(), kv.detach().clone(), m.detach().clone()]
        if dtype is torch.float32:
            ins = [x.float() for x in ins]
        ins = [x.requires_grad_() for x in ins]
        sk = sinks.detach().clone().requires_grad_()
        out = (attn(*ins, i, window=WINDOW, compress_ratio=RATIO, sinks=sk) if dtype is None else
               golden(ins[0], ins[1], main_kv=ins[2], indices=i.long(), dtype=dtype,
                      level="sparse", window=WINDOW, compress_ratio=RATIO, sinks=sk)[0])
        out.backward(do.float() if dtype is torch.float32 else do)
        grads = [x.grad for x in ins] + [sk.grad]
        if m.shape[1] > groups:  # fold the cloned entry's gradient back onto entry 0
            dmain = grads[2][:, :groups].clone()
            dmain[:, 0] += grads[2][:, groups]
            grads[2] = dmain
        return grads

    for name, g, r, t in zip(NAMES + ["dsinks"], run(main_kv, idx),
                             run(main2, idx2, torch.float32), run(main2, idx2, torch.bfloat16)):
        assert_within_2x_torch(g, r, t, name)


def test_grads_all_visible_is_csa_attn():
    """Feature-off: hand the kernel *every* group-causally visible entry in group order and
    it must reproduce ``csa_attn``'s dense backward. ``dq``/``dkv`` are the same programs on
    the same tiles, so they are expected bit-for-bit; ``dmain_kv`` is a scatter-add against
    ``csa_attn``'s split-buffer reduce, so it only matches to fp32 summation-order noise."""
    from mzoo.layers.attn.csa_attn.attn import attn as csa_attn

    batch, seq_len, heads, dim, groups = 2, 512, 16, 128, 256
    q, kv, main_kv, sinks, do = _leaves(batch, heads, seq_len, dim, groups, use_sink=True)
    t = torch.arange(seq_len, device="cuda").view(-1, 1)
    g = torch.arange(groups, device="cuda").view(1, -1)
    visible = g < ((t + 1) // RATIO)  # the dense group-causal rule, as an index list
    idx = torch.where(visible, g.expand_as(visible), torch.full_like(g.expand_as(visible), -1))
    idx = idx.unsqueeze(0).expand(batch, -1, -1).contiguous().int()

    def run(fn):
        ins = [x.detach().clone().requires_grad_() for x in (q, kv, main_kv)]
        sk = sinks.detach().clone().requires_grad_()
        fn(*ins, sk).backward(do)
        return [x.grad for x in ins] + [sk.grad]

    kw = dict(window=WINDOW, compress_ratio=RATIO)
    got = run(lambda a, b, m, s: attn(a, b, m, idx, sinks=s, **kw))
    exp = run(lambda a, b, m, s: csa_attn(a, b, m, sinks=s, **kw))
    assert torch.equal(got[0], exp[0]), "dq must be bit-for-bit csa_attn's"
    assert torch.equal(got[1], exp[1]), "dkv must be bit-for-bit csa_attn's"
    assert torch.equal(got[3], exp[3]), "dsinks is the same torch reduce"
    # dmain_kv: fp32 atomics vs fp32 split buffers -- same sum, different order
    err = (got[2].float() - exp[2].float()).abs().max().item()
    scale = exp[2].float().abs().max().item()
    assert err <= 5e-3 * scale, f"dmain_kv drifted too far from csa_attn: {err:.3e} vs {scale:.3e}"


@pytest.mark.parametrize("use_sink", [False, True])
def test_grads_all_empty_is_swa(use_sink):
    """Feature-off: all indices ``-1``. Every gathered tile contributes an exact zero, so
    ``dq``/``dkv`` must be bit-for-bit ``swa_attn``'s backward and ``dmain_kv`` exactly 0."""
    from mzoo.layers.attn.swa_attn.attn import attn as swa_attn

    q, kv, main_kv, sinks, do = _leaves(2, 16, 512, 128, 256, use_sink)
    idx = torch.full((2, 512, 64), -1, device="cuda", dtype=torch.int32)

    def run(fn, *leaves):
        ins = [x.detach().clone().requires_grad_() for x in leaves]
        sk = sinks.detach().clone().requires_grad_() if use_sink else None
        fn(*ins, sk).backward(do)
        return [x.grad for x in ins] + ([sk.grad] if use_sink else [])

    got = run(lambda a, b, m, s: attn(a, b, m, idx, window=WINDOW, compress_ratio=RATIO, sinks=s),
              q, kv, main_kv)
    exp = run(lambda a, b, s: swa_attn(a, b, window=WINDOW, sinks=s), q, kv)
    assert torch.equal(got[0], exp[0]), "dq must be bit-for-bit swa_attn's"
    assert torch.equal(got[1], exp[1]), "dkv must be bit-for-bit swa_attn's"
    assert not got[2].any(), "no index is valid, so no main entry may receive a gradient"


def test_backward_saves_only_q_kv_main_indices_o_lse():
    """FA2 recompute: the forward saves exactly (q, kv, main_kv, indices, o, lse), never an
    S x (S + topk) matrix. ``window``/``compress_ratio`` ride on ``ctx`` as plain ints, and
    ``indices`` -- the only new save -- gets no gradient."""
    batch, heads, seq_len, dim, groups, topk = 1, 16, 1024, 64, 256, 64
    q, kv, main_kv, _, _ = _leaves(batch, heads, seq_len, dim, groups, use_sink=False)
    q, kv, main_kv = (x.requires_grad_() for x in (q, kv, main_kv))
    idx = _indices(batch, seq_len, groups, 4, topk).int()
    o = attn(q, kv, main_kv, idx, window=WINDOW, compress_ratio=4)

    saved = o.grad_fn.saved_tensors
    assert len(saved) == 6
    assert saved[0].shape == saved[4].shape == (batch, seq_len, heads, dim)
    assert saved[1].shape == (batch, seq_len, 1, dim)
    assert saved[2].shape == (batch, groups, 1, dim)
    assert saved[3].shape == (batch, seq_len, topk) and not saved[3].is_floating_point()
    assert saved[5].shape == (batch, heads, seq_len)

    total = sum(t.numel() for t in saved)
    expect = ((2 * heads + 1) * batch * seq_len * dim + batch * groups * dim
              + batch * seq_len * topk + batch * heads * seq_len)
    assert total == expect
    assert total < batch * heads * seq_len * (seq_len + topk)  # an S x (S + topk) P matrix

    sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
    saved_s = attn(q, kv, main_kv, idx, window=WINDOW, compress_ratio=4,
                   sinks=sinks).grad_fn.saved_tensors
    assert len(saved_s) == 7 and saved_s[6].shape == (heads,)
    assert sum(t.numel() for t in saved_s) == expect + heads
    assert o.grad_fn.next_functions[3][0] is None  # indices is not differentiable
