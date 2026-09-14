"""GPU tests for the indexer bf16 score kernel against the golden fp32 reference.

The reference is ``golden_ref.indexer_scores`` (steps 2-3 of
``DeepseekV41Indexer.forward``) run on **fp32** q/k; the acceptance criterion is the
shared ``dense_attn.ref.assert_within_2x_torch``, with the bf16 baseline being the
same einsum done with a **bf16 matmul and an fp32 head reduce** -- exactly what the
kernel does, so the comparison isolates tiling, not dtype.

``-inf`` (group-causally invisible) entries are not part of that numeric comparison:
they are checked separately as an exact mask equality, which is the stronger claim.
"""

import pytest
import torch

from mzoo.layers.attn.dense_attn.ref import assert_within_2x_torch
from mzoo.layers.attn.golden_ref import indexer_scores
from mzoo.layers.attn.indexer.fwd import fwd


def inputs(batch, seq_len, heads, dim, seq_kv, seed=0):
    """bf16 q/k and fp32 weights, ``w`` already carrying the ``Hi**-0.5`` factor."""
    g = torch.Generator("cuda").manual_seed(seed)
    kw = dict(device="cuda", generator=g)
    q = torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16, **kw)
    k = torch.randn(batch, seq_kv, dim, dtype=torch.bfloat16, **kw)
    w = torch.randn(batch, seq_len, heads, dtype=torch.float32, **kw) * heads**-0.5
    return q, k, w


def bf16_baseline(q, k, w, compress_ratio):
    """``indexer_scores`` with the q.k matmul in bf16 (fp32 accumulate) and everything
    after it in fp32 -- torch's own version of what the kernel computes."""
    raw = torch.einsum("bshd,btd->bsht", q, k).float().relu_() * q.shape[-1]**-0.5
    scores = (raw * w.float().unsqueeze(-1)).sum(dim=2)
    ref = indexer_scores(q.float(), k.float(), w, compress_ratio)
    return scores.masked_fill(torch.isinf(ref), float("-inf"))


def check(batch, seq_len, heads, dim, seq_kv, compress_ratio):
    q, k, w = inputs(batch, seq_len, heads, dim, seq_kv)
    got = fwd(q, k, w, compress_ratio=compress_ratio)
    assert got.shape == (batch, seq_len, seq_kv) and got.dtype == torch.float32
    ref = indexer_scores(q.float(), k.float(), w, compress_ratio)
    torch_bf16 = bf16_baseline(q, k, w, compress_ratio)
    masked = torch.isinf(ref)
    assert torch.equal(torch.isinf(got), masked), "group-causal -inf mask differs from the reference"
    finite = ~masked
    assert_within_2x_torch(got[finite], ref[finite], torch_bf16[finite], "index_scores")


@pytest.mark.parametrize("heads", [4, 32])
@pytest.mark.parametrize("dim", [64, 96, 128, 256])
def test_scores_match_reference(dim, heads):
    check(batch=2, seq_len=256, heads=heads, dim=dim, seq_kv=128, compress_ratio=2)


@pytest.mark.parametrize("compress_ratio", [1, 2, 4])
def test_compress_ratios(compress_ratio):
    check(batch=1, seq_len=256, heads=32, dim=128, seq_kv=256 // compress_ratio, compress_ratio=compress_ratio)


@pytest.mark.parametrize("seq_kv", [128, 400, 7])
def test_kv_lengths(seq_kv):
    """``T`` exactly the visible count, far past it (the whole tail must be ``-inf``),
    and far short of it (``T`` not a multiple of ``block_T`` -- the key padding path)."""
    check(batch=1, seq_len=256, heads=32, dim=128, seq_kv=seq_kv, compress_ratio=2)


def test_empty_kv():
    q, k, w = inputs(1, 256, 32, 128, 0)
    assert fwd(q, k, w, compress_ratio=2).shape == (1, 256, 0)


def test_long_seq():
    check(batch=1, seq_len=1024, heads=32, dim=128, seq_kv=256, compress_ratio=4)


def test_rejects_bad_seq_len():
    """``S`` must be a multiple of ``block_M // Hi`` (= 4 at Hi=32, block_M=128)."""
    q, k, w = inputs(1, 258, 32, 128, 128)
    with pytest.raises(AssertionError, match="S %"):
        fwd(q, k, w, compress_ratio=2)
