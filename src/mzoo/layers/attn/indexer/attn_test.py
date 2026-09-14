"""GPU tests for the public indexer entry: score kernel -> ``torch.topk`` -> ``Indices``.

Two levels:

1. Against ``golden_ref.topk_indices`` on the fp32 reference scores -- exact **set**
   equality per query row (``topk`` is unsorted, and the kernel's bf16 rounding can
   reorder near-equal picks; inputs are gaussian so exact ties are improbable).
2. End-to-end against the vendored model: the tiny 2-layer config of
   ``golden_ref_test`` is run once, the indexer's own ``q``/``k``/``weights`` and its
   published ``shared["topk_bias"]`` are captured, and our ``Indices`` must select the
   same set of groups the model's ``topk`` did. The capture is deliberately minimal:
   a spy on the module-level ``_fake_quant_fp4_block`` (whose ``block_size=32`` calls
   are exactly the indexer's k then q, post-quant -- so the kernel sees the same values
   the model scored), a forward hook on ``weights_proj``, and a wrapper around
   ``DeepseekV41Indexer.forward`` to read ``shared["topk_bias"]`` back out.
"""

import pytest
import torch

from mzoo.archs.dsv4 import modeling_deepseek_v41 as modeling
from mzoo.layers.attn.golden_ref import indexer_scores, topk_indices
from mzoo.layers.attn.golden_ref_test import _build_tiny_model
from mzoo.layers.attn.indexer.attn import attn
from mzoo.layers.attn.indexer.fwd_test import inputs


def assert_same_sets(got: torch.Tensor, ref: torch.Tensor, name: str = "indices") -> None:
    """Per-row set equality of two ``[B, S, topk]`` index tensors (``-1`` = empty)."""
    assert got.shape == ref.shape and got.dtype == torch.int32
    a, b = got.long().sort(dim=-1).values, ref.long().sort(dim=-1).values
    bad = (a != b).any(dim=-1).nonzero()
    assert len(bad) == 0, f"{name}: {len(bad)} rows differ, first {bad[0].tolist()}: {a[tuple(bad[0])]} vs {b[tuple(bad[0])]}"


def check(batch, seq_len, heads, dim, seq_kv, compress_ratio, topk):
    q, k, w = inputs(batch, seq_len, heads, dim, seq_kv)
    got = attn(q, k, w, compress_ratio=compress_ratio, topk=topk)
    ref = topk_indices(indexer_scores(q.float(), k.float(), w, compress_ratio), topk)
    assert_same_sets(got, ref)


@pytest.mark.parametrize("topk", [64, 135])  # 135 = T + 7: more picks than columns -> -1 padding
@pytest.mark.parametrize("compress_ratio", [1, 2, 4])
def test_indices_match_reference(compress_ratio, topk):
    check(batch=2, seq_len=256, heads=32, dim=128, seq_kv=128, compress_ratio=compress_ratio, topk=topk)


def test_indices_long_seq():
    check(batch=1, seq_len=1024, heads=4, dim=64, seq_kv=256, compress_ratio=4, topk=64)


def test_empty_kv_is_all_minus_one():
    q, k, w = inputs(1, 256, 32, 128, 0)
    idx = attn(q, k, w, compress_ratio=2, topk=64)
    assert idx.shape == (1, 256, 64) and torch.equal(idx, torch.full_like(idx, -1))


def _capture_indexer(model, input_ids):
    """One prefill forward, recording the indexer's post-fp4 q/k, its fp32 weights and
    the ``topk_bias`` it published."""
    cap, quant = {}, []
    orig_quant, orig_fwd = modeling._fake_quant_fp4_block, modeling.DeepseekV41Indexer.forward

    def spy_quant(x, block_size, e4m3_scales=False):
        out = orig_quant(x, block_size, e4m3_scales)
        if block_size == 32:  # the indexer's two calls (k then q); the compressor uses 16
            quant.append(out.detach().clone())
        return out

    def spy_fwd(self, hidden_states, *args, **kwargs):
        h = self.weights_proj.register_forward_hook(
            lambda _m, _i, o: cap.update(w=o.detach().float() * self.heads_scaling))
        try:
            out = orig_fwd(self, hidden_states, *args, **kwargs)
        finally:
            h.remove()
        shared = kwargs.get("shared", args[-1] if args else None)
        cap.update(bias=shared["topk_bias"].detach().clone(), ratio=self.compress_ratio, topk=self.index_topk)
        return out

    modeling._fake_quant_fp4_block, modeling.DeepseekV41Indexer.forward = spy_quant, spy_fwd
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        modeling._fake_quant_fp4_block, modeling.DeepseekV41Indexer.forward = orig_quant, orig_fwd
    assert len(quant) == 2 and quant[0].ndim == 3 and quant[1].ndim == 4, f"unexpected capture {[t.shape for t in quant]}"
    cap["k"], cap["q"] = quant
    return cap


def test_matches_vendored_model_indexer():
    """Selection equality with ``DeepseekV41Indexer``'s own top-k, on its own inputs.

    Compared by the **score values** of the two chosen sets, not by index set: the
    freshly-initialised tiny model produces ~7% exactly-zero scores (all four heads
    relu'd to 0), so which of the tied entries each ``torch.topk`` keeps is arbitrary.
    Both sets are scored with the same fp32 reference, so an exact match of the sorted
    score vectors is the strongest tie-free statement available (observed diff: 0.0).
    """
    torch.manual_seed(6)
    model = _build_tiny_model()
    input_ids = torch.randint(0, 64, (2, 160), device="cuda")
    c = _capture_indexer(model, input_ids)
    q, k, w, topk = c["q"], c["k"], c["w"], c["topk"]
    idx = attn(q.bfloat16(), k.bfloat16(), w, compress_ratio=c["ratio"], topk=topk).long()
    ref = indexer_scores(q.float(), k.float(), w, c["ratio"])
    assert ref.shape[-1] > topk, "config assumption broke: no real top-k sparsity to check"
    neg_inf = torch.full_like(ref, float("-inf"))
    ours = torch.where(idx >= 0, ref.gather(-1, idx.clamp(min=0)), neg_inf[..., :topk])
    # the model publishes its selection as a 0/-inf bias [B, 1, S, G] over the groups
    theirs = torch.where(c["bias"].squeeze(1) == 0, ref, neg_inf)
    torch.testing.assert_close(ours.sort(-1, descending=True).values,
                               theirs.sort(-1, descending=True).values[..., :topk], atol=0, rtol=0)
