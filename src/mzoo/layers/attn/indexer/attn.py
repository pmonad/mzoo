"""indexer: public entry -- score matrix -> top-k -> the ``Indices`` tensor ``csa2_attn`` consumes.

``attn(q, k, w, compress_ratio=..., topk=...)`` runs ``fwd`` (the bf16 score kernel)
and then plain ``torch.topk`` on the materialised ``[B, S, T]`` fp32 scores. The
output contract is ``golden_ref.topk_indices`` verbatim: ``[B, S, topk]`` int32 with
``-1`` wherever the pick's score was ``-inf`` (fewer than ``topk`` group-causally
visible entries yet) or ``topk > T`` ran out of columns. ``T == 0`` returns all ``-1``.

``scores`` (ticket 0007) is the *differentiable* score matrix: an autograd Function
whose backward is ``bwd.py``. It saves only ``q, k, w`` -- the relu mask is recomputed
inside the backward kernel, never stored as a ``[S, Hi, T]`` tensor. Top-k stays
discrete: no gradient flows through ``attn``'s index output (the training signal is
the indexer's own auxiliary loss applied to ``scores``).

``fake_quant_fp4=True`` applies the model's ``_fake_quant_fp4_block`` round trip to
q/k with a **straight-through estimator** (forward: quantized values; backward:
identity, and the ue8m0 scales inside the quantizer are already stop-gradiented).
The model's own call site detaches, which is why the vendored indexer never trains
(``archs/dsv4/README.md``); making the estimator live here instead is a deliberate,
flagged divergence from the vendored model.
"""

import torch

from mzoo.layers.attn.indexer.bwd import bwd
from mzoo.layers.attn.indexer.fwd import fwd


def _fake_quant_fp4_ste(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """The model's fp4 round trip with the gradient put back: forward values are
    bit-identical to ``_fake_quant_fp4_block``, backward is the identity (STE)."""
    from mzoo.archs.dsv4.modeling_deepseek_v41 import _fake_quant_fp4_block
    quant = _fake_quant_fp4_block(x.detach(), block_size)
    return x - x.detach() + quant  # 0 with gradient x + quant: bits exact, grad identity


class _ScoresAutograd(torch.autograd.Function):
    """``fwd`` + ``bwd`` glued together; see module docstring for what is saved."""

    @staticmethod
    def forward(ctx, q, k, w, compress_ratio):
        ctx.save_for_backward(q, k, w)
        ctx.compress_ratio = compress_ratio
        return fwd(q, k, w, compress_ratio=compress_ratio)

    @staticmethod
    def backward(ctx, dl):
        q, k, w = ctx.saved_tensors
        dq, dk, dw = bwd(q, k, w, dl.contiguous().float(), compress_ratio=ctx.compress_ratio)
        return dq, dk, dw, None


def scores(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, *, compress_ratio: int,
           fake_quant_fp4: bool = False) -> torch.Tensor:
    """Differentiable ``fwd``: same output, plus ``dq``/``dk``/``dw`` via ``bwd.py``.

    ``fake_quant_fp4=False`` is the feature-off default: its forward is exactly
    ``fwd`` (bit-for-bit, same kernel), and its backward is the plain relu-gated
    gradient. ``fake_quant_fp4=True`` round-trips q/k through fp4 first -- STE, see
    module docstring -- so gradients reach the pre-quant projections.
    """
    if fake_quant_fp4:
        q, k = _fake_quant_fp4_ste(q), _fake_quant_fp4_ste(k)
    return _ScoresAutograd.apply(q, k, w, compress_ratio)


def attn(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, *, compress_ratio: int, topk: int) -> torch.Tensor:
    """``q`` bf16 [B, S, Hi, Di], ``k`` bf16 [B, T, Di], ``w`` fp32 [B, S, Hi] ->
    ``Indices`` int32 [B, S, topk] (``-1`` = empty slot).
    """
    batch, seq_len = q.shape[0], q.shape[1]
    seq_kv = k.shape[1]
    assert topk >= 0, f"topk must be >= 0, got {topk}"
    kept = min(topk, seq_kv)
    idx = q.new_full((batch, seq_len, topk), -1, dtype=torch.int32)
    if kept > 0:
        scores = fwd(q, k, w, compress_ratio=compress_ratio)
        top = scores.topk(kept, dim=-1, sorted=False)
        idx[..., :kept] = torch.where(top.values > float("-inf"), top.indices.int(),
                                      torch.full_like(top.indices, -1, dtype=torch.int32))
    return idx
