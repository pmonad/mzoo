"""indexer: public entry -- score matrix -> top-k -> the ``Indices`` tensor ``csa2_attn`` consumes.

``attn(q, k, w, compress_ratio=..., topk=...)`` runs ``fwd`` (the bf16 score kernel)
and then plain ``torch.topk`` on the materialised ``[B, S, T]`` fp32 scores. The
output contract is ``golden_ref.topk_indices`` verbatim: ``[B, S, topk]`` int32 with
``-1`` wherever the pick's score was ``-inf`` (fewer than ``topk`` group-causally
visible entries yet) or ``topk > T`` ran out of columns. ``T == 0`` returns all ``-1``.

Forward only -- ``bwd.py`` is ticket 0007. The top-k stays unfused in v1 (ticket 0006
"v1 keeps the top-k in ``torch.topk``"); see ``README.md`` -> Known issues for the
memory the materialised score matrix costs and what 0009 has to revisit.
"""

import torch

from mzoo.layers.attn.indexer.fwd import fwd


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
