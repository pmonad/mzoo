"""Shared golden torch reference for the whole attn kernel series (see
``csa2_attn_design.md`` -> Next). One softmax per level, built by concatenating KV
sources exactly like ``DeepseekV41Attention.forward`` does (``kv = torch.cat([kv,
compressed_kv], dim=2)``), so ``golden`` is the single reference every later
package (``latent_attn``, ``swa_attn``, ``csa_attn``, ``csa2_attn``, ``indexer``)
tests against instead of re-deriving per-step math.

Levels are cumulative supersets: dense -> latent -> window -> compressed -> sparse.
``golden`` asserts that kwargs unused by a level are ``None`` so the level switch is
real, not just a superset of ifs.

Empirically verified semantics (see ``golden_ref_test.py`` for the checks):
- ``window`` counts the query token itself: with ``sliding_window=W``,
  ``transformers.masking_utils.sliding_window_overlay`` keeps ``kv_idx > q_idx - W``,
  i.e. exactly ``W`` visible raw tokens for an interior query (``[t-W+1, t]``), not
  ``W+1``. This matches the design doc's ``q·win_kv[t-127..t]`` band for ``W=128``.
- Main/compressed entries: group ``g`` is visible to token ``t`` (0-indexed, prefill
  positions ``0..S-1``) iff ``g < (t+1) // compress_ratio`` -- lifted verbatim from
  ``DeepseekV41Indexer.forward`` step 3 (``compress_lens = (position_ids + 1) //
  ratio``).
- Sparse level trusts ``indices`` completely (no extra group-causal re-check): the
  indexer is responsible for only ever proposing group-causal entries; ``golden``
  just renders whatever it is given, matching ``eager_attention_forward`` which has
  no idea where its ``attention_mask`` bias came from.

``indexer_scores``/``topk_indices`` extract steps 2, 3 and 5 of
``DeepseekV41Indexer.forward``. Skipped on purpose (out of scope for now): the fp4
fake-quant of q/k (a QAT training detail, not indexer math) and the hierarchical
two-level top-k (``select_candidate_blocks`` / step 4) -- both can be layered on top
of ``index_scores`` later without changing this module's contract.
"""

from typing import Literal

import torch
import torch.nn.functional as F
from transformers.masking_utils import (
    causal_mask_function,
    eager_mask,
    sliding_window_causal_mask_function,
    sliding_window_overlay,
)
from transformers.models.gpt_oss.modeling_gpt_oss import eager_attention_forward as _gpt_oss_eager

Level = Literal["dense", "latent", "window", "compressed", "sparse"]

_EXTRA_KWARGS = ("v", "window", "main_kv", "compress_ratio", "indices")
_REQUIRED = {
    "dense": {"v"},
    "latent": set(),
    "window": {"window"},
    "compressed": {"window", "main_kv", "compress_ratio"},
    "sparse": {"window", "main_kv", "compress_ratio", "indices"},
}


def _check_kwargs(level: Level, **kw) -> None:
    required = _REQUIRED[level]
    for name in _EXTRA_KWARGS:
        present = kw[name] is not None
        if name in required and not present:
            raise ValueError(f"level={level!r} requires `{name}`")
        if name not in required and present:
            raise ValueError(f"level={level!r} must not receive `{name}` (got a value)")


def _causal(seq_q: int, seq_k: int, device) -> torch.Tensor:
    q_idx = torch.arange(seq_q, device=device).view(-1, 1)
    k_idx = torch.arange(seq_k, device=device).view(1, -1)
    return k_idx <= q_idx


def _band(seq: int, window: int, device) -> torch.Tensor:
    q_idx = torch.arange(seq, device=device).view(-1, 1)
    k_idx = torch.arange(seq, device=device).view(1, -1)
    return k_idx > (q_idx - window)


def _bias(visible: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(visible, dtype=torch.float32).masked_fill(~visible, float("-inf"))


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None,
            sinks: torch.Tensor | None, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """q [B,H,S,D], k/v [B,Hk,T,D] (Hk in {1, H}, broadcasts like ``eager_attention_forward``'s
    ``torch.matmul(query, key.transpose(2,3))``). mask is an additive fp32 bias, broadcastable
    to [B,H,S,T]. Returns (o [B,S,H,D] in `dtype`, lse [B,H,S] fp32)."""
    mm_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
    logits = torch.matmul(q.to(mm_dtype), k.to(mm_dtype).transpose(-1, -2)).float() * q.shape[-1]**-0.5
    if mask is not None:
        logits = logits + mask
    if sinks is not None:
        col = sinks.float().view(1, -1, 1, 1).expand(logits.shape[0], -1, logits.shape[2], 1)
        logits = torch.cat([logits, col], dim=-1)
    lse = torch.logsumexp(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    if sinks is not None:
        probs = probs[..., :-1]  # sink column only feeds the denominator
    o = torch.matmul(probs.to(mm_dtype), v.to(mm_dtype))
    return o.to(mm_dtype).transpose(1, 2), lse


def golden(q: torch.Tensor, kv: torch.Tensor, *, level: Level, causal: bool = True,
           sinks: torch.Tensor | None = None, v: torch.Tensor | None = None,
           window: int | None = None, main_kv: torch.Tensor | None = None,
           compress_ratio: int | None = None, indices: torch.Tensor | None = None,
           dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """BSHD in, BSHD out. Scale is always ``D**-0.5``. See module docstring for the
    exact per-level semantics and ``sinks``: [H] fp32 optional, one extra logit column
    dropped from the numerator (``eager_attention_forward`` semantics) -- rows that are
    otherwise fully masked get a finite result when a sink is present.

    - dense: ``kv`` is K [B,S,H,D], ``v`` [B,S,H,D] required (separate per-head K/V).
    - latent: ``kv`` [B,S,1,D] is K==V, shared by all H query heads (``v`` must be None).
    - window: latent + a sliding band of ``window`` raw tokens (see module docstring
      for the exact off-by-one, verified against ``create_sliding_window_causal_mask``).
    - compressed: window + a second KV source ``main_kv`` [B,G,1,D]; entry ``g`` visible
      to token ``t`` iff ``g < (t+1) // compress_ratio`` (group-causal, one softmax over
      window + main + sink, exactly like the model's KV-axis concatenation).
    - sparse: compressed + ``indices`` [B,S,topk] int64 (-1 = empty slot) restricts the
      main entries to exactly the listed ones; group-causality of ``indices`` is the
      indexer's job, not golden's.
    """
    _check_kwargs(level, v=v, window=window, main_kv=main_kv, compress_ratio=compress_ratio, indices=indices)
    b, s, h, d = q.shape
    device = q.device
    q_bhsd = q.transpose(1, 2)

    if level == "dense":
        assert kv.shape[2] == h, "dense expects a per-head K [B,S,H,D]"
        mask = _bias(_causal(s, s, device)).view(1, 1, s, s) if causal else None
        return _attend(q_bhsd, kv.transpose(1, 2), v.transpose(1, 2), mask, sinks, dtype)

    assert kv.shape[2] == 1, "latent+ levels expect a shared K==V latent [B,S,1,D]"
    lat = kv.transpose(1, 2)  # [B,1,S,D]

    if level == "latent":
        mask = _bias(_causal(s, s, device)).view(1, 1, s, s) if causal else None
        return _attend(q_bhsd, lat, lat, mask, sinks, dtype)

    win_visible = _band(s, window, device)
    if causal:
        win_visible = win_visible & _causal(s, s, device)
    win_mask = _bias(win_visible).view(1, 1, s, s)

    if level == "window":
        return _attend(q_bhsd, lat, lat, win_mask, sinks, dtype)

    g = main_kv.shape[1]
    main = main_kv.transpose(1, 2)  # [B,1,G,D]
    t_idx = torch.arange(s, device=device).view(-1, 1)
    g_idx = torch.arange(g, device=device).view(1, -1)
    group_visible = g_idx < ((t_idx + 1) // compress_ratio)
    main_mask = _bias(group_visible).view(1, 1, s, g)

    if level == "compressed":
        k = v = torch.cat([lat, main], dim=2)
        mask = torch.cat([win_mask, main_mask], dim=-1)
        return _attend(q_bhsd, k, v, mask, sinks, dtype)

    assert level == "sparse"
    safe = torch.where(indices < 0, torch.full_like(indices, g), indices)
    sparse_bias = q.new_zeros(b, s, g + 1, dtype=torch.float32).fill_(float("-inf"))
    sparse_bias.scatter_(-1, safe, 0.0)
    sparse_mask = sparse_bias[..., :g].view(b, 1, s, g)
    k = v = torch.cat([lat, main], dim=2)
    mask = torch.cat([win_mask.expand(b, 1, s, s), sparse_mask], dim=-1)
    return _attend(q_bhsd, k, v, mask, sinks, dtype)


def gpt_oss_ref(q: torch.Tensor, kv: torch.Tensor, *, causal: bool = True,
                 sinks: torch.Tensor | None = None, window: int | None = None,
                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Second, implementation-independent reference for the latent/window levels:
    calls transformers' real ``gpt_oss`` ``eager_attention_forward`` directly instead
    of re-deriving the math in ``_attend``. Worth having because DSV4.1's own eager
    path (what ``golden`` was cross-checked against above) is a copy of gpt-oss's --
    this checks against the upstream original, not the vendored copy.

    q [B,S,H,D], kv [B,S,1,D] (K==V latent, shared by all H heads via
    ``num_key_value_groups=H``, mirroring MQA). ``sinks`` [H] or None (all ``-inf`` ->
    the extra logit column contributes zero probability, a documented no-op: gpt-oss
    subtracts the row max, which is finite from the real logits, before softmax, so
    the ``-inf`` sink never becomes the max and never produces a NaN). ``window``
    follows ``golden(level="window")``'s convention (``sliding_window_overlay`` keeps
    ``kv_idx > q_idx - window``, i.e. W tokens including self).

    Mask: built with ``transformers.masking_utils.eager_mask`` (the library's own
    float-mask factory) fed ``causal_mask_function`` / ``sliding_window_causal_mask_function``
    directly -- no config object needed. Cast to `dtype` so the bf16 path mirrors the
    real model's bf16 mask+logits (unlike ``golden``, which always keeps logits fp32).
    """
    b, s, h, d = q.shape
    device = q.device
    q_bhsd, k_bhsd = q.to(dtype).transpose(1, 2), kv.to(dtype).transpose(1, 2)
    mask = None
    if window is not None:
        fn = sliding_window_causal_mask_function(window) if causal else sliding_window_overlay(window)
    elif causal:
        fn = causal_mask_function
    else:
        fn = None
    if fn is not None:
        mask = eager_mask(b, s, s, mask_function=fn, dtype=dtype, device=device)
    module = type("_GptOssStub", (), {})()
    module.num_key_value_groups = h
    module.training = False
    module.sinks = sinks.to(dtype) if sinks is not None else torch.full((h,), float("-inf"), dtype=dtype, device=device)
    o, _ = _gpt_oss_eager(module, q_bhsd, k_bhsd, k_bhsd, mask, scaling=d**-0.5, dropout=0.0)
    return o.to(dtype)


def indexer_scores(q_idx: torch.Tensor, k_idx: torch.Tensor, weights: torch.Tensor,
                    compress_ratio: int) -> torch.Tensor:
    """``DeepseekV41Indexer.forward`` steps 2-3, fp4 fake-quant and hierarchical
    candidates skipped (see module docstring). ``q_idx`` [B,S,Hi,Di] (already rotated),
    ``k_idx`` [B,G,Di], ``weights`` [B,S,Hi] (already scaled by ``Hi**-0.5``) ->
    ``[B,S,G]`` fp32: ``sum_h weights[b,s,h] * relu(q·k) * Di**-0.5``, group-causal
    ``-inf`` for ``g >= (s+1) // compress_ratio``."""
    di = q_idx.shape[-1]
    raw = torch.einsum("bshd,bgd->bshg", q_idx.float(), k_idx.float()).relu_() * di**-0.5
    scores = (raw * weights.float().unsqueeze(-1)).sum(dim=2)  # [B,S,G]
    s, g = q_idx.shape[1], k_idx.shape[1]
    s_idx = torch.arange(s, device=scores.device).view(-1, 1)
    g_idx = torch.arange(g, device=scores.device).view(1, -1)
    visible = g_idx < ((s_idx + 1) // compress_ratio)
    return scores.masked_fill(~visible.unsqueeze(0), float("-inf"))


def topk_indices(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """``DeepseekV41Indexer.forward`` step 5. ``[B,S,G]`` fp32 scores -> ``[B,S,topk]``
    int64: the top-k group indices per query, ``-1`` where the pick was ``-inf``
    (fewer than ``topk`` groups reachable yet) or ``topk > G`` ran out of columns --
    mirrors the model's "safe" dummy-slot logic without materialising the dummy
    column itself."""
    g = scores.shape[-1]
    k = min(topk, g)
    top = scores.topk(k, dim=-1, sorted=False)
    idx = torch.where(top.values > float("-inf"), top.indices, torch.full_like(top.indices, -1))
    if k < topk:
        pad = idx.new_full((*idx.shape[:-1], topk - k), -1)
        idx = torch.cat([idx, pad], dim=-1)
    return idx
