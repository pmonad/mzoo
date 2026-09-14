"""csa2_attn: public entry -- the gathered forward, **forward only**.

Ticket 0004 is the forward half of step 5; the scatter-add ``dKV`` backward is its own
ticket (0005), which brings ``bwd.py`` forward from ``csa_attn`` and turns this wrapper into
a real autograd function. Until then the node exists but its ``backward`` raises
``NotImplementedError``: building the graph is allowed (so a caller can run under
``torch.no_grad()`` or check shapes), calling ``.backward()`` is not.

``indices`` is an integer tensor and never differentiable; ``window`` and ``compress_ratio``
are plain ints carried on ``ctx``, compile-time constants of the kernel.
"""

import torch

from mzoo.layers.attn.csa2_attn.fwd import fwd


class _Attn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, kv, main_kv, indices, window, compress_ratio, causal, sinks):
        o, _ = fwd(q, kv, main_kv, indices, window=window, compress_ratio=compress_ratio,
                   causal=causal, sinks=sinks)
        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError(
            "csa2_attn is forward-only (ticket 0004); the sparse backward is ticket 0005 -- "
            "use csa_attn.attn for a differentiable dense main source in the meantime")


def attn(q: torch.Tensor, kv: torch.Tensor, main_kv: torch.Tensor, indices: torch.Tensor, *,
         window: int, compress_ratio: int, causal: bool = True,
         sinks: torch.Tensor | None = None) -> torch.Tensor:
    """Sliding-window + top-k gathered shared-latent MQA attention. **No backward yet.**

    ``q`` bf16 [B, S, H, D], ``kv`` bf16 [B, S, 1, D] (raw latent, K == V),
    ``main_kv`` bf16 [B, G, 1, D] (compressed latents), ``indices`` int32/int64 [B, S, topk]
    (``-1`` = empty slot) -> ``o`` bf16 [B, S, H, D]. See ``fwd`` for the full contract,
    including that ``compress_ratio`` is documentation only (group-causality lives in
    ``indices``).
    """
    return _Attn.apply(q, kv, main_kv, indices, window, compress_ratio, causal, sinks)
