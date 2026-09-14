"""csa2_attn: autograd wrapper tying the gathered ``fwd`` and ``bwd`` together.

Same FA2 recompute contract as ``csa_attn``, one tensor wider: the forward saves exactly
``(q, kv, main_kv, indices, o, lse)`` -- plus the ``[H]`` sink when one is given -- and the
backward recomputes the score/probability tiles of both sources from them, so saved memory
is O(B*(S + G + S*topk/H/D)*H*D) and never O(S^2) or O(S*G). The ``indices`` tensor is the
only new save and is O(B*S*topk) int32; it is **not** differentiable (top-k selection is
discrete -- the indexer's own gradient path is ticket 0007), so its grad slot is ``None``.

``window`` and ``compress_ratio`` are plain ints carried on ``ctx``; they are compile-time
constants of the kernels, not differentiable inputs.
"""

import torch

from mzoo.layers.attn.csa2_attn.bwd import bwd
from mzoo.layers.attn.csa2_attn.fwd import fwd


class _Attn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, kv, main_kv, indices, window, compress_ratio, causal, sinks):
        o, lse = fwd(q, kv, main_kv, indices, window=window, compress_ratio=compress_ratio,
                     causal=causal, sinks=sinks)
        saved = (q, kv, main_kv, indices, o, lse)
        ctx.save_for_backward(*(saved if sinks is None else saved + (sinks,)))
        ctx.window, ctx.ratio, ctx.causal = window, compress_ratio, causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, kv, main_kv, indices, o, lse, *rest = ctx.saved_tensors
        dq, dkv, dmain, dsinks = bwd(q, kv, main_kv, indices, o, lse, do, window=ctx.window,
                                     compress_ratio=ctx.ratio, causal=ctx.causal,
                                     sinks=rest[0] if rest else None)
        return dq, dkv, dmain, None, None, None, None, dsinks


def attn(q: torch.Tensor, kv: torch.Tensor, main_kv: torch.Tensor, indices: torch.Tensor, *,
         window: int, compress_ratio: int, causal: bool = True,
         sinks: torch.Tensor | None = None) -> torch.Tensor:
    """Sliding-window + top-k gathered shared-latent MQA attention, with backward.

    ``q`` bf16 [B, S, H, D], ``kv`` bf16 [B, S, 1, D] (raw latent, K == V),
    ``main_kv`` bf16 [B, G, 1, D] (compressed latents), ``indices`` int32/int64 [B, S, topk]
    (``-1`` = empty slot) -> ``o`` bf16 [B, S, H, D]. See ``fwd`` for the full contract,
    including that ``compress_ratio`` is documentation only (group-causality lives in
    ``indices``).

    Gradients come back for ``q``, ``kv``, ``main_kv`` and ``sinks``; ``indices`` gets
    ``None``. ``dmain_kv`` is accumulated with fp32 atomics and is therefore **not**
    bitwise reproducible run to run (``bwd.py``, ``README.md`` -> Known issues).
    """
    return _Attn.apply(q, kv, main_kv, indices, window, compress_ratio, causal, sinks)
