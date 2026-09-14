"""dense_attn: autograd wrapper tying ``fwd`` and ``bwd`` together.

FA2 autograd wrapper: forward saves exactly ``(q, k, v, o, lse)`` -- plus the
``[H]`` sink when one is given -- and backward recomputes the score/probability
tiles from them, so saved memory is O(B*S*H*D), never O(S^2).
"""

import torch

from mzoo.layers.attn.dense_attn.bwd import bwd
from mzoo.layers.attn.dense_attn.fwd import fwd


class _Attn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sinks):
        o, lse = fwd(q, k, v, causal=causal, sinks=sinks)
        ctx.save_for_backward(*((q, k, v, o, lse) if sinks is None else (q, k, v, o, lse, sinks)))
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, *rest = ctx.saved_tensors
        dq, dk, dv, dsinks = bwd(q, k, v, o, lse, do, causal=ctx.causal, sinks=rest[0] if rest else None)
        return dq, dk, dv, None, dsinks


def attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = True,
         sinks: torch.Tensor | None = None) -> torch.Tensor:
    """Dense FA2 attention with backward. q/k/v bf16 [B, S, H, D] -> o bf16 [B, S, H, D].

    ``sinks``: optional per-head learnable sink ``[H]`` fp32; its grad (fp32) comes
    back in the ``sinks`` slot.
    """
    return _Attn.apply(q, k, v, causal, sinks)
