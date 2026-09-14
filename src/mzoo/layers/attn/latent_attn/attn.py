"""latent_attn: autograd wrapper tying ``fwd`` and ``bwd`` together.

Same FA2 recompute contract as ``dense_attn``, one tensor lighter: the forward
saves exactly ``(q, kv, o, lse)`` -- plus the ``[H]`` sink when one is given --
and the backward recomputes the score/probability tiles from them, so saved
memory is O(B*S*H*D), never O(S^2).
"""

import torch

from mzoo.layers.attn.latent_attn.bwd import bwd
from mzoo.layers.attn.latent_attn.fwd import fwd


class _Attn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, kv, causal, sinks):
        o, lse = fwd(q, kv, causal=causal, sinks=sinks)
        ctx.save_for_backward(*((q, kv, o, lse) if sinks is None else (q, kv, o, lse, sinks)))
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, kv, o, lse, *rest = ctx.saved_tensors
        dq, dkv, dsinks = bwd(q, kv, o, lse, do, causal=ctx.causal, sinks=rest[0] if rest else None)
        return dq, dkv, None, dsinks


def attn(q: torch.Tensor, kv: torch.Tensor, *, causal: bool = True,
         sinks: torch.Tensor | None = None) -> torch.Tensor:
    """Shared-latent MQA attention with backward.

    ``q`` bf16 [B, S, H, D], ``kv`` bf16 [B, S, 1, D] (K == V) -> ``o`` bf16 [B, S, H, D].
    ``sinks``: optional per-head learnable sink ``[H]`` fp32; its grad comes back
    in the ``sinks`` slot. ``kv``'s grad is ``dKV = dK + dV``.
    """
    return _Attn.apply(q, kv, causal, sinks)
