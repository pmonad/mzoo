"""csa_attn: autograd wrapper tying ``fwd`` and ``bwd`` together.

Same FA2 recompute contract as ``swa_attn``, one tensor wider: the forward saves
exactly ``(q, kv, main_kv, o, lse)`` -- plus the ``[H]`` sink when one is given --
and the backward recomputes the score/probability tiles of **both** sources from
them, so saved memory is O(B*(S+G)*H*D), never O(S^2) or O(S*G). ``window`` and
``compress_ratio`` are plain ints carried on ``ctx``; they are compile-time
constants of the kernels, not differentiable inputs.
"""

import torch

from mzoo.layers.attn.csa_attn.bwd import bwd
from mzoo.layers.attn.csa_attn.fwd import fwd


class _Attn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, kv, main_kv, window, compress_ratio, causal, sinks):
        o, lse = fwd(q, kv, main_kv, window=window, compress_ratio=compress_ratio,
                     causal=causal, sinks=sinks)
        saved = (q, kv, main_kv, o, lse)
        ctx.save_for_backward(*(saved if sinks is None else saved + (sinks,)))
        ctx.window, ctx.ratio, ctx.causal = window, compress_ratio, causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, kv, main_kv, o, lse, *rest = ctx.saved_tensors
        dq, dkv, dmain, dsinks = bwd(q, kv, main_kv, o, lse, do, window=ctx.window,
                                     compress_ratio=ctx.ratio, causal=ctx.causal,
                                     sinks=rest[0] if rest else None)
        return dq, dkv, dmain, None, None, None, dsinks


def attn(q: torch.Tensor, kv: torch.Tensor, main_kv: torch.Tensor, *, window: int,
         compress_ratio: int, causal: bool = True, sinks: torch.Tensor | None = None) -> torch.Tensor:
    """Sliding-window + dense compressed shared-latent MQA attention, with backward.

    ``q`` bf16 [B, S, H, D], ``kv`` bf16 [B, S, 1, D] (raw latent, K == V),
    ``main_kv`` bf16 [B, G, 1, D] (compressed latents) -> ``o`` bf16 [B, S, H, D].

    ``window``: visible raw KV tokens per query, self included (``t - window + 1 <= k <= t``).
    ``compress_ratio``: main entry ``g`` is visible to token ``t`` iff ``g < (t + 1) // ratio``.
    ``G == 0`` reduces to ``swa_attn``. ``causal=False`` is rejected.
    ``sinks``: optional per-head learnable sink ``[H]`` fp32; its grad comes back in
    the ``sinks`` slot. Both KV gradients are ``dK + dV``.
    """
    return _Attn.apply(q, kv, main_kv, window, compress_ratio, causal, sinks)
