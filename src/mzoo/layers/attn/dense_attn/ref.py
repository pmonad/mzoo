"""Torch references for the attn kernels. BSHD in, BSHD out.

``sinks [H]`` is optional and follows the DSV4.1 ``eager_attention_forward``
semantics: the per-head scalar joins the logits as one extra column, softmax runs
over the concatenation, and the column is dropped from the numerator -- so it
only enlarges the denominator (and the LSE). Everything is fp32, and the sink
path is plain autograd-friendly torch so ``dsinks`` can be checked too.
"""

import torch
import torch.nn.functional as F


def _logits(q: torch.Tensor, k: torch.Tensor, causal: bool, sinks: torch.Tensor | None) -> torch.Tensor:
    """fp32 attention logits [B, H, S, S (+1 sink column)] from BSHD q/k."""
    qt, kt = (x.transpose(1, 2).float() for x in (q, k))
    s = qt @ kt.transpose(-1, -2) * q.shape[-1]**-0.5
    if causal:
        n = q.shape[1]
        s = s.masked_fill(torch.ones(n, n, device=s.device, dtype=torch.bool).triu(1), float("-inf"))
    if sinks is not None:
        col = sinks.float().view(1, -1, 1, 1).expand(s.shape[0], -1, s.shape[2], 1)
        s = torch.cat([s, col], dim=-1)
    return s


def sdpa_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
             sinks: torch.Tensor | None = None) -> torch.Tensor:
    """Dense SDPA in fp32. q/k/v are [B, S, H, D]; returns [B, S, H, D] in fp32."""
    if sinks is None:
        qt, kt, vt = (x.transpose(1, 2).float() for x in (q, k, v))
        o = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
        return o.transpose(1, 2).contiguous()
    p = _logits(q, k, causal, sinks).softmax(dim=-1)[..., :-1]  # sink column feeds the denominator only
    return (p @ v.transpose(1, 2).float()).transpose(1, 2).contiguous()


def sdpa_ref_lse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
                 sinks: torch.Tensor | None = None) -> torch.Tensor:
    """Natural-log log-sum-exp of the attention logits, fp32. q/k are [B, S, H, D] -> [B, H, S]."""
    return torch.logsumexp(_logits(q, k, causal, sinks), dim=-1)
