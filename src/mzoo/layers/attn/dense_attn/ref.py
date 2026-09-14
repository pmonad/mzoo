"""Torch references for the attn kernels. BSHD in, BSHD out.

``sinks [H]`` is optional and follows the DSV4.1 ``eager_attention_forward``
semantics: the per-head scalar joins the logits as one extra column, softmax runs
over the concatenation, and the column is dropped from the numerator -- so it
only enlarges the denominator (and the LSE). Everything is fp32, and the sink
path is plain autograd-friendly torch so ``dsinks`` can be checked too.

``torch_bf16_ref`` runs the same computation in bf16 (torch's own kernel) and
``assert_within_2x_torch`` is the shared acceptance criterion: a kernel may be at
most ``factor`` times as far from the fp32 reference as torch's bf16 kernel is on
the same inputs. Every attn package should reuse these two instead of a fixed atol.
"""

import torch
import torch.nn.functional as F


def _logits(q: torch.Tensor, k: torch.Tensor, causal: bool, sinks: torch.Tensor | None,
            dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Attention logits [B, H, S, S (+1 sink column)] from BSHD q/k, computed in ``dtype``."""
    qt, kt = (x.transpose(1, 2).to(dtype) for x in (q, k))
    s = qt @ kt.transpose(-1, -2) * q.shape[-1]**-0.5
    if causal:
        n = q.shape[1]
        s = s.masked_fill(torch.ones(n, n, device=s.device, dtype=torch.bool).triu(1), float("-inf"))
    if sinks is not None:
        col = sinks.to(dtype).view(1, -1, 1, 1).expand(s.shape[0], -1, s.shape[2], 1)
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


def torch_bf16_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
                    sinks: torch.Tensor | None = None) -> torch.Tensor:
    """Same computation as ``sdpa_ref`` but run in bf16 -- the baseline for the acceptance
    criterion. No sinks: ``F.scaled_dot_product_attention`` on bf16 q/k/v. With sinks: bf16
    matmul for the logits, softmax in fp32, probabilities cast back to bf16 before the PV
    matmul, mirroring what a bf16 kernel does.
    """
    if sinks is None:
        qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
        o = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
        return o.transpose(1, 2).contiguous()
    p = _logits(q, k, causal, sinks, dtype=torch.bfloat16).float().softmax(dim=-1)[..., :-1].to(torch.bfloat16)
    return (p @ v.transpose(1, 2).to(torch.bfloat16)).transpose(1, 2).contiguous()


def assert_within_2x_torch(got: torch.Tensor, ref: torch.Tensor, torch_bf16: torch.Tensor, name: str,
                            factor: float = 2.0, floor: float = 1e-5) -> None:
    """FlashAttention acceptance criterion: ``got``'s max-abs error against the fp32 ``ref`` may be
    at most ``factor`` times torch's own bf16 kernel's (``torch_bf16``) error against that same
    ``ref``, plus ``floor`` so a zero torch error cannot make the bound zero.
    """
    err_got = (got.float() - ref.float()).abs().max().item()
    err_torch = (torch_bf16.float() - ref.float()).abs().max().item()
    ratio = err_got / err_torch if err_torch > 0 else (float("inf") if err_got > 0 else 1.0)
    assert err_got <= factor * err_torch + floor, (
        f"{name}: err={err_got:.3e} torch_err={err_torch:.3e} ratio={ratio:.2f}x factor={factor}"
    )
