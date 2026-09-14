"""Bench swa_attn against latent_attn (full causal) and torch SDPA with an explicit band mask.

    python -m mzoo.layers.attn.swa_attn.bench --dim 128 --heads 64
    python -m mzoo.layers.attn.swa_attn.bench --dim 128 --heads 64 --mode bwd

``--mode fwd`` times ``fwd`` (forward only); ``--mode bwd`` times fwd+bwd through
torch autograd via the ``attn`` wrapper. Three rows, all on the same inputs:

- ``swa``    -- this package, ``window`` visible tokens per query.
- ``latent`` -- ``latent_attn`` with full causal attention: the same kernel family
  without the band, i.e. what the window buys.
- ``sdpa``   -- ``F.scaled_dot_product_attention`` fed ``kv.expand(B, S, H, D)`` and
  an explicit boolean band mask ``(k <= t) & (k > t - window)``. That is torch's
  own windowed path; note it still evaluates the full S x S score matrix, which is
  precisely why an in-kernel band is worth having.

TFLOPS are reported against the *banded* work (``sum_t min(t+1, window)`` visible
score entries), so every row is measured against the math the window actually
requires -- the two baselines do more than that by construction. Our rows always
run with a random per-head sink; SDPA has no sink.
"""

import fire
import torch
import torch.nn.functional as F
from tilelang.profiler import do_bench

from mzoo.layers.attn.latent_attn.attn import attn as latent_attn
from mzoo.layers.attn.latent_attn.fwd import fwd as latent_fwd
from mzoo.layers.attn.swa_attn.attn import attn
from mzoo.layers.attn.swa_attn.fwd import fwd


def _expand(kv, heads):
    return kv.expand(kv.shape[0], kv.shape[1], heads, kv.shape[3])


def _band_mask(seq_len, window, device):
    t = torch.arange(seq_len, device=device)
    return (t.view(1, -1) <= t.view(-1, 1)) & (t.view(1, -1) > t.view(-1, 1) - window)


def _fwd_runs(q, kv, sinks, window):
    heads = q.shape[2]
    mask = _band_mask(q.shape[1], window, q.device)
    qt, kt = (x.transpose(1, 2).contiguous() for x in (q, _expand(kv, heads)))
    return 4.0, {
        "swa": lambda: fwd(q, kv, window=window, sinks=sinks),
        "latent": lambda: latent_fwd(q, kv, causal=True, sinks=sinks),
        "sdpa": lambda: F.scaled_dot_product_attention(qt, kt, kt, attn_mask=mask),
    }


def _bwd_runs(q, kv, sinks, window):
    """fwd+bwd, graph rebuilt each call so every side pays the same autograd cost."""
    heads = q.shape[2]
    mask = _band_mask(q.shape[1], window, q.device)
    q, kv = (x.detach().requires_grad_() for x in (q, kv))
    sinks = sinks.detach().requires_grad_()  # so dsinks is timed too
    do = torch.randn_like(q)

    def step(fn):
        for x in (q, kv, sinks):
            x.grad = None
        fn().backward(do)

    def sdpa():
        k = _expand(kv, heads).transpose(1, 2)
        return F.scaled_dot_product_attention(q.transpose(1, 2), k, k, attn_mask=mask).transpose(1, 2)

    # 2 matmuls fwd + 3 bwd, in units of the 2*B*H*S*S*D "flops_per_matmul"
    return 10.0, {
        "swa": lambda: step(lambda: attn(q, kv, window=window, sinks=sinks)),
        "latent": lambda: step(lambda: latent_attn(q, kv, causal=True, sinks=sinks)),
        "sdpa": lambda: step(sdpa),
    }


def main(dim: int = 128, seqlen: int = 4096, batch: int = 1, heads: int = 64, window: int = 128,
         mode: str = "fwd"):
    q = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(batch, seqlen, 1, dim, device="cuda", dtype=torch.bfloat16)
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
    factor, runs = (_fwd_runs if mode == "fwd" else _bwd_runs)(q, kv, sinks, window)

    w = min(window, seqlen)
    band = w * seqlen - w * (w - 1) // 2  # sum_t min(t+1, window) -- already summed over query tokens
    flops = factor * batch * heads * dim * band
    print(f"{mode}: B={batch} H={heads} S={seqlen} D={dim} window={window} bf16 "
          f"(swa/latent with per-head sink, sdpa without; TFLOPS vs banded work)")
    base = None
    for name, fn in runs.items():
        ms = do_bench(fn, warmup=25, rep=50)
        base = base or ms
        print(f"{name:>6}: {ms:8.3f} ms  {flops / ms * 1e-9:8.2f} TFLOPS  {ms / base:5.2f}x swa")


if __name__ == "__main__":
    fire.Fire(main)
