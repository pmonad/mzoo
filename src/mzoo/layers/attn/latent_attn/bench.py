"""Bench latent_attn against torch SDPA with the latent expanded to H heads.

    python -m mzoo.layers.attn.latent_attn.bench --dim 128 --heads 64
    python -m mzoo.layers.attn.latent_attn.bench --mode bwd --dim 128

``--mode fwd`` times ``fwd`` (forward only); ``--mode bwd`` times fwd+bwd through
torch autograd via the ``attn`` wrapper.

The baseline is ``F.scaled_dot_product_attention`` fed ``kv.expand(B, S, H, D)``
-- torch materialises nothing but it does read the same latent H times, which is
exactly the traffic ``latent_attn`` saves. Both modes always run with a random
per-head sink on our side; SDPA has no sink, so that row is sink-free.
"""

import fire
import torch
import torch.nn.functional as F
from tilelang.profiler import do_bench

from mzoo.layers.attn.latent_attn.attn import attn
from mzoo.layers.attn.latent_attn.fwd import fwd

NAME = "latent"


def _expand(kv, heads):
    return kv.expand(kv.shape[0], kv.shape[1], heads, kv.shape[3])


def _fwd_runs(q, kv, sinks, causal):
    heads = q.shape[2]
    qt, kt = (x.transpose(1, 2).contiguous() for x in (q, _expand(kv, heads)))
    return 4.0, {
        NAME: lambda: fwd(q, kv, causal=causal, sinks=sinks),
        "sdpa": lambda: F.scaled_dot_product_attention(qt, kt, kt, is_causal=causal),
    }


def _bwd_runs(q, kv, sinks, causal):
    """fwd+bwd, graph rebuilt each call so both sides pay the same autograd cost."""
    heads = q.shape[2]
    q, kv = (x.detach().requires_grad_() for x in (q, kv))
    sinks = sinks.detach().requires_grad_()  # so dsinks is timed too
    do = torch.randn_like(q)

    def step(fn):
        for x in (q, kv, sinks):
            x.grad = None
        fn().backward(do)

    def sdpa():
        k = _expand(kv, heads).transpose(1, 2)
        return F.scaled_dot_product_attention(q.transpose(1, 2), k, k, is_causal=causal).transpose(1, 2)

    # 2 matmuls fwd + 3 bwd, in units of the 2*B*H*S*S*D "flops_per_matmul"
    return 10.0, {NAME: lambda: step(lambda: attn(q, kv, causal=causal, sinks=sinks)), "sdpa": lambda: step(sdpa)}


def main(dim: int = 128, seqlen: int = 4096, batch: int = 1, heads: int = 64, causal: bool = True, mode: str = "fwd"):
    q = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(batch, seqlen, 1, dim, device="cuda", dtype=torch.bfloat16)
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
    factor, runs = (_fwd_runs if mode == "fwd" else _bwd_runs)(q, kv, sinks, causal)

    flops = factor * batch * heads * seqlen * seqlen * dim * (0.5 if causal else 1.0)
    print(f"{mode}: B={batch} H={heads} S={seqlen} D={dim} causal={causal} bf16 ({NAME} with per-head sink, sdpa without)")
    for name, fn in runs.items():
        ms = do_bench(fn, warmup=50, rep=100)
        print(f"{name:>6}: {ms:8.3f} ms  {flops / ms * 1e-9:8.2f} TFLOPS")


if __name__ == "__main__":
    fire.Fire(main)
