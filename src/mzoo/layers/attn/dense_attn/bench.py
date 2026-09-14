"""Bench dense_attn against torch SDPA.

    python -m mzoo.layers.attn.dense_attn.bench --dim 128 --seqlen 4096
    python -m mzoo.layers.attn.dense_attn.bench --mode bwd --dim 128

``--mode fwd`` times ``fwd`` (forward only); ``--mode bwd`` times fwd+bwd
through torch autograd via the ``attn`` wrapper.

Both modes always run with a random per-head sink; torch SDPA has no sink, so the
``sdpa`` row is the sink-free baseline (said so in the printed header).
"""

import fire
import torch
import torch.nn.functional as F
from tilelang.profiler import do_bench

from mzoo.layers.attn.dense_attn.attn import attn
from mzoo.layers.attn.dense_attn.fwd import fwd

NAME = "dense"


def _fwd_runs(q, k, v, sinks, causal):
    qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    return 4.0, {
        NAME: lambda: fwd(q, k, v, causal=causal, sinks=sinks),
        "sdpa": lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal),
    }


def _bwd_runs(q, k, v, sinks, causal):
    """fwd+bwd, graph rebuilt each call so both sides pay the same autograd cost."""
    q, k, v = (x.detach().requires_grad_() for x in (q, k, v))
    sinks = sinks.detach().requires_grad_()  # so dsinks is timed too
    do = torch.randn_like(q)

    def step(fn):
        for x in (q, k, v, sinks):
            x.grad = None
        fn().backward(do)

    sdpa = lambda: F.scaled_dot_product_attention(  # noqa: E731
        *(x.transpose(1, 2) for x in (q, k, v)), is_causal=causal).transpose(1, 2)
    # 2 matmuls fwd + 3 bwd, in units of the 2*B*H*S*S*D "flops_per_matmul"
    return 10.0, {NAME: lambda: step(lambda: attn(q, k, v, causal=causal, sinks=sinks)), "sdpa": lambda: step(sdpa)}


def main(dim: int = 128, seqlen: int = 4096, batch: int = 1, heads: int = 16, causal: bool = True, mode: str = "fwd"):
    q, k, v = (torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
    factor, runs = (_fwd_runs if mode == "fwd" else _bwd_runs)(q, k, v, sinks, causal)

    flops = factor * batch * heads * seqlen * seqlen * dim * (0.5 if causal else 1.0)
    print(f"{mode}: B={batch} H={heads} S={seqlen} D={dim} causal={causal} bf16 ({NAME} with per-head sink, sdpa without)")
    for name, fn in runs.items():
        ms = do_bench(fn, warmup=50, rep=100)
        print(f"{name:>6}: {ms:8.3f} ms  {flops / ms * 1e-9:8.2f} TFLOPS")


if __name__ == "__main__":
    fire.Fire(main)
