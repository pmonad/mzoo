"""Bench csa_attn against swa_attn (window only) on the same inputs.

    python -m mzoo.layers.attn.csa_attn.bench --dim 128 --heads 64
    python -m mzoo.layers.attn.csa_attn.bench --dim 128 --heads 64 --mode bwd

``--mode fwd`` times ``fwd``; ``--mode bwd`` times fwd+bwd through torch autograd
via the ``attn`` wrapper. One row per ``compress_ratio`` in ``--ratios`` with
``G = S // ratio`` compressed entries, plus the ``swa_attn`` baseline.

**The only baseline is ``swa_attn``**: it is the same kernel family on the same
``q``/``kv`` with the compressed source removed, so the difference is exactly the
cost of the second source -- which is the number this bench exists to produce.
(No SDPA row: torch has no kernel built for "sliding band + group-causal second KV
source", so an SDPA number would measure a mask-materialising fallback, not a
comparable implementation.)

Note the GPU may be shared with other work; re-run if a row looks anomalous.
"""

import fire
import torch
from tilelang.profiler import do_bench

from mzoo.layers.attn.csa_attn.attn import attn
from mzoo.layers.attn.csa_attn.fwd import fwd
from mzoo.layers.attn.swa_attn.attn import attn as swa_attn
from mzoo.layers.attn.swa_attn.fwd import fwd as swa_fwd


def _fwd_step(q, kv, main_kv, sinks, window, ratio):
    return lambda: fwd(q, kv, main_kv, window=window, compress_ratio=ratio, sinks=sinks)


def _bwd_step(q, kv, main_kv, sinks, window, ratio):
    """fwd+bwd, graph rebuilt each call so every row pays the same autograd cost."""
    leaves = [x.detach().requires_grad_() for x in (q, kv, main_kv)] + [sinks.detach().requires_grad_()]
    do = torch.randn_like(q)

    def step():
        for x in leaves:
            x.grad = None
        attn(leaves[0], leaves[1], leaves[2], window=window, compress_ratio=ratio, sinks=leaves[3]).backward(do)

    return step


def _swa_step(q, kv, sinks, window, mode):
    if mode == "fwd":
        return lambda: swa_fwd(q, kv, window=window, sinks=sinks)
    leaves = [x.detach().requires_grad_() for x in (q, kv)] + [sinks.detach().requires_grad_()]
    do = torch.randn_like(q)

    def step():
        for x in leaves:
            x.grad = None
        swa_attn(leaves[0], leaves[1], window=window, sinks=leaves[2]).backward(do)

    return step


def main(dim: int = 128, seqlen: int = 4096, batch: int = 1, heads: int = 64, window: int = 128,
         mode: str = "fwd", ratios: tuple = (1, 2, 4)):
    q = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(batch, seqlen, 1, dim, device="cuda", dtype=torch.bfloat16)
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
    factor = 4.0 if mode == "fwd" else 10.0  # 2 matmuls fwd, +3 bwd, per visible score entry

    w = min(window, seqlen)
    band = w * seqlen - w * (w - 1) // 2  # sum_t min(t+1, window)
    print(f"{mode}: B={batch} H={heads} S={seqlen} D={dim} window={window} bf16, per-head sink; "
          f"TFLOPS vs the entries each row actually attends")
    base = do_bench(_swa_step(q, kv, sinks, window, mode), warmup=25, rep=50)
    print(f"  swa_attn (window only): {base:8.3f} ms  {factor * batch * heads * dim * band / base * 1e-9:7.2f} TFLOPS")
    for ratio in ratios:
        groups = seqlen // ratio
        main_kv = torch.randn(batch, groups, 1, dim, device="cuda", dtype=torch.bfloat16)
        step = (_fwd_step if mode == "fwd" else _bwd_step)(q, kv, main_kv, sinks, window, ratio)
        ms = do_bench(step, warmup=25, rep=50)
        # visible main entries: sum_t min(G, (t+1)//ratio)
        t = torch.arange(seqlen)
        main_entries = torch.clamp((t + 1) // ratio, max=groups).sum().item()
        flops = factor * batch * heads * dim * (band + main_entries)
        print(f"  csa_attn ratio={ratio} G={groups:5d}: {ms:8.3f} ms  {flops / ms * 1e-9:7.2f} TFLOPS  "
              f"{ms / base:5.2f}x swa  (+{100 * (ms - base) / base:6.1f}% for the second source)")


if __name__ == "__main__":
    fire.Fire(main)
