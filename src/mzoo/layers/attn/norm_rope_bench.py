"""Bench + memory probe for the QK prologue (ticket 0011).

    python -m mzoo.layers.attn.norm_rope_bench --dim 128 --rope_dim 64
    python -m mzoo.layers.attn.norm_rope_bench --memory

Eager norm-then-rope vs the fused TileLang kernel; median ms and effective
GB/s counting q read + out write. ``--memory`` reports the autograd footprint
of a fwd-with-grad for q ``[1, 4096, 64, 128]``.
"""

import fire
import torch
from tilelang.profiler import do_bench

from mzoo.layers.attn.norm_rope import qk_norm_rope, qk_norm_rope_eager


def _inputs(batch, seq, heads, dim, rope_dim, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(batch, seq, heads, dim, device="cuda", dtype=dtype)
    weight = torch.randn(dim, device="cuda", dtype=dtype)
    pos = torch.arange(seq, device="cuda", dtype=torch.float32)
    freqs = pos[:, None] / (10000.0 ** (torch.arange(0, rope_dim, 2, device="cuda") / rope_dim))
    cos, sin = freqs.cos()[None], freqs.sin()[None]
    return x, weight, cos.expand(batch, -1, -1), sin.expand(batch, -1, -1)


def main(dim: int = 128, rope_dim: int | None = None, seqlen: int = 4096, heads: int = 64, batch: int = 1):
    rd = rope_dim or (16 if dim == 64 else 64)
    args = _inputs(batch, seqlen, heads, dim, rd)
    traffic = args[0].numel() * args[0].element_size() * 2  # read x + write out
    print(f"fwd: B={batch} H={heads} S={seqlen} D={dim} rd={rd} bf16 ({traffic / 1e6:.0f} MB min traffic)")
    for name, fn in (("eager", qk_norm_rope_eager), ("tilelang", qk_norm_rope)):
        ms = do_bench(lambda: fn(*args), warmup=50, rep=100)
        print(f"{name:>8}: {ms:8.3f} ms  {traffic / ms * 1e-6:8.0f} GB/s")


def memory():
    """Allocations of one fwd-with-grad (saved tensors for the bwd), fused vs eager."""
    for name, fn in (("eager", qk_norm_rope_eager), ("tilelang", qk_norm_rope)):
        torch.manual_seed(0)
        x, weight, cos, sin = _inputs(1, 4096, 64, 128, 64)
        x, weight = (t.clone().requires_grad_() for t in (x, weight))
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        out = fn(x, weight, cos, sin)
        torch.cuda.synchronize()
        delta = (torch.cuda.memory_allocated() - before) / 1e6
        print(f"{name:>8}: fwd retained {delta:6.1f} MB (out 67.1 + saved-for-bwd)")
        del out


if __name__ == "__main__":
    fire.Fire({"main": main, "memory": memory})
