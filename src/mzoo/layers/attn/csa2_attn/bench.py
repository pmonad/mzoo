"""Bench csa2_attn (top-k gather) against csa_attn (dense main loop) on the same inputs.

    python -m mzoo.layers.attn.csa2_attn.bench --dim 128 --heads 64
    python -m mzoo.layers.attn.csa2_attn.bench --dim 128 --heads 64 --groups "(4096, 16384)"

Forward only (the backward is ticket 0005). **The only baseline is ``csa_attn``**: the
previous package, same ``q``/``kv``/``main_kv``, with the main source walked densely instead
of gathered. The point of the table is the scaling: the gather walks exactly ``topk`` entries
per token whatever ``G`` is, while the dense loop walks every group-causally visible entry.

Note what the dense loop *can* walk: ``sum_t min(G, (t+1) // ratio)`` with ``ratio=1``, so at
``G >= S`` it saturates at ``S(S+1)/2`` -- at ``G = 16384, S = 4096`` csa_attn is capped by
group-causality at 4096 visible entries per token and cannot attend the rest of the cache at
all, while csa2 picks its 512 from anywhere in it. Both counts are printed per row.

A third column times the same csa2 kernel with ``topk=0`` (the gather dropped at compile
time, i.e. the window branch alone), so ``gather ms`` is the gathered source's own cost and
``GB/s`` is the effective bandwidth of the ``B*S*topk*D`` bf16 rows it pulls -- the number
that says whether the random-access gather is latency-bound on this GPU (273 GB/s peak).

Note the GPU may be shared with other work; re-run if a row looks anomalous.
"""

import fire
import torch
from tilelang.profiler import do_bench

from mzoo.layers.attn.csa2_attn.fwd import fwd
from mzoo.layers.attn.csa_attn.fwd import fwd as csa_fwd
from mzoo.layers.attn.golden_ref import topk_indices


def _indices(batch, seq_len, groups, ratio, topk):
    """Random valid group-causal top-k sets (``golden_ref``'s contract, ``-1`` = empty)."""
    scores = torch.rand(batch, seq_len, groups, device="cuda")
    t = torch.arange(seq_len, device="cuda").view(-1, 1)
    g = torch.arange(groups, device="cuda").view(1, -1)
    scores.masked_fill_(~(g < ((t + 1) // ratio)), float("-inf"))
    return topk_indices(scores, topk).int()


def main(dim: int = 128, seqlen: int = 4096, batch: int = 1, heads: int = 64, window: int = 128,
         topk: int = 512, ratio: int = 1, groups: tuple = (4096, 16384)):
    q = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(batch, seqlen, 1, dim, device="cuda", dtype=torch.bfloat16)
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
    t = torch.arange(seqlen)
    print(f"fwd: B={batch} H={heads} S={seqlen} D={dim} window={window} topk={topk} bf16, "
          f"per-head sink; csa_attn walks every visible main entry, csa2_attn gathers topk")
    for g in groups:
        main_kv = torch.randn(batch, g, 1, dim, device="cuda", dtype=torch.bfloat16)
        idx = _indices(batch, seqlen, g, ratio, topk)
        none = idx[:, :, :0]
        kw = dict(window=window, compress_ratio=ratio, sinks=sinks)
        dense = do_bench(lambda: csa_fwd(q, kv, main_kv, **kw), warmup=25, rep=50)
        sparse = do_bench(lambda: fwd(q, kv, main_kv, idx, **kw), warmup=25, rep=50)
        win = do_bench(lambda: fwd(q, kv, main_kv, none, **kw), warmup=25, rep=50)
        visible = torch.clamp((t + 1) // ratio, max=g).sum().item()
        gathered = batch * seqlen * topk
        gb = batch * seqlen * topk * dim * 2 / (sparse - win) * 1e-6  # bf16 rows pulled per second
        print(f"  G={g:6d}: csa_attn {dense:8.3f} ms ({visible / seqlen:7.1f} main entries/token)   "
              f"csa2_attn {sparse:7.3f} ms ({gathered / seqlen:5.0f}/token)   "
              f"{dense / sparse:5.2f}x   window-only {win:6.3f} ms -> gather {sparse - win:6.3f} ms "
              f"= {gb:6.1f} GB/s")


if __name__ == "__main__":
    fire.Fire(main)
