"""Bench csa2_attn (top-k gather) against csa_attn (dense main loop) on the same inputs.

    python -m mzoo.layers.attn.csa2_attn.bench --dim 128 --heads 64
    python -m mzoo.layers.attn.csa2_attn.bench --dim 128 --heads 64 --groups "(4096, 16384)"
    python -m mzoo.layers.attn.csa2_attn.bench --fwd_only

**The only baseline is ``csa_attn``**: the previous package, same ``q``/``kv``/``main_kv``,
with the main source walked densely instead of gathered. The point of the tables is the
scaling: the gather walks exactly ``topk`` entries per token whatever ``G`` is, while the
dense loop walks every group-causally visible entry.

Note what the dense loop *can* walk: ``sum_t min(G, (t+1) // ratio)`` with ``ratio=1``, so at
``G >= S`` it saturates at ``S(S+1)/2`` -- at ``G = 16384, S = 4096`` csa_attn is capped by
group-causality at 4096 visible entries per token and cannot attend the rest of the cache at
all, while csa2 picks its 512 from anywhere in it. Both counts are printed per row.

Table 1 (forward) has a third column timing the same csa2 kernel with ``topk=0`` (the gather
dropped at compile time, i.e. the window branch alone), so ``gather ms`` is the gathered
source's own cost and ``GB/s`` is the effective bandwidth of the ``B*S*topk*D`` bf16 rows it
pulls -- the number that says whether the random-access gather is latency-bound on this GPU
(273 GB/s peak).

Table 2 (backward) times ``bwd`` on the forward's own ``(o, lse)`` and splits it into its
three kernels: ``preprocess`` (delta), ``bwd_kv`` (the window slice's dKV, byte-for-byte
``csa_attn``'s) and ``bwd_dq`` (dQ **and** the scatter-added dmain_kv). The last columns are
the **atomic ablation** (``bwd_dq.py``'s ``scatter`` knob): ``dmain ms`` is ``atomic`` minus
``none`` -- the whole cost of producing dmain_kv, its two GEMMs included -- and
``atomic-over-store`` is ``atomic`` minus a plain store to the *same* addresses, i.e. the
read-modify-write and the contention on their own.

Note the GPU may be shared with other work; re-run if a row looks anomalous.
"""

import fire
import torch
from tilelang.profiler import do_bench

from mzoo.layers.attn.csa2_attn.bwd import CONFIGS, bwd
from mzoo.layers.attn.csa2_attn.bwd_dq import bwd_dq
from mzoo.layers.attn.csa2_attn.bwd_kernels import bwd_kv, preprocess
from mzoo.layers.attn.csa2_attn.fwd import fwd
from mzoo.layers.attn.csa_attn.bwd import bwd as csa_bwd
from mzoo.layers.attn.csa_attn.fwd import fwd as csa_fwd
from mzoo.layers.attn.golden_ref import topk_indices

BENCH = dict(warmup=25, rep=50)


def _indices(batch, seq_len, groups, ratio, topk):
    """Random valid group-causal top-k sets (``golden_ref``'s contract, ``-1`` = empty)."""
    scores = torch.rand(batch, seq_len, groups, device="cuda")
    t = torch.arange(seq_len, device="cuda").view(-1, 1)
    g = torch.arange(groups, device="cuda").view(1, -1)
    scores.masked_fill_(~(g < ((t + 1) // ratio)), float("-inf"))
    return topk_indices(scores, topk).int()


def _kernel_split(q, kv, main_kv, idx, o, lse, do, window, ratio):
    """ms for ``preprocess``, ``bwd_kv`` and ``bwd_dq`` in its three scatter modes.

    Mirrors ``bwd``'s packing rather than importing it, so the three launches can be timed
    on their own; the numbers must add up to the ``bwd`` column plus the torch epilogue.
    """
    batch, seq_len, heads, dim = q.shape
    groups, topk = main_kv.shape[1], idx.shape[2]
    kv_cfg, dq_cfg = CONFIGS[dim]["kv"], CONFIGS[dim]["dq"]
    rows = seq_len * heads
    qp, op, dop = (x.contiguous().view(batch, rows, dim) for x in (q, o, do))
    kvp = kv.contiguous().view(batch, seq_len, dim)
    mainp = main_kv.contiguous().view(batch, groups, dim)
    lsep = lse.permute(0, 2, 1).reshape(batch, rows).contiguous()
    tiles = -(-(kv_cfg["block_M"] + window - 1) // (kv_cfg["block_N"] // heads))
    splits = max(1, min(12, tiles, -(-768 // (batch * -(-seq_len // kv_cfg["block_M"])))))

    pre = preprocess(batch, rows, dim)
    delta = pre(op, dop)
    parts = torch.empty(splits, batch, seq_len, dim, device=q.device, dtype=torch.float32)
    kvk = bwd_kv(batch, heads, seq_len, dim, window, True, splits, **kv_cfg)
    dmain = torch.zeros(batch, groups, dim, device=q.device, dtype=torch.float32)
    dqk = [bwd_dq(batch, heads, seq_len, dim, groups, topk, window, ratio, True, s, **dq_cfg)
           for s in ("atomic", "store", "none")]
    return (do_bench(lambda: pre(op, dop), **BENCH),
            do_bench(lambda: kvk(qp, kvp, dop, lsep, delta, parts), **BENCH),
            *[do_bench(lambda k=k: k(qp, kvp, mainp, idx, dop, lsep, delta, dmain), **BENCH)
              for k in dqk])


def main(dim: int = 128, seqlen: int = 4096, batch: int = 1, heads: int = 64, window: int = 128,
         topk: int = 512, ratio: int = 1, groups: tuple = (4096, 16384), fwd_only: bool = False):
    q = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(batch, seqlen, 1, dim, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
    t = torch.arange(seqlen)
    print(f"B={batch} H={heads} S={seqlen} D={dim} window={window} topk={topk} bf16, "
          f"per-head sink; csa_attn walks every visible main entry, csa2_attn gathers topk")
    print("fwd:")
    rows = {}
    for g in groups:
        main_kv = torch.randn(batch, g, 1, dim, device="cuda", dtype=torch.bfloat16)
        idx = _indices(batch, seqlen, g, ratio, topk)
        none = idx[:, :, :0]
        kw = dict(window=window, compress_ratio=ratio, sinks=sinks)
        dense = do_bench(lambda: csa_fwd(q, kv, main_kv, **kw), **BENCH)
        sparse = do_bench(lambda: fwd(q, kv, main_kv, idx, **kw), **BENCH)
        win = do_bench(lambda: fwd(q, kv, main_kv, none, **kw), **BENCH)
        visible = torch.clamp((t + 1) // ratio, max=g).sum().item()
        gathered = batch * seqlen * topk
        gb = batch * seqlen * topk * dim * 2 / (sparse - win) * 1e-6  # bf16 rows pulled per second
        print(f"  G={g:6d}: csa_attn {dense:8.3f} ms ({visible / seqlen:7.1f} main entries/token)   "
              f"csa2_attn {sparse:7.3f} ms ({gathered / seqlen:5.0f}/token)   "
              f"{dense / sparse:5.2f}x   window-only {win:6.3f} ms -> gather {sparse - win:6.3f} ms "
              f"= {gb:6.1f} GB/s")
        rows[g] = (main_kv, idx, dense, sparse)
    if fwd_only:
        return

    print("bwd (same inputs, dO random; csa2's three kernels split out, then the dmain_kv "
          "ablations -- see the module docstring):")
    for g, (main_kv, idx, dense_f, sparse_f) in rows.items():
        kw = dict(window=window, compress_ratio=ratio, sinks=sinks)
        o_c, lse_c = csa_fwd(q, kv, main_kv, **kw)
        o_s, lse_s = fwd(q, kv, main_kv, idx, **kw)
        dense = do_bench(lambda: csa_bwd(q, kv, main_kv, o_c, lse_c, do, **kw), **BENCH)
        sparse = do_bench(lambda: bwd(q, kv, main_kv, idx, o_s, lse_s, do, **kw), **BENCH)
        pre, kvk, dq, dq_st, dq_no = _kernel_split(q, kv, main_kv, idx, o_s, lse_s, do, window, ratio)
        print(f"  G={g:6d}: csa_attn {dense:8.3f} ms   csa2_attn {sparse:8.3f} ms   {dense / sparse:5.2f}x"
              f"   [preprocess {pre:5.3f} + dKV {kvk:6.3f} + dQ/dmain {dq:7.3f}]"
              f"   dmain {dq - dq_no:6.3f} ms of which atomic-over-store {dq - dq_st:6.3f} ms"
              f" ({(dq - dq_no) / dq * 100:4.1f}% / {(dq - dq_st) / dq * 100:4.1f}% of dQ)")
        print(f"           fwd+bwd: csa_attn {dense_f + dense:8.3f} ms   "
              f"csa2_attn {sparse_f + sparse:8.3f} ms   {(dense_f + dense) / (sparse_f + sparse):5.2f}x")


if __name__ == "__main__":
    fire.Fire(main)
