"""Bench the indexer score kernel and its top-k against a torch einsum baseline.

    python -m mzoo.layers.attn.indexer.bench

Two rows only (ticket 0006): B=1, S=4096, Hi=32, Di=128, T in {4096, 16384},
``compress_ratio=1`` so T=4096 is fully visible and T=16384 exercises the
group-causal early-out (only the first 4096 entries can ever be visible).

Score kernel and top-k are timed **separately** so ticket 6b's fuse/no-fuse call is
data-driven. The baseline is the same math in torch -- bf16 ``q @ k^T``, relu, fp32
weighted head reduce, group-causal ``-inf`` -- chunked over the query axis, because
the model's unchunked ``einsum("bshd,btd->bsht")`` materialises a ``[B,S,Hi,T]``
intermediate (8.6 GB fp32 at the T=16384 row) that would measure allocator traffic
rather than the operation.
"""

import fire
import torch
from tilelang.profiler import do_bench

from mzoo.layers.attn.indexer.fwd import fwd

CHUNK = 512  # query rows per torch-baseline chunk


def torch_scores(q, k, w, compress_ratio):
    """bf16 matmul + fp32 relu/weight/reduce + group-causal mask, chunked over S."""
    b, s, h, d = q.shape
    t = k.shape[1]
    out = q.new_empty((b, s, t), dtype=torch.float32)
    t_idx = torch.arange(t, device=q.device).view(1, -1)
    for i in range(0, s, CHUNK):
        j = min(i + CHUNK, s)
        raw = torch.einsum("bshd,btd->bsht", q[:, i:j], k).float().relu_() * d**-0.5
        chunk = (raw * w[:, i:j].unsqueeze(-1)).sum(dim=2)
        vis = t_idx < ((torch.arange(i, j, device=q.device).view(-1, 1) + 1) // compress_ratio)
        out[:, i:j] = chunk.masked_fill(~vis, float("-inf"))
    return out


def main(batch: int = 1, seqlen: int = 4096, heads: int = 32, dim: int = 128,
         compress_ratio: int = 1, topk: int = 512, kv_lens: tuple = (4096, 16384)):
    torch.manual_seed(0)
    q = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(batch, seqlen, heads, device="cuda") * heads**-0.5
    print(f"B={batch} S={seqlen} Hi={heads} Di={dim} ratio={compress_ratio} topk={topk} bf16 q/k, fp32 scores")
    print(f"{'T':>7} {'kernel':>10} {'torch':>10} {'speedup':>8} {'topk(ours)':>11} {'topk(torch)':>12} {'scores MB':>10}")
    for t in kv_lens:
        k = torch.randn(batch, t, dim, device="cuda", dtype=torch.bfloat16)
        ours, ref = fwd(q, k, w, compress_ratio=compress_ratio), torch_scores(q, k, w, compress_ratio)
        kept = min(topk, t)
        ms_k = do_bench(lambda: fwd(q, k, w, compress_ratio=compress_ratio), warmup=10, rep=50)
        ms_t = do_bench(lambda: torch_scores(q, k, w, compress_ratio), warmup=5, rep=20)
        ms_tk = do_bench(lambda: ours.topk(kept, dim=-1, sorted=False), warmup=10, rep=50)
        ms_tk_ref = do_bench(lambda: ref.topk(kept, dim=-1, sorted=False), warmup=10, rep=50)
        mb = batch * seqlen * t * 4 / 2**20
        print(f"{t:>7} {ms_k:>9.3f}m {ms_t:>9.3f}m {ms_t / ms_k:>7.2f}x {ms_tk:>10.3f}m {ms_tk_ref:>11.3f}m {mb:>10.0f}")


if __name__ == "__main__":
    fire.Fire(main)
