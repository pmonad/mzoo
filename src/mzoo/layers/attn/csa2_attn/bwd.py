"""csa2_attn: sparse (window + top-k gathered) shared-latent MQA FA2 backward driver.

``bwd(q, kv, main_kv, indices, o, lse, do, *, window, compress_ratio, causal, sinks)
-> (dq, dkv, dmain_kv, dsinks)`` with ``q/o/do`` bf16 [B, S, H, D], ``kv``/``dkv``
bf16 [B, S, 1, D], ``main_kv``/``dmain_kv`` bf16 [B, G, 1, D], ``indices`` [B, S, topk],
``lse`` fp32 [B, H, S]. No gradient flows to ``indices`` (top-k is discrete).

**Three** kernels, where ``csa_attn`` had four:

1. ``preprocess`` -> ``delta = rowsum(dO * O)``, fp32, packed rows (``bwd_kernels.py``).
2. ``bwd_kv``     -> the window slice's ``dKV = dK + dV``, ``csa_attn``'s kernel and
   ``csa_attn``'s config **unchanged** -- ``lse`` is an input, so it does not care that
   the second source is now gathered. fp32 split buffers, no atomics.
3. ``bwd_dq``     -> ``dQ`` **and** ``dmain_kv`` from one query-owning CTA
   (``bwd_dq.py``).

``csa_attn``'s fourth kernel, the group-owning ``bwd_main``, has no analogue: a block of
main entries cannot enumerate the tokens that picked it without an inverse index, so
``dmain_kv`` is scattered out of the query owner with fp32 ``T.atomic_add`` into a zeroed
``[B, G, D]`` fp32 buffer, cast to bf16 here. **This is the one non-deterministic part of
the package**: atomic ordering varies run to run, so ``dmain_kv`` is not bitwise
reproducible (~1 ulp of fp32 per contributing token). Documented, not fixed -- see
``README.md``.

Everything else is inherited: ``dKV = dK + dV`` because K == V, the reduction over the H
heads sharing the latent is free (heads are query-tile rows), and
``dsinks[h] = -sum_{b,s} exp(sink[h] - lse[b,h,s]) * delta[b,h,s]`` stays a torch reduce
because ``lse`` already carries the sink column.

``CONFIGS`` is **untuned**: ``csa_attn``'s table verbatim, except that the ``dq`` kernel's
``block_M``/``threads`` are forced to ``T_q * H`` rows / 128 threads by the per-token
gather and by the layout-inference limits of ``tickets/0001-tilelang-issues.md``, exactly
as in ``fwd.py``. No sweep was run for this ticket; nothing had to shrink.
"""

import torch
import torch.nn.functional as F

from mzoo.layers.attn.csa2_attn.bwd_dq import bwd_dq
from mzoo.layers.attn.csa2_attn.bwd_kernels import bwd_kv, preprocess
from mzoo.layers.attn.csa2_attn.fwd import check_shapes, tokens_per_block

# head dim -> kv=(block_M latent tokens, block_N packed query rows, stages, threads),
#             dq=(block_N window KV tokens / gathered entries, stages); the dq kernel's
# block_M is T_q * H (= 64 rows) and its threads 128, both forced -- see fwd.py.
CONFIGS = {
    64: dict(kv=dict(block_M=64, block_N=128, num_stages=2, threads=128),
             dq=dict(block_N=64, num_stages=3, gather_stages=2, threads=128)),
    96: dict(kv=dict(block_M=64, block_N=64, num_stages=3, threads=128),
             dq=dict(block_N=64, num_stages=3, gather_stages=2, threads=128)),
    128: dict(kv=dict(block_M=64, block_N=64, num_stages=2, threads=128),
              dq=dict(block_N=64, num_stages=2, gather_stages=1, threads=128)),
    256: dict(kv=dict(block_M=64, block_N=64, num_stages=1, threads=128),
              dq=dict(block_N=32, num_stages=2, gather_stages=1, threads=128)),
}

TARGET_CTAS = 768  # inherited from swa_attn/csa_attn; drives the query-loop split count
MAX_SPLITS = 12


def pick_splits(batch: int, blocks_per_batch: int, tiles: int, max_splits: int = MAX_SPLITS) -> int:
    """Enough query-loop splits to fill the GPU, capped by the tiles actually walked."""
    blocks = batch * blocks_per_batch
    return max(1, min(max_splits, tiles, -(-TARGET_CTAS // blocks)))


def bwd(q, kv, main_kv, indices, o, lse, do, *, window: int, compress_ratio: int,
        causal: bool = True, sinks=None, cfg: dict | None = None, splits: int | None = None,
        scatter: str = "atomic"):
    """Sparse two-source shared-latent MQA FA2 backward.

    -> ``(dq [B,S,H,D], dkv [B,S,1,D], dmain_kv [B,G,1,D], dsinks [H] | None)``, bf16
    grads (``dsinks`` fp32). ``window`` and ``compress_ratio`` are the same compile-time
    constants the forward was called with, and ``indices`` must be the same tensor.

    ``scatter`` is the bench's atomic-cost ablation and is ``"atomic"`` for every real call;
    ``"store"`` / ``"none"`` return a meaningless ``dmain_kv`` (see ``bwd_dq.py``).
    """
    assert causal, "csa2_attn is causal-only"
    assert q.shape == o.shape == do.shape, "shape mismatch"
    assert isinstance(window, int) and window >= 1, f"window must be a positive int, got {window!r}"
    assert isinstance(compress_ratio, int) and compress_ratio >= 1, (
        f"compress_ratio must be a positive int, got {compress_ratio!r}")
    dim = q.shape[-1]
    assert dim in CONFIGS, f"unsupported head dim {dim}"
    cfg = cfg or CONFIGS[dim]
    kv_cfg, dq_cfg = cfg["kv"], cfg["dq"]
    batch, seq_len, heads, dim, groups, topk = check_shapes(q, kv, main_kv, indices)
    assert kv_cfg["block_N"] % heads == 0, (
        f"csa2_attn bwd at D={dim} needs heads | {kv_cfg['block_N']} (the window dKV kernel's "
        f"query tile), got H={heads}")
    tq_kv = kv_cfg["block_N"] // heads  # query tokens per tile of the window dKV kernel
    blk = max(kv_cfg["block_M"], tq_kv, tokens_per_block(heads))
    assert seq_len % blk == 0, f"csa2_attn bwd needs seq_len % {blk} == 0, got {seq_len}"
    window = min(window, seq_len)
    splits = splits or pick_splits(batch, -(-seq_len // kv_cfg["block_M"]),
                                   -(-(kv_cfg["block_M"] + window - 1) // tq_kv))

    idx = indices.to(torch.int32)
    pad = -topk % dq_cfg["block_N"] if topk else 1  # whole gathered tiles; the tail is -1
    if pad:
        idx = F.pad(idx, (0, pad), value=-1)
    main = main_kv.contiguous().view(batch, groups, dim) if groups else \
        torch.zeros(batch, 1, dim, device=q.device, dtype=q.dtype)

    rows = seq_len * heads
    pack = lambda x: x.contiguous().view(batch, rows, dim)  # noqa: E731
    q, o, do = (pack(x) for x in (q, o, do))
    kv = kv.contiguous().view(batch, seq_len, dim)
    lse = lse.permute(0, 2, 1).reshape(batch, rows).contiguous()  # [B, H, S] -> packed token-major

    delta = preprocess(batch, rows, dim)(o, do)
    dkv_parts = torch.empty(splits, batch, seq_len, dim, device=q.device, dtype=torch.float32)
    bwd_kv(batch, heads, seq_len, dim, window, causal, splits, **kv_cfg)(q, kv, do, lse, delta, dkv_parts)

    # fp32 scatter target: zeroed here, atomically accumulated in the dq kernel, cast below
    dmain_f32 = torch.zeros(batch, max(groups, 1), dim, device=q.device, dtype=torch.float32)
    dq = bwd_dq(batch, heads, seq_len, dim, groups, topk + pad if topk else 0, window,
                compress_ratio, causal, scatter, **dq_cfg)(
                    q, kv, main, idx.contiguous(), do, lse, delta, dmain_f32)
    dmain = dmain_f32.to(torch.bfloat16).view(batch, max(groups, 1), 1, dim)[:, :groups]

    dsinks = None
    if sinks is not None:
        d = delta.view(batch, seq_len, heads).permute(0, 2, 1)  # packed -> [B, H, S]
        l = lse.view(batch, seq_len, heads).permute(0, 2, 1)
        dsinks = -(torch.exp(sinks.detach().float().view(1, heads, 1) - l) * d).sum(dim=(0, 2))

    dkv = dkv_parts.sum(0).to(torch.bfloat16).view(batch, seq_len, 1, dim)
    return dq.view(batch, seq_len, heads, dim), dkv, dmain, dsinks
