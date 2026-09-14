"""latent_attn: shared-latent MQA FlashAttention-2 backward driver.

``bwd(q, kv, o, lse, do, *, causal, sinks) -> (dq, dkv, dsinks)`` with
``q/o/do`` bf16 [B, S, H, D], ``kv``/``dkv`` bf16 [B, S, 1, D], ``lse`` fp32
[B, H, S]. The kernels live in ``bwd_kernels.py`` and ``bwd_dq.py``.

Three kernels, and **no atomics anywhere**:

1. ``preprocess`` -> ``delta = rowsum(dO * O)``, fp32, over packed rows.
2. ``bwd_kv``     -> a CTA owns a latent block + a contiguous chunk of the query
   tiles, accumulates ``dKV = dK + dV`` in one fp32 register tile (the reduction
   over all H heads sharing the latent is free, heads are rows), and writes an
   fp32 slice of ``[splits, B, S, D]``; torch sums over ``splits`` and casts.
3. ``bwd_dq``     -> a CTA owns a packed query tile, recomputes S and dS against
   the (L2-resident) latent, stores ``dQ`` bf16 exactly once.

``dense_attn`` fused 3 into 2 with an fp32 ``atomic_add`` scatter; that costs
~4.3 GB of read-modify-write traffic at B1 H64 S4096 D128 on a 227 GB/s part.
Recomputing S/dS in a third kernel is +40% FLOPs and ~2x faster overall here.

``sinks`` adds no kernel: ``lse`` already carries the sink column, so the
recomputed P is sink-normalised and only
``dsinks[h] = -sum_{b,s} exp(sink[h] - lse[b,h,s]) * delta[b,h,s]`` is left --
O(B*H*S) elementwise, so it stays a plain fp32 torch reduce.

``CONFIGS`` is re-tuned per head dim (see ``README.md``), and is the same at
H=16 and H=64 -- the sweep's H=16 winner is within 1.3% of the H=64 pick at
every dim, so there is no per-(dim, H) table. In ``bwd_kv`` ``block_M`` counts
latent tokens and ``block_N`` packed query rows; in ``bwd_dq`` it is the other
way round, matching ``fwd.py``. Whichever of the two counts packed query rows
must be a multiple of ``H``, and both are 64 at D=256, so H must divide 64
there (H <= 64, the model's shape, still fits). The tilelang layout-inference
limits of ``tickets/0001-tilelang-issues.md`` still apply: ``block_M=32`` never
compiles, and 64-row tiles need 128 threads in both backward kernels.
"""

import torch

from mzoo.layers.attn.latent_attn.bwd_dq import bwd_dq
from mzoo.layers.attn.latent_attn.bwd_kernels import bwd_kv, preprocess
from mzoo.layers.attn.latent_attn.fwd import check_shapes

# head dim -> kv=(block_M latent tokens, block_N packed query rows, stages, threads),
#             dq=(block_M packed query rows, block_N latent tokens, stages, threads)
CONFIGS = {
    64: dict(kv=dict(block_M=256, block_N=64, num_stages=2, threads=256),
             dq=dict(block_M=128, block_N=128, num_stages=2, threads=256)),
    96: dict(kv=dict(block_M=256, block_N=64, num_stages=2, threads=256),
             dq=dict(block_M=128, block_N=128, num_stages=2, threads=256)),
    128: dict(kv=dict(block_M=128, block_N=64, num_stages=2, threads=256),
              dq=dict(block_M=128, block_N=64, num_stages=2, threads=256)),
    256: dict(kv=dict(block_M=64, block_N=64, num_stages=1, threads=128),
              dq=dict(block_M=64, block_N=32, num_stages=2, threads=128)),
}

TARGET_CTAS = 256  # >5 waves on GB10's 48 SMs; drives the query-loop split count
MAX_SPLITS = 8


def pick_splits(batch: int, seq_len: int, block_M: int, max_splits: int = MAX_SPLITS) -> int:
    """Enough query-loop splits to fill the GPU: heads are rows, so the KV grid alone is tiny."""
    blocks = batch * -(-seq_len // block_M)
    return max(1, min(max_splits, -(-TARGET_CTAS // blocks)))


def bwd(q, kv, o, lse, do, *, causal: bool = True, sinks=None, cfg: dict | None = None, splits: int | None = None):
    """Shared-latent MQA FA2 backward -> ``(dq [B,S,H,D], dkv [B,S,1,D], dsinks [H] | None)``, bf16 grads."""
    assert q.shape == o.shape == do.shape, "shape mismatch"
    dim = q.shape[-1]
    assert dim in CONFIGS, f"unsupported head dim {dim}"
    cfg = cfg or CONFIGS[dim]
    kv_cfg, dq_cfg = cfg["kv"], cfg["dq"]
    batch, seq_len, heads, dim = check_shapes(q, kv, dq_cfg, "latent_attn bwd (dq)")
    assert kv_cfg["block_N"] % heads == 0, (
        f"latent_attn bwd at D={dim} needs heads | {kv_cfg['block_N']} (the dKV kernel's query tile), got H={heads}")
    tq = kv_cfg["block_N"] // heads
    blk = max(kv_cfg["block_M"], tq)
    assert seq_len % blk == 0, f"latent_attn bwd needs seq_len % {blk} == 0, got {seq_len}"
    splits = splits or pick_splits(batch, seq_len, kv_cfg["block_M"])

    rows = seq_len * heads
    pack = lambda x: x.contiguous().view(batch, rows, dim)  # noqa: E731
    q, o, do = (pack(x) for x in (q, o, do))
    kv = kv.contiguous().view(batch, seq_len, dim)
    lse = lse.permute(0, 2, 1).reshape(batch, rows).contiguous()  # [B, H, S] -> packed token-major

    delta = preprocess(batch, rows, dim)(o, do)
    dkv_parts = torch.empty(splits, batch, seq_len, dim, device=q.device, dtype=torch.float32)
    bwd_kv(batch, heads, seq_len, dim, causal, splits, **kv_cfg)(q, kv, do, lse, delta, dkv_parts)
    dq = bwd_dq(batch, heads, seq_len, dim, causal, **dq_cfg)(q, kv, do, lse, delta)

    dsinks = None
    if sinks is not None:
        d = delta.view(batch, seq_len, heads).permute(0, 2, 1)  # packed -> [B, H, S]
        l = lse.view(batch, seq_len, heads).permute(0, 2, 1)
        dsinks = -(torch.exp(sinks.detach().float().view(1, heads, 1) - l) * d).sum(dim=(0, 2))

    dkv = dkv_parts.sum(0).to(torch.bfloat16).view(batch, seq_len, 1, dim)
    return dq.view(batch, seq_len, heads, dim), dkv, dsinks
