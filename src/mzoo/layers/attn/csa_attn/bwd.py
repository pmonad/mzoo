"""csa_attn: two-source (window + compressed) shared-latent MQA FA2 backward driver.

``bwd(q, kv, main_kv, o, lse, do, *, window, compress_ratio, causal, sinks)
-> (dq, dkv, dmain_kv, dsinks)`` with ``q/o/do`` bf16 [B, S, H, D], ``kv``/``dkv``
bf16 [B, S, 1, D], ``main_kv``/``dmain_kv`` bf16 [B, G, 1, D], ``lse`` fp32 [B, H, S].

**Four** kernels now, and still no atomics anywhere:

1. ``preprocess``  -> ``delta = rowsum(dO * O)``, fp32, over packed rows (unchanged).
2. ``bwd_kv``      -> the window slice's ``dKV = dK + dV``, ``swa_attn``'s kernel
   **unchanged**: ``lse`` is an input, so the fact that it now also normalises the
   main source changes nothing about this kernel.
3. ``bwd_main``    -> ``dMainKV``, a second KV-owner launch whose rows are group
   indices and whose visibility is group-causal (``bwd_main.py``).
4. ``bwd_dq``      -> ``dQ``, one query-owning CTA walking window tiles then visible
   main tiles, storing once.

Both KV-owner kernels write fp32 ``[splits, ...]`` slices that torch reduces, which
is why the two sources can be two launches instead of one fused kernel: they own
disjoint outputs and share only their (read-only) ``lse``/``delta`` inputs.

``main_kv`` is zero-padded to a multiple of the group-tile sizes so every kernel
reads whole tiles; the pad rows are killed by the ``g < min(G, ...)`` masks, and the
gradient slice past ``G`` is dropped. ``G == 0`` skips kernel 3 entirely and returns
an empty ``dmain_kv``, so the result is exactly ``swa_attn``'s.

``sinks`` adds no kernel, as before: ``lse`` already carries the sink column, so
``dsinks[h] = -sum_{b,s} exp(sink[h] - lse[b,h,s]) * delta[b,h,s]`` stays a torch reduce.

``CONFIGS`` is **untuned**: ``swa_attn``'s table verbatim (``kv``/``dq``), with the
main source's KV-owner reusing the ``kv`` tile. No sweep was run for this ticket, and
nothing had to shrink. ``bwd_main`` gets its own split count -- its grid is ``G /
block_M`` blocks each walking up to ``S / T_q`` query tiles, a different shape from the
window's band. See ``README.md``.
"""

import math

import torch

from mzoo.layers.attn.csa_attn.bwd_dq import bwd_dq
from mzoo.layers.attn.csa_attn.bwd_kernels import bwd_kv, preprocess
from mzoo.layers.attn.csa_attn.bwd_main import bwd_main
from mzoo.layers.attn.csa_attn.fwd import check_shapes, pack_main

# head dim -> kv=(block_M latent tokens, block_N packed query rows, stages, threads),
#             dq=(block_M packed query rows, block_N KV tokens/groups, stages, threads);
# the main source's KV-owner kernel reuses the ``kv`` tile (see README -> Tile configs).
CONFIGS = {
    64: dict(kv=dict(block_M=64, block_N=128, num_stages=2, threads=128),
             dq=dict(block_M=128, block_N=64, num_stages=3, threads=256)),
    96: dict(kv=dict(block_M=64, block_N=64, num_stages=3, threads=128),
             dq=dict(block_M=128, block_N=64, num_stages=3, threads=256)),
    128: dict(kv=dict(block_M=64, block_N=64, num_stages=2, threads=128),
              dq=dict(block_M=128, block_N=64, num_stages=2, threads=256)),
    256: dict(kv=dict(block_M=64, block_N=64, num_stages=1, threads=128),
              dq=dict(block_M=64, block_N=32, num_stages=2, threads=128)),
}

TARGET_CTAS = 768  # inherited from swa_attn; drives the query-loop split count
MAX_SPLITS = 12


def pick_splits(batch: int, blocks_per_batch: int, tiles: int, max_splits: int = MAX_SPLITS) -> int:
    """Enough query-loop splits to fill the GPU, capped by the tiles actually walked."""
    blocks = batch * blocks_per_batch
    return max(1, min(max_splits, tiles, -(-TARGET_CTAS // blocks)))


def bwd(q, kv, main_kv, o, lse, do, *, window: int, compress_ratio: int, causal: bool = True,
        sinks=None, cfg: dict | None = None, splits: int | None = None):
    """Two-source shared-latent MQA FA2 backward.

    -> ``(dq [B,S,H,D], dkv [B,S,1,D], dmain_kv [B,G,1,D], dsinks [H] | None)``, bf16
    grads. ``window`` and ``compress_ratio`` are the same compile-time constants the
    forward was called with.
    """
    assert causal, "csa_attn is causal-only"
    assert q.shape == o.shape == do.shape, "shape mismatch"
    dim = q.shape[-1]
    assert dim in CONFIGS, f"unsupported head dim {dim}"
    cfg = cfg or CONFIGS[dim]
    kv_cfg, dq_cfg = cfg["kv"], cfg["dq"]
    batch, seq_len, heads, dim, groups = check_shapes(q, kv, main_kv, dq_cfg, "csa_attn bwd (dq)")
    assert kv_cfg["block_N"] % heads == 0, (
        f"csa_attn bwd at D={dim} needs heads | {kv_cfg['block_N']} (the dKV kernels' query tile), got H={heads}")
    tq = kv_cfg["block_N"] // heads
    blk = max(kv_cfg["block_M"], tq)
    assert seq_len % blk == 0, f"csa_attn bwd needs seq_len % {blk} == 0, got {seq_len}"
    window = min(window, seq_len)
    splits = splits or pick_splits(batch, -(-seq_len // kv_cfg["block_M"]),
                                   -(-(kv_cfg["block_M"] + window - 1) // tq))

    # one padding for both group-tiled kernels: bwd_dq's block_N and bwd_main's block_M
    main, gpad = pack_main(main_kv, batch, dim, math.lcm(dq_cfg["block_N"], kv_cfg["block_M"]))
    rows = seq_len * heads
    pack = lambda x: x.contiguous().view(batch, rows, dim)  # noqa: E731
    q, o, do = (pack(x) for x in (q, o, do))
    kv = kv.contiguous().view(batch, seq_len, dim)
    lse = lse.permute(0, 2, 1).reshape(batch, rows).contiguous()  # [B, H, S] -> packed token-major

    delta = preprocess(batch, rows, dim)(o, do)
    dkv_parts = torch.empty(splits, batch, seq_len, dim, device=q.device, dtype=torch.float32)
    bwd_kv(batch, heads, seq_len, dim, window, causal, splits, **kv_cfg)(q, kv, do, lse, delta, dkv_parts)
    dq = bwd_dq(batch, heads, seq_len, dim, groups, gpad, window, compress_ratio, causal, **dq_cfg)(
        q, kv, main, do, lse, delta)

    if gpad:
        msplits = pick_splits(batch, -(-gpad // kv_cfg["block_M"]), -(-seq_len // tq))
        dmain_parts = torch.empty(msplits, batch, gpad, dim, device=q.device, dtype=torch.float32)
        bwd_main(batch, heads, seq_len, dim, groups, gpad, compress_ratio, causal, msplits, **kv_cfg)(
            q, main, do, lse, delta, dmain_parts)
        dmain = dmain_parts.sum(0)[:, :groups].to(torch.bfloat16).view(batch, groups, 1, dim)
    else:
        dmain = main_kv.new_zeros(batch, 0, 1, dim)

    dsinks = None
    if sinks is not None:
        d = delta.view(batch, seq_len, heads).permute(0, 2, 1)  # packed -> [B, H, S]
        l = lse.view(batch, seq_len, heads).permute(0, 2, 1)
        dsinks = -(torch.exp(sinks.detach().float().view(1, heads, 1) - l) * d).sum(dim=(0, 2))

    dkv = dkv_parts.sum(0).to(torch.bfloat16).view(batch, seq_len, 1, dim)
    return dq.view(batch, seq_len, heads, dim), dkv, dmain, dsinks
