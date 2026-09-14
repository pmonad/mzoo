# source: tile shape from tilelang@v0.1.14 examples/dsa_hisa/block_sparse_mqa_fp8.py
# (KV rows in M, query heads in N, weighted head reduce over the last dim).
"""indexer: DSV4.1 lightning-indexer score matrix, bf16 q/k, fp32 out (forward only).

``index_scores[b, s, t] = sum_h w[b,s,h] * relu(q[b,s,h,:] . k[b,t,:]) * Di**-0.5``

with a group-causal mask: entry ``t`` is visible to query ``s`` iff
``t < (s + 1) // compress_ratio`` (absolute prefill positions ``0..S-1``); masked
entries are ``-inf``. The relu runs **before** the weighting, so ``w`` may be
negative and no monotonicity trick is available.

Tile shape (from ``block_sparse_mqa_fp8.py``, generalised from 1 to ``T_s`` query
tokens per block): a block owns ``block_M = T_s * Hi`` **query rows** (``T_s``
consecutive tokens x ``Hi`` heads, the natural BSHD order of ``q.view(B, S*Hi, Di)``)
and walks the key axis in ``block_T``-row tiles. The GEMM is
``K_shared [block_T, Di] @ Q_shared [block_M, Di]^T -> s [block_T, block_M]``, i.e.
**keys in M, query rows in N**, so the per-token head reduce is a plain
``T.reduce_sum(dim=-1)`` over the last axis of ``s.reshape(block_T, T_s, Hi)`` --
the same trick the template uses. Accumulate and reduce are fp32 throughout.

The key loop is cut at the group-causal frontier like FA2's causal loop
(``loop_range = ceildiv(((bx+1)*T_s) // ratio, block_T)``) and the invisible tail of
each row is filled with ``-inf`` by a short serial loop, so a large ``T`` with a
small visible prefix costs writes but no GEMMs.

Constraints (asserted in ``fwd``): ``block_M % Hi == 0`` and ``S % (block_M // Hi) == 0``.
``T`` is free -- keys are zero-padded to a multiple of ``block_T`` and the extra score
columns are sliced off.

Target GB10 (sm121). Configs are **untuned** (one smem-safe tile per head dim that
compiles and passes); see ``README.md``.
"""

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

CONFIGS = {  # index head dim -> (block_M = T_s * Hi query rows, block_T = key rows, stages, threads)
    32: dict(block_M=128, block_T=64, num_stages=2, threads=128),  # extra: the vendored tiny model
    64: dict(block_M=128, block_T=64, num_stages=2, threads=128),
    96: dict(block_M=128, block_T=64, num_stages=2, threads=128),
    128: dict(block_M=128, block_T=64, num_stages=2, threads=128),
    256: dict(block_M=64, block_T=64, num_stages=1, threads=128),  # smem-capped, see README
}


@tilelang.jit(out_idx=[3], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def index_score(batch, seq_len, seq_kv, heads, dim, compress_ratio,
                block_M=128, block_T=64, num_stages=2, threads=128):
    assert block_M % heads == 0, f"block_M={block_M} must be a multiple of index heads={heads}"
    tokens = block_M // heads  # query tokens per block; rows are (token, head) in BSHD order
    scale = dim**-0.5
    dtype, accum = T.bfloat16, T.float32
    num_t = T.ceildiv(seq_kv, block_T)

    @T.prim_func
    def main(
            Q: T.Tensor([batch, seq_len * heads, dim], dtype),
            K: T.Tensor([batch, seq_kv, dim], dtype),
            Weights: T.Tensor([batch, seq_len, heads], accum),
            Scores: T.Tensor([batch, seq_len, seq_kv], accum),
    ):
        with T.Kernel(T.ceildiv(seq_len, tokens), batch, threads=threads) as (bx, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_T, dim], dtype)
            O_shared = T.alloc_shared([block_T, tokens], accum)  # transpose staging for a coalesced store
            w = T.alloc_fragment([tokens, heads], accum)
            s = T.alloc_fragment([block_T, block_M], accum)
            s3 = T.reshape(s, (block_T, tokens, heads))
            logits = T.alloc_fragment([block_T, tokens], accum)

            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.copy(Weights[bz, bx * tokens:(bx + 1) * tokens, :], w)

            # entries [0, visible) can be seen by *some* query of this block (the last one)
            visible = ((bx + 1) * tokens) // compress_ratio
            loop_range = T.min(num_t, T.ceildiv(visible, block_T))

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, k * block_T:(k + 1) * block_T, :], K_shared)
                T.gemm(K_shared, Q_shared, s, transpose_B=True, clear_accum=True,
                       policy=T.GemmWarpPolicy.FullRow)
                for n, i, h in T.Parallel(block_T, tokens, heads):  # relu BEFORE the weighting
                    s3[n, i, h] = T.max(s3[n, i, h], 0) * scale * w[i, h]
                T.reduce_sum(s3, logits, dim=-1, clear=True)
                T.copy(logits, O_shared)
                for i, n in T.Parallel(tokens, block_T):  # group-causal mask, folded into the store
                    Scores[bz, bx * tokens + i, k * block_T + n] = T.if_then_else(
                        k * block_T + n < (bx * tokens + i + 1) // compress_ratio,
                        O_shared[n, i], -T.infinity(accum))

            for k in T.serial(num_t - loop_range):  # nothing here is visible to any query of the block
                for i, n in T.Parallel(tokens, block_T):
                    Scores[bz, bx * tokens + i, (loop_range + k) * block_T + n] = -T.infinity(accum)

    return main


def check_shapes(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, cfg: dict) -> tuple:
    """Validate the indexer call shapes and return ``(batch, seq_len, heads, dim, seq_kv)``."""
    assert q.dtype == k.dtype == torch.bfloat16, f"q/k must be bf16, got {q.dtype}/{k.dtype}"
    batch, seq_len, heads, dim = q.shape
    assert k.shape[0] == batch and k.shape[2] == dim, f"k must be [B, T, {dim}], got {tuple(k.shape)}"
    assert w.shape == (batch, seq_len, heads), f"w must be [B, S, Hi]={(batch, seq_len, heads)}, got {tuple(w.shape)}"
    assert cfg["block_M"] % heads == 0, f"indexer needs Hi | block_M={cfg['block_M']}, got Hi={heads}"
    tokens = cfg["block_M"] // heads
    assert seq_len % tokens == 0, f"indexer needs S % {tokens} == 0 (block_M/Hi tokens per block), got {seq_len}"
    return batch, seq_len, heads, dim, k.shape[1]


def fwd(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, *, compress_ratio: int) -> torch.Tensor:
    """Lightning-indexer score matrix.

    ``q`` bf16 [B, S, Hi, Di] (already rotated / fp4-round-tripped by the caller),
    ``k`` bf16 [B, T, Di] (one shared key per compressed group), ``w`` fp32 [B, S, Hi]
    (already scaled by ``Hi**-0.5``) -> ``scores`` fp32 [B, S, T], ``-inf`` where
    ``t >= (s + 1) // compress_ratio``.
    """
    dim = q.shape[-1]
    assert dim in CONFIGS, f"unsupported index head dim {dim} (have {sorted(CONFIGS)})"
    assert compress_ratio >= 1, f"compress_ratio must be >= 1, got {compress_ratio}"
    cfg = CONFIGS[dim]
    batch, seq_len, heads, dim, seq_kv = check_shapes(q, k, w, cfg)
    if seq_kv == 0:
        return q.new_empty((batch, seq_len, 0), dtype=torch.float32)

    pad = -seq_kv % cfg["block_T"]  # keys are zero-padded; the extra score columns are sliced off
    k_pad = F.pad(k.contiguous(), (0, 0, 0, pad)) if pad else k.contiguous()
    kernel = index_score(batch, seq_len, seq_kv + pad, heads, dim, compress_ratio, **cfg)
    scores = kernel(q.contiguous().view(batch, seq_len * heads, dim), k_pad, w.float().contiguous())
    return scores[:, :, :seq_kv]
