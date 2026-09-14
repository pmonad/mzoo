# source: the forward kernel of ``fwd.py`` with the softmax replaced by the FA2 dS recompute
"""latent_attn: the query-owning ``dQ`` kernel of the shared-latent MQA backward.

``dense_attn`` produced ``dQ`` inside the KV-block-owning kernel with an fp32
``atomic_add`` scatter. That is fine when ``dQ`` fits in L2; at H=64 it does not
(134 MB at S4096 D128), and each of the 16-32 KV blocks read-modify-writes all of
it -- ~4.3 GB of traffic on a 227 GB/s part, measured at 9 of the 28 ms the fused
kernel took.

So ``dQ`` gets its own kernel with the **query tile as the owner**: the CTA holds
``block_M`` packed query rows (``T_q = block_M // H`` tokens x H heads, same
layout and same causal rule as ``fwd.py``), walks the visible KV tiles, and
stores ``dQ`` exactly once in bf16. The latent it re-reads is only
``S * D`` bf16 (2 MB at S4096 D128), which lives in L2.

The cost is recomputing ``S`` and ``dP``: 3 GEMMs here plus 4 in ``bwd_kv``
instead of 5 fused, i.e. +40% FLOPs to delete 4.3 GB of atomics. On this machine
that trade is strongly positive (see ``README.md`` -> Bench).

Structurally this is ``fwd.py``'s kernel with the online softmax replaced by
``P = exp2(S * scale - lse)`` (``lse`` is already known) and
``dS = P * (dP - delta) * sm_scale``, so it inherits the forward's tiling
behaviour -- which is the fast path on this target.
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[5], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def bwd_dq(batch, heads, seq_len, dim, is_causal, block_M=128, block_N=64, num_stages=2, threads=256):
    """Query-owning ``dQ``. ``block_M`` = packed query rows, ``block_N`` = latent (KV) tokens."""
    assert block_M % heads == 0, f"block_M={block_M} must be a multiple of heads={heads}"
    tq = block_M // heads  # query tokens per block
    sm_scale = (1.0 / dim)**0.5
    scale = sm_scale * 1.44269504  # log2(e), folded into the exp2
    log2e = 1.44269504  # lse arrives in natural log, the kernel works in base 2
    q_shape = [batch, seq_len * heads, dim]
    kv_shape = [batch, seq_len, dim]
    dtype, accum_dtype = T.bfloat16, T.float32

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),
            KV: T.Tensor(kv_shape, dtype),
            dO: T.Tensor(q_shape, dtype),
            LSE: T.Tensor([batch, seq_len * heads], accum_dtype),
            Delta: T.Tensor([batch, seq_len * heads], accum_dtype),
            dQ: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, tq), batch, threads=threads) as (bx, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            dO_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)  # serves S = Q K^T, dP = dO K^T and dQ += dS K
            dQ_shared = T.alloc_shared([block_M, dim], dtype)
            lse = T.alloc_shared([block_M], accum_dtype)
            delta = T.alloc_shared([block_M], accum_dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            dp = T.alloc_fragment([block_M, block_N], accum_dtype)
            ds_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_dq = T.alloc_fragment([block_M, dim], accum_dtype)

            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.copy(dO[bz, bx * block_M:(bx + 1) * block_M, :], dO_shared)
            T.copy(LSE[bz, bx * block_M:(bx + 1) * block_M], lse)
            T.copy(Delta[bz, bx * block_M:(bx + 1) * block_M], delta)
            T.clear(acc_dq)

            loop_range = (T.ceildiv((bx + 1) * tq, block_N) if is_causal else T.ceildiv(seq_len, block_N))

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(KV[bz, k * block_N:(k + 1) * block_N, :], K_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_M, block_N):  # P, already sink-normalised via lse
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - lse[i] * log2e)
                if is_causal:  # mask from the row's TOKEN (row // H), not the block index
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(bx * tq + i // heads >= k * block_N + j, acc_s[i, j], 0)

                T.clear(dp)
                T.gemm(dO_shared, K_shared, dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)  # V == K
                for i, j in T.Parallel(block_M, block_N):
                    ds_cast[i, j] = acc_s[i, j] * (dp[i, j] - delta[i]) * sm_scale
                T.gemm(ds_cast, K_shared, acc_dq, policy=T.GemmWarpPolicy.FullRow)  # dQ += dS K

            T.copy(acc_dq, dQ_shared)
            T.copy(dQ_shared, dQ[bz, bx * block_M:(bx + 1) * block_M, :])

    return main
