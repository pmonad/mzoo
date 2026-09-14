# source: copied from ../swa_attn/bwd_dq.py (the forward kernel with the softmax replaced by
# the FA2 dS recompute); gains the main source's second loop, mirroring fwd.py
"""csa_attn: the query-owning ``dQ`` kernel of the two-source backward.

Same owner split as ``swa_attn``: the CTA holds ``block_M`` packed query rows
(``T_q = block_M // H`` tokens x H heads), walks the KV tiles it can see, and
stores ``dQ`` exactly once in bf16 -- no atomics.

One change from ``swa_attn``: after the window tiles it walks the **visible main
tiles**, with the same ``P = exp2(S * scale - lse)`` recompute and the group-causal
mask ``g < min(G, (t + 1) // ratio)`` instead of the band mask. ``lse`` already
normalised window + main + sink together, so both legs use the same ``lse`` and
``delta`` and simply add into one ``acc_dq``.

The masks are correctness, not speed, in both legs: an unmasked recomputed ``P``
would include columns the forward never saw while ``lse`` was normalised only over
the ones it did.
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[6], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def bwd_dq(batch, heads, seq_len, dim, groups, gpad, window, ratio, is_causal,
           block_M=128, block_N=64, num_stages=2, threads=256):
    """Query-owning ``dQ``. ``block_M`` = packed query rows, ``block_N`` = KV tokens/groups."""
    assert is_causal, "csa_attn is causal-only"
    assert block_M % heads == 0, f"block_M={block_M} must be a multiple of heads={heads}"
    assert gpad % block_N == 0, f"padded groups={gpad} must be a multiple of block_N={block_N}"
    tq = block_M // heads  # query tokens per block
    has_main = groups > 0
    main_tiles = -(-groups // block_N)  # tiles holding a real entry
    sm_scale = (1.0 / dim)**0.5
    scale = sm_scale * 1.44269504  # log2(e), folded into the exp2
    log2e = 1.44269504  # lse arrives in natural log, the kernel works in base 2
    q_shape = [batch, seq_len * heads, dim]
    kv_shape = [batch, seq_len, dim]
    main_shape = [batch, max(gpad, 1), dim]
    dtype, accum_dtype = T.bfloat16, T.float32

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),
            KV: T.Tensor(kv_shape, dtype),
            MainKV: T.Tensor(main_shape, dtype),
            dO: T.Tensor(q_shape, dtype),
            LSE: T.Tensor([batch, seq_len * heads], accum_dtype),
            Delta: T.Tensor([batch, seq_len * heads], accum_dtype),
            dQ: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, tq), batch, threads=threads) as (bx, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            dO_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)  # S = Q K^T, dP = dO K^T, dQ += dS K
            M_shared = T.alloc_shared([block_N, dim], dtype)  # same three roles for the main source
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

            # --- source 1: the window band, same tiles as fwd.py
            loop_st = T.floordiv(T.max(bx * tq - window + 1, 0), block_N)
            loop_ed = T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * tq, block_N))
            for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                T.copy(KV[bz, k * block_N:(k + 1) * block_N, :], K_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_M, block_N):  # P, already sink-normalised via lse
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - lse[i] * log2e)
                for i, j in T.Parallel(block_M, block_N):  # band mask from the row's TOKEN (row // H)
                    acc_s[i, j] = T.if_then_else(
                        T.And(bx * tq + i // heads >= k * block_N + j,
                              k * block_N + j > bx * tq + i // heads - window), acc_s[i, j], 0)
                T.clear(dp)
                T.gemm(dO_shared, K_shared, dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)  # V == K
                for i, j in T.Parallel(block_M, block_N):
                    ds_cast[i, j] = acc_s[i, j] * (dp[i, j] - delta[i]) * sm_scale
                T.gemm(ds_cast, K_shared, acc_dq, policy=T.GemmWarpPolicy.FullRow)  # dQ += dS K

            # --- source 2: the visible main entries, group-causal
            if has_main:
                # min of ceildivs, not ceildiv of a min (tilelang codegen; see fwd.py)
                main_ed = T.min(main_tiles, T.ceildiv(T.floordiv((bx + 1) * tq, ratio), block_N))
                # T.serial, NOT T.Pipelined: the main loop's software pipeline miscompiles
                # here (silently wrong results, see tickets/0001-tilelang-issues.md).
                for kg in T.serial(0, main_ed):
                    T.copy(MainKV[bz, kg * block_N:(kg + 1) * block_N, :], M_shared)
                    T.clear(acc_s)
                    T.gemm(Q_shared, M_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - lse[i] * log2e)
                    for i, j in T.Parallel(block_M, block_N):  # g < min(G, (t + 1) // ratio)
                        acc_s[i, j] = T.if_then_else(
                            kg * block_N + j < T.min(groups, T.floordiv(bx * tq + i // heads + 1, ratio)),
                            acc_s[i, j], 0)
                    T.clear(dp)
                    T.gemm(dO_shared, M_shared, dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)  # V == K
                    for i, j in T.Parallel(block_M, block_N):
                        ds_cast[i, j] = acc_s[i, j] * (dp[i, j] - delta[i]) * sm_scale
                    T.gemm(ds_cast, M_shared, acc_dq, policy=T.GemmWarpPolicy.FullRow)  # dQ += dS K

            T.copy(acc_dq, dQ_shared)
            T.copy(dQ_shared, dQ[bz, bx * block_M:(bx + 1) * block_M, :])

    return main
