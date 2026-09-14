# source: ../csa_attn/bwd_dq.py with the dense main loop replaced by fwd.py's per-token gather,
# plus the scatter-add dMainKV of tile-ai/tilelang@v0.1.14
# examples/dsa_sparse_finetune/sparse_mla_bwd.py
"""csa2_attn: the query-owning kernel that produces **both** ``dQ`` and ``dmain_kv``.

``csa_attn`` needed three GEMM kernels: a window ``dKV`` owner, a *group* ``dKV`` owner
(``bwd_main.py``) and this query-owning ``dQ``. Gathered rows have no query range a group
block could walk -- entry ``g`` is touched by whichever tokens happened to pick it, which
is only known through ``indices`` -- so the group owner is gone and its output is produced
here instead, as a **scatter-add** out of the same tile that already holds ``P`` and ``dS``:

    dQ      += dS   M                       (stored once, bf16, no atomics)
    dMainKV += dS^T Q + P^T dO   scattered into rows ``indices[b, t, :]``   (fp32 atomics)

``K == V``, so the two halves of the gathered gradient land in **one** accumulator, exactly
as in ``bwd_kernels.py``. The scatter target is an fp32 ``[B, G, D]`` buffer the caller
zeroes and casts; **never bf16 atomics** (the upstream template asserts bf16 inputs but
accumulates ``dKV`` in fp32 for the same reason).

Structure, per gathered tile, is ``fwd.py``'s gather with the online softmax replaced by
the FA2 recompute:

- ``M_shared[i, :] = MainKV[b, indices[b, t, kt * block_N + i], :]``, the loaded row clamped
  into range and validity carried on the logit (``fwd.py``'s convention);
- ``P = exp2(S * scale - lse)`` -- ``lse`` already normalised window + gathered + sink
  together, so the same ``lse``/``delta`` feed both legs and both simply add into
  ``acc_dq``;
- an invalid (``-1`` / out-of-range) column, and every row that does not belong to this
  tile's token when ``T_q > 1``, is zeroed in ``P``; so its ``dS`` is zero, its ``dQ``
  contribution is zero, and its scattered row is zero. The atomic is skipped for those
  columns anyway (``valid`` predicate) so the invalid rows never touch memory.

**Duplicate indices are summed, not deduplicated** -- if a token lists the same entry
twice, the forward gives it two softmax columns and this kernel scatters two
contributions. That is self-consistent (this *is* the gradient of that forward) but it is
*not* what ``golden(level="sparse")`` computes, which scatters a 0.0 bias and so sees the
entry once. See ``README.md`` -> Known issues.

Tiles: ``block_M = T_q * H`` rows with ``T_q = ceil(64 / H)`` and ``threads = 128``, the
same two compile-time-forced choices as ``fwd.py`` (a 16/32-row tile never compiles here
and ``threads = 256`` needs ``block_M % 128 == 0``; ``tickets/0001-tilelang-issues.md``).
``block_N`` / ``num_stages`` are ``csa_attn``'s ``dq`` entries, **untuned**. The gathered
loop is ``T.Pipelined`` for the same reason as in the forward: its trip count is the
compile-time constant ``ceil(topk / block_N)``, not a nested division.

``scatter`` is the atomic-cost ablation knob for ``bench.py`` and is ``"atomic"`` for every
real call: ``"store"`` replaces the ``T.atomic_add`` with a plain store to the *same*
addresses (so the difference is the read-modify-write and the contention, at an identical
access pattern) and ``"none"`` drops the scatter entirely, which also lets the two
``dMainKV`` GEMMs die -- so ``atomic - none`` is the whole cost of producing ``dmain_kv``
and ``atomic - store`` is the atomic's own. Both leave ``dmain_kv`` garbage.
"""

import tilelang
import tilelang.language as T

from mzoo.layers.attn.csa2_attn.fwd import tokens_per_block


@tilelang.jit(out_idx=[8], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def bwd_dq(batch, heads, seq_len, dim, groups, topk, window, ratio, is_causal, scatter="atomic",
           block_N=64, num_stages=2, gather_stages=2, threads=128):
    """Query-owning ``dQ`` + gathered-row scatter-add ``dMainKV``.

    ``block_M = T_q * heads`` packed query rows (``T_q = ceil(64 / heads)`` tokens),
    ``block_N`` = window KV tokens **and** gathered entries per tile.
    """
    assert is_causal, "csa2_attn is causal-only"
    assert scatter in ("atomic", "store", "none"), f"unknown scatter mode {scatter!r}"
    assert window >= 1, f"window must be >= 1, got {window}"
    assert ratio >= 1, f"compress_ratio must be >= 1, got {ratio}"
    assert topk % block_N == 0, f"topk={topk} must be padded to a multiple of block_N={block_N}"
    tq = tokens_per_block(heads)
    block_M = tq * heads
    has_main = groups > 0 and topk > 0
    gather_tiles = topk // block_N
    sm_scale = (1.0 / dim)**0.5
    scale = sm_scale * 1.44269504  # log2(e), folded into the exp2
    log2e = 1.44269504  # lse arrives in natural log, the kernel works in base 2
    q_shape = [batch, seq_len * heads, dim]
    kv_shape = [batch, seq_len, dim]
    main_shape = [batch, max(groups, 1), dim]
    idx_shape = [batch, seq_len, max(topk, 1)]
    dtype, accum_dtype = T.bfloat16, T.float32

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),
            KV: T.Tensor(kv_shape, dtype),
            MainKV: T.Tensor(main_shape, dtype),
            Indices: T.Tensor(idx_shape, T.int32),
            dO: T.Tensor(q_shape, dtype),
            LSE: T.Tensor([batch, seq_len * heads], accum_dtype),
            Delta: T.Tensor([batch, seq_len * heads], accum_dtype),
            dMainKV: T.Tensor(main_shape, accum_dtype),  # zeroed by the caller, scattered into
            dQ: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, tq), batch, threads=threads) as (bx, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            dO_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)  # S = Q K^T, dP = dO K^T, dQ += dS K
            M_shared = T.alloc_shared([block_N, dim], dtype)  # the gathered tile, same three roles
            dQ_shared = T.alloc_shared([block_M, dim], dtype)
            # the two dMainKV GEMMs read their A operand transposed, which is a different
            # fragment layout from the dQ GEMM's -- one bf16 *shared* staging tile serves both
            # (P first, then dS), as in sparse_mla_bwd.py. A second fragment would hit
            # "Get different layout for cast" in layout inference.
            aT_shared = T.alloc_shared([block_M, block_N], dtype)
            lse = T.alloc_shared([block_M], accum_dtype)
            delta = T.alloc_shared([block_M], accum_dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            dp = T.alloc_fragment([block_M, block_N], accum_dtype)
            cast = T.alloc_fragment([block_M, block_N], dtype)  # P, then dS -- one buffer, in order
            acc_dq = T.alloc_fragment([block_M, dim], accum_dtype)
            acc_dkv = T.alloc_fragment([block_N, dim], accum_dtype)  # dK + dV of one gathered tile

            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.copy(dO[bz, bx * block_M:(bx + 1) * block_M, :], dO_shared)
            T.copy(LSE[bz, bx * block_M:(bx + 1) * block_M], lse)
            T.copy(Delta[bz, bx * block_M:(bx + 1) * block_M], delta)
            T.clear(acc_dq)

            # --- source 1: the window band, same tiles as fwd.py; dKV is bwd_kernels.py's job
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
                    cast[i, j] = acc_s[i, j] * (dp[i, j] - delta[i]) * sm_scale  # dS
                T.gemm(cast, K_shared, acc_dq, policy=T.GemmWarpPolicy.FullRow)  # dQ += dS K

            # --- source 2: the gathered top-k entries; dQ *and* the scattered dMainKV
            if has_main:
                for ti in T.serial(0, tq):  # one gathered pass per token of the block (tq == 1 usually)
                    tok = bx * tq + ti
                    # T.Pipelined is safe: the trip count is the constant ceil(topk / block_N),
                    # not csa_attn's nested division (tickets/0001-tilelang-issues.md).
                    for kt in T.Pipelined(0, gather_tiles, num_stages=gather_stages):
                        for i, j in T.Parallel(block_N, dim):
                            g = Indices[bz, tok, kt * block_N + i]
                            # the row loaded is clamped in range; validity lands on P below
                            M_shared[i, j] = MainKV[bz, T.if_then_else(T.And(g >= 0, g < groups), g, 0), j]
                        T.clear(acc_s)
                        T.gemm(Q_shared, M_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - lse[i] * log2e)
                        for i, j in T.Parallel(block_M, block_N):
                            g = Indices[bz, tok, kt * block_N + j]
                            # index out of [0, G) -> no gradient; and only THIS token's rows
                            # attend its gathered entries (a no-op when tq == 1)
                            acc_s[i, j] = T.if_then_else(
                                T.And(T.And(g >= 0, g < groups), i // heads == ti), acc_s[i, j], 0)
                        T.clear(dp)
                        T.gemm(dO_shared, M_shared, dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        T.copy(acc_s, aT_shared)  # P in bf16
                        T.clear(acc_dkv)
                        T.gemm(aT_shared, dO_shared, acc_dkv, transpose_A=True,
                               policy=T.GemmWarpPolicy.FullRow)  # the dV half: P^T dO
                        for i, j in T.Parallel(block_M, block_N):
                            cast[i, j] = acc_s[i, j] * (dp[i, j] - delta[i]) * sm_scale  # dS
                        T.gemm(cast, M_shared, acc_dq, policy=T.GemmWarpPolicy.FullRow)  # dQ += dS M
                        T.copy(cast, aT_shared)
                        T.gemm(aT_shared, Q_shared, acc_dkv, transpose_A=True,
                               policy=T.GemmWarpPolicy.FullRow)  # the dK half: dS^T Q
                        if scatter != "none":
                            for i, j in T.Parallel(block_N, dim):
                                g = Indices[bz, tok, kt * block_N + i]
                                if T.And(g >= 0, g < groups):
                                    if scatter == "atomic":
                                        T.atomic_add(dMainKV[bz, g, j], acc_dkv[i, j])
                                    else:  # ablation only: same addresses, plain store
                                        dMainKV[bz, g, j] = acc_dkv[i, j]

            T.copy(acc_dq, dQ_shared)
            T.copy(dQ_shared, dQ[bz, bx * block_M:(bx + 1) * block_M, :])

    return main
