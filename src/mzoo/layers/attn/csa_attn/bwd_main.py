# source: ../swa_attn/bwd_kernels.py::bwd_kv with the band replaced by group-causal visibility
"""csa_attn: the ``dmain_kv`` kernel -- the compressed source's KV-owner backward.

A second KV-owner launch, structurally ``bwd_kv`` (``bwd_kernels.py``) with two
substitutions, because the compressed source's rows are **group indices**, not
tokens:

- **Which query tiles a block walks.** A block owns groups ``[g0, g0 + block_M - 1]``.
  Token ``t`` sees group ``g`` iff ``g < (t + 1) // ratio``, i.e.
  ``t >= ratio * (g + 1) - 1``. The smallest such ``t`` over the block is the one
  for ``g0``, so the query loop starts at tile
  ``floor((ratio * g0 + ratio - 1) / T_q)`` and runs to the end of the sequence --
  unlike the window, the visible set grows without bound to the right.
- **The mask.** ``g < min(G, (t + 1) // ratio)`` per (group row, query row), the
  same expression ``fwd.py`` and ``bwd_dq.py`` use. The ``min`` also drops the
  zero-padded tail rows of ``MainKV``.

Everything else is ``bwd_kv`` verbatim: ``dKV = dK + dV`` in one fp32 fragment
(K == V), the reduction over the H heads sharing the latent is free because heads
are query-tile rows, the query loop is split across a ``splits`` grid dimension
with each split writing its own fp32 slice of ``[splits, B, G, D]``, and torch sums
over ``splits``. No atomics.

``lse`` and ``delta`` are inputs, already covering window + main + sink, so nothing
about this kernel needs to know the window exists.
"""

import tilelang
import tilelang.language as T

_JIT = dict(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})


@tilelang.jit(**_JIT)
def bwd_main(batch, heads, seq_len, dim, groups, gpad, ratio, is_causal, splits=1,
             block_M=128, block_N=64, num_stages=1, threads=256):
    """Group-block-owning ``dMainKV``. ``block_M`` = main entries, ``block_N`` = packed query rows."""
    assert is_causal, "csa_attn is causal-only"
    assert block_N % heads == 0, f"block_N={block_N} must be a multiple of heads={heads}"
    assert gpad % block_M == 0, f"padded groups={gpad} must be a multiple of block_M={block_M}"
    tq = block_N // heads  # query tokens per query tile
    sm_scale = (1.0 / dim)**0.5
    scale = sm_scale * 1.44269504  # log2(e), folded into the exp2 softmax
    log2e = 1.44269504  # lse arrives in natural log, the kernel works in base 2
    q_shape = [batch, seq_len * heads, dim]
    main_shape = [batch, gpad, dim]
    dtype, accum_dtype = T.bfloat16, T.float32

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),
            MainKV: T.Tensor(main_shape, dtype),
            dO: T.Tensor(q_shape, dtype),
            LSE: T.Tensor([batch, seq_len * heads], accum_dtype),
            Delta: T.Tensor([batch, seq_len * heads], accum_dtype),
            dMainKV: T.Tensor([splits, batch, gpad, dim], accum_dtype),
    ):
        with T.Kernel(splits, gpad // block_M, batch, threads=threads) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim], dtype)  # K and V at once
            q = T.alloc_shared([block_N, dim], dtype)
            do = T.alloc_shared([block_N, dim], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dkv = T.alloc_fragment([block_M, dim], accum_dtype)  # dK + dV, K == V

            T.copy(MainKV[bz, by * block_M:(by + 1) * block_M, :], K_shared)
            T.clear(dkv)

            # first token that can see group g0 is ratio * g0 + ratio - 1; no right edge
            loop_st = T.min(T.ceildiv(seq_len, tq), T.floordiv(by * block_M * ratio + ratio - 1, tq))
            loop_ed = T.ceildiv(seq_len, tq)
            chunk = T.ceildiv(loop_ed - loop_st, splits)  # contiguous query-block chunk per split
            for k in T.Pipelined(loop_st + bx * chunk, T.min(loop_ed, loop_st + (bx + 1) * chunk), num_stages=num_stages):
                T.copy(Q[bz, k * block_N:(k + 1) * block_N, :], q)
                T.clear(qkT)
                T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(LSE[bz, k * block_N:(k + 1) * block_N], lse_shared)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j] * log2e)
                for i, j in T.Parallel(block_M, block_N):  # query row j lives at token k * tq + j // H
                    qkT[i, j] = T.if_then_else(
                        by * block_M + i < T.min(groups, T.floordiv(k * tq + j // heads + 1, ratio)), qkT[i, j], 0)

                T.copy(dO[bz, k * block_N:(k + 1) * block_N, :], do)
                T.clear(dsT)
                T.gemm(K_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)  # V == K
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dkv, policy=T.GemmWarpPolicy.FullRow)  # the dV half

                T.copy(Delta[bz, k * block_N:(k + 1) * block_N], delta)
                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.gemm(dsT_cast, q, dkv, policy=T.GemmWarpPolicy.FullRow)  # the dK half, same accumulator

            T.copy(dkv, dMainKV[bx, bz, by * block_M:(by + 1) * block_M, :])

    return main
