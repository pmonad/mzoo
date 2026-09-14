# source: copied from ../latent_attn/bwd_kernels.py
# (originally tile-ai/tilelang@v0.1.14 examples/flash_attention/example_mha_bwd_bshd.py)
"""swa_attn: the ``preprocess`` and ``dKV`` kernels of the sliding-window shared-latent MQA backward.

The ``dQ`` kernel lives in ``bwd_dq.py``, the python driver in ``bwd.py``.

All query-side tensors are the **packed** ``[B, S*H, D]`` view of BSHD, so a
query tile of ``block_N`` rows is ``T_q = block_N // H`` consecutive tokens x
``H`` heads, exactly as in ``fwd.py``. The latent is ``[B, S, D]``.

One change from ``latent_attn``: this CTA owns latent tokens
``[k0, k0 + block_M - 1]`` and only the query tiles whose tokens can *see* them --
``k <= t <= k + window - 1``, so ``t in [k0, k0 + block_M - 1 + window - 1]``,
i.e. ``ceil((block_M + window) / T_q)`` query tiles instead of every tile from
``k0`` to the end. The per-row band mask ``t - window < k <= t`` applies to the
recomputed ``P`` exactly as in ``fwd.py``/``bwd_dq.py``.

Differences from ``dense_attn``'s backward:

- ``K_shared`` is used as both K and V (K == V), so there is no ``V_shared``.
- ``dK`` and ``dV`` accumulate into **one** fp32 fragment ``dkv`` (``dKV = dK +
  dV``), reduced over every head and query token the block sees -- for free,
  because heads are rows of the query tile.
- **No ``dQ`` here.** ``dense_attn`` scattered ``dQ`` from this same kernel with
  an fp32 ``atomic_add``; on GB10 (227 GB/s measured) that read-modify-writes the
  whole 134 MB ``dQ`` once per KV block, ~4.3 GB of traffic at B1 H64 S4096 D128,
  and it cost 9 of the kernel's 28 ms. ``bwd_dq.py`` recomputes S and dS from a
  query-owning CTA instead and stores ``dQ`` exactly once. See ``README.md``.
- Grid gains a **split** dimension over the query loop. With heads folded into
  rows, the natural grid is only ``B * S/block_M`` CTAs (16-32 at B1 S4096, on
  48 SMs) each doing H times the work; each split owns a contiguous chunk of
  query tiles and writes its own fp32 slice of ``dKV_partial [splits, B, S, D]``,
  summed in torch. fp32 split buffers, never bf16 atomics.

Dropping ``dQ`` also frees the ``dsT_shared`` staging tile and the ``dq``
accumulator, which is what lets ``block_M`` reach 256: this kernel re-reads Q and
dO once per KV block, so halving the number of KV blocks halves its dominant
memory cost.
"""

import tilelang
import tilelang.language as T

_JIT = dict(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})


@tilelang.jit(out_idx=[2], **_JIT)
def preprocess(batch, rows, dim, blk=32):
    """``delta[b, r] = rowsum(dO[b, r] * O[b, r])`` in fp32 over the packed ``rows = S*H``."""
    dtype, accum_dtype = T.bfloat16, T.float32
    shape = [batch, rows, dim]

    @T.prim_func
    def main(
            O: T.Tensor(shape, dtype),
            dO: T.Tensor(shape, dtype),
            Delta: T.Tensor([batch, rows], accum_dtype),
    ):
        with T.Kernel(T.ceildiv(rows, blk), batch) as (by, bz):
            o = T.alloc_fragment([blk, blk], accum_dtype)  # fp32: bf16 products would lose ~3 digits
            do = T.alloc_fragment([blk, blk], accum_dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(O[bz, by * blk:(by + 1) * blk, k * blk:(k + 1) * blk], o)
                T.copy(dO[bz, by * blk:(by + 1) * blk, k * blk:(k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, by * blk:(by + 1) * blk])

    return main


@tilelang.jit(**_JIT)
def bwd_kv(batch, heads, seq_len, dim, window, is_causal, splits=1, block_M=128, block_N=64, num_stages=1, threads=256):
    """KV-block-owning ``dKV = dK + dV``. ``block_M`` = latent tokens, ``block_N`` = packed query rows."""
    assert is_causal, "swa_attn is causal-only"
    assert block_N % heads == 0, f"block_N={block_N} must be a multiple of heads={heads}"
    tq = block_N // heads  # query tokens per query tile
    sm_scale = (1.0 / dim)**0.5
    scale = sm_scale * 1.44269504  # log2(e), folded into the exp2 softmax
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
            dKV: T.Tensor([splits, batch, seq_len, dim], accum_dtype),
    ):
        with T.Kernel(splits, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
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

            T.copy(KV[bz, by * block_M:(by + 1) * block_M, :], K_shared)
            T.clear(dkv)

            # band: query tokens seeing this KV block are [k0, k0 + block_M - 1 + window - 1]
            loop_st = T.floordiv(by * block_M, tq)
            loop_ed = T.min(T.ceildiv(seq_len, tq), T.ceildiv((by + 1) * block_M + window - 1, tq))
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
                        T.And(by * block_M + i <= k * tq + j // heads,
                              k * tq + j // heads < by * block_M + i + window), qkT[i, j], 0)

                T.copy(dO[bz, k * block_N:(k + 1) * block_N, :], do)
                T.clear(dsT)
                T.gemm(K_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)  # V == K
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dkv, policy=T.GemmWarpPolicy.FullRow)  # the dV half

                T.copy(Delta[bz, k * block_N:(k + 1) * block_N], delta)
                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.gemm(dsT_cast, q, dkv, policy=T.GemmWarpPolicy.FullRow)  # the dK half, same accumulator

            T.copy(dkv, dKV[bx, bz, by * block_M:(by + 1) * block_M, :])

    return main
