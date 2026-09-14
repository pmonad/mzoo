# source: tile-ai/tilelang@v0.1.14 examples/flash_attention/example_mha_bwd_bshd.py (adapted)
"""dense_attn: dense FlashAttention-2 backward, bf16 in/out, fp32 accumulate, BSHD [B, S, H, D].

Three kernels, exactly as the upstream example:

1. ``preprocess``  -> ``delta[b, h, s] = rowsum(dO * O)`` in fp32.
2. ``bwd``         -> one CTA owns a KV block (``block_M`` rows of K/V), loops over
   query blocks (``block_N`` rows), accumulates ``dK``/``dV`` in registers and
   scatters ``dQ`` with ``atomic_add`` into an fp32 buffer.
3. ``postprocess`` -> cast the fp32 ``dQ`` buffer to bf16 (and undo the atomic-add
   layout permutation; ``atomicAdd`` cannot be vectorized, so ``dQ`` is stored in
   the 8x8 mma fragment order, see ``_dq_layout``).

An optional per-head sink adds no kernel: ``dsinks`` is a torch reduce over the
fp32 ``delta``/``lse`` the kernels already produce (see ``bwd``).

Note the transposed block naming inherited from the example: inside the bwd
kernel ``block_M`` counts **KV** rows and ``block_N`` counts **query** rows.

``lse`` is the natural-log LSE produced by ``fwd`` (FA2 convention);
the kernel's softmax is base 2, so it is scaled by ``log2(e)`` on load.

Tiles (``CONFIGS``) are per head dim, picked by a sweep at B1 H16 S4096 causal on
GB10 (sm121, 99 KB smem/block, ``mma.sync`` only). Live smem is roughly
``K + V (block_M x dim)`` plus ``num_stages x (Q + dO) (block_N x dim)``, so the
Q tile shrinks as ``dim`` grows: D=256 only fits at ``block_M=64, block_N=16``.

Two tilelang layout-inference limits shape the search space on this stack (see
``tickets/0001-tilelang-issues.md``): ``block_M=32`` never compiles, and
``block_M=64`` needs ``threads=128``, so ``threads=256`` implies ``block_M=128``.
"""

import torch
import tilelang
import tilelang.language as T

CONFIGS = {  # head dim -> (block_M = KV rows, block_N = query rows, num_stages, threads)
    64: dict(block_M=128, block_N=64, num_stages=2, threads=256),
    96: dict(block_M=128, block_N=32, num_stages=2, threads=256),
    128: dict(block_M=128, block_N=32, num_stages=1, threads=256),
    256: dict(block_M=64, block_N=16, num_stages=1, threads=128),
}

_JIT = dict(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})


@tilelang.jit(out_idx=[2], **_JIT)
def _preprocess(batch, heads, seq_len, dim, blk=32):
    dtype, accum_dtype = T.bfloat16, T.float32
    shape = [batch, seq_len, heads, dim]

    @T.prim_func
    def main(
            O: T.Tensor(shape, dtype),
            dO: T.Tensor(shape, dtype),
            Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], accum_dtype)  # fp32: bf16 products would lose ~3 digits
            do = T.alloc_fragment([blk, blk], accum_dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(O[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], o)
                T.copy(dO[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk:(by + 1) * blk])

    return main


def _dq_layout(dQ):
    # atomicAdd cannot be vectorized, so dQ is kept in the 8x8 gemm fragment order
    return T.Layout(dQ.shape, lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2])


@tilelang.jit(out_idx=[1], **_JIT)
def _postprocess(batch, heads, seq_len, dim, blk=64):
    dtype, accum_dtype = T.bfloat16, T.float32
    shape = [batch, seq_len, heads, dim]

    @T.prim_func
    def main(dQ: T.Tensor(shape, accum_dtype), dQ_out: T.Tensor(shape, dtype)):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            T.annotate_layout({dQ: _dq_layout(dQ)})
            T.copy(dQ[bz, bx * blk:(bx + 1) * blk, by, :], dQ_out[bz, bx * blk:(bx + 1) * blk, by, :])

    return main


@tilelang.jit(**_JIT)
def _bwd(batch, heads, seq_len, dim, is_causal, block_M=64, block_N=64, num_stages=2, threads=128):
    sm_scale = (1.0 / dim)**0.5
    scale = sm_scale * 1.44269504  # log2(e), folded into the exp2 softmax
    log2e = 1.44269504  # lse arrives in natural log, the kernel works in base 2
    shape = [batch, seq_len, heads, dim]
    dtype, accum_dtype = T.bfloat16, T.float32

    @T.prim_func
    def main(
            Q: T.Tensor(shape, dtype),
            K: T.Tensor(shape, dtype),
            V: T.Tensor(shape, dtype),
            dO: T.Tensor(shape, dtype),
            LSE: T.Tensor([batch, heads, seq_len], accum_dtype),
            Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
            dQ: T.Tensor(shape, accum_dtype),
            dK: T.Tensor(shape, dtype),
            dV: T.Tensor(shape, dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim], dtype)
            V_shared = T.alloc_shared([block_M, dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            q = T.alloc_shared([block_N, dim], dtype)
            do = T.alloc_shared([block_N, dim], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dv = T.alloc_fragment([block_M, dim], accum_dtype)
            dk = T.alloc_fragment([block_M, dim], accum_dtype)
            dq = T.alloc_fragment([block_N, dim], accum_dtype)
            dv_shared = T.alloc_shared([block_M, dim], dtype)
            dk_shared = T.alloc_shared([block_M, dim], dtype)

            T.annotate_layout({dQ: _dq_layout(dQ)})
            T.copy(K[bz, by * block_M:(by + 1) * block_M, bx, :], K_shared)
            T.copy(V[bz, by * block_M:(by + 1) * block_M, bx, :], V_shared)
            T.clear(dv)
            T.clear(dk)

            loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
            loop_ed = T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                T.copy(Q[bz, k * block_N:(k + 1) * block_N, bx, :], q)
                T.clear(qkT)
                T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(LSE[bz, bx, k * block_N:(k + 1) * block_N], lse_shared)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j] * log2e)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(by * block_M + i <= k * block_N + j, qkT[i, j], 0)

                T.copy(dO[bz, k * block_N:(k + 1) * block_N, bx, :], do)
                T.clear(dsT)
                T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                T.copy(Delta[bz, bx, k * block_N:(k + 1) * block_N], delta)
                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                for i, j in T.Parallel(block_N, dim):
                    T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])

            T.copy(dv, dv_shared)
            T.copy(dk, dk_shared)
            T.copy(dv_shared, dV[bz, by * block_M:(by + 1) * block_M, bx, :])
            T.copy(dk_shared, dK[bz, by * block_M:(by + 1) * block_M, bx, :])

    return main


def bwd(q, k, v, o, lse, do, *, causal: bool = True, sinks=None, cfg: dict | None = None):
    """Dense FA2 backward. All tensors bf16 [B, S, H, D] (lse fp32 [B, H, S]) -> (dq, dk, dv, dsinks).

    With ``sinks [H]`` the three kernels are unchanged: ``lse`` already carries the
    sink column, so the recomputed P is the sink-normalised one. Only the extra
    ``dsinks[h] = -sum_{b,s} exp(sink[h] - lse[b,h,s]) * delta[b,h,s]`` is needed;
    that is O(B*H*S) elementwise, so it stays in fp32 torch rather than a kernel.
    """
    assert q.shape == k.shape == v.shape == o.shape == do.shape, "shape mismatch"
    assert q.dtype == torch.bfloat16, f"expected bf16, got {q.dtype}"
    batch, seq_len, heads, dim = q.shape
    assert dim in CONFIGS, f"unsupported head dim {dim}"
    cfg = cfg or CONFIGS[dim]
    blk = max(cfg["block_M"], cfg["block_N"], 64)
    assert seq_len % blk == 0, f"dense_attn bwd needs seq_len % {blk} == 0, got {seq_len}"

    q, k, v, o, do = (x.contiguous() for x in (q, k, v, o, do))
    delta = _preprocess(batch, heads, seq_len, dim)(o, do)
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk, dv = torch.empty_like(k), torch.empty_like(v)
    _bwd(batch, heads, seq_len, dim, causal, **cfg)(q, k, v, do, lse, delta, dq, dk, dv)
    dsinks = None
    if sinks is not None:
        dsinks = -(torch.exp(sinks.detach().float().view(1, heads, 1) - lse) * delta).sum(dim=(0, 2))
    return _postprocess(batch, heads, seq_len, dim)(dq), dk, dv, dsinks
