# source: tile shape and gemm orientations from tilelang@v0.1.14
# examples/dsa_sparse_finetune/indexer_bwd.py (bf16 upstream), re-based on our fwd.py
# tiling (keys in M, packed query-head rows in N) instead of its per-token top-k loop.
"""indexer backward: dq / dk / dw for the relu-gated weighted reduce, bf16, no softmax.

Forward (``fwd.py``): ``L[b,s,t] = sum_h w[b,s,h] * relu(q[b,s,h,:] . k[b,t,:]) * scale``
with ``scale = Di**-0.5`` and group-causal masking. Given ``dL`` (fp32 ``[B, S, T]``,
gradient w.r.t. the *visible* scores only -- masked entries are gated to zero inside the
kernel, so whatever the loss put there is ignored):

- ``dw[b,s,h] = sum_t relu(r) * scale * dL``                            (fp32, exclusive)
- ``dr[n=(t), m=(s,h)] = w * dL * scale * (r > 0)``   (relu subgradient: 0 at r == 0,
  matching torch's ``relu`` backward)
- ``dq[s,h,:] = sum_t dr @ k``                                         (bf16 out)
- ``dk[t,:]   = sum_{s,h} dr^T @ q``  (fp32 atomics over the query axis -- every query
  block contributes to every visible key tile)

The relu mask is **recomputed**, not stored: the backward re-runs the score GEMM
(``K_shared @ Q_shared^T -> s [block_T, block_M]``, exactly ``fwd.py``'s orientation)
and re-derives the gate, so no ``[S, Hi, T]`` mask tensor is saved by the autograd
wrapper. Accumulation is fp32; only the two gradient GEMM operands are cast to bf16.

No softmax, no ``lse`` -- this is not an attention backward. Top-k selection is
discrete and gets no gradient (the training signal is the indexer's own auxiliary
loss); see ``attn.py`` for the autograd wrapper and the fp4 straight-through estimator.

Config ``CONFIGS``/constraints are shared with ``fwd.py``. Untuned (2026-09-14 call).
"""

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

from mzoo.layers.attn.indexer.fwd import CONFIGS, check_shapes


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def index_score_bwd(batch, seq_len, seq_kv, heads, dim, compress_ratio,
                    block_M=128, block_T=64, num_stages=2, threads=128):
    assert block_M % heads == 0
    tokens = block_M // heads
    scale = dim**-0.5
    dtype, accum = T.bfloat16, T.float32
    num_t = T.ceildiv(seq_kv, block_T)

    @T.prim_func
    def main(
            Q: T.Tensor([batch, seq_len * heads, dim], dtype),
            K: T.Tensor([batch, seq_kv, dim], dtype),
            Weights: T.Tensor([batch, seq_len, heads], accum),
            DL: T.Tensor([batch, seq_len, seq_kv], accum),
            DQ: T.Tensor([batch, seq_len * heads, dim], dtype),
            DW: T.Tensor([batch, seq_len, heads], accum),
            DK: T.Tensor([batch, seq_kv, dim], accum),
    ):
        with T.Kernel(T.ceildiv(seq_len, tokens), batch, threads=threads) as (bx, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_T, dim], dtype)
            D_shared = T.alloc_shared([tokens, block_T], accum)  # dL tile, [token, key]
            w = T.alloc_fragment([tokens, heads], accum)
            s = T.alloc_fragment([block_T, block_M], accum)
            s3 = T.reshape(s, (block_T, tokens, heads))
            dr = T.alloc_fragment([block_T, block_M], accum)
            dr3 = T.reshape(dr, (block_T, tokens, heads))
            dr_shared = T.alloc_shared([block_T, block_M], dtype)  # bf16 staging for the gemms
            dq = T.alloc_fragment([block_M, dim], accum)
            dw = T.alloc_fragment([tokens, heads], accum)
            dk = T.alloc_fragment([block_T, dim], accum)

            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.copy(Weights[bz, bx * tokens:(bx + 1) * tokens, :], w)
            T.fill(dq, 0)
            T.fill(dw, 0)

            visible = ((bx + 1) * tokens) // compress_ratio
            loop_range = T.min(num_t, T.ceildiv(visible, block_T))

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, k * block_T:(k + 1) * block_T, :], K_shared)
                T.copy(DL[bz, bx * tokens:(bx + 1) * tokens, k * block_T:(k + 1) * block_T], D_shared)
                T.gemm(K_shared, Q_shared, s, transpose_B=True, clear_accum=True,
                       policy=T.GemmWarpPolicy.FullRow)
                for n, i, h in T.Parallel(block_T, tokens, heads):
                    t = k * block_T + n
                    d = T.if_then_else(t < (bx * tokens + i + 1) // compress_ratio, D_shared[i, n], 0.0)
                    r_relu = T.alloc_var(accum)
                    r_relu = T.max(s3[n, i, h], 0)
                    s3[n, i, h] = r_relu * scale * d  # dw contribution
                    dr3[n, i, h] = d * w[i, h] * scale * T.if_then_else(r_relu > 0, 1.0, 0.0)
                T.reduce_sum(s3, dw, dim=0, clear=False)
                T.copy(dr, dr_shared)
                T.gemm(dr_shared, K_shared, dq, transpose_A=True, clear_accum=False)   # dq [M, dim]
                T.gemm(dr_shared, Q_shared, dk, transpose_A=False, clear_accum=True)   # dk [T, dim]
                for n, j in T.Parallel(block_T, dim):
                    T.atomic_add(DK[bz, k * block_T + n, j], dk[n, j])

            T.copy(dq, DQ[bz, bx * block_M:(bx + 1) * block_M, :])
            T.copy(dw, DW[bz, bx * tokens:(bx + 1) * tokens, :])

    return main


def bwd(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, dl: torch.Tensor,
        *, compress_ratio: int) -> tuple:
    """Indexer score backward. ``q`` bf16 [B, S, Hi, Di], ``k`` bf16 [B, T, Di], ``w``
    fp32 [B, S, Hi], ``dl`` fp32 [B, S, T] -> ``(dq bf16, dk bf16, dw fp32)``.

    ``dl`` entries at group-causally masked positions are ignored (zero contribution),
    whatever the caller put there -- the mask is re-derived, not read off ``dl``.
    """
    dim = q.shape[-1]
    assert dim in CONFIGS, f"unsupported index head dim {dim} (have {sorted(CONFIGS)})"
    cfg = CONFIGS[dim]
    batch, seq_len, heads, dim, seq_kv = check_shapes(q, k, w, cfg)
    assert dl.shape == (batch, seq_len, seq_kv) and dl.dtype == torch.float32, \
        f"dl must be [B,S,T] fp32, got {tuple(dl.shape)} {dl.dtype}"
    assert seq_kv > 0, "bwd of an empty key axis is all zeros -- handle at the caller"
    q, k, w, dl = (x.contiguous() for x in (q, k, w, dl))

    pad = -seq_kv % cfg["block_T"]
    if pad:  # keys and dl zero-padded; the pad columns get zero gradient by construction
        k = F.pad(k, (0, 0, 0, pad))
        dl = F.pad(dl, (0, pad))
    kernel = index_score_bwd(batch, seq_len, seq_kv + pad, heads, dim, compress_ratio, **cfg)
    # DK is accumulated with atomics and DQ/DW are written block-exclusively, but all
    # three are in-out buffers (no out_idx): tilelang's out_idx tensors are
    # empty-allocated, which would leave the atomically-accumulated DK full of garbage.
    dq = torch.empty(batch, seq_len * heads, dim, dtype=torch.bfloat16, device=q.device)
    dw = torch.empty(batch, seq_len, heads, dtype=torch.float32, device=q.device)
    dk = torch.zeros(batch, seq_kv + pad, dim, dtype=torch.float32, device=q.device)
    kernel(q.view(batch, seq_len * heads, dim), k, w, dl, dq, dw, dk)
    return dq.view(batch, seq_len, heads, dim), dk[:, :seq_kv].to(q.dtype), dw
