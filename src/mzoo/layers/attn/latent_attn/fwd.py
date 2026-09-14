# source: copied from ../dense_attn/fwd.py (tile-ai/tilelang@v0.1.14 examples/flash_attention/example_mha_fwd_bshd.py);
# heads-in-rows packing follows examples/dsa_sparse_finetune/sparse_mla_fwd.py
"""latent_attn: shared-latent MQA FlashAttention-2 forward, bf16 in/out, fp32 accumulate, BSHD.

One change from ``dense_attn``: K and V are the *same* latent tensor with a
single KV head, ``kv [B, S, 1, D]``, shared by all ``H`` query heads.

Two consequences:

- **Heads go into the M (row) dimension.** A block owns ``block_M`` rows that
  are ``T_q = block_M // H`` consecutive *tokens* x ``H`` heads, in the natural
  BSHD memory order, so the whole tile is one contiguous ``[block_M, D]`` slice
  of ``q.view(B, S*H, D)``. The causal mask therefore derives from the row's
  **token** (``bx * T_q + i // H``), not from the block index. The general rule
  is ``T_q = block_M // H`` with ``block_M % H == 0`` asserted, which covers
  ``H in {4, 8, 16, 32, 64}`` for every ``block_M`` in the config table
  (128/256), and ``seq_len % T_q == 0`` is asserted too.
- **One smem KV tile serves both GEMMs.** ``S = Q K^T`` and ``O += P K`` read
  the same ``K_shared``; the V load is gone, which halves KV smem traffic and
  lets the pipeline run more stages at the same smem budget.

Everything else is ``dense_attn``: online base-2 softmax, fp32 ``lse [B, H, S]``
out, optional per-head sink (``sinks [H]`` fp32) as one extra denominator-only
softmax column with ``has_sink`` a static flag.

Target GB10 (sm121, SM120 family: ``mma.sync`` tensor cores, 99 KB smem/block).
Tiles are per head dim (``CONFIGS``), re-tuned from scratch for this layout --
see ``README.md`` for the sweep. D=64 and D=128 only.
"""

import torch
import tilelang
import tilelang.language as T

CONFIGS = {  # head dim -> (block_M = T_q * H rows, block_N = KV tokens, num_stages, threads)
    64: dict(block_M=256, block_N=128, num_stages=2, threads=256),
    128: dict(block_M=256, block_N=64, num_stages=2, threads=256),
}


@tilelang.jit(out_idx=[2, 3], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn(batch, heads, seq_len, dim, is_causal, has_sink=False, block_M=128, block_N=128, num_stages=2, threads=256):
    assert block_M % heads == 0, f"block_M={block_M} must be a multiple of heads={heads}"
    tq = block_M // heads  # query tokens per block; rows are (token, head) in BSHD order
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e), folded into the exp2 softmax
    log2e = 1.44269504  # the sink is an unscaled natural-log logit, so it needs its own conversion
    ln2 = 0.6931471805599453  # exp2-domain lse -> natural log
    q_shape = [batch, seq_len * heads, dim]  # packed rows: q.view(B, S*H, D)
    kv_shape = [batch, seq_len, dim]  # the single shared latent head
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.macro
    def body(Q, KV, Output, LSE, Sinks):
        with T.Kernel(T.ceildiv(seq_len, tq), batch, threads=threads) as (bx, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)  # serves both S = Q K^T and O += P K
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            if has_sink:
                sink = T.alloc_fragment([block_M], accum_dtype)  # per row -> per head, row % H
                for i in T.Parallel(block_M):
                    sink[i] = Sinks[i % heads]

            loop_range = (T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * tq, block_N)) if is_causal else T.ceildiv(seq_len, block_N))

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(KV[bz, k * block_N:(k + 1) * block_N, :], K_shared)
                if is_causal:  # mask from the row's TOKEN (row // H), not the block index
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(bx * tq + i // heads >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                else:
                    T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.gemm(acc_s_cast, K_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)  # V == K, no V load

            if has_sink:  # extra softmax column, dropped from the numerator -> denominator only
                for i in T.Parallel(block_M):
                    logsum[i] += T.exp2(sink[i] * log2e - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, :])
            for i in T.Parallel(block_M):
                logsum[i] = (T.log2(logsum[i]) + scores_max[i] * scale) * ln2
            T.copy(logsum, LSE[bz, bx * block_M:(bx + 1) * block_M])

    # Sinks trails the outputs so that ``out_idx`` is the same with and without it.
    if has_sink:

        @T.prim_func
        def main(
                Q: T.Tensor(q_shape, dtype),
                KV: T.Tensor(kv_shape, dtype),
                Output: T.Tensor(q_shape, dtype),
                LSE: T.Tensor([batch, seq_len * heads], accum_dtype),
                Sinks: T.Tensor([heads], accum_dtype),
        ):
            body(Q, KV, Output, LSE, Sinks)
    else:

        @T.prim_func
        def main(
                Q: T.Tensor(q_shape, dtype),
                KV: T.Tensor(kv_shape, dtype),
                Output: T.Tensor(q_shape, dtype),
                LSE: T.Tensor([batch, seq_len * heads], accum_dtype),
        ):
            body(Q, KV, Output, LSE, None)

    return main


def check_shapes(q: torch.Tensor, kv: torch.Tensor, cfg: dict, what: str = "latent_attn") -> tuple:
    """Validate the MQA call shapes and return ``(batch, seq_len, heads, dim)``."""
    assert q.dtype == torch.bfloat16, f"expected bf16, got {q.dtype}"
    batch, seq_len, heads, dim = q.shape
    assert kv.shape == (batch, seq_len, 1, dim), f"kv must be [B, S, 1, D]={(batch, seq_len, 1, dim)}, got {tuple(kv.shape)}"
    assert cfg["block_M"] % heads == 0, f"{what} needs heads | block_M={cfg['block_M']}, got H={heads}"
    tq = cfg["block_M"] // heads
    assert seq_len % tq == 0, f"{what} needs seq_len % {tq} == 0 (block_M/H tokens per block), got {seq_len}"
    return batch, seq_len, heads, dim


def fwd(q: torch.Tensor, kv: torch.Tensor, *, causal: bool = True, sinks: torch.Tensor | None = None):
    """Shared-latent MQA FA2 forward.

    ``q`` bf16 [B, S, H, D], ``kv`` bf16 [B, S, 1, D] (K == V, one head for all H)
    -> ``(o bf16 [B, S, H, D], lse fp32 [B, H, S])``.

    ``sinks``: optional per-head learnable sink ``[H]`` (cast to fp32); one extra
    softmax column that only enlarges the denominator. ``lse`` includes it.
    """
    dim = q.shape[-1]
    assert dim in CONFIGS, f"unsupported head dim {dim}"
    cfg = CONFIGS[dim]
    batch, seq_len, heads, dim = check_shapes(q, kv, cfg)

    kernel = flashattn(batch, heads, seq_len, dim, causal, sinks is not None, **cfg)
    args = [q.contiguous().view(batch, seq_len * heads, dim), kv.contiguous().view(batch, seq_len, dim)]
    if sinks is not None:
        assert sinks.shape == (heads,), f"sinks must be [H]={heads}, got {tuple(sinks.shape)}"
        args.append(sinks.detach().float().contiguous())
    o, lse = kernel(*args)
    # lse comes out packed [B, S*H] (token-major); the public layout is [B, H, S]
    return o.view(batch, seq_len, heads, dim), lse.view(batch, seq_len, heads).permute(0, 2, 1).contiguous()
