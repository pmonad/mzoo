# source: tile-ai/tilelang@v0.1.14 examples/flash_attention/example_mha_fwd_bshd.py (adapted) + lse output
"""dense_attn: dense FlashAttention-2 forward with LSE, bf16 in/out, fp32 accumulate, BSHD.

FA2 forward that also returns ``lse [B, H, S]`` fp32, the natural-log
log-sum-exp per query row that FA2 saves for the backward pass. The kernel runs
the online softmax in base 2 (``exp2``), so the conversion at the end is
``lse = (log2(logsum) + m * scale) * ln2`` with ``scale = d**-0.5 * log2(e)``,
which is ``ln(sum_j exp(s_j * d**-0.5))``.

Optional per-head learnable attention sink (``sinks [H]`` fp32, DSV4.1 /
gpt-oss semantics, see ``examples/attention_sink/example_mha_sink_fwd_bhsd.py``):
a scalar logit that joins the softmax as one extra column and is then dropped,
so it only enlarges the denominator. The sink is an *unscaled* natural-log
logit, hence its own ``* log2(e)`` against ``scores_max * scale``. It is added
to ``logsum`` before the normalize, so the emitted ``lse`` includes it too and
the backward's recomputed P is automatically the sink-normalised one.
``has_sink`` is a static flag: without a sink the generated kernel is byte-for-
byte the sink-free one (no extra argument, no extra register).

Baseline for the CSA2 attention series (see ``csa2_attn_design.md`` step 1a):
no MQA, no windows. Target GB10 (sm121, SM120 family: ``mma.sync``
tensor cores, 99 KB smem/block), so ``T.gemm`` lowers to the sm80/89 path.

Head dims {64, 96, 128, 256} are supported. ``D=96`` needs no special handling
on this stack: the tilelang smem layout/swizzle accepts the non-power-of-two
minor extent, so there is no D-padding path.

Tiling is per-dim (``CONFIGS``), picked by a small sweep at B1 H16 S4096 causal.
The binding constraint is the 99 KB smem budget: live tiles are Q + stages x
(K, V), so ``D=256`` only fits at ``block_M=block_N=64, num_stages=1`` (96 KB).
``block_M=64`` with 256 threads hits a tilelang layout-inference conflict
(see ``tickets/0001-tilelang-issues.md``), hence 128 threads at ``D=256``.

``seq_len`` must be a multiple of the config's ``block_M`` (asserted).
"""

import torch
import tilelang
import tilelang.language as T

CONFIGS = {  # head dim -> (block_M, block_N, num_stages, threads)
    64: dict(block_M=128, block_N=128, num_stages=2, threads=256),
    96: dict(block_M=128, block_N=128, num_stages=1, threads=256),
    128: dict(block_M=128, block_N=128, num_stages=1, threads=256),
    256: dict(block_M=64, block_N=64, num_stages=1, threads=128),
}


@tilelang.jit(out_idx=[3, 4], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn(batch, heads, seq_len, dim, is_causal, has_sink=False, block_M=128, block_N=128, num_stages=1, threads=256):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e), folded into the exp2 softmax
    log2e = 1.44269504  # the sink is an unscaled natural-log logit, so it needs its own conversion
    ln2 = 0.6931471805599453  # exp2-domain lse -> natural log
    shape = [batch, seq_len, heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.macro
    def body(Q, K, V, Output, LSE, Sinks):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            if has_sink:
                sink = T.alloc_local([1], accum_dtype)  # one scalar load per block, hoisted above the KV loop
                sink[0] = Sinks[by]

            loop_range = (T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N))

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
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

                T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            if has_sink:  # extra softmax column, dropped from the numerator -> denominator only
                for i in T.Parallel(block_M):
                    logsum[i] += T.exp2(sink[0] * log2e - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])
            for i in T.Parallel(block_M):
                logsum[i] = (T.log2(logsum[i]) + scores_max[i] * scale) * ln2
            T.copy(logsum, LSE[bz, by, bx * block_M:(bx + 1) * block_M])

    # Sinks trails the outputs so that ``out_idx`` is the same with and without it.
    if has_sink:

        @T.prim_func
        def main(
                Q: T.Tensor(shape, dtype),
                K: T.Tensor(shape, dtype),
                V: T.Tensor(shape, dtype),
                Output: T.Tensor(shape, dtype),
                LSE: T.Tensor([batch, heads, seq_len], accum_dtype),
                Sinks: T.Tensor([heads], accum_dtype),
        ):
            body(Q, K, V, Output, LSE, Sinks)
    else:

        @T.prim_func
        def main(
                Q: T.Tensor(shape, dtype),
                K: T.Tensor(shape, dtype),
                V: T.Tensor(shape, dtype),
                Output: T.Tensor(shape, dtype),
                LSE: T.Tensor([batch, heads, seq_len], accum_dtype),
        ):
            body(Q, K, V, Output, LSE, None)

    return main


def fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = True, sinks: torch.Tensor | None = None):
    """Dense FA2 forward. q/k/v bf16 [B, S, H, D] -> (o bf16 [B, S, H, D], lse fp32 [B, H, S]).

    ``sinks``: optional per-head learnable sink ``[H]`` (bf16/fp32, cast to fp32);
    one extra softmax column that only enlarges the denominator. ``lse`` includes it.
    """
    assert q.shape == k.shape == v.shape, f"shape mismatch: {q.shape} {k.shape} {v.shape}"
    assert q.dtype == torch.bfloat16, f"expected bf16, got {q.dtype}"
    batch, seq_len, heads, dim = q.shape
    assert dim in CONFIGS, f"unsupported head dim {dim}"
    cfg = CONFIGS[dim]
    assert seq_len % cfg["block_M"] == 0, f"dense_attn needs seq_len % {cfg['block_M']} == 0, got {seq_len}"

    kernel = flashattn(batch, heads, seq_len, dim, causal, sinks is not None, **cfg)
    qkv = [x.contiguous() for x in (q, k, v)]
    if sinks is None:
        return kernel(*qkv)
    assert sinks.shape == (heads,), f"sinks must be [H]={heads}, got {tuple(sinks.shape)}"
    return kernel(*qkv, sinks.detach().float().contiguous())
