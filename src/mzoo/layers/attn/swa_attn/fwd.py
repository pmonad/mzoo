# source: copied from ../latent_attn/fwd.py (tile-ai/tilelang@v0.1.14 examples/flash_attention/example_mha_fwd_bshd.py);
# heads-in-rows packing follows examples/dsa_sparse_finetune/sparse_mla_fwd.py
"""swa_attn: sliding-window shared-latent MQA FlashAttention-2 forward, bf16 in/out, fp32 accumulate, BSHD.

One change from ``latent_attn``: every query token sees only the ``window`` most
recent raw KV tokens, **itself included** -- ``t - window + 1 <= k <= t``. That is
``golden(level="window", window=W)`` and transformers'
``sliding_window_overlay`` (which keeps ``kv_idx > q_idx - W``), verified in
``golden_ref_test.py`` against the vendored model's ``compress_ratio == 0`` layer.

Two consequences:

- **The KV loop is restricted, not just masked.** A block owns tokens
  ``[t0, t0 + T_q - 1]`` with ``t0 = bx * T_q``, so the only KV tiles it can ever
  touch are the ones overlapping ``[t0 - window + 1, t0 + T_q - 1]``. The loop
  runs from ``floor(max(0, t0 - window + 1) / block_N)`` to the causal end, i.e.
  ``ceil(window / block_N) + ceil(T_q / block_N)`` tiles instead of
  ``ceil((t0 + T_q) / block_N)``. Work becomes O(S * window), not O(S^2).
- **The mask gains a lower edge**, still per *row token*: a row at token ``t``
  keeps column ``k`` iff ``t - window < k <= t``. ``window >= seq_len`` degenerates
  to plain causal and reproduces ``latent_attn`` to within one bf16 ulp (the two
  packages pick different tiles, so the softmax accumulates in a different order).

``window`` is a compile-time constant (part of the jit key), as are ``heads`` and
``seq_len``. ``causal=False`` is not supported: a non-causal sliding window is not
a shape the model has, and the online softmax's loop bounds assume the causal end.

Everything else is ``latent_attn``: K == V shared latent ``kv [B, S, 1, D]`` with
one KV head serving ``H`` query heads, heads packed into the M dimension
(``T_q = block_M // H`` tokens x H heads per block), one smem KV tile for both
GEMMs, online base-2 softmax, fp32 ``lse [B, H, S]``, optional per-head sink as one
denominator-only softmax column.

The fp8 window cache of the model is a *caller-side* fake-quant round trip in v1
(ticket 0002): the kernel only ever sees bf16, so there is nothing to do here.
In-kernel dequant is ticket 0010.

Target GB10 (sm121, SM120 family: ``mma.sync`` tensor cores, 99 KB smem/block).
Tiles are per head dim (``CONFIGS``), re-swept for the banded loop -- see
``README.md``. D in {64, 96, 128} are tuned; 256 is shape support only.
"""

import torch
import tilelang
import tilelang.language as T

CONFIGS = {  # head dim -> (block_M = T_q * H rows, block_N = KV tokens, num_stages, threads)
    64: dict(block_M=256, block_N=64, num_stages=3, threads=256),
    96: dict(block_M=256, block_N=64, num_stages=3, threads=256),
    128: dict(block_M=256, block_N=32, num_stages=3, threads=256),
    256: dict(block_M=128, block_N=32, num_stages=2, threads=256),  # smem-capped, see README
}


@tilelang.jit(out_idx=[2, 3], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn(batch, heads, seq_len, dim, window, is_causal, has_sink=False, block_M=128, block_N=128, num_stages=2, threads=256):
    assert is_causal, "swa_attn is causal-only (see module docstring)"
    assert block_M % heads == 0, f"block_M={block_M} must be a multiple of heads={heads}"
    assert window >= 1, f"window must be >= 1, got {window}"
    tq = block_M // heads  # query tokens per block; rows are (token, head) in BSHD order
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e), folded into the exp2 softmax
    log2e = 1.44269504  # the sink is an unscaled natural-log logit, so it needs its own conversion
    ln2 = 0.6931471805599453  # exp2-domain lse -> natural log
    # Running rowmax floor. Unlike plain causal, a banded row can meet a KV tile it sees
    # *nothing* in (its band starts above the tile), and then both the running max and the
    # tile max are -inf, so the FA2 rescale computes -inf - (-inf) = NaN. Flooring the running
    # max at a finite, unreachably small value makes that step exp2(0) = 1 (no rescale) while
    # every masked logit still gives exp2(-inf - floor*scale) = 0. Any real logit dominates it.
    neg_floor = -1e30
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
            T.fill(scores_max, neg_floor)
            if has_sink:
                sink = T.alloc_fragment([block_M], accum_dtype)  # per row -> per head, row % H
                for i in T.Parallel(block_M):
                    sink[i] = Sinks[i % heads]

            # only the KV tiles overlapping [t0 - window + 1, t0 + tq - 1], t0 = bx * tq
            loop_st = T.floordiv(T.max(bx * tq - window + 1, 0), block_N)
            loop_ed = T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * tq, block_N))

            for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                T.copy(KV[bz, k * block_N:(k + 1) * block_N, :], K_shared)
                # band mask from the row's TOKEN (row // H): t - window < k <= t
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        T.And(bx * tq + i // heads >= k * block_N + j,
                              k * block_N + j > bx * tq + i // heads - window), 0, -T.infinity(acc_s.dtype))
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, neg_floor)  # floor, not -inf: see neg_floor above
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


def check_shapes(q: torch.Tensor, kv: torch.Tensor, cfg: dict, what: str = "swa_attn") -> tuple:
    """Validate the MQA call shapes and return ``(batch, seq_len, heads, dim)``."""
    assert q.dtype == torch.bfloat16, f"expected bf16, got {q.dtype}"
    batch, seq_len, heads, dim = q.shape
    assert kv.shape == (batch, seq_len, 1, dim), f"kv must be [B, S, 1, D]={(batch, seq_len, 1, dim)}, got {tuple(kv.shape)}"
    assert cfg["block_M"] % heads == 0, f"{what} needs heads | block_M={cfg['block_M']}, got H={heads}"
    tq = cfg["block_M"] // heads
    assert seq_len % tq == 0, f"{what} needs seq_len % {tq} == 0 (block_M/H tokens per block), got {seq_len}"
    return batch, seq_len, heads, dim


def fwd(q: torch.Tensor, kv: torch.Tensor, *, window: int, causal: bool = True,
        sinks: torch.Tensor | None = None):
    """Sliding-window shared-latent MQA FA2 forward.

    ``q`` bf16 [B, S, H, D], ``kv`` bf16 [B, S, 1, D] (K == V, one head for all H)
    -> ``(o bf16 [B, S, H, D], lse fp32 [B, H, S])``.

    ``window``: number of visible raw KV tokens per query, self included
    (``t - window + 1 <= k <= t``). Compile-time constant, part of the jit key.
    ``window >= seq_len`` is plain causal.

    ``causal=False`` is rejected: the model only ever has the causal band.

    ``sinks``: optional per-head learnable sink ``[H]`` (cast to fp32); one extra
    softmax column that only enlarges the denominator. ``lse`` includes it.
    """
    assert causal, "swa_attn is causal-only: a non-causal sliding window is not a model shape"
    assert isinstance(window, int) and window >= 1, f"window must be a positive int, got {window!r}"
    dim = q.shape[-1]
    assert dim in CONFIGS, f"unsupported head dim {dim}"
    cfg = CONFIGS[dim]
    batch, seq_len, heads, dim = check_shapes(q, kv, cfg)

    kernel = flashattn(batch, heads, seq_len, dim, min(window, seq_len), causal, sinks is not None, **cfg)
    args = [q.contiguous().view(batch, seq_len * heads, dim), kv.contiguous().view(batch, seq_len, dim)]
    if sinks is not None:
        assert sinks.shape == (heads,), f"sinks must be [H]={heads}, got {tuple(sinks.shape)}"
        args.append(sinks.detach().float().contiguous())
    o, lse = kernel(*args)
    # lse comes out packed [B, S*H] (token-major); the public layout is [B, H, S]
    return o.view(batch, seq_len, heads, dim), lse.view(batch, seq_len, heads).permute(0, 2, 1).contiguous()
