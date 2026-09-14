# source: copied from ../swa_attn/fwd.py (itself ../latent_attn/fwd.py, tile-ai/tilelang@v0.1.14
# examples/flash_attention/example_mha_fwd_bshd.py); two-source loop shape follows
# examples/deepseek_v4/sparse_attn_fwd_sm90.py
"""csa_attn: compressed (main) KV as a **second source** in the sliding-window MQA FA2 forward.

One change from ``swa_attn``: after the window tiles of the raw latent
``kv [B, S, 1, D]``, the same loop walks the visible entries of a second KV source
``main_kv [B, G, 1, D]`` (the compressor's latents, entry ``g`` at group index
``g``), with **group-causal** visibility -- entry ``g`` is visible to token ``t``
iff ``g < (t + 1) // compress_ratio`` (``compress_lens`` in
``DeepseekV41Indexer.forward`` step 3). Dense: every visible entry, no top-k yet.

One running ``(m, l, acc)`` spans both sources, so there is **no LSE merge**: the
main tiles simply continue the online softmax the window tiles started, exactly as
if the KV axis were the model's ``torch.cat([kv, compressed_kv], dim=2)``. The sink
stays one denominator-only column at the very end, and ``lse`` covers window + main
+ sink.

Loop bounds. A block owns tokens ``[t0, t0 + T_q - 1]``, ``t0 = bx * T_q``, so the
highest group index any of its rows can see is ``(t0 + T_q) // ratio`` and the main
loop runs ``ceil(min(G, (t0 + T_q) // ratio) / block_N)`` tiles. The per-row tail is
masked with ``g < min(G, (t + 1) // ratio)`` -- one comparison, because ``G`` is a
python int, and the ``min`` also kills the rows of the zero-padded tail (see below).

``G == 0`` (no compressed entries yet) drops the second loop entirely at compile
time, so the kernel *is* ``swa_attn``'s. ``compress_ratio`` and ``G`` are
compile-time constants, part of the jit key, like ``window``/``heads``/``seq_len``.

The compressor, the latent-position RoPE and the fp4 fake-quant all stay in torch
and out of the kernel path (ticket 0003 v1); the kernel consumes bf16 ``main_kv``.
In-kernel e2m1 + e4m3-per-16 dequant is ticket 0010 -- see ``README.md`` for the
cache layout this package fixes.

Everything else is ``swa_attn`` verbatim: K == V shared latent, heads packed into
the M dimension (``T_q = block_M // H`` tokens x H heads per block), one smem tile
per source serving both GEMMs, the ``neg_floor`` rowmax floor, fp32 ``lse``,
``causal=False`` rejected.

Target GB10 (sm121). ``CONFIGS`` is **untuned**: ``swa_attn``'s table verbatim, no
sweep run for this ticket (correctness first). Nothing had to shrink -- the main
source's tile is single-buffered because its loop is ``T.serial``, which is itself a
workaround for a tilelang pipeline miscompile (``tickets/0001-tilelang-issues.md``,
``README.md`` -> Known issues).
"""

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

CONFIGS = {  # head dim -> (block_M = T_q * H rows, block_N = KV tokens, num_stages, threads)
    64: dict(block_M=256, block_N=64, num_stages=3, threads=256),
    96: dict(block_M=256, block_N=64, num_stages=3, threads=256),
    128: dict(block_M=256, block_N=32, num_stages=3, threads=256),
    256: dict(block_M=128, block_N=32, num_stages=2, threads=256),
}


@tilelang.jit(out_idx=[3, 4], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn(batch, heads, seq_len, dim, groups, gpad, window, ratio, is_causal, has_sink=False,
              block_M=128, block_N=128, num_stages=2, threads=256):
    assert is_causal, "csa_attn is causal-only (see module docstring)"
    assert block_M % heads == 0, f"block_M={block_M} must be a multiple of heads={heads}"
    assert window >= 1, f"window must be >= 1, got {window}"
    assert ratio >= 1, f"compress_ratio must be >= 1, got {ratio}"
    assert gpad % block_N == 0, f"padded groups={gpad} must be a multiple of block_N={block_N}"
    assert 0 <= groups <= gpad, f"groups={groups} must fit the padded extent {gpad}"
    tq = block_M // heads  # query tokens per block; rows are (token, head) in BSHD order
    has_main = groups > 0
    main_tiles = -(-groups // block_N)  # tiles holding a real entry; the rest of gpad is zero pad
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e), folded into the exp2 softmax
    log2e = 1.44269504  # the sink is an unscaled natural-log logit, so it needs its own conversion
    ln2 = 0.6931471805599453  # exp2-domain lse -> natural log
    neg_floor = -1e30  # running-rowmax floor; see swa_attn/README.md -> Decisions
    q_shape = [batch, seq_len * heads, dim]  # packed rows: q.view(B, S*H, D)
    kv_shape = [batch, seq_len, dim]  # the single shared latent head
    main_shape = [batch, max(gpad, 1), dim]  # zero-padded [B, gpad, D]; a 1-row dummy when G == 0
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.macro
    def body(Q, KV, MainKV, Output, LSE, Sinks):
        with T.Kernel(T.ceildiv(seq_len, tq), batch, threads=threads) as (bx, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)  # serves both S = Q K^T and O += P K
            M_shared = T.alloc_shared([block_N, dim], dtype)  # the main source's tile (liveness-aliased)
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

            # --- source 1: the raw window, KV tiles overlapping [t0 - window + 1, t0 + tq - 1]
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
                T.fill(scores_max, neg_floor)  # floor, not -inf: a row can see nothing in a tile
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

            # --- source 2: the compressed/main entries, group-causal, same running (m, l, acc)
            if has_main:
                # min of two ceildivs, not ceildiv of a min: the latter also crashes codegen
                # (`Downcast from tirx.Sub to ir.IntImm`). See tickets/0001-tilelang-issues.md.
                main_ed = T.min(main_tiles, T.ceildiv(T.floordiv((bx + 1) * tq, ratio), block_N))
                # T.serial, NOT T.Pipelined: the main loop's software pipeline miscompiles
                # here (silently wrong results, see tickets/0001-tilelang-issues.md).
                for kg in T.serial(0, main_ed):
                    T.copy(MainKV[bz, kg * block_N:(kg + 1) * block_N, :], M_shared)
                    # group-causal mask: g < min(G, (t + 1) // ratio). G is the TRUE entry count,
                    # so the min also drops MainKV's zero-padded tail rows.
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            kg * block_N + j < T.min(groups, T.floordiv(bx * tq + i // heads + 1, ratio)),
                            0, -T.infinity(acc_s.dtype))
                    T.gemm(Q_shared, M_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, neg_floor)  # floor, not -inf: a row can see nothing in a tile
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
                    T.gemm(acc_s_cast, M_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)  # V == K, no V load

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
                MainKV: T.Tensor(main_shape, dtype),
                Output: T.Tensor(q_shape, dtype),
                LSE: T.Tensor([batch, seq_len * heads], accum_dtype),
                Sinks: T.Tensor([heads], accum_dtype),
        ):
            body(Q, KV, MainKV, Output, LSE, Sinks)
    else:

        @T.prim_func
        def main(
                Q: T.Tensor(q_shape, dtype),
                KV: T.Tensor(kv_shape, dtype),
                MainKV: T.Tensor(main_shape, dtype),
                Output: T.Tensor(q_shape, dtype),
                LSE: T.Tensor([batch, seq_len * heads], accum_dtype),
        ):
            body(Q, KV, MainKV, Output, LSE, None)

    return main


def check_shapes(q: torch.Tensor, kv: torch.Tensor, main_kv: torch.Tensor, cfg: dict,
                 what: str = "csa_attn") -> tuple:
    """Validate the MQA call shapes and return ``(batch, seq_len, heads, dim, groups)``."""
    assert q.dtype == torch.bfloat16, f"expected bf16, got {q.dtype}"
    batch, seq_len, heads, dim = q.shape
    assert kv.shape == (batch, seq_len, 1, dim), f"kv must be [B, S, 1, D]={(batch, seq_len, 1, dim)}, got {tuple(kv.shape)}"
    groups = main_kv.shape[1]
    assert main_kv.shape == (batch, groups, 1, dim), f"main_kv must be [B, G, 1, D], got {tuple(main_kv.shape)}"
    assert cfg["block_M"] % heads == 0, f"{what} needs heads | block_M={cfg['block_M']}, got H={heads}"
    tq = cfg["block_M"] // heads
    assert seq_len % tq == 0, f"{what} needs seq_len % {tq} == 0 (block_M/H tokens per block), got {seq_len}"
    return batch, seq_len, heads, dim, groups


def pack_main(main_kv: torch.Tensor, batch: int, dim: int, block: int) -> tuple[torch.Tensor, int]:
    """``[B, G, 1, D]`` -> a ``[B, ceil(G/block)*block, D]`` zero-padded contiguous view.

    The kernel's main loop reads whole ``block``-row tiles, so the last tile must be
    in bounds; the padded rows are killed by the ``g < min(G, ...)`` mask, not by
    their (zero) values. ``G == 0`` yields a 1-row dummy the kernel never reads.
    """
    groups = main_kv.shape[1]
    if groups == 0:
        return torch.zeros(batch, 1, dim, device=main_kv.device, dtype=main_kv.dtype), 0
    padded = -(-groups // block) * block
    m = main_kv.contiguous().view(batch, groups, dim)
    if padded != groups:
        m = F.pad(m, (0, 0, 0, padded - groups))
    return m.contiguous(), padded


def fwd(q: torch.Tensor, kv: torch.Tensor, main_kv: torch.Tensor, *, window: int,
        compress_ratio: int, causal: bool = True, sinks: torch.Tensor | None = None):
    """Compressed-source sliding-window shared-latent MQA FA2 forward.

    ``q`` bf16 [B, S, H, D], ``kv`` bf16 [B, S, 1, D] (raw latent, K == V),
    ``main_kv`` bf16 [B, G, 1, D] (compressed latents, K == V)
    -> ``(o bf16 [B, S, H, D], lse fp32 [B, H, S])``.

    ``window``: visible raw KV tokens per query, self included (``t - window + 1 <= k <= t``).
    ``compress_ratio``: entry ``g`` of ``main_kv`` is visible to token ``t`` iff
    ``g < (t + 1) // compress_ratio``. Both are compile-time constants (jit key), as is ``G``.
    ``G == 0`` reduces exactly to ``swa_attn``.

    ``causal=False`` is rejected: the model only ever has the causal band.

    ``sinks``: optional per-head learnable sink ``[H]``; one extra softmax column over
    the *combined* window+main logits that only enlarges the denominator. ``lse``
    includes it.
    """
    assert causal, "csa_attn is causal-only: a non-causal sliding window is not a model shape"
    assert isinstance(window, int) and window >= 1, f"window must be a positive int, got {window!r}"
    assert isinstance(compress_ratio, int) and compress_ratio >= 1, (
        f"compress_ratio must be a positive int, got {compress_ratio!r}")
    dim = q.shape[-1]
    assert dim in CONFIGS, f"unsupported head dim {dim}"
    cfg = CONFIGS[dim]
    batch, seq_len, heads, dim, groups = check_shapes(q, kv, main_kv, cfg)
    main, gpad = pack_main(main_kv, batch, dim, cfg["block_N"])

    kernel = flashattn(batch, heads, seq_len, dim, groups, gpad, min(window, seq_len), compress_ratio,
                       causal, sinks is not None, **cfg)
    args = [q.contiguous().view(batch, seq_len * heads, dim), kv.contiguous().view(batch, seq_len, dim), main]
    if sinks is not None:
        assert sinks.shape == (heads,), f"sinks must be [H]={heads}, got {tuple(sinks.shape)}"
        args.append(sinks.detach().float().contiguous())
    o, lse = kernel(*args)
    # lse comes out packed [B, S*H] (token-major); the public layout is [B, H, S]
    return o.view(batch, seq_len, heads, dim), lse.view(batch, seq_len, heads).permute(0, 2, 1).contiguous()
