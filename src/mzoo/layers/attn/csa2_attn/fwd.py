# source: copied from ../csa_attn/fwd.py (itself ../swa_attn/fwd.py, ../latent_attn/fwd.py,
# tile-ai/tilelang@v0.1.14 examples/flash_attention/example_mha_fwd_bshd.py); the gather loop
# follows examples/dsa_sparse_finetune/sparse_mla_fwd.py + examples/deepseek_v4/sparse_attn_fwd_sm90.py
"""csa2_attn: the main KV source is **gathered** by per-token top-k indices (forward only).

One change from ``csa_attn``: after the window tiles of the raw latent ``kv [B, S, 1, D]``,
instead of walking every group-causally visible entry of ``main_kv [B, G, 1, D]``, the loop
walks ``ceil(topk / block_N)`` **gathered** tiles built from ``indices [B, S, topk]``::

    M_shared[i, :] = main_kv[b, indices[b, t, kt * block_N + i], :]

An index ``< 0`` (the ``-1`` empty slot) or ``>= G`` gives the whole column a ``-inf``
logit, so it contributes nothing to either GEMM (``sparse_mla_fwd.py``'s convention); the
row actually loaded for such a column is clamped to ``0`` so the gather never reads out of
bounds. Cost is therefore ``O(topk)`` per token, **flat in G**, where ``csa_attn`` was
``O(G)``.

The kernel does **not** re-apply the group-causal rule ``g < (t + 1) // ratio``: the indexer
owns it, exactly like ``golden(level="sparse")``, which "trusts ``indices`` completely".
``compress_ratio`` stays in the signature for call-shape parity with ``csa_attn`` and is
only asserted/documented.

**One token per block where possible.** Indices are per token, so a gathered tile serves
exactly one token's rows. ``block_M = T_q * heads`` is kept general (``csa_attn``'s packed
rows) but ``T_q`` is now the fewest tokens whose ``T_q * heads`` rows reach 64 (a smaller tile
fails layout inference, ``tickets/0001``): ``T_q = 1`` at ``H = 64``, 4 at ``H = 16``, 16 at
``H = 4``.
When ``T_q > 1`` the gathered section is repeated once per token of the block with the rows
of the other tokens masked off -- correct, and ``T_q`` times the gather work, which only
happens at head counts the model does not use. Rows that see nothing in a tile are exactly
the case ``csa_attn``'s ``neg_floor`` rowmax already handles.

``block_M = 64`` rows and ``threads=128``: a 16-row tile fails layout inference on this stack
and ``threads=256`` needs ``block_M % 128 == 0`` (``tickets/0001-tilelang-issues.md``), so both
of ``csa_attn``'s 256-row / 256-thread choices are out. ``block_N`` and ``num_stages`` are
``csa_attn``'s, **untuned** (see ``README.md``). The gathered loop *is* pipelined -- its trip
count is a constant, so ``csa_attn``'s miscompile trigger is absent.

``topk`` is ``-1``-padded to a multiple of ``block_N`` in the caller; ``G`` needs no
padding at all any more (rows are addressed one by one), which removes ``csa_attn``'s
``pack_main``. ``G``, ``topk`` and ``compress_ratio`` are compile-time constants (jit key).
``G == 0`` or ``topk == 0`` drops the gather entirely, giving ``swa_attn``'s kernel.

Everything else is ``csa_attn`` verbatim: K == V shared latent, one smem tile serving both
GEMMs, one running ``(m, l, acc)`` over window + gathered tiles (no LSE merge), the sink as a
denominator-only column, fp32 ``lse``, the ``-1e30`` rowmax floor, ``causal=False`` rejected.

Backward is ticket 0005 (``attn.py`` raises).
"""

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

CONFIGS = {  # head dim -> (block_N = KV tokens / gathered entries per tile, num_stages, threads)
    64: dict(block_N=64, num_stages=3, threads=128),
    96: dict(block_N=64, num_stages=3, threads=128),
    128: dict(block_N=32, num_stages=3, threads=128),
    256: dict(block_N=32, num_stages=2, threads=128),
}
MIN_ROWS = 64  # smallest tile M that compiles here (tickets/0001 layout inference); drives T_q


def tokens_per_block(heads: int) -> int:
    """``T_q``: the fewest whole tokens whose ``T_q * heads`` rows reach ``MIN_ROWS``."""
    return max(1, -(-MIN_ROWS // heads))


@tilelang.jit(out_idx=[4, 5], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn(batch, heads, seq_len, dim, groups, topk, window, ratio, is_causal, has_sink=False,
              block_N=64, num_stages=2, threads=128):
    assert is_causal, "csa2_attn is causal-only (see module docstring)"
    assert window >= 1, f"window must be >= 1, got {window}"
    assert ratio >= 1, f"compress_ratio must be >= 1, got {ratio}"
    assert topk % block_N == 0, f"topk={topk} must be padded to a multiple of block_N={block_N}"
    tq = tokens_per_block(heads)
    block_M = tq * heads  # rows are (token, head) in BSHD order, exactly csa_attn's packing
    has_main = groups > 0 and topk > 0
    gather_tiles = topk // block_N
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e), folded into the exp2 softmax
    log2e = 1.44269504  # the sink is an unscaled natural-log logit, so it needs its own conversion
    ln2 = 0.6931471805599453  # exp2-domain lse -> natural log
    neg_floor = -1e30  # running-rowmax floor; see swa_attn/README.md -> Decisions
    q_shape = [batch, seq_len * heads, dim]  # packed rows: q.view(B, S*H, D)
    kv_shape = [batch, seq_len, dim]  # the single shared latent head
    main_shape = [batch, max(groups, 1), dim]  # a 1-row dummy when G == 0; no tile padding needed
    idx_shape = [batch, seq_len, max(topk, 1)]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.macro
    def body(Q, KV, MainKV, Indices, Output, LSE, Sinks):
        with T.Kernel(T.ceildiv(seq_len, tq), batch, threads=threads) as (bx, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)  # serves both S = Q K^T and O += P K
            M_shared = T.alloc_shared([block_N, dim], dtype)  # the gathered tile (liveness-aliased)
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

            # --- source 2: the top-k main entries, GATHERED, same running (m, l, acc)
            if has_main:
                for ti in T.serial(0, tq):  # one gathered pass per token of the block (tq == 1 usually)
                    tok = bx * tq + ti
                    # T.Pipelined is safe here (unlike csa_attn's main loop): the trip count is
                    # the compile-time constant ceil(topk/block_N), so the miscompile trigger --
                    # a division nested in the bound -- is absent. Verified against golden at
                    # every tested shape; the fallback is T.serial(0, gather_tiles). See README.
                    for kt in T.Pipelined(0, gather_tiles, num_stages=2):
                        for i, j in T.Parallel(block_N, dim):
                            g = Indices[bz, tok, kt * block_N + i]
                            # the row loaded is clamped in range; validity lands on the logit below
                            M_shared[i, j] = MainKV[bz, T.if_then_else(T.And(g >= 0, g < groups), g, 0), j]
                        for i, j in T.Parallel(block_M, block_N):
                            g = Indices[bz, tok, kt * block_N + j]
                            # index out of [0, G) -> -inf; and only THIS token's rows attend its
                            # gathered entries (a no-op when tq == 1, i.e. i // heads == ti == 0)
                            acc_s[i, j] = T.if_then_else(
                                T.And(T.And(g >= 0, g < groups), i // heads == ti), 0,
                                -T.infinity(acc_s.dtype))
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
                        T.gemm(acc_s_cast, M_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)  # V == K

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
                Indices: T.Tensor(idx_shape, T.int32),
                Output: T.Tensor(q_shape, dtype),
                LSE: T.Tensor([batch, seq_len * heads], accum_dtype),
                Sinks: T.Tensor([heads], accum_dtype),
        ):
            body(Q, KV, MainKV, Indices, Output, LSE, Sinks)
    else:

        @T.prim_func
        def main(
                Q: T.Tensor(q_shape, dtype),
                KV: T.Tensor(kv_shape, dtype),
                MainKV: T.Tensor(main_shape, dtype),
                Indices: T.Tensor(idx_shape, T.int32),
                Output: T.Tensor(q_shape, dtype),
                LSE: T.Tensor([batch, seq_len * heads], accum_dtype),
        ):
            body(Q, KV, MainKV, Indices, Output, LSE, None)

    return main


def check_shapes(q: torch.Tensor, kv: torch.Tensor, main_kv: torch.Tensor,
                 indices: torch.Tensor) -> tuple:
    """Validate the MQA + indices call shapes and return ``(B, S, H, D, G, topk)``."""
    assert q.dtype == torch.bfloat16, f"expected bf16, got {q.dtype}"
    batch, seq_len, heads, dim = q.shape
    assert kv.shape == (batch, seq_len, 1, dim), f"kv must be [B, S, 1, D]={(batch, seq_len, 1, dim)}, got {tuple(kv.shape)}"
    groups = main_kv.shape[1]
    assert main_kv.shape == (batch, groups, 1, dim), f"main_kv must be [B, G, 1, D], got {tuple(main_kv.shape)}"
    assert indices.dtype in (torch.int32, torch.int64), f"indices must be int32/int64, got {indices.dtype}"
    assert indices.dim() == 3 and indices.shape[:2] == (batch, seq_len), (
        f"indices must be [B, S, topk]=[{batch}, {seq_len}, *], got {tuple(indices.shape)}")
    tq = tokens_per_block(heads)
    assert seq_len % tq == 0, f"csa2_attn needs seq_len % {tq} == 0 ({tq} tokens per block), got {seq_len}"
    return batch, seq_len, heads, dim, groups, indices.shape[2]


def fwd(q: torch.Tensor, kv: torch.Tensor, main_kv: torch.Tensor, indices: torch.Tensor, *,
        window: int, compress_ratio: int, causal: bool = True, sinks: torch.Tensor | None = None):
    """Sliding-window + **top-k gathered** compressed shared-latent MQA FA2 forward.

    ``q`` bf16 [B, S, H, D], ``kv`` bf16 [B, S, 1, D] (raw latent, K == V),
    ``main_kv`` bf16 [B, G, 1, D] (compressed latents, K == V),
    ``indices`` int32/int64 [B, S, topk] (``-1`` = empty slot, as produced by
    ``golden_ref.topk_indices`` / ``indexer.attn``)
    -> ``(o bf16 [B, S, H, D], lse fp32 [B, H, S])``.

    ``window``: visible raw KV tokens per query, self included (``t - window + 1 <= k <= t``).
    ``compress_ratio``: **documentation only** here -- group-causal visibility is baked into
    ``indices`` by the indexer and the kernel does not re-apply it (same contract as
    ``golden(level="sparse")``). Kept in the signature for parity with ``csa_attn``.
    An index outside ``[0, G)`` contributes nothing. ``G == 0`` / ``topk == 0`` reduce to
    ``swa_attn``. ``window``, ``G``, ``topk`` and ``compress_ratio`` are jit constants.

    ``causal=False`` is rejected: the model only ever has the causal band.

    ``sinks``: optional per-head learnable sink ``[H]``; one extra softmax column over the
    *combined* window+gathered logits that only enlarges the denominator. ``lse`` includes it.
    """
    assert causal, "csa2_attn is causal-only: a non-causal sliding window is not a model shape"
    assert isinstance(window, int) and window >= 1, f"window must be a positive int, got {window!r}"
    assert isinstance(compress_ratio, int) and compress_ratio >= 1, (
        f"compress_ratio must be a positive int, got {compress_ratio!r}")
    dim = q.shape[-1]
    assert dim in CONFIGS, f"unsupported head dim {dim}"
    cfg = CONFIGS[dim]
    batch, seq_len, heads, dim, groups, topk = check_shapes(q, kv, main_kv, indices)

    idx = indices.to(torch.int32)
    pad = -topk % cfg["block_N"] if topk else 1  # whole tiles; the tail is -1 (= empty slot)
    if pad:
        idx = F.pad(idx, (0, pad), value=-1)
    main = main_kv.contiguous().view(batch, groups, dim) if groups else \
        torch.zeros(batch, 1, dim, device=q.device, dtype=q.dtype)

    kernel = flashattn(batch, heads, seq_len, dim, groups, topk + pad if topk else 0,
                       min(window, seq_len),
                       compress_ratio, causal, sinks is not None, **cfg)
    args = [q.contiguous().view(batch, seq_len * heads, dim), kv.contiguous().view(batch, seq_len, dim),
            main, idx.contiguous()]
    if sinks is not None:
        assert sinks.shape == (heads,), f"sinks must be [H]={heads}, got {tuple(sinks.shape)}"
        args.append(sinks.detach().float().contiguous())
    o, lse = kernel(*args)
    # lse comes out packed [B, S*H] (token-major); the public layout is [B, H, S]
    return o.view(batch, seq_len, heads, dim), lse.view(batch, seq_len, heads).permute(0, 2, 1).contiguous()
