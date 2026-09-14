# 0003 csa_attn: dense compressed KV as a second source (fwd + bwd)

- status: todo
- depends on: 0002 (`swa_attn`)
- package: `src/mzoo/layers/attn/csa_attn/`

## Goal

Copy `swa_attn/` to `csa_attn/` and add the compressed (main) KV as a second source in the same
online-softmax loop: after the window tiles, walk every *visible* main entry, no sparsity yet. One
running `(m, l, acc)` across both sources, so no LSE merge. Fixes the main-cache layout that every
later package reuses.

## Semantics

Read `DeepseekV41Attention.forward` (compressed branch) and `DeepseekV41Compressor.forward` in
`archs/dsv4/modeling_deepseek_v41.py`.

- KV axis is the concatenation `[window_kv (T entries) ; compress_kv (G entries)]` (`kv =
  torch.cat([kv, compressed_kv], dim=2)`), one shared K==V head.
- Compressed entries are pre-RoPE latents pooled by the compressor (`ratio > 1`: fp32 gated softmax
  pool; `ratio == 1`: plain per-token projection), rotated at the *latent* positions
  `first_group_position + ratio*k`, then `_fake_quant_fp4_block(x, block_size=16, e4m3_scales=True)`
  (e2m1 grid, one e4m3 scale per 16 channels, no per-tensor scale, amax clamped to `6*2**-9`).
- Group-causal visibility, separate from the window band mask: entry `k` is visible to token `t` iff
  `k < (t+1) // ratio` (`compress_lens` in `DeepseekV41Indexer.forward` step 3; the eager path gets
  it via the additive mask). Window rows keep `t-127 <= j <= t`.
- Sink and softmax exactly as 0002: one extra column feeding the denominator only.
- Cache layout fixed here: main KV `[T, D]` fp4-packed + `[T, D/16]` e4m3 scales; v1 dequantizes in
  torch, so the kernel still consumes bf16 (0010 does in-kernel).
- Backward: same loop structure; `dKV = dK + dV` accumulated separately for the window slice and the
  main slice, group-causal mask applied to the main part.
- `ratio == 1` with an empty window is the uncompressed full-attention special case and doubles as a
  correctness test against causal `latent_attn`.

## Deliverables

`fwd.py`, `bwd.py`, `attn.py`, `ref.py`, `bench.py`, `*_test.py`, `README.md` (what/how, configs,
bench, accuracy, decisions, **Known issues**, next). Tests via
`dense_attn/ref.py::assert_within_2x_torch` + `torch_bf16_ref`; extend `just smoke`.
Compressor/rope/quant stay in torch, in `ref.py`.

## Acceptance

- Reference: `attn/golden_ref.py::golden` at the `compressed` level, a `compress_ratio=4` layer of
  the smoke config, and the `ratio=1` equality test vs causal `latent_attn`.
- `D in {64, 96, 128, 256}`, 64/128 tuned, 96/256 pass-only; H=64/D=256 shape check.
- `o/dq/dkv/dsinks` within 2x torch bf16; `lse` within ~1e-6 fp32.
- Bench vs `swa_attn` (window only) and vs `dense_attn` causal at B1 S4096, one row per `ratio in
  {1, 2, 4}`; report the cost of the second source.

## References

- tilelang `examples/deepseek_v4/sparse_attn_fwd_sm90.py` (two-source loop shape),
  `examples/flash_attention/example_mha_bwd_bshd.py`.
- `csa2_attn_design.md` step 4; `fp_attn_survey.md` §4 (FlashMLA's `kv_cache_format.h` is the byte
  layout to copy), §3 (PV stays bf16).

## Risks / open questions

- smem budget: window tile + main tile + Q resident is 3 x 32 KB at D=256 against 99 KB; may force
  `num_stages=1` or a smaller `block_N` for the main source.
- Two masks in one loop (band + group-causal) can cost more than the GEMM at small `ratio`; consider
  hoisting the fully-visible main tiles out of the mask.
- fp4 dequant is torch-side fake-quant here; bit-faithfulness is `ref.py`'s job, the kernel only has
  to agree numerically.
