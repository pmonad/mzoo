# 0002 swa_attn: sliding-window branch (fwd + bwd)

- status: todo
- depends on: `latent_attn` (step 2, next session)
- package: `src/mzoo/layers/attn/swa_attn/`

## Goal

Copy `latent_attn/` to `swa_attn/` and restrict the KV loop to the 128-token sliding window every
DSV4.1 layer has, keeping the shared-latent layout (K==V, 1 KV head, heads packed into tile rows).
Forward, backward and autograd wrapper land together; this is the `compress_ratio == 0` layer type,
so it is the first package testable against the real model.

## Semantics

Read `DeepseekV41Attention.forward` + `eager_attention_forward` in
`archs/dsv4/modeling_deepseek_v41.py`.

- `kv = kv_norm(wkv(x))`, RoPE'd, then `_fake_quant_fp8_block(kv, 32)` (ue8m0 per 32 channels,
  `_pow2_ceil_scale` ceil rule, amax clamped 1e-4). Same tensor is K and V, `[B, 1, T, D]`,
  broadcast over `H=64` query heads.
- Mask: HF `create_sliding_window_causal_mask`, `sliding_window=128`, i.e. `t-127 <= k <= t` (self
  included). Band mask per *row token*, not per block.
- `logits = q k^T * D**-0.5` + mask; the per-head `attn_sink[H]` joins as one extra column, rowmax
  subtracted, softmax, sink column dropped from the numerator -- it only enlarges the denominator
  and the LSE.
- v1 dequantizes the fp8 window cache in torch (caller-side fake-quant round trip); the kernel sees
  bf16. In-kernel dequant is ticket 0010.
- Backward: dQ as in `dense_attn`; dK/dV reduce over the 64 heads sharing the latent and `dKV = dK +
  dV`, fp32 split buffers not bf16 atomics; `dsinks` stays the torch fp32 reduction. Band mask
  applies in both passes.

## Deliverables

`fwd.py`, `bwd.py`, `attn.py` (autograd, saves `q, k, v, o, lse[, sinks]`, recomputes S/P),
`ref.py`, `bench.py`, `*_test.py`, `README.md` in the standard section order (what/how, configs,
bench, accuracy, decisions, **Known issues**, next). Tests reuse
`dense_attn/ref.py::assert_within_2x_torch` + `torch_bf16_ref`, no fixed atols; extend `just smoke`.

## Acceptance

- Reference: `attn/golden_ref.py::golden` at the `window` level, plus a
  layer-by-layer check against the 6-layer smoke config (`archs/dsv4/model.py`, H=4, D=64).
- `D in {64, 96, 128, 256}`: 64/128 tuned, 96/256 pass-only; H=64 shape check at D=256; `S` a
  multiple of `block_M`.
- `o/dq/dkv/dsinks` within 2x torch bf16; `lse` within ~1e-6 fp32.
- Bench vs `dense_attn` causal at B1 S4096: work is O(S*128), so expect a large win over dense
  causal; report both fwd and fwd+bwd.

## References

- tilelang `examples/flash_attention/example_mha_{fwd,bwd}_bshd.py`,
  `examples/attention_sink/example_mha_sink_bwd_bhsd.py` (dsink kernel).
- `dense_attn/README.md`, `csa2_attn_design.md` step 3, `fp_attn_survey.md` §4.

## Risks / open questions

- With 1 token x 64 heads per block the window is exactly 2 tiles of 64; check whether a `T_q x H`
  multi-token block still pays once the band mask is applied.
- Re-tune tiles from scratch: `block_M=32` and `block_M=64`+`threads=256` fail layout inference here
  (`tickets/0001-tilelang-issues.md`), and `dense_attn`'s table does not carry over to MQA row
  packing.
- Do not stash a bf16 copy of the dequantized cache for the backward; recompute.
