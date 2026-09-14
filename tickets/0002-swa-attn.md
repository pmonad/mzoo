# 0002 swa_attn: sliding-window branch (fwd + bwd)

- status: done (2026-09-14): 39 swa tests green (152 in `attn/`), fwd+bwd at D128 H64 S4096 W128 is 5.10x `latent_attn` full causal and 34.4x SDPA with a band mask; model layer 0 replay passes the 2x criterion.
- depends on: `latent_attn` (step 2, done and verified)
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

## Result

- package `src/mzoo/layers/attn/swa_attn/`: `fwd.py`, `bwd.py` + `bwd_kernels.py` + `bwd_dq.py`,
  `attn.py`, `bench.py`, `fwd_test.py`, `attn_test.py`, `model_test.py`, `README.md`.
- final `CONFIGS` -- fwd (block_M/block_N/stages/threads): 64 -> 256/64/3/256, 96 -> 256/64/3/256,
  128 -> 256/32/3/256, 256 -> 128/32/2/256. bwd `kv`: 64 -> 64/128/2/128, 96 -> 64/64/3/128,
  128 -> 64/64/2/128, 256 -> 64/64/1/128; bwd `dq`: 64/96/128 -> 128/64/{3,3,2}/256, 256 -> 64/32/2/128.
  `splits` from `pick_splits` (`TARGET_CTAS` 768, `MAX_SPLITS` 12; 12 at the headline shape).
- bench B1 S4096 W128 D128 H64: fwd 0.624 ms vs `latent_attn` 2.903 (4.66x) vs SDPA+band-mask 21.27
  (34.1x); fwd+bwd 3.073 vs 15.673 (5.10x) vs 105.6 (34.4x). Range over D{64,96,128,256} x H{16,64}:
  3.5-5.1x `latent_attn`, 17-42x SDPA. Ideal band ratio is 16.3x; the gap is that the banded kernel
  is launch/memory-bound (27 vs ~95 effective TFLOPS).
- accuracy (B2 S512, per-head sink, vs fp32 `golden(level="window")`): `o` <= 0.60x, `dq` <= 0.93x,
  `dkv` <= 0.77x, `lse` <= 1.4e-6 abs. `dsinks` reaches 2.07x at D=256 H=4 -- a pre-existing property
  of that reduce (`latent_attn` hits 2.24x on the same shape across seeds), see README Known issues.
- deviations from the ticket: `causal=False` asserts instead of being supported (documented, tested);
  the "D 96/256 pass-only" line is superseded -- 64/96/128 are tuned, 256 is shape support;
  no `ref.py` (the series-wide `golden_ref.py` replaced it, as in `latent_attn`); the model check is
  its own `model_test.py` reusing `golden_ref_test.py`'s capture helpers.
- one real problem hit, parked in `docs/evolution/attn/attention-kernels-impl.md`: a banded row can
  meet a KV tile it sees nothing in, so FA2's `-inf - (-inf)` rescale produced NaN; fixed by flooring
  the running rowmax at -1e30 instead of -inf.
