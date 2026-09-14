# 0008 indexer_fp4: MXFP4 x MXFP4 score matmul

- status: todo
- depends on: 0006, 0007 (`indexer` fwd + bwd in bf16)
- package: `src/mzoo/layers/attn/indexer_fp4/`

## Goal

Copy `indexer/` to `indexer_fp4/` and change exactly one thing: the score GEMM runs as a real
block-scaled fp4 x fp4 MMA instead of bf16 over dequantized values. This is the *only* place in CSA2
where fp4 math (not just fp4 storage) is part of the trained model, so faithfulness to
`_fake_quant_fp4_block(..., 32)` is the whole point.

## Semantics

Read `DeepseekV41Indexer.forward` step 2 (`_fake_quant_fp4_block` calls) and `_fake_quant_fp4_block`
/ `_pow2_ceil_scale` in `archs/dsv4/modeling_deepseek_v41.py`.

- Both `q [B, S, 32, 128]` and the shared `k [B, T, 128]` are quantized to e2m1 with one **ue8m0**
  scale per **32** channels (`_pow2_ceil_scale(amax/6)`, the ceil rule, amax clamped to
  `6*2**-126`). That is OCP MXFP4, not NVFP4, and not the OCP §6.3 floor rule -- a stock encoder
  will not bit-match.
- Everything after the GEMM is unchanged from 0006: `relu`, `* head_dim**-0.5`, fp32 weighted head
  reduce, visibility mask, top-k, `-1` padding.
- TileLang's `T.mma_gemm_blockscaled` currently asserts the NVFP4 flavour only (ue4m3 scales, block
  16). Two routes, pick one and record why: (a) widen
  `src/tl_templates/cuda/instruction/mma_block_scale.h` upstream to the `kind::mxf4 / scale_vec::2X
  / ue8m0` atom; (b) re-encode each ue8m0/32 scale as two identical ue4m3/16 scales -- exact while
  the power of two is in `2**-9 .. 2**8`, otherwise pull a per-tensor power-of-two factor into
  `softmax_scale`.
- Backward stays bf16 straight-through over the dequantized values (0007); the fp4 path is
  forward-only.

## Deliverables

`fwd.py` (block-scaled MMA + the scale re-encode or the widened atom), `quant.py` (host-side packer
producing the e2m1 codes + scale words), `ref.py`, `bench.py`, `*_test.py`, `README.md` with a
**Known issues** section covering the `SM120A_ENABLED` hack and the chosen route. Reuse the flag
pattern from `src/mzoo/kernels/sm120_nvfp4_blockscaled_gemm.py`; extend `just smoke`.

## Acceptance

- Bit-exactness test: the fp4 score matrix must equal the bf16 kernel run on
  `_fake_quant_fp4_block`-round-tripped inputs, to within fp32 accumulation order (state the
  tolerance; the codes themselves must match exactly).
- Range test for route (b): assert every scale lands in e4m3's range on the smoke config, and fail
  loudly (not silently) when it does not -- `fp_attn_survey.md` §7 Q1 is exactly this histogram.
- Top-k index-set equality vs 0006 on the smoke config.
- `index_head_dim=128` tuned; 64 also tuned, 96/256 pass-only (96 is not a multiple of 64 -- ffpa
  pads fp4 tiles with zero data *and* zero scale words).
- Bench vs `indexer/` bf16 at `S=4096, T=4096/16384`; expect a bandwidth win, not a math win
  (measured fp4 mma throughput ~= fp8 on this class of part).

## References

- `src/mzoo/kernels/sm120_nvfp4_blockscaled_gemm.py`, `tickets/0001-tilelang-issues.md` (the
  `CUTLASS_ARCH_MMA_SM120A_ENABLED` workaround; sm121 is blocked upstream).
- `tilelang/language/gemm_op.py::mma_gemm_blockscaled`, `mma_block_scale.h`.
- `fp_attn_survey.md` §5 and §7 Q1/Q2/Q5; `csa2_attn_design.md` step 6a.

## Risks / open questions

- Arch flag is unsettled (`sm_121a` vs `sm_121f`); settle before starting.
- No TileLang example combines a block-scaled MMA with attention at all -- the GEMM example is the
  only in-repo call site, so expect layout surprises.
- Its own example documents two bugs: `T.copy` of an SFA slice mis-lowers on the bulk-TMA path, and
  simultaneous M *and* N tail tiles are unsupported.
