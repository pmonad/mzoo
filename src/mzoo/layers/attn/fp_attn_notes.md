# FP8/FP4 attention — raw research notes (2026-09-13)

Companion dump for `fp_attn_survey.md` (the ~150-line decision doc). Everything below is
what was actually read, including numbers, file paths, dead ends and non-findings, so a
later session does not repeat the search. Bias: **training (fwd+bwd) on GB10 sm121**.

---

## Part 1 — Hardware facts confirmed

### SM120 / SM121 (GB10, RTX 5090, RTX PRO 5000/6000)

- Tensor cores are **warp-level `mma.sync`**. No `tcgen05`/UMMA, no tensor memory, no TMA
  multicast. Consequence: the FA2 template is the right shape; `flash_attention_sm100`
  examples are not portable.
- **99 KB smem per CTA** (vs 228 KB on SM100, 163 KB on SM80). 64K regs/SM, 255 regs/thread.
- `sm_121` is a distinct cc (12.1) from `sm_120` (12.0). SageAttention3's `setup.py`
  dispatches `(12,0) → sm_120a`, `(12,1) → sm_121a`, `(10,0) → sm_100a` (vestigial), and
  `api.cu:220` runtime-checks `major==12 && (minor==0||minor==1)`.
- **Arch-flag confusion, unresolved.** Three conflicting reports:
  - ffpa-attn: must build `sm_120f` (family), **not** `sm_120a`, or `setmaxnreg` is silently
    ignored by ptxas (warning C7506).
  - DGX Spark community reports: `sm_121a` or `sm_121f` needed for FP4; plain `sm_121`
    gives "Feature not supported".
  - CUTLASS gates on `CUTLASS_ARCH_MMA_SM120A_ENABLED`; our vendored
    `kernels/sm120_nvfp4_blockscaled_gemm.py` force-defines it to 1 to let sm121 emit the MMA
    (see `tickets/0001-tilelang-issues.md`).

### FP8 / FP4 MMA instructions actually issued on sm120 (from ffpa-attn + SageAttention3 source)

| Purpose | PTX / CuTe atom | Operands | Scale | Block |
|---|---|---|---|---|
| fp8 dense | `SM89_16x8x32_F32E4M3E4M3F32_TN` | e4m3 × e4m3 → f32 | software (no `.block_scale`) | — |
| fp8 → fp16 acc | `SM120_16x8x32_TN<e4m3,e4m3,half_t>` | e4m3 | software | — |
| int8 | `SM80_16x8x32_S32S8S8S32_TN` | s8 | software | — |
| **NVFP4** | `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3` | e2m1 | **ue4m3** (e4m3 used as SF) | **16** |
| **MXFP8** | `mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0` | e4m3 | **ue8m0** | **32** |
| MXFP4 (ue8m0, block 32) | `…kind::mxf4…scale_vec::2X…ue8m0` — exists per PTX ISA, **but see TileLang gap below** | e2m1 | ue8m0 | 32 |

Conversions: `cvt.rn.satfinite.e2m1x2.f32` and `cvt.rn.satfinite.e4m3x2.f32` (2 floats per
instruction), both require `__CUDA_ARCH__ >= 1200`.

SageAttention3 fuses **four** `m16n8k64` issues into one atom
`SM120_16x32x64_TN_VS_NVFP4` (`ValTypeA/B = uint4_t`, `ValTypeSF = float_ue4m3_t`,
`SFVecSize = 16`, `Shape_MNK = <16,32,64>`), sharing A regs and stepping B regs `b0..b7`
with `tidB ∈ {0,1,2,3}` selecting the SFB thread-id field. **QK and PV use the same atom.**

### Block-scale factor layout

SageAttention3 `blackwell/blockscaled_layout.h` (copied from CUTLASS SM100, reused on SM120):

```cpp
using Blk_MN = _64;  using Blk_SF = _4;
using SfAtom = Layout< Shape< Shape<_16,_4>, Shape<Int<SFVecSize>,_4>>,
                      Stride<Stride<_16,_4>, Stride<           _0,_1>>>;
```
i.e. one indivisible block holds 4 SFs for 64 rows, "chosen to make consecutive 32 bits of
data have scale factors for only a single row" (verbatim comment). Not the `128x4` layout.

TileLang's equivalent is `sf_layout="blockscaled_chunk_kmajor"`; `"cutlass_128x4"` is
explicitly **rejected** (unit test
`test_nvf4_mma_block_scale_rejects_legacy_cutlass_128x4_layout_alias`). Host-side packing
reference is `swizzle_blockscaled_chunk_kmajor_scale_words()` in the vendored example:
`block_rows=128, block_words=4`, reshape `(row_blocks,4,32,cols)` → permute → `(row_blocks,cols,32,4)`.
`sf_tile_bytes = 128 * (K//64) * 4`; K ∈ {64,128,256}.

### TileLang gap (important, blocks the indexer plan)

`tilelang/language/gemm_op.py::mma_gemm_blockscaled` docstring:
> "Explicit SM120 warp-level block-scaled MMA GEMM. … synchronous warp-level `mma.sync`,
> does not use tensor memory or mbarriers. **The current supported instruction is SM120 NVF4:
> `m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X` with E2M1 operands, FP32 accumulation,
> and UE4M3 scale factors.**"

`src/tl_templates/cuda/instruction/mma_block_scale.h` `static_assert`s exactly that
(`sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3`). So **block-32 / ue8m0 MXFP4 has no TileLang path
today** — our indexer's OCP-MXFP4 grid must be re-encoded into ue4m3/16 (exact when the
power-of-two scale is in e4m3's range 2⁻⁹…2⁸) or dequantized.

Signature: `(A, B, C, SFA, SFB, transpose_A, transpose_B, policy, clear_accum, *, k_start,
sf_a_granularity_k, sf_b_granularity_k, sf_layout)`. `k_start` + `granularity_k` is how you
step SFs across a KV loop — the API surface to understand for FA.

Supporting files: `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py`
(`BlockScaleMmaConfig`, `SM120BlockScaleTile`), `tilelang/cuda/op/gemm/gemm_mma_sm120.py`.
Only two call sites in the whole repo: `examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py`
and `testing/python/language/test_tilelang_language_nvf4_mma_block_scale.py`.
Two live upstream bugs documented in the example's own comments: (a) simultaneous M *and* N
tail tiles unsupported; (b) `T.copy` of an SFA slice mis-lowers on the bulk-TMA path, so
scales must be staged with an explicit `T.Parallel` loop.
SM100 sibling (not usable here): `T.tcgen05_gemm_blockscaled`.

### GB10 throughput reality — FP4 is compression, not compute

- GB10: 48 SMs, 6144 CUDA cores, 128 GB LPDDR5x, **273 GB/s**, "1 PFLOP sparse FP4".
- Measured numbers **disagree between sources** — report the ratio, not the absolute:
  - NVIDIA dev forum thread 360142: FP8 188 TFLOPS, NVFP4 356 TFLOPS measured; but end-to-end
    only 1.2–1.4× because (i) no tensor memory, fp4 goes smem→regs→unpack; (ii) 99 KB smem
    forces small tiles and more reloads; (iii) 273 GB/s is the real wall.
  - `VincentKaufmann/fp4-cuda-kernel`: FP4 85–129 TFLOPS, 1.4–2.4× BF16; "Raw FP8 GEMM 85.9 TFLOPS".
  - ffpa-attn measured on RTX 5090: **the fp4 attention kernel is only ~23 % faster than its
    fp8 kernel; `mxf4nvf4` block-scaled MMA throughput ≈ fp8 dense. "the gain comes mainly
    from bandwidth."** (fp4/fp8 at D=192: self 1.47×, causal 1.29×, gqa 1.50×, cross 1.57×.)
  - hao-ai-lab `flash-attention-fp4` README: FP4 PV is **0.84–0.95× BF16 on B200** (softmax
    MUFU warp becomes the bottleneck).
- Conclusion carried into the survey: on GB10 chase **bandwidth** (fp4/fp8 *storage*), not
  fp4 *math*.

### Format definitions — short version (authoritative long version in **Part 6**)

- **e2m1 (FP4)**: max 6.0. Grid `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`. RTNE boundaries
  `0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0` with ties-to-even at `0.25/1.25/2.5/5.0`
  (see `_e2m1_codes` in `archs/dsv4/modeling_deepseek_v41.py`). Sign in bit 3.
- **e4m3 (FP8)**: max 448, min normal 2⁻⁶, min subnormal 2⁻⁹.
- **e5m2**: max 57344. Used for `dO` (FA4-FP4, TE) because e4m3 underflows gradients.
- **ue8m0**: unsigned 8-bit exponent-only, value `2^(byte-127)`, range 2⁻¹²⁷…2¹²⁷.
- **MXFP4 (OCP)** = e2m1 elements, **block 32**, shared scale **ue8m0**. OCP reference scale
  rule: `X = 2^(floor(log2 amax) − emax_elem)`, `emax_elem = 2` for e2m1 → **this clips**
  (amax=7 → X=1 → 7 clamps to 6).
- **NVFP4** = e2m1 elements, **block 16**, per-block scale **e4m3**, *plus* a per-tensor FP32
  global scale. Max representable = 448 × 6 = 2688.
- **DSV4.1 deviates from both** (see Part 2 → DeepSeek).

---

## Part 2 — Per-source notes

### DeepSeek (the target model)

**DeepSeek-V3 tech report, arXiv 2412.19437** — the FP8 training recipe, and the reason
attention is BF16:
- > "we maintain the original precision (e.g., BF16 or FP32) for the following components:
  the embedding module, the output head, MoE gating modules, normalization operators, and
  **attention operators**." → QK^T, softmax and PV are BF16 in V3 training, fwd and bwd.
- > "(1) for activations, we group and scale elements on a **1×128 tile** basis (i.e., per
  token per 128 channels); and (2) for weights, we group and scale elements on a **128×128
  block** basis."
- > "Once an interval of N_C is reached, these partial results will be copied to FP32
  registers on CUDA Cores, where full-precision FP32 accumulation is performed … setting
  **N_C = 128** elements, equivalent to 4 WGMMAs, represents the minimal accumulation interval."
- > "we adopt the E4M3 format on all tensors" (no E4M3/E5M2 hybrid).
- The only precision concession *because of* attention: > "(1) Inputs of the Linear after the
  attention operator. These activations are also used in the backward pass of the attention
  operator, which makes it sensitive to precision. We adopt a customized **E5M6** data format
  exclusively for these activations… converted from a 1×128 quantization tile to a 128×1 tile
  in the backward pass… all the scaling factors are **round scaled, i.e. integral power of 2**."
  Verified in DeepSeek's own training kernels: `deepseek-ai/TileKernels`
  `tile_kernels/quant/per_token_cast_to_e5m6_kernel.py` (max_value 65024, per-token scale,
  optional ue8m0 `round_sf`) + `per_channel_cast_and_transpose_kernel.py` for the 1×128→128×1
  flip. **TileKernels has no attention kernels at all** (only MoE/gating/quant/transpose/
  engram/mHC).
- The 1×128 + 128×128 + "2xAcc" scheme is reproduced in TileLang in
  `DeepSeek-V3.2-Exp/inference/kernel.py::fp8_gemm_kernel` (`scales_a [M,K/128]`,
  `scales_b [N/128,K/128]`, `# Promote to enable 2xAcc` every `block_K=128`).

**DeepSeek-V3.2-Exp** (github.com/deepseek-ai/DeepSeek-V3.2-Exp):
- FP8 appears in the paper **exactly once**: "Given that the lightning indexer has a small
  number of heads and can be implemented in FP8, its computational efficiency is remarkable."
  Nothing about FP8 training, fake-quant or STE. The FP8 indexer is an **inference** kernel.
- `inference/model.py`: indexer q/k use `act_quant(..., block_size=128, scale_fmt="ue8m0")`;
  `k_cache` is `float8_e4m3fn [B,S,128]` with `k_scale_cache [B,S,1]` fp32 → **one ue8m0 scale
  per token** (head_dim == block_size == 128). Q gets one scale per (token, head), folded into
  the per-head gate: `weights = weights_proj(x.float()) * n_heads**-0.5;
  weights = weights.unsqueeze(-1) * q_scale * softmax_scale`.
  **`rotate_activation()` = fast Hadamard transform applied before quantization.**
- Main MLA KV in that file is *fake*-quantized: `# we use fp8 kv cache in actual deployment,
  so here we simulate the precision by casting kv to fp8 and then back to bf16.`
- Indexer training: > "we detach the indexer input from the computational graph for separate
  optimization. The training signal of the indexer is from only L_I" — KL to the head-summed,
  L1-normalized main-attention distribution. Dense warm-up 1000 steps / 2.1B tokens, then
  sparse 15000 steps / 943.7B tokens, top-k 2048.

**DeepSeek-V4, arXiv 2606.19348**:
- FP4 QAT applied to "MoE expert weights and the **indexer QK path**". Indexer QK activations
  are "cached, loaded and multiplied **entirely in FP4**"; index scores further quantized
  FP32→BF16; **2× speedup for the top-k selector at 99.7 % KV-entry recall**.
- STE description: FP32 master weights → FP4 → dequantized **back to FP8** for compute
  (FP4→FP8 dequant is lossless); backward computes gradients w.r.t. the same FP8 weights and
  propagates straight to the FP32 master weights — "equivalent to applying the Straight-Through
  Estimator (STE) through the quantization operation".
- V4 KV cache: "BF16 precision for the RoPE dimensions, FP8 for the remaining dimensions."
  (V4.1 changes this — see below.)
- Could not extract block sizes/scale dtypes for V4's FP8 KV from the PDF.

**DeepSeek-V4.1-Flash** (our target; local paper copy in `archs/dsv4/paper/`):
- §2.4.4 (local `015-2.4.4-fp4-main-kv-cache.md`), the load-bearing quotes:
  - MXFP4 for the indexer "to support as many hardware platforms as possible, **despite the
    higher accuracy of alternative formats in our experiments**".
  - main KV: "**E2M1 with one E4M3 scale per 16 channels, following NVFP4 but omitting its
    second-level global scale**". Justification: format reaches 448 × 6 = 2688; RMSNorm weight
    max ≈ 1, so ‖512-ch latent‖₂ ≲ √512 ≈ 22.6 after RMSNorm, RoPE preserves the norm, and the
    max observed in training ≈ 10.
  - "**Dequantizing cached values before attention** allows us to use a more accurate format
    without requiring native matrix-multiplication support for that format."
  - Quantize **after** RoPE ("quantizing before RoPE yields only a marginal accuracy
    improvement … and would introduce additional overhead during decoding"). Non-RoPE and RoPE
    components use the **same** format (differs from V4!).
  - "We retain FP8 for the SWA KV cache **due to its sensitivity to quantization**."
- §1: "At the cache-precision level, we use **FP4 global KV caches during training** with only
  marginal performance degradation." → this is QAT, stated.
- DSV4.1 decode-FLOP figure weights BF16/FP8/FP4 ops by 1 / 0.5 / 0.25.
- HF reference `deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py`:
  `fp4_quant_kernel(..., block_size=32, scale_dtype=FE8M0, inplace=False)` —
  "Block-wise FP4 with power-of-2 or E4M3 scales; optionally dequantize in place";
  `fp4_act_quant` docstring "FP4 with E8M0 scales for the indexer or E4M3 scales for compressed
  KV. `inplace=True` writes the dequantized values back to x." In-source comment:
  `# Training's compressed KV: keep even an all-zero group's scale nonzero.`
  `inference/model.py` uses `inplace=True` (QDQ round-trip = fake quant) at every KV site, with
  `# Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.`

**Our vendored quantizers** (`archs/dsv4/modeling_deepseek_v41.py`) — the grid the kernel must match:
```python
_FP4_MAX = 6.0 ; _FP8_MAX = 448.0
def _pow2_ceil_scale(t):   # 2^ceil(log2 t), via exponent field + (mantissa != 0)
# fp8 window KV (block 32):  scale = _pow2_ceil_scale(amax.clamp_min(1e-4) / 448)
# fp4 indexer  (block 32):   scale = _pow2_ceil_scale(amax.clamp_min(6*2**-126) / 6)
# fp4 main KV  (block 16):   scale = e4m3(amax.clamp_min(6*2**-9) / 6)
# then clamp to ±max, RTNE onto the grid, multiply back by scale
```
Call sites: L713 `_fake_quant_fp8_block(kv, 32)` on the post-RoPE window KV; L742
`_fake_quant_fp4_block(rotated, 16, e4m3_scales=True)` on the post-RoPE compressed latent;
L519 / L542 `_fake_quant_fp4_block(k|q, 32)` on indexer K and Q.
**Note the `ceil` rule never clips, unlike OCP's `floor` rule** — a stock MXFP4/MXFP8 encoder
will not bit-match us. `archs/dsv4/README.md` already records that the indexer never trains
because `_fake_quant_fp4_block` detaches the graph (needs an STE).

### FlashAttention-3 FP8 — arXiv 2407.08608

- Block quantization: **one scalar per `Br×d` / `Bc×d` block** for Q, K, V (not per-tensor);
  "scale each block of S to account for this block quantization at no computation cost".
- Incoherent processing: `Q←QM, K←KM` with `M = diag(±1) · Hadamard / √d`, so `(QM)(KM)ᵀ=QKᵀ`;
  O(d log d), per head, fusible with RoPE.
- RMSE table 3: per-tensor fp8 **2.4e-2** → +block quant **9.3e-3** → +incoherent **9.1e-3**
  (2.6× better than baseline fp8). ⇒ **block quantization does ~all the work; Hadamard adds ~2 %.**
- fp8 WGMMA needs k-major operands ⇒ in-kernel transpose of V tiles in smem via LDSM/STSM.
- P: the paper text does not present P as "quantized", but the second GEMM is fp8, so P̃ is cast
  to e4m3. The code uses a **fixed 2⁸ = 256 offset**:
  `flash::Softmax<..., Max_offset = !Is_FP8 ? 0 : 8>` in `hopper/flash_fwd_kernel_sm90.h:409`.
- **FP8 is forward-only.** README: "FP16 / BF16 forward and backward, **FP8 forward**".
  `mha_bwd` has `TORCH_CHECK(q_type == kFloat16 || q_type == kBFloat16)`.
  FlashAttention-4 (CuTeDSL, `pip install flash-attn-4`) raises
  `NotImplementedError("FA4 CuTe FP8 backward is not supported yet (forward-only)")`.
  No FP4 anywhere in the README.
- Open issue #1848: users see `HMMA.16816.F32.BF16` instead of native fp8 instructions — the
  fp8 path may not always activate.

### SageAttention family (thu-ml/SageAttention)

**SageAttention 1, arXiv 2410.02367** (sm80/89): Q,K INT8 per-tile; `K ← K − mean_tokens(K)`
(exact, softmax is shift-invariant); **P·V stays FP16 with an FP16 accumulator**; adaptive
per-layer selection. 2.1× / 2.7× vs FA2 / xformers.

**SageAttention2, arXiv 2411.10958** (v6 is the one to read):
- Per-thread INT4 Q/K: query tokens grouped `{i, 8+i, 16+i, 24+i}` (8 tokens per scale),
  key tokens `{j, 1+j, 8+j, 9+j, …}` (4 pairs per scale) — aligned to `mma.m16n8k64` C-fragment
  ownership so each thread needs exactly one Q scale and one K scale. "32× and 4× finer
  granularity than per-block" for query / key tokens.
- Smoothing: `γ(Q_i) = Q_i − q̄_i`, `γ(K_j) = K_j − k̄`, `q̄_i = mean(Q_i)` (per block),
  `k̄ = mean(K)`, "the mean is conducted along the token axis".
- **P: static `δ_P = 1/448`** — verbatim: "We quantize P with a static scale: δ_P = 1/448
  since the original P elements are already in [0,1]."
- **V: per-channel** — "We quantize V per-channel to address the channel-wise outliers".
- Two-level accumulation: `mma(f32f8f8f32)` gives only **22 effective bits** (FP22: 1s/8e/13m),
  "sufficient since we only accumulate over a small number of `b_k` tokens (e.g. `b_k=64`).
  Then R_ij is accumulated to O_ij in high FP32 precision."
- Accuracy: SageAttn2-8b matches FP32 metrics; 4b loses 1–3 % on some benchmarks. 481 TOPS on
  RTX4090, 3× FA2, 4.5× xformers.
- **Evaluated on long context** (rare): WikiText ppl, LAMBADA, MMLU, **Longbench** on
  Llama2-7B / Llama3.1-8B / GLM4-9B, plus **Llama-3-262k on InfiniBench and NIAH up to 262k**.
- Inference only; Algorithm 1 is forward only.

**SageAttention2++, arXiv 2505.21136**: same accuracy, swaps in "FP8 Matmul accumulated in
FP16", which is 2× the fp8 matmul of SageAttention2 → 3.9× FA. Granularity unchanged.

**SageAttention3, arXiv 2505.11594** (RTX5090):
- Q,K,P,V **all NVFP4** (1×16, e2m1, e4m3 SF). NVFP4 chosen over MXFP4: **cos-sim 99.52 % vs
  98.37 %, L1 0.077 vs 0.294** (Table 1a).
- Two-level P: `s_P1 = rowmax(P̃)/(448×6)`, `P̃₂ = P̃/s_P1`, `(s_P2, P̂₂) = φ(P̃₂)`,
  `O = FP4MM(P̂₂, s_P2, V̂, s_V) × s_P1`. Direct per-block P gives SFs in [0, 0.167] which
  "vastly underutilizes E4M3's range": **93.32 % → 99.52 % cos-sim**.
- 1038 TOPS on RTX5090, 5× FA2, 8–11× xformers; 3× e2e HunyuanVideo, 2.4× CogVideoX.
- K mean-subtracted, Q per-block mean-subtracted, both **before** quantization; correction
  folded in as a rank-1 `delta_s` added to the S accumulator (host-side `triton_group_mean`).
- Kernel `sageattention3_blackwell/` is **sm_120a / sm_121a only** — builds for GB10 today.
  `softmax_fused.h` has the log-domain fusion:
  ```cpp
  fp8_scalexfp4_scale      = 1.f/(448*6);
  fp8_scalexfp4_scale_log2 = -11.392317422778762f;   // log2(1/2688)
  fp4_scale_log2           = -2.584962500721156f;    // log2(1/6)
  ```
  group max via `__shfl_xor_sync(...,1)` (16 lanes), row max via `__shfl_xor_sync(...,2)`;
  `max_scaled = row_max*softmax_scale_log2 + fp8_scalexfp4_scale_log2`;
  `P = exp2(S*scale_log2 - max_scaled)`;
  `AbsMaxP = exp2(AbsMaxP*scale_log2 - max_scaled + fp4_scale_log2)`.
  Packing in `mainloop_tma_ws.h:750-800` via `packed_float_to_ue4m3` / `packed_float_to_e2m1`
  + `__shfl_xor_sync(-1, local_sfp, 2)` byte-mask fixups for the MMA's `tidA/tidB` fields.
- **No backward anywhere in the repo** — `setup.py` builds only `_qattn_sm80/89/90` + `_fused`.

**SageBwd, arXiv 2603.02170** (also §5 of the SageAttention3 paper): the INT8 *training* attention.
- 6 of 7 matmuls INT8. Forward: `QKᵀ` INT8, `PV` INT8 (P **per-token**, V per-block).
  Backward: `dV = PᵀdO` INT8, `dS K → dQ` INT8, `dSᵀQ → dK` INT8, and
  **`dP = dO·Vᵀ` kept FP16** — "we maintain dOVᵀ in FP16 while accelerating the other four …
  using INT8 per-block quantization"; the reason is error amplification into the sensitive dS.
- Granularity: per-block INT8 for Q,K,V,dO,dS (scale `max|X|/127`), per-token for P̃ and Q.
  Smooth-K retained.
- Results: 325M Llama, 78B tokens. Matches full precision at 260K tokens/step; **"clear gap" at
  2.1M tokens/step needing QK-norm**. Verdict: "lossless … when fine-tuning base models for
  instruction-following tasks, but is **not suitable for pretraining**." Up to 1.67× fwd+bwd
  vs FA2. Context length fixed at 4096; **no long-context eval, no downstream benchmarks**.
- **No public kernel** — not in thu-ml/SageAttention, no repo in the paper. sm support unverified.
- One useful positive result: models fine-tuned with SageBwd then served with SageAttention3
  beat BF16-trained + FP4-served on GSM8K/MMLU — i.e. QAT helps.

### Attn-QAT — arXiv 2603.00040 (Feb→Aug 2026, Peiyuan Zhang, …, Hao Zhang)

The single most relevant paper. RTX 5090 (sm120) + B200/B300.
- NVFP4 only, `X_ij ∈ ℝ^{1×16}`, `s_ij = max|X_ij| / 6`, dequant `X'_ij = s_ij · X̂_ij`.
  Optional global `s_enc = 6·448 / globalmax`. Softmax stays FP32.
- **The failure mode**: naive FP4 forward + BF16 FA backward has *exploding gradients*, because
  FA's backward uses the identity `Pᵢᵀ dPᵢ = dOᵢᵀ Oᵢ`, which assumes a high-precision O.
- **Fix 1**: the backward's recomputation of P from the LSE must be fake-quantized to the same
  precision as the forward (Alg. 3 line 11).
- **Fix 2**: carry a second forward output `O'ᵢ = Σⱼ P̃_ij V^F_j` (high-precision P, quantized V),
  store it *only* for gradients, and use `D = rowsum(dO ⊙ O')`.
- Forward (Alg. 2): fake-quant Q,K,V; per KV tile `S = Q^F(K^F)ᵀ/√d`, online softmax,
  `(P̃^F, ŝ) = φ⁻¹(φ(P̃))`, `O ← diag(α)O + P̃^F V^F`, `O' ← diag(α)O' + P̃ V^F`; return O, L, O'.
- Backward (Alg. 3): `D = rowsum(dO⊙O')`; per tile `P = exp(S − L)`, `P^F = φ⁻¹(φ(P))`;
  `dV += (P^F)ᵀ dO`; `dP = dO (V^F)ᵀ`; `dS = P ⊙ (dP − D)/√d` ← **unquantized P**;
  `dQ += dS K^F`; `dK += dSᵀ Q^F`. **Backward GEMMs are BF16, not FP4.**
- STE: `φ⁻¹(φ(·))` on the inputs of both matmuls; scales are static statistics, implicitly
  stop-gradiented (no scale derivative).
- No two-level P rescale (explicitly drops SageAttention3's heuristic; "QAT's fake quantization
  automatically learns to handle P's distribution"). Their shipped kernel keeps it as an option.
- Speed: RTX5090 1.1–1.5× SageAttention3 (by eliminating smoothing + two-level quant overhead);
  B200 1.31× BF16 FA4 (NVFP4 QK + BF16 PV); B300 1.74× (NVFP4 QK + FP8 PV). "softmax bottleneck"
  limits B200.
- Accuracy: Wan 2.1 1.3B/14B VBench + 99-case human eval; Qwen3-14B & Llama3.1-70B pretraining
  (WikiText, HellaSwag, PIQA, WinoGrande, ARC-C) and fine-tuning (MMLU-Redux, GPQA, MATH-500,
  GSM8K, IFEval). Near-BF16 on Qwen3-14B; partial recovery on Llama-70B (limited budget).
  **No long-context eval** (B200 benches go to s=32768 but only measure throughput).
- Code: `github.com/jzhang38/Attn-QAT` is **PDF only**. Real code is:
  - **Training (this is the answer for us)**:
    `hao-ai-lab/FastVideo` → `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_train.py`
    (1553 lines Triton). *All BF16 `tl.dot` over fake-quantized values; no FP4 tensor-core op in
    the training path at all.* Q/K/V fake-quantized in prologue kernels
    (`quant_utils.py: fake_quantize_q / fake_quantize_kv`, quant dim = HEAD_DIM, block 16, E4M3 SF);
    **the fake tensors are what is saved for backward**.
    Forward detail the paper omits: `l_ij = tl.sum(high_prec_p, 1)` — the softmax denominator
    comes from the **high-precision** P.
    `JOIN_QAT_PV` (env `FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV`, default on): `tl.join(p, high_prec_p)`
    → one `2·BLOCK_M × BLOCK_N` dot against shared V, so O and O' cost one GEMM.
    Backward: `o_for_bwd = high_prec_o if use_high_prec_o else o` (default True);
    `_attn_bwd_preprocess` computes `delta = sum(o*do, axis=1)`.
    `_attn_bwd_dkdv`: `dv += dot(trans(p_quant), do)` but `ds = p * (dp - Di)` — the asymmetry.
    `_attn_bwd_dq` never quantizes P.
    sm120 tuning: `NUM_STAGES = 2 if consumer_blackwell and seq >= 8192 else 3`;
    `BLOCK_M1=BLOCK_N1=BLOCK_M2=BLOCK_N2=32`; `BLK_SLICE_FACTOR=1` (must be 1 for QAT);
    the 64×64 split dQ/dKdV fast path is sm100-only; warp specialization force-disabled on
    Blackwell ("Triton 3.7's NVWS pass aborts").
    Two-level P in `nvfp4_utils.py::_compute_quant_and_scale`, credited to SageAttention3.
  - **Inference**: `fastvideo-kernel/attn_qat_infer/` = vendored SageAttention3-blackwell fork,
    `sm_120a` only, 8 files changed; adds a `single_level_p_quant` flag (`params.h:165`,
    `softmax_fused.h:39-42`) and fixes an upstream dangling `tile_count_semaphore` on the causal path.
  - **B200/B300 forward**: `hao-ai-lab/flash-attention-fp4`, branch `fp4`,
    `flash_attn/cute/flash_fwd_sm100_fp4.py` (4743 lines CuTe DSL). QK = NVFP4(sf_vec 16, ue4m3)
    or MXFP8(sf_vec 32, e8m0); PV independently BF16 / FP8 / NVFP4 / MXFP8 (two separate
    `make_blockscaled_trivial_tiled_mma` calls, lines 635 and 655). P quant in
    `flash_attn/cute/softmax.py:458 scale_subtract_rowmax_fp4` with the *identical* constants
    `-11.392317422778762`, `-2.584962500721156`.
    **No `flash_bwd_*_fp4.py` exists.** `flash_bwd_sm120.py` (55 lines) is a BF16-only subclass
    of the SM80 backward that only overrides the smem budget (99 KB vs 163 KB), header comment
    "Blackwell GeForce / DGX Spark".

### Hardware-Aware FP4 FlashAttention-4 — arXiv 2609.04105 (Sep 2026, sm100/103 only)

- Inference (noncausal): Q,K **NVFP4 block-16 e4m3** folded offline via MSE; P,V **MXFP4
  block-32 e8m0**.
- **Training (causal): Q,K NVFP4 row-by-K16; P,V FP8; dQ,dK,dV,dP,dS e4m3; dO e5m2.**
  Reason for e5m2 dO: "E4M3 rounded roughly 97 % of observed dO values to zero … E5M2 reduced
  the zero fraction to roughly 14 %."
- **"every tested MXFP4 probability/value training trajectory diverges."** Both MXFP4-P/V arms
  separate from FP8 controls near 0.1B tokens; one run "tracks FP8 through update 300, then its
  loss rises from 7.01 at update 325 to 16.25 at update 350." ⇒ "Training stability selects FP8 P/V."
- "Direct-P": map log-softmax scores straight to e2m1 codes via an affine classifier
  `max(0, Ax+B)` (generic A=1.50 B=1.20; Wan-tuned A=1.60 B=0.95), skipping the exponential.
- Backward saves: NVFP4 Q/K payload bytes + block scales, per-head Q–K global factors, row-wise
  LSE. Rebuilds P from LSE + quantized Q/K; `dS = P⊙(dP − r·1ᵀ)`.
- Speed: GB200 up to 2.13× peak (2998 TFLOP/s), geomean 2.023× over 9 D=128 shapes;
  B300 D128 S8192 H64 = 3116 TFLOP/s. Training: backward-only 1.125×, projection-inclusive
  attention 1.245×, full update at B4/S4096 1.141×, 64-GPU/100B-token median 1.112× throughput.
- Accuracy (cos / rel-L2 vs BF16): Direct-P fast 0.9438 / 0.3363; HAO NV/NV and NV/FP8 0.9899.
- Explicitly **not** consumer Blackwell: "SageAttention3 … targets a consumer Blackwell interface
  that differs from the asynchronous tensor-core and tensor-memory interface on the data-center
  Blackwell GPUs evaluated here."

### ffpa-attn (xlite-dev/ffpa-attn) — the most detailed sm120 fp8/fp4 attention source

"Kernel Library for Large Headdim Attention (64~1024, BF16/FP8/FP4), 1.5x~15x↑ vs PyTorch SDPA."
**Both quantized paths are forward-only**: README "`sm_120`, forward only" for FP8 (2026-07) and
FP4 (2026-08); backends table `|CUDA FP8|sm_120{a,f}|✔|✖️|` (Fwd|Bwd); code comment
"Currenly, fp8/fp4 attention are only supported on sm_120, forward only." Backward exists only
for BF16/FP16 (`src/ffpa_attn/triton/_ffpa_bwd.py`, `src/ffpa_attn/cute/_dkdv_*`, `_dq_*`).

FP8 path (`csrc/cuffpa/cute/fp8/`):

| Operand | Format | Granularity |
|---|---|---|
| Q | e4m3 or int8 | `per_block` (1 per (b,h,kBr-row block)) **default**, or `per_thread` (64 scales per 128-row block = 1 per C-fragment row-pair {r, r+8}) |
| K | e4m3/int8 | `per_block` default, or `per_thread` (4 per kBc block, grouped by `(row%8)/2`) |
| V | e4m3 | **`per_channel` default** (1 per (b,h,d), amax over all N), or per_block |
| **P** | e4m3 | **Mode B default: fixed `1/448`**; Mode A opt-in `rowmax/448` |

`quantize_fp8.cuh`: `float s = kQKInt8 ? amax/127.0f : amax/448.0f;` (Q/K), `amax/448.0f` (Vᵗ).
`fp8_pscale.cuh`:
```cpp
constexpr float kE4m3Max = 448.0f;
constexpr float kFP8FixedPScale = 1.0f / kE4m3Max;   // Mode B: max(P) <= 1
// P8  = round(P * vs / p_scale)   (v_scale folded into P's quantization)
// MMA = P8 @ V8                   (V8 = V / vs, pre-quantized in gmem, untouched)
// O  += MMA * p_scale
```
Scale-folding algebra: `k_scale` folded into the log2-domain softmax exponent; `v_scale` folded
into P's quantization; `q_scale * p_scale` applied in the epilogue. Mode B is the fast path
because the dequant constant is global → the PV MMA accumulates **directly** into `o_acc`
(Mode A needs a quad row-max reduce, a separate `o_tile` fragment, and a rescale pass, and is
mutually exclusive with lazy rescale).
Two tricks worth stealing:
- **Prescaled softmax fold**: `online_softmax_fp8_fixed` emits `P*vs*448` directly by adding
  `exp_offset = log2f(vs*448)` into the exp2 argument, so quantization is just
  `cvt.rn.satfinite.e4m3x2.f32`.
- **Tensor-core row-sum**: `pscale_rowsum_mma` runs `mma.sync…m16n8k32…e4m3.e4m3.f32` with an
  all-ones e4m3 B operand (`0x38383838`) over the *same* registers the PV MMA consumes, so
  `row_sum` is exact w.r.t. the quantized P and the fp32 FADD chain disappears.
- Overflow caveat: with FA4-style lazy rescale (`FFPA_RESCALE_THRESHOLD_FP8 = 4` in log2) values
  can inflate 2⁴, so fixed mode needs `amax(V) ≤ 28`; `satfinite` clamps residuals. Lazy rescale
  disabled on the m4n2 fp8 family for this reason.
Accuracy aids: `smooth_k` **default ON** (per-(b,h) K sequence mean; exact for O, LSE corrected
by `+scale*dot(q_row, km)`), `smooth_v` (per-D mean, added back in epilogue), optional
Walsh–Hadamard Q/K rotation (`hadamard.cuh`, cites FA3 §3.3), and **hybrid** (fp16 kernel does
the first `n_early=256` rows, fp8 does the rest; auto-on for causal).
Defaults (`src/ffpa_attn/functional.py`): `fp8_q/k_quant_method='per_block'`,
`fp8_v_quant_method='per_channel'`, `fp8_pv_acc_type='f32'`, `fp8_smooth_k=True`,
`fp8_hadamard=False`, `fp8_hybrid=None` (auto=causal).

FP4 path (`csrc/cuffpa/cute/fp4/`) — a port of SageAttention3's Blackwell kernel onto ffpa's
producer/consumer skeleton. Uses the **hardware block-scaled MMA** (both PTX strings in Part 1).
Q/K/Vᵗ = e2m1 + ue4m3 per 16; opt-in `fp4_pv_mm_type='fp8'` makes Vᵗ e4m3 + **ue8m0 per 32**.
P = e2m1 + ue4m3 per 16 columns, two-level.
```cpp
// quantize_fp4.cuh  (~286-295, 456-460, 1519-1522)
float SFValue = vecMax / 6.0f;
reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));  // round-trip ⇒ exact dequant
float SFValueInv = (SFValue == 0.f) ? 0.f : 1.f / SFValue;
// MXFP8 Vᵗ ue8m0 (~625-640)
const float groupMax = fmaxf(vecMax, __shfl_xor_sync(0xFFFFFFFFu, vecMax, 1));  // 32-token group
float sf = groupMax / 448.f;  int e = sf > 0.f ? int(ceilf(log2f(sf))) : -127;
const float scale = ldexpf(1.f, -e);  uint8_t byte = uint8_t(e + 127);
```
Note this ue8m0 rule is **ceil**, like ours.
Smooth-Q *and* smooth-K are **mandatory** on fp4 ("e2m1's ±6 range makes mean-centering a
correctness-level necessity"), which forces a separate wmma kernel for the rank-1 correction
`delta_s[b,h,mb,n] = qm @ (K − km)ᵀ`, so `S = Q̂K̂ᵀ + Δs` (`delta_s.cuh`). `qm` is the
per-128-row-block Q mean, `km` the per-(b,h) K mean. LSE:
`(m*L + log2(row_sum) + log2(1/2688))*ln2 + scale*qkm` (`log2(1/448)` instead under MXFP8 PV —
that domain constant was a latent bug fixed in their FC-6).
Layout gotchas: **`kv_perm32`** — K/Vᵗ workspaces use a 32-row interleave
`[0,1,8,9,16,17,24,25,2,3,…]`, so causal masking and attn_bias must index through `kv_perm32(j)`;
they note upstream SageAttention3 masks on the *original* column index and "causal 直接坏 —
max_abs 3.3". Mirror on the fp8 side: `VTPermInv32` / `PackC8bitToA8bitPermVT` lets P be packed
into the PV A operand with **zero cross-lane shuffles** ("reorg-free PV pack"). Subbyte pitfall:
e2m1 smem tensors need `make_smem_ptr<Element>(void*)`; `reinterpret_cast<Element*>` scales by 1
byte and overruns 2×.

Accuracy/speed reported:
- **No published accuracy table.** Bench metric is relative Frobenius error
  `||out−ref||_F/||ref||_F` + `max|·|` against **bf16 SDPA-FA2** (not fp32 — would OOM). No
  committed numbers; only FLUX.1-dev image grids (seed 42, 28 steps, 1024²/2048²).
- Test tolerances (`tests/test_ffpa_fp8.py`): dense `atol=rtol=4e-2`, causal `1e-1`,
  m4n2 causal `2e-1`, lse `5e-2`. Asserts `rel_int8 < rel_fp8` and `rel_int8 < 0.10`
  — **int8 QK is more accurate than e4m3 QK**. fp4 bench tol: dense 0.15 / causal 0.70,
  `mean_abs 0.014–0.03` as quality indicator.
- **Error decomposition** (B1H32N8192D128 randn, `fp8/sm_120/persist_d.cuh:22-52`): causal
  `max_abs 0.22` vs dense `0.015` (15×) but **relative error ~5 % for both**; the gap is output
  amplitude via effective sample size `ESS_i = 1/Σ_j P[i,j]²` (causal row 0 has ESS=1, amp ~3.1;
  dense ESS ~3000, amp ~0.05). Per-source contributions: **V quant 0.19 > QK 0.13 > P 0.11.**
  Their conclusion: "early-row fp16 QK does NOT fix causal accuracy — V quant is the dominant
  source". Known unfixed: causal early-row **LSE** error up to ~6e-2 (e4m3 rounding of near-1
  probabilities); dense ~4e-3.
- Speed: FP8 3–6× SDPA, FP4 4–7× (RTX 5090). FP4 absolute **850–980 TOPS** (D=128–256) on
  RTX 5090, 3.8–4.4× SDPA; ~1000 TOPS at D=128 on PRO 6000. vs Sage: "FFPA FP8 comparable or
  slightly better than SageAttention-2 at D=128; FFPA FP4 significantly better than
  SageAttention-3 at D=128". FLUX e2e on RTX PRO 5000 @2048²: SDPA 91.25 s, FFPA-FP8 79.91 s,
  Sage-3 80.39 s, FFPA-FP4 75.73 s. Caveat: "not suitable for … small models or short seqlen";
  the preprocess chain costs ~1.1 ms.
- BF16 contrast: B200 CuTe-DSL tcgen05 2-CTA D=512 → 1517 TFLOPS fwd / 763 TFLOPS bwd.
- Split-D / smem: persist-D keeps Q resident at O(D) — "D=512 ⇒ 192 KB > 99 KB per-CTA limit on
  sm_8x/sm_120". Split-D chunks D keeping SRAM at `Br × 16`. TiledMMA **M4N2** fixes register
  pressure: PV has N=D so O acc costs `D/(2·Nw)` regs/thread; M8N1 ⇒ O(D/2) (256 regs at D=512,
  over the 255 limit), M4N2 ⇒ O(D/4). M8N1 for D≤512, M4N2 above; on RTX5090 M4N2 is 1.55× M8N1
  at D=1024 (154 T vs 100 T).
  Per-family D ranges: fp8 persist-D D≤224 (%32, kBc=128 for D≤128 else 64), split-D M8N1
  224<D<768, M4N2 D≥768. fp4 persist-D **D ∈ {64,128,192,256} (%64 — the SF atom is 64-wide on
  both MN and K)**, split-D M8N1 (256,768), M4N2 [768,1024]. Any `D%8==0` works via pad to
  `(D+63)&~63` with zero data **and zero SF** (contributes 0 to the MMA); only O is padded and
  sliced back. So D=120 and D=96 work.
  smem: fp4 persist-D `kStages=3` for D≤192, 2 at D=256 ("3 stages needs 130,560 B"). With
  MXFP8 PV, 3 stages fit only D=128 (89,600 B). fp8 persist-D aliases K stage 0 onto the dead Q
  smem tile: **80 KB → 64 KB**, and the measured gain is **L1 capacity (~4–5 µs per 16 KB freed),
  not occupancy** (register-limited to 1 CTA/SM either way). Same trick is off by default on fp4
  (+1.8 % slower — fp4's L1 is already 96.75 % sector-hit).
  Dropout unsupported on all fp8/fp4 families; attn_bias added Aug 2026.
- Best single document: `https://raw.githubusercontent.com/xlite-dev/ffpa-attn/main/.github/skills/ffpa-cuda-understand/SKILL.md`
  (108 KB, Chinese) — §5 fp8, §6 fp4, §8 split-D gaps, §11.5/11.6 quantization & scale-folding
  algebra, §11.7 ESS error model, §11.8 smoothing math, §11.9 WHT, §11.10 NVFP4 blockscale +
  two-level P, §11.11 reduction-axis permutation invariance, §11.15 per-stage numeric-format
  cheat sheet. Plus `references/rfc-future-optimizations.md` (166 KB) with a falsified-
  optimizations appendix.
- `xlite-dev/LeetCUDA` was checked: `kernels/flash-attn` is fp16/bf16 teaching kernels only,
  no fp8/fp4 attention. ffpa-attn is the only place these live; upstream for both is
  thu-ml/SageAttention (`csrc/qattn/*`, `csrc/mma.cuh::rowsum_f8f8f32`, and
  `sageattention3_blackwell/sageattn3/blackwell/{cute_extension.h,blockscaled_layout.h,
  softmax_fused.h,utils.h}`, `quantization/fp4_quantization_4d.cu`).

### NVIDIA TransformerEngine / cuDNN — the only production FP8 attention *backward*

- `transformer_engine/common/fused_attn/fused_attn_fp8.h::fused_attn_fp8_bwd` → cuDNN
  `mha_graph->sdpa_fp8_backward(...)`. Python entry
  `transformer_engine/pytorch/attention/dot_product_attention/backends.py:1798 FusedAttnFunc.backward`.
  Env `NVTE_FP8_DPA_BWD` (default 1). Arch gate `(9,0) <= cc < (12,0)` ⇒ **not available on
  sm120/121.** TE disables FA2/FA3/FA4/unfused under fp8 attention.
- Scale tensors are all **per-tensor fp32 `(1,1,1,1)`**. Fwd: `descale_q/k/v`, `descale_s`,
  `scale_s`, `scale_o`, out `amax_s`, `amax_o`. Bwd adds `descale_o/do/dp`,
  `scale_dq/dk/dv/dp`, out `amax_dq/dk/dv/dp`.
- S and dP are **never materialized** — created as shape-`{0}` fp32 tensors carrying only
  scale/scale_inv/amax (`csrc/extensions/attention.cpp:370`). Dedicated meta slots
  `META_S = FP8FwdTensorIdx.GEMM3_OUTPUT` (`scaling_fwd[8]`),
  `META_DP = FP8BwdTensorIdx.GRAD_INPUT3` (`scaling_bwd[5]`). **S is E4M3, dP is E5M2** under HYBRID.
- Even when the linear layers use Float8CurrentScaling / MXFP8 / NVFP4, S and dP are **forced
  onto a synthetic DelayedScaling recipe** with `NVTE_DPA_FP8DS_AMAX_HISTLEN=1`, algo
  "most_recent" ⇒ dP's scale is literally the previous step's amax.
- dQ/dK/dV are actually fp8 (E5M2, uint8 storage) **only under DelayedScaling**; under
  CurrentScaling and MXFP8 they come out BF16.
- **MXFP8 attention** (sm100+, cuDNN ≥ 9.21): block 32, E8M0, on Q/K/V/dO. A distinct
  `sdpa_fp8_backward` overload takes `Q,Q_t,K,K_t,V,O,dO_f16,dO,dO_t` — **dual orientations
  because dS must be quantized twice**: blocked along `s_kv` for `dQ = dS·K`, along `s_q` for
  `dK = dSᵀ·Q`. TE nulls the S/dP quantizers there ("the kernel handles S/dP internally for
  MXFP8"). **P is not block-scaled — fixed scale 256.0, "because softmax output is bounded
  [0,1]"** (https://nvidia.github.io/cudnn-frontend/mxfp8-attention-scaling/).
- **NVFP4 attention does not exist** — TE disables FusedAttention under an NVFP4 recipe.
- Limitations: fp8 attention cannot work with THD layout, bias, or sliding window.
- cuDNN also ships a DeepSeek Sparse Attention module (CuTe-DSL kernels, SM90/SM100+):
  https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/dsa.html

### Serving stacks (granularity reference only — training scope de-prioritizes these)

- **vLLM**: `kv_cache_dtype=fp8_e4m3|fp8_e5m2`; `layer._k_scale` / `_v_scale` are **0-dim**
  params ⇒ per-tensor. `--calculate-kv-scales` removed in PR #49389 (Aug 2026).
  `compressed-tensors` `kv_cache_scheme = {num_bits: 8, type: "float", strategy: "tensor"|
  "attn_head", symmetric: true, dynamic: false}`; `strategy: attn_head` = one scalar per KV head
  (FA3+ only). FA3 backend takes `q/k/v_descale` of shape `(batch, num_kv_heads)`.
- **SGLang**: raises "Only support per-tensor scaling factor for fp8 KV cache".
- **FlashInfer**: `NVFP4_SF_VEC_SIZE = 16` (verified in
  `include/flashinfer/attention/prefill.cuh`) ⇒ NVFP4 KV = one e4m3 scale per 16 channels per
  token per head; MXFP8 KV = per-32 ue8m0. **The FA2 path repacks an fp8/fp4 KV tile into a
  16-bit smem staging buffer and then does bf16 `ldmatrix` + MMA** — i.e. dequant on load.
  No attention autograd at all.
- **FlashMLA**: dequantizes fp4/fp8 → bf16 **in registers**, then bf16 UMMA.
- True fp8 MMA with no dequant exists only in FA3/FA4 with an fp8 Q, and in trtllm-gen
  (per-tensor `bmm1_scale` / `bmm2_scale`).
- AMD `ck_tile`: `BWD_DTYPE_MAP` is {fp32, fp16, bf16} only, while FWD has fp8/mxfp8/mxfp4.
- PyTorch SDPA: fp8 only via the FA3 backend; backward rejects it.

### KV-cache QAT / fake-quant with an STE

- **LLM-QAT, arXiv 2305.17888** (facebookresearch/LLM-QAT) — the canonical one.
  STE site `models/modeling_llama_quant.py:321-327`, on raw `k_proj`/`v_proj` outputs ⇒
  **pre-RoPE** (RoPE applied at L339). Attention scores are *not* quantized (dead
  `# TODO: attention score matrix` branch). Scale explicitly detached:
  ```python
  max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
  s = (2**(num_bits-1)-1)/(max_input+1e-6); output = torch.round(input*s).div(s+1e-6)
  ```
  The `(x_q-x).detach()+x` idiom appears only in their ≤2-bit *weight* path. The STE is
  **clipped and buggily so**: backward zeroes grads outside a hard-coded ±2.0 clip the forward
  never applies. Granularity: per-token symmetric over the **concatenated** hidden dim
  (`[bsz, q_len, num_heads*head_dim]`, reduce `dim=-1`) — one scale per token for *all heads
  together*, coarser than anything in production. **No long-context eval** (WikiText2/C4 ppl,
  zero-shot, MMLU, TriviaQA).
- **NVIDIA TensorRT Model Optimizer** — the substrate to copy if we implement fake-quant.
  `modelopt/torch/quantization/plugins/huggingface.py::_QuantAttention._setup()` creates four
  quantizers: `q_bmm_quantizer`, `k_bmm_quantizer`, `v_bmm_quantizer`, **`p_bmm_quantizer`**.
  `_quantized_attention()` applies them **post-RoPE**; `_eager_p_qdq_attention` fake-quantizes
  the softmax probabilities (`return _pq(_orig_softmax(...))`), with a Triton FA kernel for
  NVFP4/FP8 P. STE in `tensor_quant.py`:
  ```python
  def _fake_tensor_quant_backward(inputs, amax, grad_outputs):
      if amax is None: return grad_outputs
      return torch.where(inputs.abs() <= amax, grad_outputs, zero)   # clipped STE
  ```
  `pass_through_bwd=True` (default for dynamic block quant) gives an identity STE.
  Scale detached at `tensor_quantizer.py:748-750` (`reduce_amax(...).detach()`), except
  `enable_lsq` where amax is trainable (used for weight/input quantizers, not KV).
  Recipes `modelopt_recipes/configs/ptq/units/`: `kv_fp8.yaml` → E4M3, `axis: null` = per-tensor;
  `kv_nvfp4.yaml` → E2M1 `block_sizes: {-1: 16, type: dynamic, scale_bits: e4m3}` = per-16-channel
  group along head_dim per token per head. Also `kv_nvfp4_affine`, `kv_nvfp4_rotate` (Hadamard),
  `kv_fp8_cast`. QAT is live: `examples/llm_qat/{train.py,simple_qat_train.py}` with
  `general/ptq/nvfp4_default-kv_fp8` and three QAD recipes `modelopt_recipes/general/qad/*-kv_fp8.yaml`.
  Deliberate trick in the legacy path `plugins/attention.py:177-181`: a double `transpose(-1,-2)`
  around the K quantizer so K is quantized **per-token, not per-channel**.
  Also `KitchenFlashAttentionModule` with `pv_dot_precisions="mxfp8_e4m3_emulation@bf16"` wired to
  `p_bmm_quantizer` (`plugins/huggingface.py:90-125`) — a real MXFP8 P·V inside an FA module
  during training.
- **ReQAT, arXiv 2606.15682** (ICML 2026, aiha-lab/ReQAT) — FP4 W4A4KV4 QAT, forks modelopt.
  `training/modelopt/torch/quantization/config.py:670-692 NVFP4_W4A4_E1M2_KV4_FAKE_CFG`:
  `"*[kv]_bmm_quantizer": {"num_bits": (1,2), "block_sizes": {-1: 16, "type": "dynamic",
  "scale_bits": (4,3)}, "axis": None}` — **KV uses E1M2, not E2M1**; the paper says E1M2 gives
  lower training loss for KV. Plus `training/reqat/k_shift_attention.py` (channel-wise post-RoPE
  K shift + pre-RoPE scaling). No long-context eval (AIME/MATH-500/GPQA/LiveCodeBench).
- **arXiv 2609.04263 / saifmb0/kvlora** — *the only* KV fake-quant-training work with long-context
  evals. STE verbatim from §3.3: "Newly appended reconstructed states use a straight-through
  gradient: `x̂ + x − stopgrad(x)`. Forward attention therefore sees the stored reconstruction while
  K and V adapters receive an identity backward path." Identity STE, not clipped. Base weights +
  quantizer frozen; only LoRA adapters on Q/K/V train; objective = KL to a float-cache teacher.
  Granularity: affine over groups of **64** along head_dim, FP16 scale + FP16 offset,
  `s_g = max((max−min)/(2^b−1), 1e-8)`. Evals: **RULER 4K/8K, NIAH, 180-case associative
  retrieval**. **Key number: 2-bit ppl recovers 576.10 → 11.40 (float 10.40) but only 11–12 of
  180 retrieval cases are restored.** ⇒ perplexity recovery ≠ long-context retrieval recovery.
- **Full-Stack FP4, arXiv 2607.04422** (Aug 2026): real low-precision attention in *pretraining*.
  NVFP4 on `QKᵀ` and the dS products; Q/K/V stored as quantized views reused fwd+bwd;
  **PV, PᵀdO and dOVᵀ kept BF16**. 3B params / 64B tokens, within 0.838 % of BF16. Runs are
  fake-quant simulation on A800s. Explicitly positions against Attn-QAT ("QAT for inference" vs
  "pretraining operator"). Appendix D has the tiled algorithm.
- Aug-2026 QAT survey arXiv 2608.29667 §3.4: "The direct relevant method involving KV cache is
  LLM-QAT … KV-cache QAT is still an emerging direction and the methods remain limited."

**Traps — things that look like KV QAT and are not:**
- **KIVI** (jy-yuan/KIVI): `models/utils_quant.py` has SymQuantizer/AsymQuantizer autograd
  Functions copied from LLM-QAT, but they are **dead code, imported nowhere**. KIVI is
  tuning-free PTQ. (Its actual contribution: per-channel K, per-token V.)
- **llm-compressor / compressed-tensors**: gradients structurally impossible —
  `compressed_tensors/quantization/lifecycle/forward.py` decorates `fake_quantize`,
  `_process_quantization`, `forward_quantize` with `@torch.no_grad()` (L36/73/142/175); no
  `autograd.Function` in `src/`. KV quant lives in `compressed_tensors/modeling/kvcache.py::
  QuantizedKVCache.forward` (post-RoPE). The "quantization aware training (QAT)" string in
  `modifiers/quantization/quantization/base.py` is a stale SparseML docstring.
- **torchao QAT**: **no** KV-cache or attention fake-quant. `torchao/quantization/qat/` is
  linear + embedding + fake_quantizer only; `prototype/qat/{mx,nvfp4}.py` is linear-only. But
  `_NVFP4QuantizedForwardFakeQuantizedBackward` (real NVFP4 addmm forward, dequantized BF16
  matmuls backward) is a clean STE pattern worth copying.
- KVQuant, QAQ, GEAR, SKVQ, ZipCache, QJL, CacheGen, OScaR, RotateKV: all PTQ, no autograd.
  Useful as **long-context baselines**: KVQuant has 1M passkey; SKVQ and QJL have LongBench;
  OScaR (arXiv 2605.19660) is near-lossless INT2 training-free and is probably the PTQ number to beat.

### Other papers read

- **INT-FlashAttention, arXiv 2409.16997** (sm80): first fully-INT8 FA forward, token-level PTQ,
  72 % faster than FA-FP16, up to 82 % smaller quantization error. INT4-compatible. Forward only.
- **NVFP4 pretraining, arXiv 2509.25149** (NVIDIA): 12B params / 10T tokens, "longest publicly
  documented 4-bit run", loss and downstream accuracy comparable to an FP8 baseline. Recipe:
  Random Hadamard Transforms for block outliers, 2-D quantization for fwd/bwd consistency,
  **stochastic rounding for unbiased gradients**, selective high-precision layers.
- **Characterization and Mitigation of Training Instabilities in Microscaling Formats,
  arXiv 2506.20752**: ~1000 LMs, 2e17–4.8e19 FLOPs. "sharp, stochastic instabilities in the loss,
  particularly at larger compute scales", traced to "multiplicative gradient bias introduced by
  the quantization of layer-norm affine parameters and a small fraction of activations".
  Mitigation: switch precision schemes mid-training.
- **Why Low-Precision Transformer Training Fails, arXiv 2510.04212**: the culprit is the `P̄V`
  computation when `P̄ = 1.0` exactly (multiple tied row maxima) multiplying predominantly
  negative V; the BF16 add's right-shift rounding is systematically negative, biasing Ō, which
  biases `(δ_lp − δ_hp)` positive and corrupts gradients. Combined with recurring low-rank `P̄K`
  patterns, errors accumulate instead of cancelling. Fix: **dynamic softmax adjustment** — when a
  row has multiple identical maxima, scale the normalizer by `β > 1` so all `P̄ < 1`. Setup:
  GPT-2 12L/768, BF16 forward / FP32 backward. Small scale only.
- **Practical FP4 Training for Large-Scale MoE on Hopper, arXiv 2603.02731** (Zhejiang Lab):
  direct FP8↔FP4 quant/dequant + scaling-aware FP4 row↔column conversion; **core MoE compute in
  FP8, activations and expert-parallel comms compressed to MXFP4**. 671B scale: −14.8 % peak
  activation memory (11.8 GB), +12.5 % throughput (1157 → 1302 tok/GPU/s). Same
  "FP4 for storage/bandwidth, FP8/BF16 for math" shape as DSV4.1's KV cache.
- **lemyx/tilelang-dsa** — "DeepSeek-V3.2-Exp DSA Warmup Lightning Indexer **training** operator
  based on tilelang". One-pass KL-divergence fwd/bwd. **bf16 only**: "we chose to use bf16 to
  develop Lightning Indexer first, and later support the fp8 data type"; and "the gradient between
  fp8 and bf16 requires additional consideration; and Lightning Indexer may use the bf16 data type
  during training and the fp8 data type during inference". Files:
  `kernel_bf16_training_dsa_warmup_lightning_indexer.py`, `bert_padding.py`, `varlen_utils.py`.
- **HISA, arXiv 2603.28458**: hierarchical block indexer + token-level rerank; 3.75× over the DSA
  kernel. Relevant to design step 6c, not to precision.

### TileLang attention examples — precision audit

| Path | QK | PV | Bwd |
|---|---|---|---|
| `examples/flash_attention/example_mha_{fwd,bwd}_b{s,h}d.py`, `example_gqa_*` | bf16/fp16 | same | yes, bf16 |
| `examples/flash_attention_sm100/{mha,gqa}_{fwd,bwd}_bshd.py` | bf16 | same | yes, bf16 |
| `examples/deepseek_v32/sparse_mla_{fwd,fwd_pipelined,fwd_seesaw,bwd}.py` | bf16 | bf16 | yes, bf16 |
| `examples/dsa_sparse_finetune/{sparse_mla_fwd,sparse_mla_bwd,indexer_bwd}.py` | bf16 | bf16 | yes, bf16 |
| `examples/deepseek_v32/fp8_lighting_indexer.py`, `examples/dsa_hisa/{block_sparse_mqa_fp8,pool_mqa_fp8}.py` | **fp8 e4m3**, f32 accum, per-token f32 dequant scale | n/a (scoring only) | **no** |
| `examples/blockscaled_gemm_sm100/*`, `examples/deepseek_v4/fp8_fp4_gemm_1d1d_sm100.py` | GEMM only | — | — |

Every DSA backward asserts `dtype == T.bfloat16` (multiple sites), fp32 accum,
`dkv = torch.zeros_like(kv, dtype=torch.float32)`. `dsa_sparse_finetune/dsa.py` builds the whole
finetune graph in bf16 including `index_q`, `index_k`, `weights`. **No fp8 in any DSA backward.**
**No TileLang example combines a block-scaled MMA with attention.**

---

## Part 3 — What we looked for and did NOT find (do not repeat)

1. **Any open-source FP4 or INT8 attention *backward* kernel for sm_120/sm_121.** Does not exist
   in SageAttention, flash-attention-fp4, FastVideo, ffpa-attn, or TileLang. Attn-QAT's sm120
   training backward is **BF16 Triton over fake-quantized tensors**.
2. **`flash_bwd_*_fp4.py`** in `hao-ai-lab/flash-attention-fp4` — absent. The paper's B200/B300
   contribution is forward-only. `flash_bwd_sm120.py` exists but is BF16.
3. **SageBwd source code.** Not in thu-ml/SageAttention; its paper lists no repo; sm support
   unverified from source.
4. **Attn-QAT's own GitHub** (`jzhang38/Attn-QAT`) contains only the PDF; repo pointers come from
   the Hao AI Lab blog.
5. **FP8 attention backward on sm120.** TE's arch gate is `(9,0) ≤ cc < (12,0)` — Blackwell
   consumer is excluded. No other implementation found.
6. **NVFP4 attention in TransformerEngine** — TE explicitly disables FusedAttention under an
   NVFP4 recipe.
7. **Evidence that DeepSeek-V3.2's indexer is trained with FP8 fake-quant** — no. FP8 is an
   inference kernel; the indexer trains in bf16 via a detached KL loss.
8. **A DeepSeek-V4.1 tech-report statement that the compressed KV is QAT'd** — §2.4.4 and §1 say
   QAT/"FP4 global KV caches during training", but the *mechanism* (STE placement) is not stated;
   the HF `inference/kernel.py` comment "`# Training's compressed KV: …`" is the strongest code-level hint.
9. **Long-context task accuracy for FP4 attention.** Nobody reports it. SageAttention2 (INT4 QK /
   FP8 PV) has Longbench + NIAH + InfiniBench@262k; Attn-QAT, SageAttention3 and FA4-FP4 report
   only cos-sim / VBench / short-context LM benchmarks. Among KV-cache work, only arXiv 2609.04263
   reports RULER/NIAH for a *trained* fake-quant cache.
10. **An accuracy table for ffpa-attn.** None published — only relative Frobenius error against a
    bf16 reference and FLUX image grids.
11. **NVIDIA staff confirmation of the GB10 FP4-vs-FP8 ratio** — the dev-forum thread is community
    analysis only, and the absolute TFLOPS numbers there (188/356) disagree with other reports
    (85.9/129). Trust the ratio, not the absolutes.
12. ~~OCP MX / NVFP4 / PTX exact formulas~~ — **found, see Part 6.** Retrieval notes for next time:
    the OCP spec URL 403s to non-browser clients (use
    `https://web.archive.org/web/2024/https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf`);
    the PTX ISA page must be pulled with `curl` (3.9 MB) because WebFetch truncates it;
    "Alvarez et al. 2025" for NVFP4 is a **blog**, not a paper — the paper is arXiv 2509.25149.
13. **Any FP4/FP8 attention *backward* in FlashMLA.** `_flash_attn_varlen_backward` allocates
    `dq/dk/dv` as `dtype=q.dtype`; `grep fp8` over `dense_bwd.cpp` and
    `fmha_cutlass_bwd_sm100.cu` returns nothing.
14. **A per-channel (per-head_dim) KV scale mode in vLLM main** — does not exist. Five granularities
    exist (per-tensor, per-attn-head static, per-(token,head) dynamic, per-16-channel NVFP4,
    per-128-tile `fp8_ds_mla`), none per-channel.
15. **Accuracy data for NVFP4 KV on RTX PRO 6000 / DGX Spark** — the NVIDIA forum thread
    (377425) has memory-capacity ratios only (1.69× / 1.68× tokens vs FP8), zero accuracy numbers.
16. **A TensorRT-LLM doc entry for NVFP4 KV** — the precision matrix still lists only INT8/FP8 KV;
    the Dec-2025 blog is ahead of the docs.

## Part 4 — Dead ends

- `arxiv.org/pdf/...` WebFetch on FA3, SageAttention2 v6, Attn-QAT and DeepSeek-V4 all failed
  (PDF too large or no text layer). Use `ar5iv.labs.arxiv.org/html/<id>` or
  `arxiv.org/html/<id>v<n>` instead — both worked reliably.
- `docs.nvidia.com/cuda/parallel-thread-execution/index.html` returns only the ToC through
  WebFetch; the block-scaled MMA tables are in sub-sections that need direct deep links.
- Searching for "SageAttention long context" surfaces SpargeAttention (sparse, different paper);
  the long-context evals are in the SageAttention2 paper itself.
- `xlite-dev/LeetCUDA` has no fp8/fp4 attention (checked) — don't re-check.
- `deepseek-ai/TileKernels` has no attention kernels (checked).
- The DeepSeek-V4 PDF's FP4-QAT section could not be extracted by WebFetch; the details in Part 2
  came from search-result summaries of that paper and should be re-verified against the PDF text
  if quoted precisely.

## Part 5 — URL list

Papers
- FlashAttention-3 https://arxiv.org/abs/2407.08608
- SageAttention https://arxiv.org/abs/2410.02367
- SageAttention2 https://arxiv.org/abs/2411.10958 (read v6)
- SageAttention2++ https://arxiv.org/abs/2505.21136
- SageAttention3 https://arxiv.org/abs/2505.11594
- SageBwd https://arxiv.org/abs/2603.02170
- Attn-QAT https://arxiv.org/abs/2603.00040 · blog https://haoailab.com/blogs/attn-qat/ · docs https://haoailab.com/FastVideo/training/attn_qat/
- Hardware-Aware FP4 FlashAttention-4 https://arxiv.org/abs/2609.04105
- INT-FlashAttention https://arxiv.org/abs/2409.16997
- OCP Microscaling formats https://arxiv.org/abs/2310.10537
- NVFP4 pretraining https://arxiv.org/abs/2509.25149
- MX training instabilities https://arxiv.org/abs/2506.20752
- Why low-precision transformer training fails https://arxiv.org/abs/2510.04212
- Practical FP4 Training for MoE on Hopper https://arxiv.org/abs/2603.02731
- Full-Stack FP4 https://arxiv.org/abs/2607.04422
- LLM-QAT https://arxiv.org/abs/2305.17888
- ReQAT https://arxiv.org/abs/2606.15682
- KV-LoRA / quantized-cache recovery (RULER) https://arxiv.org/abs/2609.04263
- OScaR INT2 KV PTQ https://arxiv.org/abs/2605.19660
- QAT survey https://arxiv.org/abs/2608.29667
- HISA https://arxiv.org/abs/2603.28458
- DeepSeek-V3 https://arxiv.org/abs/2412.19437 · DeepSeek-V4 https://arxiv.org/abs/2606.19348
- NVFP4 QAD report https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf

Repos
- https://github.com/Dao-AILab/flash-attention
- https://github.com/thu-ml/SageAttention (+ `sageattention3_blackwell/`)
- https://github.com/xlite-dev/ffpa-attn (+ `.github/skills/ffpa-cuda-understand/SKILL.md`)
- https://github.com/hao-ai-lab/FastVideo · https://github.com/hao-ai-lab/flash-attention-fp4 (branch `fp4`)
- https://github.com/tile-ai/tilelang · https://github.com/lemyx/tilelang-dsa
- https://github.com/deepseek-ai/DeepSeek-V3.2-Exp · https://github.com/deepseek-ai/FlashMLA · https://github.com/deepseek-ai/TileKernels
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash (`inference/kernel.py`, `inference/model.py`)
- https://github.com/NVIDIA/TransformerEngine · https://github.com/NVIDIA/TensorRT-Model-Optimizer
- https://github.com/flashinfer-ai/flashinfer · https://github.com/vllm-project/vllm
- https://github.com/facebookresearch/LLM-QAT · https://github.com/aiha-lab/ReQAT · https://github.com/saifmb0/kvlora · https://github.com/jy-yuan/KIVI

Docs
- cuDNN FP8 SDPA fwd/bwd tensors https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.9.0/operations/Attention.html
- cuDNN MXFP8 attention scaling https://nvidia.github.io/cudnn-frontend/mxfp8-attention-scaling/
- cuDNN DSA https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/dsa.html
- GB10 FP4 scaling thread https://forums.developer.nvidia.com/t/fp4-on-dgx-spark-why-it-doesnt-scale-like-youd-expect/360142
- GB10 1-PFLOPS reproduction thread https://forums.developer.nvidia.com/t/question-on-reproducing-dgx-spark-gb10-fp4-1-pflops-performance-using-cutlass-profiler/357249

---

## Part 6 — Format and ISA definitions, from the primary specs (authoritative)

### 6.1 OCP MX v1.0 spec (Sep 2023)

Canonical URL `https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf`
**returns HTTP 403 to non-browser clients.** Retrieved via
`https://web.archive.org/web/2024/https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf`
(16 pages). Companion paper: *Microscaling Data Formats for Deep Learning*, Rouhani et al.,
arXiv:2310.10537 v3.

**Table 1 (§5.2)** — all four concrete formats use **k = 32** and an **E8M0** scale (w = 8):

| Format | Element type(s) | d | k | Scale | w |
|---|---|---|---|---|---|
| MXFP8 | E5M2, E4M3 | 8 | 32 | E8M0 | 8 |
| MXFP6 | E3M2, E2M3 | 6 | 32 | E8M0 | 8 |
| **MXFP4** | **E2M1** | 4 | **32** | **E8M0** | 8 |
| MXINT8 | INT8 | 8 | 32 | E8M0 | 8 |

**Table 5 (§5.3.3) — FP4 = E2M1**: exponent bias 1; **no Inf, no NaN**;
max normal `S 11 1₂ = ±2²×1.5 = ±6.0`; min normal `±2⁰×1.0 = ±1.0`;
max subnorm = min subnorm = `±2⁰×0.5 = ±0.5`. **emax_elem = 2**; largest *power of two*
representable in E2M1 is **4.0**, not 6.0. 15 distinct values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.

**Table 7 (§5.4.1) — E8M0**: "an unsigned representation of a conventional biased Float32
exponent… no representation for Inf and only a single NaN encoding is reserved." Bias 127,
exponent range **−127…127**, NaN = `0xFF`, **Zeros N/A** ⇒ **scale 0 is not representable**
(range 2⁻¹²⁷…2¹²⁷).

**Scale-selection rule, §6.3, verbatim:**
> 1. Set 𝑋 to be the largest power-of-two less than or equal to max(|𝑉ᵢ|), **divided by the
>    largest power-of-two representable in the element data type**.
> 2. Set 𝑃ᵢ to be the scaled inputs 𝑉ᵢ / 𝑋 quantized to the element data type. For this
>    quantization, normal numbers that exceed the max normal representation of the element data
>    type should be **clamped to the max normal, preserving the sign**.

Rounding: roundTiesToEven **must** be supported; other algorithms may be.
Closed form (companion paper Alg. 1, which says it "follows the semantics outlined in Section 6.3"):
```
shared_exp ← ⌊log2( max_i |V_i| )⌋ − emax_elem        # emax_elem = 2 for e2m1
X ← 2^shared_exp
P_i = quantize_to_element_format(V_i / X)              # clamp normals, preserve sign
```
Additional rules stated only in the paper, not the spec: Infs/NaNs are not clamped; `P_i = 0` if
`V_i` is an FP32 subnormal. Also noted: **MX conversion and transpose do not commute** (the
shared-scale axis changes) — relevant if we ever need dS in two orientations.

**⚠️ The OCP rule divides by 2^emax = 4, not by 6, so it saturates.** With amax = 2^e·m,
m ∈ [1,2): X = 2^(e−2) and amax/X = 4m ∈ [4,8); since e2m1 max is 6, **any block whose amax
mantissa exceeds 1.5 has its maximum element clamped**. Verified numerically: amax = 3.0 →
6.0 exactly (fine); amax = 3.0001 → 6.0002 → clamps; amax = 7 → clamps to 6.
**Our `_pow2_ceil_scale(amax/6)` does NOT do this** — it is the round-up-to-power-of-two recipe,
which never clips but wastes up to one binade.

### 6.2 NVFP4 (NVIDIA)

Primary: *Pretraining Large Language Models with NVFP4*, arXiv:2509.25149 (v1 Sep 2025, v2 Mar 2026).
"Alvarez et al. 2025" in DSV4.1's bibliography is the **blog**
https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/.

Paper **Table 1** (Blackwell tensor cores):

| Format | Element | Scale | Block | GB200 vs BF16 | GB300 |
|---|---|---|---|---|---|
| MXFP8 | E5M2/E4M3 | UE8M0 | 32 | 2× | 2× |
| MXFP6 | E3M2/E2M3 | UE8M0 | 32 | 2× | 2× |
| MXFP4 | E2M1 | UE8M0 | 32 | 4× | 6× |
| **NVFP4** | **E2M1** | **E4M3** | **16** | **4×** | **6×** |

**Appendix B, exact two-level formulas (verbatim):**
```
(1) s_enc  = (6 · 448) / amax_x        # amax_x over the whole tensor;  s_dec = 1/s_enc  (fp32)
(2) s_dec,b = amax_b / 6               # amax_b over the 16-element block
(3) s_dec,b,e4m3 = e4m3( s_dec,b · s_enc )              # RNE
    s_enc,b      = 1 / ( fp32(s_dec,b,e4m3) · s_dec )
(4) x̂_i = q( x_i · s_enc,b )
(5) tensor-core partial dot = s^x_dec,b,e4m3 · s^y_dec,b,e4m3 · Σ_{k∈b} x_k y_k
    then the global s^x_dec · s^y_dec is applied to the GEMM output
```
Design goal `s_enc,b · s_dec · s_dec,b,e4m3 ≈ 1`. "6 and 448 are the maximum representable
magnitudes in the E2M1 and E4M3 formats."
**Sign-convention gotcha**: the paper's `s_enc` is a multiply-by scale; the reference
implementation stores the reciprocal —
`TensorRT-Model-Optimizer/modelopt/torch/quantization/qtensor/nvfp4_tensor.py`:
`weights_scaling_factor_2 = global_amax / (E2M1_MAX * 448)`, block scales **clamped to
[2⁻⁹, 448]** before the e4m3 cast.
NVFP4 "encodes at least 6.25 % of values in a block (the amax of each 16-element block) at
near-FP8 precision."
**Appendix B.4 describes the "MXFP4" recipe as rounding `amax/6` UP to the next power of two**
(citing Mishra et al. 2025) — i.e. *not* OCP §6.3. For amax = 3+δ that picks X=1, quantizes the
block max to 3, and uses only `log2(3/0.5) = 2.58` of the available 3.58 binades. **Two
incompatible numerics share the name MXFP4; always state which conversion.**
**Attention is not quantized in that paper**: "attention components, including softmax and the
query-key and attention score-value batched GEMMs" are kept at original precision, along with
embeddings, the output head, normalization, non-linearities and the final ~8 blocks (~16 % of
linear layers). Master weights / weight grads / optimizer states FP32; TP reductions BF16.
Blog numbers: E4M3-scale MSE **0.08 vs 0.72** for E8M0 in their illustrative example;
DeepSeek-R1-0528 FP8→NVFP4 MMLU-PRO 85→84, GPQA-D 81→80, AIME24 89→91, MATH-500 98→98.

### 6.3 FP8 (arXiv:2209.05433 Table 1)

| | E4M3 | E5M2 |
|---|---|---|
| bias | 7 | 15 |
| Inf | N/A | `S.11111.00` |
| NaN | `S.1111.111` | `S.11111.{01,10,11}` |
| **max normal** | `1.75·2⁸ = 448` | `1.75·2¹⁵ = 57344` |
| min normal | 2⁻⁶ | 2⁻¹⁴ |
| max subnorm | 0.875·2⁻⁶ | 0.75·2⁻¹⁴ |
| min subnorm | **2⁻⁹** | 2⁻¹⁶ |

E4M3's 448 comes from reclaiming special-value patterns ("we gain … 256, 288, 320, 352, 384,
416, 448"); without that the max would be 240.

### 6.4 PTX ISA 9.4 — warp-level block-scaled `mma.sync`

§9.7.16.3 *Block Scaling for `mma.sync`* (`#warp-level-block-scaling`), §9.7.16.5.14 *mma*.
Computes `D = (A · scale_A) · (B · scale_B) + C`. **The page must be fetched with `curl`;
WebFetch truncates it.**

**Table 45 — valid `.scale_vec_size` × `.kind` combinations (verbatim):**

| `.kind::*` | `.atype`/`.btype` | `.stype` | `.scale_vec_size` |
|---|---|---|---|
| `.kind::mxf8f6f4` | `.e4m3 .e5m2 .e3m2 .e2m3 .e2m1` | `.ue8m0` | `.scale_vec::1X` |
| `.kind::mxf4` | `.e2m1` | `.ue8m0` | `.scale_vec::2X` |
| `.kind::mxf4nvf4` | `.e2m1` | `.ue8m0` | `.scale_vec::2X`, `.scale_vec::4X` |
| `.kind::mxf4nvf4` | `.e2m1` | `.ue4m3` | `.scale_vec::4X` |

Syntax:
```
mma.sync.aligned.m16n8k64.row.col.<kind>.block_scale{.scale_vec_size}.f32.e2m1.e2m1.f32.<stype>
    d, a, b, c, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};
mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.<t>.<t>.f32.ue8m0 ...
```
Defaults: `mxf4`→2X, `mxf8f6f4`→1X; **`mxf4nvf4` requires an explicit `.scale_vec_size`**.
With `scale_vec::4X`, `byte-id-a/b` **must be 0** (all four bytes of the b32 metadata contribute).
`scale_vec::NX` = N scale factors across the atom's K ⇒ at K=64, `2X` = block 32 (MXFP4),
`4X` = block 16 (NVFP4); at K=32, `1X` = block 32. PTX documents `.block32`/`.block16` as
aliases in the tcgen05 section.

**Target/ISA notes (verbatim):**
> `.e3m2`, `.e2m3` and `.e2m1` alternate floating point type mma operation requires **sm_120a**
> and is supported on **sm_120f** from PTX ISA version 8.8.
> Support for `.kind`, `.block_scale`, `.scale_vec_size` qualifier **requires sm_120a** and are
> supported on **sm_120f or higher in the same family** from PTX ISA version 8.8.

`.kind`/`.block_scale`/`.scale_vec_size` introduced in PTX **8.7**;
`.scale_vec::4X` with `.ue8m0` for `.kind::mxf4nvf4` added in PTX **9.1**.
Sparse variant: `mma.sp::ordered_metadata` `.e2m1` with `.ue8m0`/`.ue4m3` at **m16n8k128** (PTX 8.7).

**⚠️ Corollary: warp-level `mma.sync` block-scaling does NOT exist on sm_100a.** Datacenter
Blackwell must use `tcgen05.mma` with Tensor Memory, whose table (§9.7.18.10.7 Table 69) is
*wider*: `mxf4nvf4` accepts UE8M0, **UE4M3 and UE5M3** at both `.block32` and `.block16`, and
K=96 gets 3X/6X layouts (UE5M3 needs sm_107f+; UE4M3 with `.block32` needs sm_107f+).
This is why SageAttention3 and ffpa-attn's fp4 kernels are sm120-shaped and FA4-FP4 is not
portable down to us.

**§5.2.3 alternate-format bit widths (verbatim, note the surprises):**
- `e2m1`: 4-bit, no Inf, no NaN; must be packed as `e2m1x2` in a `.b8`.
- `ue8m0`: "**8-bit** unsigned … 8 bits exponent, 0 mantissa. No infinity. NaN limited to `0xff`."
  Packed `ue8m0x2` in a `.b16`.
- `ue4m3`: "a **7-bit** unsigned floating-point format with 4 bits exponent and 3 mantissa …
  NaN limited to `0x7f`. A register variable containing a single `ue4m3` value must be declared
  with `.b8` type having **MSB padded with zero**." (Colfax calls it "UE4M3 = nonnegative E4M3";
  same max 448, but PTX counts 7 bits.)
- `ue5m3`: 8-bit unsigned, NaN `0xff`.
- Under `kind::mxf8f6f4`, an `.e2m1` operand must be **padded into the central 4 bits of an
  8-bit container**; `kind::mxf4`/`mxf4nvf4` need no padding.

---

## Part 7 — Serving stacks in detail (granularity reference; training is out of scope here)

Kept because it is the only place the *conventions* are pinned down, and because several file
paths in the older literature are now stale.

### FlashInfer
- Legacy per-tensor scales never enter the kernel — folded in Python (`flashinfer/prefill.py`,
  `decode.py`): `if q_scale is not None: sm_scale *= q_scale`, `... *= k_scale`, and
  `out *= v_scale` **after** the kernel. Only valid for per-tensor; this folding trick is *why*
  per-tensor is so entrenched.
- A newer per-head API exists on `single_prefill_with_kv_cache` (`scale_q [num_qo_heads]`,
  `scale_k/scale_v [num_kv_heads]`) but could **not** be found applied in
  `include/flashinfer/attention/prefill.cuh` or `variants.cuh` (FA2 path has only
  `sm_scale_log2`). Issue #742 still tracks "generalize tensorwise qk_scale, v_scale to headwise"
  and "move v_scale application inside the kernel".
- Dequant, verified in `prefill.cuh`: **no fp8 MMA**. Either
  `struct KVRepackSmem { alignas(16) DTypeQ kv_smem_repack[...] }` filled by
  `repack_kv_tile_to_16b` — "dequantize[s] one K or V tile … into the same 16-bit staging layout
  so the QK/PV MMAs can read it with the native 16-bit `ldmatrix` path" — or an in-loop
  `vec_cast<DTypeQ, DTypeKV>::cast<8>(...)` inside `compute_qk`.
- NVFP4 KV: `constexpr uint32_t NVFP4_SF_VEC_SIZE = 16;`. Python `kv_cache_sf` =
  "(k_scales, v_scales) … shape `[num_pages, page_size, num_kv_heads, head_dim // 16]`",
  dtype `float8_e4m3fn` ⇒ **one e4m3 scale per 16 channels, per token, per KV head**, no global
  scale in the KV path. Backends fa2 + trtllm-gen; packed uint8 `head_dim // 2`.
  PR #4769: "FP8 KV already dequantizes a whole tile into the 16-bit staging buffer … NVFP4 was
  excluded from that path and kept dequantizing inside the fully unrolled loop" — pre-SM100 only
  (SM80 1.403 → 0.552 ms). **So NVFP4 KV works on Ampere as a pure memory optimization.**
- True fp4 attention (Q too): `flashinfer/nvfp4_attention_sm120.py`,
  `nvfp4_attention_sm120_quantize_qkv` → packed uint8 fp4 with per-vector e4m3 scales
  `[batch, heads, seq, head_dim/16]`, no global scale, sequences padded to 128, **plus an fp32 QK
  correction term for per-block-mean centering of Q**. Forward only.
- **Open RFC #3628 — directly adjacent to our work**: MXFP8 block-scaled prefill for SM120a,
  "Q/K/V in e4m3 with ue8m0 block-32 scales" and "**P quantized in-kernel to e4m3 + ue8m0 block
  scales between the two matmuls**", via `mma.sync.aligned.kind::mxf8f6f4.block_scale…m16n8k32…ue8m0`
  (SASS `QMMA.SF`) / CUTLASS `SM120_16x8x32_TN_VS`. Motivation: on consumer Blackwell "the
  block-scaled tensor instruction is not throttled" on the FP32-accumulate path. Unclaimed.

### vLLM (main @ `b7e0cdac5d`, 2026-09-13) — many older paths are gone
Stale → current: `vllm/attention/layer.py` → `vllm/model_executor/layers/attention/attention.py`;
`quantization/schema.py` **deleted** (PR #42889); `--quantization-param-path` **gone**;
`csrc/cache_kernels.cu` → `csrc/libtorch_stable/cache_kernels.cu`;
`csrc/attention/paged_attention_v1/v2.cu` **deleted** (PR #47361 "Delete PagedAttention");
`--calculate-kv-scales` **removed** (PR #49389). Source of truth: `vllm/config/cache.py:39`,
20 `CacheDType` entries incl. `nvfp4`, `nvfp4_4over6`, `fp8_ds_mla`, `nvfp4_ds_mla`,
`{int4,int8,fp8}_per_token_head`, `turboquant_*`.

| granularity | modes | evidence |
|---|---|---|
| per-tensor scalar | `fp8`, `fp8_e4m3`, `fp8_e5m2` | `quantization/kv_cache.py`, `KVCacheScaleParameter` is 0-dim; `raise ValueError("Only support per-tensor scaling factor for fp8 KV cache")` |
| per-attention-head static | compressed-tensors `strategy:"attn_head"` | `compressed_tensors.py:1093` `n_scales = int(layer.num_kv_heads) if strategy=="attn_head" else 1`; `supports_per_head_quant_scales()` overridden by `FlashAttentionBackend` **only** |
| per-(token,head) dynamic | `*_per_token_head` | `triton_reshape_and_cache_flash.py:144`, grid `(num_tokens, NUM_KV_HEADS)`, absmax over head_size, **fp32 scale stored inline in the page**, `layer._k_scale` forced to 1.0 |
| per-16-channel | `nvfp4`, `nvfp4_4over6` | `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu:4` `// Per page layout: [K_data | K_scale | V_data | V_scale]`; V scales **swizzled** for SM100 trtllm-gen, **linear on SM12x** |
| per-128-element tile dynamic | `fp8_ds_mla` | `cache_kernels.cu:515` "For the NoPE part, each tile of 128 elements … 4 total tiles", `tile_scale = fmaxf(max_abs/kFp8ScaleDivisor, FLT_MIN)` |

**No per-channel (per-head_dim) KV mode exists.** `nvfp4_ds_mla` (352 B/token):
`[0,256)` 512×e2m1 NoPE packed 2/byte, `[256,320)` 64×e4m3 RoPE **unscaled**,
`[320,352)` 32 e4m3 NoPE SFs "one per 16 elements, permuted",
`sf = e4m3(max(amax_16/6, 2⁻⁹))`.
Live compressed-tensors JSON (verified against `nm-testing/TinyLlama-1.1B-compressed-tensors-kv-cache-scheme`):
`{"num_bits":8,"type":"float","strategy":"tensor","symmetric":true,"dynamic":false,"observer":"minmax","block_structure":null,"group_size":null}`.
`validate_kv_cache_scheme` accepts `strategy ∈ {tensor, attn_head}`, `type=="float"`,
`num_bits==8`; zero-points discarded. llm-compressor stores per-head scales `[num_heads,1,1]`,
`_tp_aware_loader` reduces to `[num_kv_heads]` by max over the GQA group.
**AMD Quark is per-tensor only** (`if qscheme != "per_tensor": raise NotImplementedError`).
Where dequant happens — three coexisting strategies:
- FA3/FA4: **true fp8 MMA**, `flash_attn.py:1038` `key_cache.view(fp8_dtype())`,
  `descale_shape = (cu_seqlens_q.shape[0]-1, self.num_kv_heads)`, `q/k/v_descale` expanded to that
  shape. Gate: "FP8 KV cache requires FA3 on SM90 or FA4 on SM100"; `fp8_e5m2` unsupported.
- FlashInfer/trtllm-gen: host scalars, `flashinfer.py:1992`
  `bmm1_scale = scale*q_scale_float*k_scale_float`, `bmm2_scale = v_scale_float`; per-head arrays
  get `.max().item()`.
- The one real pre-pass dequant: `_trtllm_prefill_attn_kvfp8_dequant` (`flashinfer.py:159`) —
  "TRTLLM prefill attention does not support BF16 Q and fp8 kv cache", materializes a BF16 mock KV
  for prefill pages.
- Triton unified attention `_cast_kv_tile`: per-token-head scales are applied to **the scores**,
  not the tiles — `S += tl.dot(Q,K) * (score_scale * k_token_head_scales[None,:])`,
  `P_v = (P * v_token_head_scales[None,:]).to(V.dtype)`.
FP8 *attention* (Q quantized) is first-class: `attention.py:465` builds `QuantFP8` with
`GroupShape(-1, head_size*num_heads//num_kv_heads)` for per-head else `PER_TENSOR`, applied in
`forward()` as plain torch ops **so `torch.compile` can fuse it into the preceding op**. There is
also a `prob_scale` slot (a scale for `P = softmax(QKᵀ)`) on the ROCm/Quark path.
FP4 KV is merged, not an RFC: #37332 (store kernel), #49818/#49891/#50085/#50288 (SM120/RTX 5090),
#45187 (`nvfp4_4over6`: try both `max/6` and `max/4` per 16-element group, keep the lower
reconstruction error). In flight: #53770 `fp8_k_nvfp4_v` (FP8 K + NVFP4 V = 12.5 bits/pair),
layer-wise mixed KV #56155/#56156/#56116. Open bugs: #55673 (garbage output, trtllm NVFP4 KV on
Blackwell), #50416 (**NVFP4 causal prefill 1.7–1.8× *slower* than FP8 on SM120**),
#56081 (per-token-head inline scales break 128 B alignment → 6.3× slowdown).
Also PR #46329 (open, needs rebase): NVFP4 KV on **SM120/SM121** via the FlashInfer FA2 paged
reader because the TRT-LLM path is SM100-only and Triton cannot read NVFP4. Gemma-4-E2B on
RTX 5090: KV pool **18.1M tokens NVFP4 vs 5.1M BF16 = 3.52×**, needle detection within ±0.3
nats/token of BF16, decode tok/s at parity. Docs (`docs.vllm.ai/…/quantized_kvcache.html`) are
stale — read `vllm/config/cache.py` and `vllm/v1/kv_cache_interface.py`.

### SGLang (main @ `6220f45d8e9a`)
`python/sglang/srt/layers/attention/nsa/*` are now 10-line deprecation shims →
`.../attention/dsa/` + `python/sglang/kernels/ops/attention/dsa/`.
Plain fp8 is per-layer per-tensor with the same guard as vLLM; `MHATokenToKVPool` has **no scale
buffer** (`k_scale_buffer = None`), quantize-on-write is `cache_k.div_(k_scale).to(self.dtype)`.
MLA has two layouts: plain `MLATokenToKVPool` quantizes **all 576 dims incl. RoPE**, but
```python
class DSATokenToKVPool(MLATokenToKVPool):
    quant_block_size = 128
    rope_storage_dtype = torch.bfloat16  # rope is always stored in bf16
```
row = `kv_lora_rank + kv_lora_rank//128*4 + qk_rope_head_dim*2` = 512+16+128 = **656 B/token/layer**,
byte-identical to FlashMLA's V3.2 format. `dsa/quant_k_cache.py`: `y_s = tl.max(tl.abs(y))/448.0`
per GROUP_SIZE=128 ⇒ per-token × per-128-channel fp32 scale, fully dynamic, no calibration.
`flashinfer_mla_backend.py` has **zero fp8 references** (BF16 only in SGLang).
DSA indexer (`dsa/dsa_indexer.py`, `scale_fmt="ue8m0"`, `block_size=128`, `assert index_head_dim==128`):
```python
amax = tl.maximum(tl.max(x_abs, axis=1), 1e-4)
if round_scale: scale = tl.exp2(tl.ceil(tl.log2(amax * (1/448.0))))
y = tl.minimum(tl.maximum(x / scale[:, None], -448.0), 448.0)
```
⇒ K one scale per token, Q one per (token, head). Indexer K row = 128+4 = 132 B/token.
**Nothing dequantizes**: `deep_gemm.fp8_mqa_logits(q_padded, (k_fp8, k_scale), w_padded, ks, ke)`
eats fp8 + scales; the Q scale is never applied to Q, it is folded into the gate
(`weights.unsqueeze(-1) * q_scale * softmax_scale`).
Best fused kernel found anywhere: `flashmla_sparse_q8` — "paged_kv_cache (fp8, 656 B/token) is
gathered, dequantized per group, and requantized to per-tensor fp8 in one fused Triton kernel
(`gather_dequant_requant_fp8_paged`) — no intermediate bf16 materialization", then identity
per-tensor scales (1.0) because "a raw bf16→fp8 cast of q/kv is accurate on real DeepSeek-V3
magnitudes".
FP4/MXFP8 modes: `nvfp4` (`SCALE_BLOCK_SIZE=16`, "two-level scaling: global FP32 + per-block FP8
E4M3", SM100/SM120); `fp4_mx_block16` with a useful in-source warning — "**This is intentionally
not called MXFP4: standard MXFP4 uses a block size of 32, while this KV cache recipe stores one
scale per 16 FP4 values**"; `mxfp8` (FA4 only, `MXFP8_SCALE_BLOCK_SIZE=32`,
`torch.float8_e8m0fnu`, interleaved into FA4's `BlockScaledBasicChunk`); DeepSeek-V4 pools
(`deepseek_v4_memory_pool.py`) main KV `quantize_block_size=64` / 584 B/token, fp4 indexer
`index_head_dim//2 + index_head_dim//32` = 68 B (block-32 MX-style).
Gotcha: on SM100 `k_scale *= E2M1_MAX` (6.0) because "SM100 uses TRT-LLM XQA kernels that expect
KV scales as amax/448, but the calibrated checkpoint stores amax/(6*448)".

### FlashMLA (HEAD `07a1089` "Add kernels for DeepSeek v4.1 (#221)") — the reference implementation
`csrc/kernels/kv_cache_format.h` + `flash_mla/flash_mla_interface.py`:

| model | layout | quant tile | scale dtype | B/token |
|---|---|---|---|---|
| V3.2 | 512 e4m3 NoPE + 4 fp32 scales + 64 **bf16** RoPE | 128 | fp32 (ue8m0 values) | 656 |
| V4 | 448 e4m3 NoPE + 64 bf16 RoPE ‖ 7 scales + 1 pad | 64 | e8m0 | 584 |
| V4.1 | 512 e4m3 (**RoPE quantized too**) ‖ 16 scales | 32 | e8m0 | 528 |
| **V4.1 fp4** | 512 e2m1 packed 2/byte ‖ 32 scales | **16** | **e4m3** | **288** |

V3.2 doc: "The first float32 is the scale for the first 128 float8_e4m3 values … The 'RoPE' part,
containing 64 bfloat16 values. **This part is not quantized for accuracy.**"
V4.1 fp4 doc: "32 Bytes of float8_e4m3 scales, each covering 16 consecutive e2m1 values";
`tests/quant.py`: "The scale is amax / 6 … rounded to e4m3, **without a per-tensor scale**."
The V4.1 fp4 cache is only valid as `extra_k_cache` **alongside** the V4.1 fp8 `k_cache` —
a two-tier fp8/fp4 cache.
Trajectory: tile 128 → 64 → 32 → 16, scale fp32 → ue8m0 → ue8m0 → e4m3, RoPE bf16-protected → quantized.
`docs/20250929-hopper-fp8-sparse-deep-dive.md`: "Inside the kernel, we first dequantize the 512
float8_e4m3 values into 512 bfloat16s. We then concatenate them with the 64 original bfloat16
values from the RoPE part. Finally, we perform the MQA calculation using MMA operations in
bfloat16 precision." **Dequantization-bound: 50 CUDA-core cycles/token vs 34 MMA cycles**, fixed
by "crossover" — clusters of 2 CTAs each dequantize half the KV and `st.async` into the sibling's
smem via DSM: **250 → 410 TFLOPS on H800**. (sm120 has no DSM/clusters.)
SM100 path `csrc/kernels/sm100/dequant_utils.cuh::KVBlockDequantizer`:
```cpp
fp4x8_to_bf16x2x4(data, data_bf16);
ku::nvbf16x2 scale = e4m3x2_to_bf16x2(__byte_perm(scale_word, 0, scale_prmt_sel11));
data_bf16[i] = __hmul2(data_bf16[i], scale);  // Exact: e2m1 x e4m3 has at most 2 + 4 significant bits
```
**No fp8/fp4 backward**: `_flash_attn_varlen_backward` allocates `dq/dk/dv` as `dtype=q.dtype`.

### DeepGEMM — DeepSeek's public fp4 indexer math
`csrc/apis/attention.hpp`: `fp8_fp4_mqa_logits`, `fp8_fp4_sparse_mqa_logits`,
`fp8_fp4_paged_mqa_logits`. When `is_mx_sf`, `q_sf` is `[seq_len, num_heads]` **int32** (4 packed
ue8m0) and `kv_sf` is `[seq_len_kv]` int32. `tests/test_attention.py` calls
`per_token_cast_to_fp4(..., use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)` ⇒ **MXFP4
(block 32, ue8m0), not NVFP4**. For the plain fp8 path `kv_sf` is a single fp32 per token and
there is no Q SF at all.
`DeepSeek-V3.2-Exp/inference/kernel.py::fp8_index_kernel`: `T.gemm(k_smem, q_smem, logits)` on
fp8 smem → fp32 accum, scales applied **after** (valid because ReLU is positive-scale-invariant):
```python
logits[i3_n, i_h] = T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]   # relu, then per-head Q scale·gate
T.reduce_sum(logits, logits_sum, dim=1)
logits_sum[i3_n] *= k_s_frag[i3_n]                                 # per-token K scale, last
```
Config `index_n_heads=64, index_head_dim=128, index_topk=2048, scale_fmt="ue8m0"`;
`fast_hadamard_transform` (`rotate_activation`) applied to Q and K **before** quantization.

### Cross-cutting observations worth keeping
1. Two rival fp4 conventions: NVFP4 (16 + e4m3) in FlashInfer / vLLM / SGLang / DeepSeek's **KV
   cache**; MXFP4 (32 + ue8m0) in DeepSeek's **indexer** and GEMMs, and SGLang's `mxfp8`.
   DeepSeek picked NVFP4-style for KV and MX-style for the indexer — consistent with the indexer
   only needing a top-k *ordering*.
2. **RoPE dims are the canary.** V3.2 and V4 keep them bf16 "as they are sensitive to precision
   loss"; SGLang hard-codes `rope_storage_dtype = bfloat16`; vLLM's `nvfp4_ds_mla` keeps them as
   unscaled e4m3. V4.1 finally quantizes them — after shrinking the tile to 32/16.
3. The scalar-folding trick (`sm_scale *= k_scale`, `out *= v_scale`) is load-bearing and is
   *why* per-tensor persists; anything finer requires kernel surgery.
4. Hadamard-before-quantization is converging independently: DeepSeek's indexer
   (`rotate_activation`), vLLM's `turboquant_*` / int4-per-token-head (RHT), modelopt's
   `kv_nvfp4_rotate`, QuaRot/SpinQuant, FA3's incoherent processing.

---

## Part 8 — Long-context evidence (the question everyone ducks)

**The central distinction**: FP4 *storage* + FP8/BF16 *math* is near-lossless on RULER; uniform
FP4 *math* is not. Every apparent contradiction in this literature is this conflation.

| Work | What | Long-context result |
|---|---|---|
| TensorRT-LLM NVFP4 KV | FP4 storage, **dequant to FP8** before both GEMMs | Qwen3-480B **RULER 64K 95.6 → 94.6** (FP8 95.5); MMLU-PRO 78.2 → 77.4; Llama-3.3-70B MMLU FP8 82.5 / NVFP4 81.9 / **MXFP4 77.8** |
| **ThriftAttention** (2605.23081) | NVFP4 Q/K/P/V math, 5 % FP16 block-pairs | RULER FP16 / FP4 / Thrift: Llama3.2-3B 0.728 / **0.321** / 0.543 · Llama3.1-8B 0.859 / **0.385** / 0.770 · Qwen3-4B 0.795 / 0.445 / 0.749 · Qwen3-8B 0.840 / 0.484 / 0.806 · Ministral-3-8B 0.848 / 0.548 / 0.866. LongBench-v1 and HELMET collapse similarly (~0.26 → ~0.13). Prefill 1.7× kernel / ~1.2× e2e at 131K; decode 3–5.5× / ~2×. Failure modes: "FP4 quality degrades relative to FP16 as sequence length increases"; PG-19 ΔNLL 0.04 short → **0.10 at 128k**; positional dependence emerges past 32k, worst at the *end* of the sequence; error concentrates on high-magnitude scores. Thrift → ΔNLL ≤ 0.02 |
| DMA (2604.03950) | MXFP4 off-diagonal + MXFP8 diagonal-128 + sink-128, **V/PV FP16** | LongBench Llama-3.1-8B 44.11 → **46.43**; Llama-3.2-3B 35.84 → 37.20. QK cos-sim 0.714 → 0.988. B200 latency 7.776 ms vs 12.980 MXFP4 / **13.404 NVFP4** / 16.771 MXFP8 (diag window 128→256 doubles to 15.720) |
| SageAttention2 | INT4 QK / FP8 PV | Longbench, **InfiniBench and NIAH up to 262k** on Llama-3-262k — the only quantized-attention paper with a clean long-context pass |
| arXiv 2609.04263 (kvlora) | trained fake-quant KV + LoRA | RULER 4K/8K, NIAH, 180-case retrieval. **2-bit ppl 576.10 → 11.40 (float 10.40) restores only 11–12 of 180 retrieval cases** |
| Block-GTQ (2606.24033) | integer 1–8 bits per RoPE 2-D frequency block for K, per-head TQ-MSE for V | Llama-3.1-8B **NIAH 70.6 → 97.4** (K2V2); **LongBench-EN 36.87 → 53.31**; R1-Distill-Qwen-7B K3V2 AIME24/25 51.7/37.5 vs fp16 54.2/37.9 vs TQ-MSE **0.0/0.0**; H800 1.34× vs fp16 FA2 at 128K, 56.31 → 19.85 GB |
| DGAP (2607.16248) | local distribution restoration; diagnoses "structured local misranking" in the top-K logit region | Llama-3.1-8B **K1V1 RULER 47.8 % → 83.2 %** |
| Minima-KV (2608.23834) | FP8 anchor/recent pages + TQ3 older pages | Qwen3.6-27B on one 96 GB RTX PRO 6000: 18.3 KiB/token = 3.50× BF16, 1.75× FP8; matches dense on 16K RULER NIAH; LongBench-v2 −0.80/−0.60/−0.40 pp at 16K/32K/64K; 0.982× throughput |
| HQMQ (2605.27646) | Hurwitz-quaternion multiplicative, 4-element chunks | **naive int4 collapses on all three RULER subtasks at 4k and 8k**; SQuAD int4 0.43 (4k) → 0.16 (8k) vs fp16 0.73 → 0.60 (deficit 42 % → 73 %). HQMQ s96_r6 @ 4.89 bits within 2 pts of fp16 |
| Alignment Collapse (2606.09864) | KV quantization vs safety | **"Mistral-7B loses 15.2 % of its refusals at only 1.03× perplexity."** Per-Channel Reduction recovers up to 97 % (AdvBench 97.8 %, L0–L1 in FP16, 7 % memory overhead) |
| Runtime-Certified Bounded-Error (2605.20868) | INT8 K / INT4 V + per-head per-step error bounds + fallback | PG-19, NIAH, RULER, Llama-3.1-8B to 128K, "comparable to FP16" — but keeps an FP16 shadow copy in system RAM, weakening the memory claim |
| SAW-INT4 (2604.19157) | asymmetric INT4, token-wise + per-head, **block-diagonal Hadamard (block 128)**; rotating K alone suffices | ⚠️ **no RULER/LongBench/NIAH despite being a "real-world serving" paper**, 32k max. Qwen3-4B-Thinking mean BF16 75.64 / **naive INT4 0.00** / BDR-128 73.11. GLM-4.7-FP8 (358B) 77.95 (−0.04). 2×H100, 32-concurrent TPS/GPU 1030.5 → 1242.2 |
| KVQuant (2401.18079) | pre-RoPE per-channel K, per-token V, nuq 2–4 bit, **token 0 in FP16** | 1M ctx on 1×A100-80GB, 10M on 8 GPUs; <0.1 PPL at 3-bit; ~1.7× |
| KIVI (2402.02750) | per-channel K, per-token V, 2-bit + FP16 recent window | 2.6× memory, 4× batch, 2.35–3.47× throughput |
| SKVQ (2405.06219) | 2-bit K / 1.5-bit V, channel reorder + clipped dynamic group + FP16 sliding window | 1M ctx on 80 GB for 7B, 7× faster decode |
| QServe/QoQ (2405.04532), Atom (2310.19102), QuaRot (2404.00456), SpinQuant (2405.16406) | W4A8KV4 / 4-bit W-A-KV | short tasks only |

**Reporting only perplexity or short tasks** (useful negatives — do not go looking for numbers
that are not there): SageAttention3, Attn-QAT, ScaleSearch (even though it quantizes KV too),
SAW-INT4, HiFA4 (MMLU only), Hardware-Aware FP4 FA4, Full-Stack FP4, FAAR (2603.22370),
ReSET (2606.13233), int4-on-Apple-Silicon (2605.05699, ≤4096 tokens), the NVIDIA forum posts,
and the llama.cpp (#26989) / SGLang (#10083) issues.

**Recurring failure modes to design against**: e2m1 has only 15 values; P ∈ [0,1] under-uses the
e4m3 scale range (hence ×448×6); softmax-normalizer vs PV precision mismatch (HiFA4 measures a
median probability-mass loss ε̄ = −0.064 across 3.6M tiles when the normalizer is not taken from
the *same* quantized P̂); error grows with sequence length **and** concentrates on the highest
attention scores — exactly what NIAH/RULER measure; attention sinks (everyone protects token 0 /
first block / first pages — we have a *learnable* sink, which is worse-understood); per-token
absmax collapse from one dominant coordinate (Qwen ΔPPL +7975 → +638.6 once scales go
per-channel); and RoPE interaction (quantize K pre-RoPE, smooth Q/K post-RoPE, or allocate bits
per RoPE frequency block).

Other FP4-attention entries not already covered above:
- **ScaleSearch** (2605.12464, Gupta/…/Tri Dao/Daniel Fu/De Sa): the only work doing full NVFP4
  Q/K/P/V **and** NVFP4 KV storage (~4.5 effective bits), with the **first KV block and the
  trailing partial block kept full precision** (explicit sink protection). WikiText-2 PPL
  Llama-3.1-70B 2.6348 (FP 2.5554, naive FP4 3.4); GPQA-Diamond Llama-3.1-8B-Instruct 32.32 vs
  SageAttention3 26.26; MATH500 Qwen3-8B +15 pts. No RULER/LongBench/NIAH.
- **HiFA4** (2607.04302, Ascend HIF4 NPUs, *not* NVIDIA, projections not hardware-validated):
  both GEMMs 4-bit, softmax state FP16; *Smooth-QK* rescales **post-RoPE**; *P-Reordering* makes
  the normalizer come from the same quantized P̂ the PV GEMM consumes. Qwen3-8B BF16-inconsistent
  MMLU predictions 16.3 % → 8.2 %; projected 35.4 % latency reduction.
- **FlashAttention-4 proper** (2603.05451, Zadouri/Hoehnerbach/Shah/Liu/Thakkar/Dao) is **BF16
  only** — 1613 TFLOPs/s (71 % util) on B200. FP4 exists only in the hao-ai-lab fork.
- **Attn-QAT accuracy numbers**: Qwen3-14B MMLU BF16 0.8044 / training-free FP4 0.7965 /
  QAT 0.7984; Llama-3.1-70B MMLU 0.7881 / 0.7577 / 0.7773, WinoGrande 0.8161 / 0.7656 / 0.7940;
  Wan 2.1 14B VBench 0.8335 / 0.8203 / 0.8279.
- **flash-attention-fp4 kernel matrix**: NVFP4-QK+FP8-PV, NVFP4-QK+BF16-PV, MXFP8-QK+FP8-PV,
  MXFP8-QK+BF16-PV. GB300 (SM103) peak **2677 TFLOPS** (NVFP4+FP8) = 1.75× BF16 FA4;
  B200 (SM100) 2018 TFLOPS.

---

## Part 9 — Search log

Queries run and what each yielded. (`WS` = WebSearch, `WF` = WebFetch.)

1. `WS "SageAttention3 FP4 microscaling attention Blackwell paper"` → arXiv 2505.11594 + the
   `sageattention3_blackwell` repo directory. Good entry point.
2. `WF arxiv.org/abs/2505.11594` → abstract only, useless for detail. **Lesson: always use
   `ar5iv.labs.arxiv.org/html/<id>` or `arxiv.org/html/<id>v<n>`.**
3. `WF ar5iv…/2505.11594` → the two-level P formula, NVFP4-vs-MXFP4 cos-sim, SageBwd matmul table.
   Jackpot.
4. `WS "FlashAttention-3 FP8 incoherent processing Hadamard block quantization"` → the RMSE
   ablation (2.4e-2 → 9.3e-3 → 9.1e-3) and the per-block granularity.
5. `WF arxiv.org/pdf/2407.08608` → **failed** (PDF binary, no text layer). `WF ar5iv…/2407.08608`
   worked.
6. `WS "SageAttention2++ per-thread quantization INT8 FP8 accumulator smoothing K mean"` →
   2411.10958, 2505.21136.
7. `WF arxiv.org/pdf/2411.10958v6` → **failed** (`maxContentLength exceeded`).
   `WF ar5iv…/2411.10958` gave the `δ_P = 1/448` quote, per-channel V, FP22 accumulator, and the
   thread-grouping indices.
8. `WS "FP8 attention training backward pass 2026 arxiv fp8 flash attention backward dQ dK dV"` →
   **the single most productive query**: surfaced FP4-FlashAttention-4 (2609.04105),
   Attn-QAT (2603.00040), and "Why Low-Precision Transformer Training Fails" (2510.04212).
9. `WF arxiv.org/html/2609.04105v1` → the MXFP4-P/V divergence result, e5m2 dO, training precision table.
10. `WF arxiv.org/pdf/2603.00040` → **failed** (too large). `WF arxiv.org/abs/…` then
    `WF arxiv.org/html/2603.00040v3` → the full Alg. 2 / Alg. 3 and the two fixes.
11. `WF arxiv.org/abs/2510.04212` then `…/html/2510.04212v1` → the `P̄ = 1.0` biased-rounding
    mechanism and the β > 1 normalizer fix.
12. `WS "NVIDIA TransformerEngine FP8 attention backward supported cudnn fused attention"` →
    `NVTE_FP8_DPA_BWD=1`, the fp8-attention limitations (no THD/bias/SWA).
13. `WF docs.nvidia.com/…/cudnn/frontend/v1.9.0/operations/Attention.html` → the complete
    per-tensor `(1,1,1,1)` scale/descale/amax tensor list for `sdpa_fp8` and `sdpa_fp8_backward`.
14. `WS "DeepSeek-V3.2-Exp lightning indexer FP8 TileLang kernel training precision bf16"` →
    `act_quant` 128-element blocks, the "FP8/BF16 training supported" framing.
15. `WS "DeepSeek-V4 tech report indexer FP4 quantization aware training MXFP4 keys queries"` →
    arXiv 2606.19348, the "entirely in FP4" / 99.7 % recall / 2× top-k claims, and the
    FP32-master → FP4 → FP8 STE description.
16. `WF arxiv.org/pdf/2606.19348` and `…/html/2606.19348v1` → **both largely failed** to surface
    the FP4-QAT section; those details came from search summaries and should be re-verified
    against the PDF text before being quoted verbatim.
17. `Bash grep` over `archs/dsv4/paper/*.md` → §2.4.4 and §1 quotes, and confirmation that the
    CSA2 / indexer sections say nothing about precision.
18. `Bash grep` over `modeling_deepseek_v41.py` → `_pow2_ceil_scale`, `_e2m1_codes`, the three
    call sites, and the `_FP4_TABLE` grid.
19. `WS '"MXFP4" OR "NVFP4" KV cache quantization aware training long context RULER 2026'` →
    the 2026 KV-cache cluster (MixKVQ, Block-GTQ, HQMQ, Kitty, EchoKV, KV-Pareto).
20. `WS "quantization-aware training KV cache straight-through estimator … 4-bit 2026"` → mostly
    PTQ; pointed at LLM-QAT as the canonical QAT-with-KV reference.
21. `WS 'flash-attention hopper fp8 "not supported" backward'` + `WF github.com/Dao-AILab/flash-attention`
    → README line "FP16 / BF16 forward and backward, FP8 forward"; FA4 exists, no FP4.
22. `WS "SageAttention SageBwd INT8 training backward … sm89"` → **SageBwd has its own paper**,
    arXiv 2603.02170. `WF arxiv.org/html/2603.02170.pdf` → the 7-matmul precision table and the
    `dP = dO·Vᵀ` FP16 exception.
23. `WS '"needle in a haystack" OR LongBench … quantized attention'` → mostly SpargeAttention
    (sparse, different paper). The SageAttention2 long-context evals had to be read from its own
    paper (`WF ar5iv…/2411.10958`, second pass): Longbench + InfiniBench + NIAH @262k.
24. `WS "GB10 DGX Spark FP4 TFLOPS dense FP8 BF16 tensor core specs sm121"` +
    `WF forums.developer.nvidia.com/t/…/360142` → the "FP4 is compression, not compute" result,
    the three architectural reasons, and the `sm_121a`/`sm_121f` confusion.
25. `WF arxiv.org/abs/2509.25149` → abstract only (RHT, stochastic rounding, 12B/10T). The
    Appendix-B formulas came from the delegated formats agent.
26. `WS '"HISA" OR "dsa_hisa" tilelang block_sparse_mqa_fp8'` → `lemyx/tilelang-dsa`
    (bf16 indexer *training* operator), cuDNN's DSA module, HISA (2603.28458).
27. `WF github.com/lemyx/tilelang-dsa` → "we chose to use bf16 … later support the fp8 data type".
28. `WF arxiv.org/abs/2506.20752` → MX training instabilities (layer-norm affine + a small
    fraction of activations; mid-training precision switching as the mitigation).
29. `WS '"Practical FP4 Training" MoE Hopper 2603.02731'` → MXFP4 for activations/comms with FP8
    compute; same "FP4 for bytes, FP8 for math" shape as our KV plan.
30. `WF docs.nvidia.com/cuda/parallel-thread-execution/index.html` → **failed** (ToC only).
    The delegated agent got Table 45 by `curl`-ing the 3.9 MB page.
31. Delegated agent *ffpa* → the complete ffpa-attn fp8/fp4 design, the PTX strings actually
    issued on sm120, the ESS error model, the V-quant-dominates finding, and the split-D/smem
    numbers.
32. Delegated agent *kvcache* → DeepSeek-V3's "attention operators stay BF16" quote, TileKernels'
    E5M6 kernel, TE/cuDNN's fp8 bwd internals (S=e4m3, dP=e5m2, forced DelayedScaling), the MXFP8
    fixed-256 P scale, FlashMLA's four KV formats and dequant-bound finding, DeepGEMM's
    `fp8_fp4_mqa_logits`, and the complete vLLM/SGLang/FlashInfer granularity survey plus the
    KV-QAT landscape (modelopt, ReQAT, kvlora) and the KIVI/llm-compressor/torchao traps.
33. Delegated agent *sm120kernels* → SageAttention3 builds for sm_121a; its MMA atom and SF
    layout; TileLang's `T.mma_gemm_blockscaled` ue4m3-only `static_assert`; Attn-QAT's real
    Triton training kernel (`JOIN_QAT_PV`, `use_high_prec_o`, the P-quantization asymmetry);
    `flash_bwd_sm120.py` being BF16-only.
34. Delegated agent *fp4fmt* → the OCP §6.3 rule and Tables 1/5/7, NVFP4 Appendix B, PTX Table 45
    with target notes, FP8 maxima, and the 2026 long-context corpus including ThriftAttention's
    RULER sweep and the TRT-LLM NVFP4-KV blog.

Not searched (deliberately out of scope): decode/paging performance, INT8 weight-only
quantization, sparse-attention selection quality, anything about MoE expert-weight FP4.

---

## Part 10 — URL addendum (sources added after Part 5 was written)

Specs / ISA
- OCP MX v1.0 spec (403s to non-browsers) https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf — use https://web.archive.org/web/2024/https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- PTX ISA 9.4 §9.7.16.3 https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-block-scaling and §9.7.16.5.14 `#warp-level-matrix-instructions-mma` (fetch with curl, not WebFetch)
- FP8 formats https://arxiv.org/abs/2209.05433
- NVFP4 blog https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- NVFP4 KV cache blog (dequant to FP8, RULER 64K) https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/
- cuDNN MXFP8 attention scaling (fixed P scale 256) https://nvidia.github.io/cudnn-frontend/mxfp8-attention-scaling/
- cuDNN DSA https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/dsa.html
- TensorRT-LLM precision matrix (no NVFP4 KV listed) https://nvidia.github.io/TensorRT-LLM/reference/precision.html

FP4 attention
- ThriftAttention https://arxiv.org/abs/2605.23081
- DMA (diagonal-tiled mixed precision) https://arxiv.org/abs/2604.03950
- ScaleSearch https://arxiv.org/abs/2605.12464
- HiFA4 (Ascend) https://arxiv.org/abs/2607.04302
- FlashAttention-4 proper (BF16 only) https://arxiv.org/abs/2603.05451
- Full-Stack FP4 https://arxiv.org/abs/2607.04422
- FAAR https://arxiv.org/abs/2603.22370 · ReSET https://arxiv.org/abs/2606.13233
- MXFP4 pretraining on native FP4 hw https://arxiv.org/abs/2605.09825 · MX+ (MICRO 58) https://arxiv.org/abs/2510.14557

Low-bit KV cache with long-context evals
- Block-GTQ / RoPE-aware bit allocation https://arxiv.org/abs/2606.24033
- DGAP https://arxiv.org/abs/2607.16248
- Minima-KV https://arxiv.org/abs/2608.23834
- HQMQ https://arxiv.org/abs/2605.27646
- Alignment collapse under KV quantization https://arxiv.org/abs/2606.09864
- Runtime-certified bounded-error quantized attention https://arxiv.org/abs/2605.20868
- SAW-INT4 https://arxiv.org/abs/2604.19157
- MixKVQ https://arxiv.org/abs/2512.19206 · int4 on Apple Silicon https://arxiv.org/abs/2605.05699

Baselines
- KVQuant https://arxiv.org/abs/2401.18079 · KIVI https://arxiv.org/abs/2402.02750 · QServe/QoQ https://arxiv.org/abs/2405.04532 · Atom https://arxiv.org/abs/2310.19102 · SKVQ https://arxiv.org/abs/2405.06219 · QuaRot https://arxiv.org/abs/2404.00456 · SpinQuant https://arxiv.org/abs/2405.16406 · OScaR https://arxiv.org/abs/2605.19660 · RotateKV https://arxiv.org/abs/2501.16383
- Unverified IDs seen only in listings (treat as unconfirmed): ZipCache, GEAR, MiniKV 2411.18077, PM-KVQ 2505.18610, Kitty 2511.18643, VQKV 2603.16435, RDKV 2605.08317, KVzap 2601.07891, EchoKV 2603.22910, KV-Pareto 2512.01953, A2ATS 2502.12665, OSCAR 2605.17757

Issues / PRs worth watching
- FlashInfer MXFP8 SM120a prefill RFC (unclaimed, P requantized in-kernel) https://github.com/flashinfer-ai/flashinfer/issues/3628
- FlashInfer headwise-scale tracking https://github.com/flashinfer-ai/flashinfer/issues/742
- FlashInfer NVFP4 tile-dequant PR https://github.com/flashinfer-ai/flashinfer/pull/4769
- vLLM NVFP4 KV on SM120/121 https://github.com/vllm-project/vllm/pull/46329 · NVFP4 causal prefill slower than FP8 on SM120 https://github.com/vllm-project/vllm/issues/50416
- NVFP4 vs FP8 KV on RTX PRO 6000 / DGX Spark (capacity only, no accuracy) https://forums.developer.nvidia.com/t/nvfp4-vs-fp8-kv-cache-on-rtx-pro-6000-blackwell-and-dgx-spark/377425
