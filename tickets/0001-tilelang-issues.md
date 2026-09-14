# tilelang issues

## sm121 (GB10) blocked from SM120 NVFP4 block-scaled MMA

- version: tilelang 0.1.14 (also unfixed on `main`; header last touched in 8f34abf, #2364)
- symptom: `T.mma_gemm_blockscaled` compiles for `sm_121a` but asserts at runtime:
  `tl::sm120_mma_sync_blockscaled requires sm_120a and CUDA 12.8 or later`
- cause: `src/tl_templates/cuda/instruction/mma_block_scale.h:42` requires `CUTLASS_ARCH_MMA_SM120A_ENABLED`.
  Everything else already allows sm121: `TargetIsSM120` (arch 120..129), and CUTLASS `cute/arch/config.hpp:168` enables
  `CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED` for SM121/SM121A (CUDA >= 12.9).
- proposed upstream fix: gate on `CUTE_ARCH_MMA_SM120_ENABLED` (covers sm120/120a/121/121a) instead of `CUTLASS_ARCH_MMA_SM120A_ENABLED`.
- workaround (in `src/mzoo/kernels/sm120_nvfp4_blockscaled_gemm.py`):
  `@tilelang.jit(compile_flags=["-DCUTLASS_ARCH_MMA_SM120A_ENABLED=1"])`.
  Blunt: also turns on other SM120A-gated CUTLASS features. Remove once fixed upstream.
- not an option: forcing target `sm_120a` (`TILELANG_DEFAULT_TARGET`) -> `CUDA_ERROR_NO_BINARY_FOR_GPU` on sm121.
- status: PR not raised yet.

## FA2 fwd: `block_M=64` + `threads=256` fails layout inference

- version: tilelang 0.1.14, GB10 (sm121), `src/mzoo/layers/attn/dense_attn/fwd.py`
- symptom: `Layout infer conflict between acc_s and acc_s_cast in T.Parallel loop` at compile time,
  for any `block_M=64` with `threads=256` (all of N=64/128, stages=1/2). `threads=128` is fine.
- impact: `D=256` must use `block_M=block_N=64, num_stages=1, threads=128` (the only config that fits
  99 KB smem), so it cannot use 256 threads. Not blocking, just caps the `D=256` config space.
- workaround: per-dim config table; 128 threads whenever `block_M=64`.

## misc

- default `/usr/bin/nvcc` is CUDA 12.0 and rejects `sm_121a`; needs `CUDA_HOME=/usr/local/cuda-13.0`.

## FA2 bwd: `block_M=32` (any threads) and `block_M=64` + `threads=256` fail layout inference

- version: tilelang 0.1.14, GB10 (sm121), `src/mzoo/layers/attn/dense_attn/bwd.py`
  (port of `examples/flash_attention/example_mha_bwd_bshd.py`; `block_M` = KV rows, `block_N` = query rows)
- symptom: `Layout infer conflict between qkT and qkT_cast in T.Parallel loop` at compile time for
  every `block_M=32` config, and for `block_M=64` with `threads=256`. Same shape of bug as the fwd
  entry above (fp32 accumulator + its bf16 cast in one `T.Parallel`), just on the transposed tile.
- impact: search space is effectively `block_M in {64 @ 128 threads, 128 @ 128/256 threads}`.
  At `D=256` that leaves only `block_M=64, block_N=16, num_stages=1, threads=128` inside the 99 KB
  smem budget (`block_M=64, block_N=32` needs 103552 B, ~2 KB over), which costs ~13 ms at
  B1 H16 S4096 causal vs ~2.9 ms for `D=128`. `D=256` is a shape-support path here, not a fast one.
- workaround: per-dim config table (`CONFIGS`), as in the fwd.
- also seen: `block_N=16` + `threads=256` at `D=96` -> `No valid warp partition for T.gemm: M=16, N=96
  cannot be evenly covered` (8 warps cannot tile a 16x96 output); not a bug, just a constraint.
