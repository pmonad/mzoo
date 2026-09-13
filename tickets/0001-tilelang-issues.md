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

## misc

- default `/usr/bin/nvcc` is CUDA 12.0 and rejects `sm_121a`; needs `CUDA_HOME=/usr/local/cuda-13.0`.
