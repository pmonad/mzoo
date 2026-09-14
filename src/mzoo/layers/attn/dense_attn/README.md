# dense_attn

Dense FlashAttention-2, the baseline package of the CSA2 series (see
`../csa2_attn_design.md`). No MQA, no windows, no sparsity.

Source: `tile-ai/tilelang@v0.1.14`
`examples/flash_attention/example_mha_fwd_bshd.py` and
`example_mha_bwd_bshd.py`; sink term from `examples/attention_sink`.

## What it is

- **Forward** (`fwd.py`): FA2 forward, bf16 in/out, fp32 accumulate, BSHD
  layout, optional `causal` flag, fp32 `lse [B, H, S]` output.
- **Sink** (optional, `sinks [H]` fp32): one extra softmax column that only
  enlarges the denominator, never the numerator; `lse` includes it, so the
  backward's recomputed P is automatically sink-normalised. `has_sink` is a
  static flag, so the sink-free kernel is byte-for-byte unchanged.
- **Backward** (`bwd.py`): three kernels, exactly the upstream shape --
  `preprocess` (`delta = rowsum(dO * O)`), the KV-block-owning main kernel
  (accumulates `dK`/`dV` in registers, scatters `dQ` via fp32 `atomic_add`),
  and `postprocess` (cast the fp32 `dQ` buffer to bf16). Sink gradient is a
  plain fp32 torch reduction, no kernel:
  `dsinks[h] = -sum_{b,s} exp(sink[h] - lse[b,h,s]) * delta[b,h,s]`.
- **Autograd wrapper** (`attn.py`): saves exactly `(q, k, v, o, lse[, sinks])`
  and recomputes S/P in the backward, so saved memory is O(B*S*H*D), never
  O(S^2) -- pinned by `test_backward_saves_only_qkvo_lse`.

## How to run

```
just src/mzoo/layers/attn/ test
just src/mzoo/layers/attn/ bench --dim 128 --mode bwd
```

(The trailing slash on the path is required by `just` for this recipe form.)
`bench` always runs with a random per-head sink; the `sdpa` row is the
sink-free baseline.

## Tile configs

`block_M`/`block_N` at D=256 are shape-support only: tilelang's layout
inference on this stack rejects `block_M=32` (bwd) and `block_M=64` with
`threads=256` (fwd and bwd) -- see `tickets/0001-tilelang-issues.md`. D=256
bwd is stuck at `block_M=64, block_N=16, threads=128` (99 KB smem is the
next config), which runs ~13 ms at B1 H16 S4096 causal. **D=64 and D=128 are
the tuned main paths; D=96 and D=256 are shape-support only.**

fwd `CONFIGS` (head dim -> block_M, block_N, num_stages, threads):

| dim | block_M | block_N | num_stages | threads |
|---|---|---|---|---|
| 64 | 128 | 128 | 2 | 256 |
| 96 | 128 | 128 | 1 | 256 |
| 128 | 128 | 128 | 1 | 256 |
| 256 | 64 | 64 | 1 | 128 |

bwd `CONFIGS` (head dim -> block_M = KV rows, block_N = query rows, num_stages, threads):

| dim | block_M | block_N | num_stages | threads |
|---|---|---|---|---|
| 64 | 128 | 64 | 2 | 256 |
| 96 | 128 | 32 | 2 | 256 |
| 128 | 128 | 32 | 1 | 256 |
| 256 | 64 | 16 | 1 | 128 |

## Bench

GB10, B=1 H=16 S=4096 causal, bf16. `sdpa` is torch's default backend,
sink-free.

Forward only:

| dim | dense_attn | sdpa |
|---|---|---|
| 64 | 0.46 ms | 0.48 ms |
| 96 | 0.64 ms | 0.69 ms |
| 128 | 0.81 ms (85 TFLOPS) | 0.91 ms |
| 256 | 1.71 ms | 1.73 ms |

With a sink at D=128: 0.81 ms (about 1% cost).

Forward + backward:

| dim | dense_attn | sdpa |
|---|---|---|
| 64 | 1.83 ms | 1.76 ms |
| 128 | 4.20 ms | 3.52 ms |

Sink cost in bwd is unmeasurable. The bwd gap to sdpa is the atomic-dQ
design, not tile tuning -- a full config sweep was done at D=64/128.

## Accuracy

Errors are exactly bf16 ulps of the output (7.8e-3 / 1.56e-2 at magnitudes
2-4 / 4-8); `lse` (fp32) is within 1.4e-6.

Ratio of our max error to torch's own bf16 kernel's max error, both against
an fp32 reference (causal, B2 H4 S512; the acceptance criterion is <= 2x):

| dim | o | dq | dk | dv | dsinks |
|---|---|---|---|---|---|
| 64 | 1.00 | 1.00 | 1.00 | 1.00 | 1.46 |
| 96 | 1.00 | 1.32 | 0.98 | 1.00 | -- |
| 128 | 1.00 | 0.76 | 0.93 | 1.00 | 1.21 |
| 256 | 1.00 | 1.00 | 1.00 | 1.00 | -- |

## Decisions

- Fast-math (`TL_ENABLE_FAST_MATH`) kept: FA1-FA4 all build with fast math
  (`Dao-AILab/flash-attention` `setup.py` and `hopper/setup.py` use
  `--use_fast_math`; `flash_attn/cute/softmax.py` uses `exp2(fastmath=True)`).
- No gradient-checkpoint guard -- only the save/recompute invariant is
  tested, not interaction with `torch.utils.checkpoint`.
- `seq_len` must be a multiple of `block_M` (asserted).
- Fixed sequence length assumed; no varlen support.

## Known issues

- D=256 bwd is shape-support only: ~13 ms at B1 H16 S4096 causal vs ~2.9 ms
  for D=128, forced onto `block_M=64, block_N=16, num_stages=1, threads=128`
  by the tilelang layout-inference limits (`block_M=32` fails, `block_M=64`
  needs `threads=128`) plus the 99 KB smem cap (`block_N=32` would need
  ~103.5 KB).
- D=256 fwd is limited to 64x64 tiles, 1 stage, 128 threads by the same
  smem cap.
- `block_M=64` + `threads=256` fails layout inference in both fwd and bwd
  -- see `tickets/0001-tilelang-issues.md` for the details.
- `seq_len` must be a multiple of `block_M` (128 for D<=128, 64 for D=256).
- bwd is ~1.05x (D=64) / ~1.2x (D=128) slower than sdpa fwd+bwd, because of
  the atomic-dQ design.

## Next

- `bwd.py` is ~200 lines; it will be split when copied forward into the next
  package.
- Next package: `latent_attn` -- shared K==V latent, heads packed into tile
  rows, dK/dV reduced across the heads sharing a latent. Because K==V,
  `dKV = dK + dV`; use fp32 split buffers, not bf16 atomics. Tiles get
  re-tuned from scratch.
