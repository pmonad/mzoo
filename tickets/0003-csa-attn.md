# 0003 csa_attn: dense compressed KV as a second source (fwd + bwd)

- status: done (2026-09-14): dense compressed KV as a second source in one online-softmax loop, fwd + bwd, `swa_attn`'s configs unchanged (untuned), 39 GPU tests green incl. a dense-compressed model replay; the main loop had to go `T.serial` around a tilelang pipeline miscompile.
- depends on: 0002 (`swa_attn`, done and independently verified 2026-09-14)
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

## Result

**Package** `src/mzoo/layers/attn/csa_attn/`: `fwd.py` (272), `bwd.py` (124),
`bwd_kernels.py` (148, byte-for-byte `swa_attn`'s `preprocess` + window `dKV`),
`bwd_main.py` (101, the new group-index KV-owner), `bwd_dq.py` (115), `attn.py` (50),
`ref.py` (44), `bench.py` (88), `fwd_test.py` (139), `attn_test.py` (161),
`model_test.py` (79), `README.md`. `bench-csa` added to
`src/mzoo/layers/attn/justfile`; `just smoke` green.

**Configs: untuned (inherited from `swa_attn`, nothing shrunk).** No sweep was run
(user decision: correctness first). All four forward configs and both backward tiles
compiled and passed with the second source added; `bwd_main` reuses the backward `kv`
tile. fwd `block_M/block_N/stages/threads`: D64 256/64/3/256, D96 256/64/3/256,
D128 256/32/3/256, D256 128/32/2/256. bwd `kv`: D64 64/128/2/128, D96 64/64/3/128,
D128 64/64/2/128, D256 64/64/1/128; bwd `dq`: D64 128/64/3/256, D96 128/64/3/256,
D128 128/64/2/256, D256 64/32/2/128.

**Bench headline** (B1 S4096 W128, vs `swa_attn` on the same inputs, sole baseline --
no SDPA row, per the user's rule about baselines not built for the task). At the
model's shape D=128 H=64 `ratio=4` (G=1024): forward 1.226 ms vs `swa_attn`'s 0.628
(**+95% for the second source**, for 9.4x more attended entries); fwd+bwd 8.397 ms vs
3.110 (**+170%**). At `ratio=1` (G=S=4096) the forward is 3.444 ms (+448%) and hits
**84.8 TFLOPS** against the band-only 26.9 -- per entry the compressed source is the
cheaper of the two, because it is a run of fully-visible tiles with no per-row edge.
The backward scales worse than the forward because `bwd_main`'s query loop has no
right edge (`O(G*S)` work vs the band's `O(S*window)`). GPU shared with the `indexer/`
worker during the run, so treat the ms as +-a few percent.

**Accuracy** (2x-torch criterion vs `golden(level="compressed")`, B2 S512, per-head
sink, `ratio in {1,2,4}` x D in {64,96,128,256} x H in {4,16,64}): max ratios
`o` 0.79, `dq` 1.18, `dkv` 0.97, `dmain_kv` 1.00, `lse` 1.4e-6 absolute. `dsinks`
peaks at 2.16 (D=64 H=16 ratio=1) -- the same statistically thin reduce `swa_attn`
already documents at D=256, not introduced here. Cross-checks: `G == 0` is
**bit-for-bit** `swa_attn` (o, lse, dq, dkv, with and without a sink); `ratio=1` with
`main_kv = kv` is the ticket's uncompressed special case; and `model_test.py` replays
the vendored model's dense-compressed layer.

**Deviations from the ticket**

- **The `ratio=1` equality test is against `golden`, not against `latent_attn`.** The
  ticket's "`ratio == 1` with an empty window" is not a callable shape (`window >= 1`
  is asserted, inherited from 0002). The two faithful readings are both covered
  instead: `ratio=1` with `main_kv = kv` (the same latent attended twice, which
  `golden_ref_test.py` hand-derives in closed form), and `G == 0` reducing to
  `swa_attn` bit-for-bit.
- **The model replay needed a config change.** The tiny smoke config's
  `compress_ratio=2` layer is *sparse* (`index_topk=64 < compressed_len=96`), so it is
  0004's shape, not this one. `model_test.py` builds the same config with
  `index_topk=4096`, which makes the indexer select every visible group, and asserts
  the captured `topk_bias` really equals `g < (t+1)//ratio` before replaying.
  (`index_source_layer_ids=[]` does not work: with no index source the model masks
  every compressed entry off.)
- **No SDPA column in the bench**, per the user's mid-ticket instruction and the new
  `CLAUDE.md` rule.
- **The main loop is `T.serial`, not `T.Pipelined`** -- a tilelang 0.1.14 miscompile,
  not a choice. See below.

**Problems parked** (`docs/evolution/attn/attention-kernels-impl.md`, two sections;
first one also in `tickets/0001-tilelang-issues.md`)

- *The second source's pipelined loop silently computes the wrong answer.* A
  `T.Pipelined` loop whose trip count nests a division by `ratio` is mis-peeled: the
  prologue `cp_async`s a **negative** global offset and the body folds to
  `for (kg = 0; kg < 1; ++kg)`, giving wrong results with no error at
  `num_stages <= 2` (0.79 abs on `o`, 57x the bf16 torch error, at D=64 H=16 ratio=4
  G=128 block_N=64) and a `Downcast from tirx.Sub to ir.IntImm` codegen crash at
  `num_stages = 3`. The bound expression itself evaluates correctly in a standalone
  kernel, and `ratio=1` (where the division collapses) is correct at every G.
  Workaround: `T.serial` for that loop. Not yet reported upstream -- needs a minimal
  repro outside the attention kernel.
- *The group-causal mask clamped to the padded `G`.* `main_kv` is zero-padded to the
  group tile; passing the padded count as `G` made the `min(G, ...)` a no-op for the
  pad rows, whose zero logits then fed the softmax denominator. Invisible for every
  power-of-two `G`; caught by the deliberate `G=100` gradient test (`dq` at 4.13x).
  Fixed by carrying the true `G` and the padded extent as two separate constants.
