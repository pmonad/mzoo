# Design: FA2-style attention for SM120 in TileLang, growing into DSV4.1 CSA2

Status: `dense_attn`, `latent_attn`, `swa_attn` complete for training (fwd + lse + sink, bwd, autograd, tests, bench) and independently verified; `csa_attn` (0003) and `indexer` (0006) done and verified (untuned); `csa2_attn` fwd+bwd (0004, 0005) done and verified; `indexer` bwd (0007) done and merged. Attention track complete through CSA2; remaining: 0008-0010, the tuning pass, 0011, 0012. Target GB10 (sm121, SM120 family), tilelang 0.1.14.
Scope: head dims `{64, 96, 128, 256}` only. `D=512` (the released V4.1-Flash size) is out of scope, so no Split-D.
Training only for now: prefill-shaped forward plus backward; no decode, no paged cache, no split-KV.

## What the kernel ultimately has to compute (from `archs/dsv4/modeling_deepseek_v41.py`)

Per layer, per query token `t` with `H=64` heads, `D=512` (last `qk_rope_head_dim=64` channels carry RoPE):

```
logits  = [ q·win_kv[t-127..t] , q·main_kv[topk(t)] , sink[h] ] * D^-0.5
p       = softmax(logits)            # sink column only feeds the denominator
o       = p[:, :-1] @ [ win_kv ; main_kv[topk(t)] ]
```

Things that make this *not* plain FA2:

| Property | Consequence for the kernel |
|---|---|
| K == V is one latent, 1 KV head shared by 64 Q heads (MQA) | pack heads into the M (row) dim; one smem KV tile serves both GEMMs |
| `D<=256` (our scope) | 64x256 bf16 tile = 32 KB, Q stays resident in smem; plain FA2 tiling works |
| Per-head learnable sink | one extra `exp2(sink - m)` added to the row sum at the end (see `examples/attention_sink`) |
| Two KV sources: SWA window (128 raw tokens, fp8 cache) + main/compressed KV (ratio `m`, fp4 cache) | one online-softmax loop that walks both sources with a single running (m, l, acc) |
| Main KV visible to `t` only for entries `< (t+1)//m` | group-causal mask, separate from the token-causal window mask |
| Top-K (512) main entries per token, indices from an indexer | gather loop over indices, exactly `examples/dsa_sparse_finetune/sparse_mla_fwd.py` |
| Cross-layer reuse (Full / Reindex / Reuse modes) | kernel takes KV / index-K / indices as pointers; reuse is a model-level scheduling choice, not a kernel feature |
| Hierarchical indexer | second indexer variant scoring a candidate list instead of the full range |

Out of kernel (elementwise, stays in torch): Q/KV RoPE, inverse RoPE on the output, `wo` projection, cache append.

## SM120 constraints to design around

- Tensor cores are `mma.sync` (no wgmma, no TMA multicast). `T.gemm` lowers to the sm80/89 path; the FA2 examples are the right template, not `flash_attention_sm100`.
- 99 KB smem per block, 64K regs per SM. At `D=256`, `acc_o[64, 256]` fp32 is 128 regs/thread at 128 threads; expect `block_M=64`, `block_N=64`, 128-256 threads.
- fp8 e4m3 MMA available; MXFP4 block-scaled MMA already works here (`src/mzoo/kernels/sm120_nvfp4_blockscaled_gemm.py`, needs the `SM120A_ENABLED` flag from `tickets/0001`). The indexer's fp4 q/k with ue8m0 per 32 channels *is* OCP MXFP4, so the indexer can use it natively. Main KV fp4 (e2m1 + e4m3 per 16, no global scale) is dequantized on load, as the paper says; Q is bf16 so no fp4 MMA there.

## Steps

| ticket | package | status | depends on |
|---|---|---|---|
| -- | `dense_attn` | done | -- |
| -- | `latent_attn` | done (2026-09-14, verified) | `dense_attn` |
| [0002](../../../../tickets/0002-swa-attn.md) | `swa_attn` (fwd+bwd) | done (2026-09-14, verified) | `latent_attn` |
| [0003](../../../../tickets/0003-csa-attn.md) | `csa_attn` (fwd+bwd) | done (2026-09-14, verified; configs untuned) | 0002 |
| [0004](../../../../tickets/0004-csa2-attn-sparse-fwd.md) | `csa2_attn` (fwd) | done (2026-09-14, verified; configs untuned; bench absolute numbers taken on a shared GPU) | 0003 |
| [0005](../../../../tickets/0005-csa2-attn-sparse-bwd.md) | `csa2_attn` (bwd) | done (2026-09-14, verified; configs untuned) | 0004 |
| [0006](../../../../tickets/0006-indexer-bf16-score.md) | `indexer` (bf16 fwd) | done (2026-09-14, verified; configs untuned) | 0004 as consumer only (contract pinned by `golden_ref.topk_indices`) |
| [0007](../../../../tickets/0007-indexer-backward.md) | `indexer` (bwd) | done (2026-09-14, verified, merged) | 0006 |
| [0008](../../../../tickets/0008-indexer-mxfp4-score.md) | `indexer_fp4` | todo | 0006, 0007 |
| [0009](../../../../tickets/0009-indexer-hierarchical.md) | `indexer_hier` | todo | 0006 |
| [0010](../../../../tickets/0010-csa2-fp-cache.md) | `csa2_fp_attn` | todo | 0005 |
| [0011](../../../../tickets/0011-qk-prologue-norm-rope.md) | `norm_rope` (fused RMSNorm+RoPE TileLang kernel, fwd+bwd) | done (2026-09-14, reviewed; untuned) | -- |
| [0012](../../../../tickets/0012-dsv4-shared-dict-cross-group.md) | dsv4 model: `shared` dict cross-group accumulation (suspected) | todo, unverified | -- |

Why this split: one ticket per package, except where a backward is a different kernel shape from its forward (0005's scatter-add dKV, 0007's relu-gated three-input reduce) -- those get their own; every other backward is folded into its package's ticket, so step 7 files no ticket of its own. The indexer's MXFP4 math path (0008) and hierarchical candidate list (0009) are separate packages, not variants, per the copy-forward rule. In-kernel fp8/fp4 dequant (steps 3v2 + 4v2) is one ticket (0010): one change, two sources, one shared risk.

Each step is its own package under `src/mzoo/layers/attn/<feature>/` holding `fwd.py`, `bwd.py`, `attn.py`, `ref.py`, `bench.py` and their tests. A step lands as a new package copied from the previous one, never by editing an earlier package (see Conventions), checked on GPU against a torch reference (`eager_attention_forward` from the modeling file for the later steps). Forward first per step, backward per step 7.

### 1. Dense FA2 forward baseline on GB10 -- done
Fwd + lse + sink, bwd, autograd wrapper, tests, bench. Design, tile configs,
bench numbers and accuracy table are in `dense_attn/README.md`. Deliverables:
`fwd.py`, `bwd.py`, `attn.py`, `ref.py`, `bench.py`, tests.

### 2. Shared-latent MQA layout -- done (see `latent_attn/README.md`)
- Switch to K==V, 1 KV head: block = `(T_q query tokens x H heads)` rows, e.g. 1 token x 64 heads. Mask per row derives from the row's token, not the block.
- Reuse the same smem KV tile for `S = Q K^T` and `O += P K`. Drop the V load entirely.
- Q resident in smem (`64 x 256` bf16 = 32 KB worst case), KV tile `64 x 256` = 32 KB, so 1-2 stages fit. No D-split needed at these dims; ffpa-attn's Split-D only matters for `D > 256`.
- Keep the RoPE tail as a plain slice of D (no separate nope/rope GEMM like MLA; V4.1 rotates in place).

### 3. Sliding-window branch (the layer type every DSV4.1 layer has) -> ticket 0002 (v2: 0010)
- Restrict the KV loop to `[t0-127, t0+T_q-1]`, band mask `t-127 <= k <= t` per row. With 1 token per block that is exactly 2 tiles of 64.
- fp8 window cache: v1 dequantize in torch (ue8m0 scale per 32 channels), v2 load fp8 + scales and dequantize into the bf16 smem tile in-kernel. v2 halves window-cache bandwidth, matters mostly for decode.
- Test against the model's eager path with `compress_ratio=0` layers.

### 4. Compressed attention (DeepSeek CSA-style dense main KV, no sparsity) -> ticket 0003 (v2: 0010)
- Second KV source in the same loop: after the window tiles, walk all visible main entries `k < (t+1)//m` (group-causal mask). One running max/sum across both sources, so no LSE merge is needed.
- `m=1` is the uncompressed full-attention special case and doubles as the correctness test against plain causal FA2 from step 2.
- Compressed rope at latent positions and the compressor itself stay in torch.
- fp4 main cache: v1 torch dequant, v2 in-kernel e2m1 + e4m3-per-16 dequant into smem. Layout choice for the cache (`[T, 512]` fp4 packed + `[T, 32]` scales) is fixed here and reused by everything after.
- Test: `compress_ratio=4` layer of the smoke config vs eager.

### 5. Sparse selection (CSA2 core attention) -> tickets 0004, 0005
- Replace the dense main loop of step 4 with a gather over `Indices[B, S, topk]` (per token, shared across heads, which is why the 1-token x 64-heads block from step 2 is the right layout). Template: `sparse_mla_fwd.py` (`KV_shared[i, :] = KV[Indices[s, i], :]`, `-1` / out-of-range masked to `-inf`). `topk=512` = 8 tiles of 64.
- Indices come from the model's indexer at first (torch `topk`), so step 5 is testable before any indexer kernel exists.
- Reuse mode is free: same kernel, indices tensor from an earlier layer. Full/Reindex/Reuse differ only in what the caller passes.

### 6. Indexer kernels (Full and Reindex modes) -> tickets 0006, 0007, 0008, 0009
- 6a. Score kernel: `scores[s, k] = sum_h w[s,h] * relu(q[s,h,:] . k[k,:]) * scale`, `q [S, 32, 128]`, `k [T, 128]`, group-causal mask. Template: `examples/dsa_hisa/block_sparse_mqa_fp8.py` (per-token block, heads in M, weighted head reduce). Start bf16, then MXFP4 x MXFP4 via `T.mma_gemm_blockscaled` since the ue8m0-per-32 format matches.
- `T.mma_gemm_blockscaled` currently asserts the NVFP4 flavor only (ue4m3 scales, block 16); the MXFP4 indexer (ue8m0, block 32) needs either a small upstream widening or re-encoding each ue8m0 scale as two ue4m3 block-16 scales (exact within the exponent range). See `fp_attn_survey.md`.
- 6b. Top-K: `torch.topk` on the score matrix. Only fuse (`dsa_sparse_finetune/indexer_topk_reducesum.py` style) if profiling says so.
- 6c. Hierarchical: block-max pool the Full-layer scores (block of 8 entries), take top-2048 blocks, emit a per-token candidate list (16384 positions). Reindex variant of 6a scores only the candidate list via gather, so cost is constant in context length. `select_candidate_blocks` in the modeling file is the reference.

### 7. Backward passes -> folded into each package's ticket (own ticket only for 0005, 0007)
- Every feature step needs a backward before it is usable for training, so each package has `bwd.py` beside `fwd.py` and an `attn.py` autograd wrapper. The forward emits `lse [B, H, S]` (`dense_attn/` does already).
- Dense + sink bwd: `examples/flash_attention/example_mha_bwd_bshd.py` and `examples/attention_sink/example_mha_sink_bwd_bhsd.py` (dsink kernel). MQA bwd: dK/dV reduce over the 64 heads sharing one latent, and since K==V, `dKV = dK + dV`.
- Sparse gather bwd: `dsa_sparse_finetune/sparse_mla_bwd.py` (scatter-add into the gathered main entries). Indexer bwd: `indexer_bwd.py` in the same dir. The indexer's detached fp4 fake-quant (README) is a model bug, not a kernel one, but the kernel path is where the straight-through estimator will have to live.
- Decode (split-KV, paged main cache) is deferred until inference matters.

## Conventions

This is where project conventions for this series live (not `CLAUDE.md`).

- **Versioning (copy-forward, freeze old, plain filenames).** The version is
  the package: `dense_attn/`, `latent_attn/`, `swa_attn/`, ... Each holds one
  `fwd.py`, `bwd.py`, `attn.py` (autograd wrapper; backward recomputes S/P,
  saves only q, k, v, o, lse), `ref.py`, `bench.py`, `README.md`, `*_test.py`
  -- no version prefixes inside a package. A new feature = copy the previous
  package to a new folder, change one thing, leave the old one untouched;
  bisecting a bug is a diff between sibling packages. No shared kernel
  helpers across packages -- duplication is the point. Every package exposes
  the same call shapes (`fwd(q, k, v, *, causal, **extras) -> (o, lse)`,
  `attn(...) -> o`) in BSHD layout. Retire a package only by deleting it, in
  its own commit.
- **Test criterion.** The FlashAttention acceptance criterion, not a fixed
  atol: a tensor's max-abs error against an fp32 reference must be <= 2x the
  max-abs error PyTorch's own bf16 kernel makes against that same reference
  on the same inputs. `lse` is the exception -- fixed fp32 tolerance, since
  it never leaves fp32. `dense_attn/ref.py::assert_within_2x_torch` (+
  `torch_bf16_ref`) is the shared helper every package reuses instead of
  reintroducing fixed atols (sparse steps feed identical indices to both).
- **References.** `golden_ref.py::golden(level=...)` is the fp32 reference
  (and, with `dtype=bf16`, the torch-bf16 baseline) for every level; it is
  pinned against the vendored model's eager path. `golden_ref.gpt_oss_ref`
  (transformers' gpt-oss eager attention, MQA + sinks + sliding window) is
  the implementation-independent second reference for the latent/window
  levels. Packages carry no `ref.py` of their own unless torch-side model
  pieces (compressor, quant) need a home.
- **Benches.** Compare only against baselines built for the same task: the
  previous package on identical inputs, or a torch path doing the same math
  (SDPA with the latent expanded for dense/latent; einsum + topk for the
  indexer). Never SDPA with masks it was not designed for (band, group-
  causal, sparse) -- those numbers say nothing. Correctness is the
  deliverable; a bench is one small table for the README and ticket.
- **Test matrix (learned 2026-09-14).** Every package's tests must include
  (a) non-tile-aligned / non-power-of-two sizes for every new extent (window
  100, G 100 and 37, topk 100): all "natural" shapes are powers of two and
  hid a padded-G mask leak in `csa_attn` that only the G=100 *gradient* test
  caught; (b) a feature-off case that must equal the previous package
  bit-for-bit (`window >= S` == `latent_attn`, `G == 0` == `swa_attn`,
  all-visible indices == `csa_attn`): the cheapest regression check there is;
  (c) the smallest AND largest head count, since packed-heads tiles count
  `T_q * H` rows and smem depends on H (`indexer` D=256 fit at Hi=32, not
  at Hi=4); (d) the vendored model replay via `golden_ref_test`'s capture
  helpers. `dsinks` is a thin reduce whose 2x ratio straddles the bound at
  small B*S in every package; check it where stable (D 64/128, H 16) and
  say so, do not loosen the criterion for the other tensors.
- **Kernel invariants (learned 2026-09-14).** Loop bounds are per block,
  masks are per row, so a row can meet a tile it sees nothing in: floor the
  running rowmax at `-1e30`, never mask logits with a finite constant. Carry
  the true extent and the padded extent as separate constants: shapes, grid
  and loops use the padded one, every mask the true one. A loop whose trip
  count nests a division (`floordiv(..., ratio)`) must be `T.serial` on
  tilelang 0.1.14 (`T.Pipelined` miscompiles it, tickets/0001). Atomics
  are cheap per instruction; repeated read-modify-write of a large buffer is
  not -- prefer an owner kernel that stores once even at +40% FLOPs.
  A `T.Parallel`-filled fragment that is then reduced needs a power-of-two last
  dim (or a per-row fragment read in the same fill loop to anchor the layout);
  `T.reduce_sum` over any dim is sound -- the 0011 `dim=0` claim was retracted.
- **Tuning (learned).** Tile tables do not transfer across loop shapes: the
  band inverted `latent_attn`'s bwd winners, the dense main loop will differ
  again. The deferred tuning pass must sweep each package on its own, and
  validate every config at the smallest supported H. Sweeps decide fit by
  compiling (smem is aliased by liveness).
- **Working mode (learned).** A stopped worker leaves an uncommitted,
  half-edited tree and the previously verified state is unrecoverable
  without git history (`latent_attn` had to be re-derived): commit or tag
  each package right after its independent verification, before the next
  worker touches the tree. Two workers on one GPU make bench numbers
  +-5%; the verifier reruns benches alone. Independent verification found
  a real gap in every package so far (window=1 compile crash, D=256 bench
  drift) that the implementing worker's own report did not.
- **Bug claims against tilelang.** Before recording one: grep upstream
  examples/tests and our packages for the same construct, and commit a
  minimal repro; no repro, no entry. Resolved entries are 3-5 lines
  (trigger, rule, pointer to the repro/test); only unresolved ones keep the
  evidence and a status.
- **Done means ticketed.** A task is done only when its ticket has the
  status line and a Result section, written by the implementing worker and
  checked by the reviewer.
- **Bugs in frozen packages.** Packages are never edited after they land,
  so a bug found in package N (e.g. the padded-G mask leak found in
  `csa_attn`, the duplicate-index double count found in `csa2_attn`'s
  forward during its backward) is fixed in the newest package, checked in
  every package copied from N since, and listed under Known issues of the
  frozen ones. Check by diff: copy-forward means the same lines exist in
  every later sibling.
- **Model divergences.** Anything the kernel path does differently from the
  vendored model (the indexer's fp4 straight-through estimator is the first)
  is listed in `archs/dsv4/README.md` with the reconcile-or-delete rule, so
  an upstream fix is never silently doubled.
- **Parking notes.** Every real problem hit while implementing goes into
  `docs/evolution/attn/attention-kernels-impl.md` as one short section
  (problem / measurement / fix), per `CLAUDE.md`; tilelang quirks go to
  `tickets/0001-tilelang-issues.md`.
- **Design docs.** Each package has its own `README.md` with the same section
  order -- what/how, configs, bench, accuracy, decisions, known issues, next
  -- and this overall doc only links to them, it does not duplicate them.
- **Tickets.** `tickets/NNNN-*.md` hold the per-step scope; the table under
  "Steps" is the status tracker; a package's `README.md` records what was built.
- **Working mode.** The main session reviews and verifies (reruns tests and
  benches); workers implement; model choice is by task difficulty.
- **Test scope.** Workers and verifiers run the changed package's tests only
  (`pytest src/mzoo/layers/attn/<pkg> -q`); `golden_ref_test.py` when
  `golden_ref.py` changed; `just smoke` only when shared files or several
  packages changed. Earlier packages are frozen, so rerunning them proves
  nothing.
- **Tooling.** `uv run --env-file .env` everywhere (machine-specific env in
  the gitignored repo-root `.env`); one justfile per folder, invoked from
  repo root with a trailing slash, e.g. `just src/mzoo/layers/attn/ test`.

## Decisions

- Training only, no decode/inference path for now.
- Fast-math kept (`TL_ENABLE_FAST_MATH`): FA1-FA4 all build with fast math.
- No gradient-checkpoint guard; only the save/recompute invariant is tested.
- Scope is `D in {64, 96, 128, 256}`; 64/96/128 are the mandatory tuned
  dims, 256 is shape-support only. `D=512` is out of scope, so no Split-D.
- Tuning sweeps deferred from `csa_attn` (0003) onward (user decision
  2026-09-14): one config per dim that compiles and passes, flagged
  "untuned" in the README and ticket; a tuning pass comes later.
- QK prologue (RMSNorm + RoPE) is a fused TileLang kernel, fwd + bwd with
  autograd, untuned: ticket 0011.
- Fixed sequence length for now; no varlen/packing.

## Next

`csa_attn` (0003) and `indexer` (0006) are running in parallel; then 0004/0005
(sparse), with the indexer track (0007-0009) continuing beside the attention
track until they meet at `csa2_attn`. The golden reference exists
(`golden_ref.py`, all five levels, checked against the vendored model layer by
layer and against gpt-oss). Deferred: the tuning pass over every package built
without sweeps, and ticket 0011 (fused norm + RoPE prologue).

Open question: variable-length / packed batches -- currently assumed no.
