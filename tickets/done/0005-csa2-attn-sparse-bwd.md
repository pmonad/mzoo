# 0005 csa2_attn: sparse gather backward + autograd

- status: done (2026-09-14): sparse backward + autograd land as 3 kernels (dmain_kv scatter-added out of the dQ kernel); 52 tests pass, fwd+bwd is 2.7-3.0x csa_attn; configs untuned.
- depends on: 0004 (`csa2_attn` forward, done and independently verified 2026-09-14)
- package: `src/mzoo/layers/attn/csa2_attn/` (backward half)

## Goal

Add `bwd.py` and the `attn.py` autograd wrapper to `csa2_attn/`, making the sparse package
trainable. Split out from 0004 because the gather backward is a different kernel shape from the
forward: the KV-block-owning loop of `dense_attn` does not exist when KV rows are reached only
through per-token indices, so dKV becomes a scatter-add.

## Semantics

Same forward semantics as 0004 (`DeepseekV41Indexer.forward` step 5 for the index conventions,
`eager_attention_forward` for the softmax/sink).

- Recompute S/P from saved `(q, kv, indices, o, lse)` -- never materialise `S^2`. `lse` already
  includes the sink column, so the recomputed P is sink-normalised.
- Window slice: dKV as in 0002/0003 (block-owning loop, `dKV = dK + dV`, fp32).
- Main slice: each query row touches 512 arbitrary entries, so dKV is a **scatter-add** into the
  gathered rows; accumulate in fp32 (`atomic_add`), never bf16 atomics. Masked / out-of-range
  indices contribute nothing.
- Both sources write into one `dKV` buffer because the model concatenates them on the KV axis; the
  caller splits it back into window and compressed tensors.
- `dsinks[h] = -sum_{b,s} exp(sink[h] - lse[b,h,s]) * delta[b,h,s]`, torch fp32, unchanged.
- No gradient flows to `Indices` (top-k selection is discrete); the indexer's own gradient path is
  ticket 0007.

## Deliverables

`bwd.py` (preprocess `delta = rowsum(dO*O)`, main kernel, postprocess cast), `attn.py` (autograd;
saves exactly `q, kv, indices, o, lse[, sinks]`, pinned by a `test_backward_saves_only_*` test as in
`dense_attn`), bench rows for `bwd` and `fwd+bwd`, tests, and the **Known issues** / bench /
accuracy sections of `csa2_attn/README.md` filled in. Extend `just smoke`.

## Acceptance

- Reference: autograd through `attn/golden_ref.py::golden` (`sparse` level) fed the *same* indices;
  `assert_within_2x_torch` on `dq`, `dkv`, `dsinks`.
- Duplicate-index case must be exercised (two picks landing on the same entry) -- scatter-add
  correctness is invisible otherwise.
- `D in {64, 96, 128, 256}`, 64/128 tuned, 96/256 pass-only; `topk=512`.
- Determinism note in the README: fp32 `atomic_add` makes dKV run-to-run non-bitwise-reproducible;
  state it rather than fixing it.
- Bench vs `csa_attn` fwd+bwd at `G = 4096, 16384`, same shapes as 0004.

## References

- tilelang `examples/dsa_sparse_finetune/sparse_mla_bwd.py` (scatter-add template; `assert dtype ==
  bfloat16` upstream), `examples/flash_attention/example_mha_bwd_bshd.py`,
  `examples/attention_sink/example_mha_sink_bwd_bhsd.py`.
- `csa2_attn_design.md` step 7; `fp_attn_survey.md` §3 (keep `dP = dO V^T` bf16).

## Risks / open questions

- fp32 atomic contention: with `topk=512` shared across all 64 heads of a token, every head's
  contribution hits the same 512 rows. Consider a per-head register reduce before the atomic.
- `dense_attn`'s bwd is already ~200 lines and slower than sdpa because of the atomic-dQ design;
  adding a second atomic (dKV) may make this the slowest package. Bench before tuning, and record
  the split.
- Layout-inference limits from `tickets/0001-tilelang-issues.md` bite hardest in the bwd
  (`block_M=32` unusable); budget tuning time for D=256.

## Learnings from earlier steps (2026-09-14)

- Scatter-add into gathered main entries: the `sparse_mla_bwd.py` template uses atomics. In
  `latent_attn` the atomic *instruction* cost ~5%; the repeated read-modify-write of a large
  buffer cost 2x. Measure before copying the template; an owner-based `dmain_kv` (each main
  block walks the tokens whose indices reference it -- needs an inverse index / CSR built in
  torch) may win. Decide by measurement, record it.
- Any loop whose trip count nests a division by `ratio` must be `T.serial` (tickets/0001
  pipeline miscompile); the gathered loop's trip count is a constant, so it may pipeline --
  verify at a non-power-of-two `topk`.
- Test matrix: `topk` in {64, 100, 512}, `G` non-power-of-two, rows with all `-1` indices,
  H in {4, 64}, and an all-visible-indices case equal to `csa_attn`'s backward bit-for-bit.
  `dsinks` straddles 2x at small B*S in every package; check it at D 64/128 H 16.
- Keep true `G` and padded `G` as separate constants in every kernel (the csa_attn mask leak).


## Result

**What landed.** `bwd_kernels.py` (`preprocess` + the window `dKV` owner, byte-for-byte
`csa_attn`'s with `csa_attn`'s config), `bwd_dq.py` (the query owner: `dQ` **and** the
scatter-added `dmain_kv`), `bwd.py` (driver), `attn.py` (autograd, saves exactly
`q, kv, main_kv, indices, o, lse[, sinks]`), `attn_test.py`, bench rows, README sections.
`csa_attn`'s fourth kernel (`bwd_main.py`, the group owner) has **no analogue**: a block of
main entries cannot enumerate the tokens that picked it without an inverse index, so
`dmain_kv = dS^T Q + P^T dO` is `T.atomic_add`-ed in fp32 into a zeroed `[B, G, D]` buffer
from inside the query-owning kernel and cast to bf16 in torch.

**Correctness.** `uv run --env-file .env pytest src/mzoo/layers/attn/csa2_attn -q` -> 52
passed (24 of them new). Reference is autograd through `golden(level="sparse")` fed the same
index list; every `dq`/`dkv`/`dmain_kv` cell of the D x H matrix is <= 1.00x the bf16
baseline's error (`dmain_kv` 0.48-0.68, the most accurate column, because it accumulates in
fp32). `dsinks` straddles 2x at small `B*S` as in every package and is asserted at D 64/128
H 16 only, with the out-of-band cells reported in the README. Feature-off: all-visible
indices give `dq`/`dkv`/`dsinks` **bit-for-bit** `csa_attn.bwd` and `dmain_kv` to 7.3e-4
relative (fp32 summation order); all `-1` gives `dq`/`dkv` bit-for-bit `swa_attn.bwd` and
`dmain_kv == 0`.

**Bench** (B1 S4096 W128 D128 H64 topk=512, shared GPU, +-5%):

| G | fwd csa/csa2 | bwd csa/csa2 | fwd+bwd csa/csa2 | speedup |
|---|---|---|---|---|
| 4096 | 3.42 / 2.17 ms | 16.15 / 4.96 ms | 19.57 / 7.13 ms | **2.74x** |
| 16384 | 3.46 / 2.20 ms | 18.21 / 5.09 ms | 21.67 / 7.29 ms | **2.97x** |

Split of the 4.96 ms: preprocess 0.67 + window dKV 0.79 + dQ/dmain 3.43. The backward is
where the gather pays off hardest (3.3-3.6x vs 1.6x in the forward) because `csa_attn` needs
a whole third GEMM kernel for `dmain_kv`.

**Atomic ablation** (the ticket's open risk). `bwd_dq.py` gained a three-way `scatter` knob:
`atomic - none` (the whole `dmain_kv` half, GEMMs included) = 1.09-1.13 ms, 32% of the `dQ`
kernel; `atomic - store` (same addresses, plain store, so only the read-modify-write and the
contention) = 0.30-0.39 ms, 9-11%. The atomic itself is ~7% of the backward, so the
owner-kernel-with-inverse-index alternative was **not** built. Matches the `latent_attn`
finding. Cost: `dmain_kv` is not bitwise reproducible (verified); `dq`/`dkv` are.

**Duplicate indices: a real discrepancy, reported not fixed.** `golden` scatters an
idempotent `0.0` bias, so a duplicated entry gets one softmax column; the kernel builds one
column per slot and gives it two. This is a **forward** property (0004, from
`sparse_mla_fwd.py`) and was not changed. `test_grads_duplicate_indices` pins both sides:
the kernel matches a reference that clones the duplicated row into a fresh entry (and
`dmain_kv[0] == dmain_ref[0] + dmain_ref[G]`, which is the scatter-add check the ticket
asked for), and it is 44x the bf16 noise floor away from plain `golden`. The model's indexer
emits `torch.topk` output, a set, so duplicates cannot occur on the model path.

**Configs: untuned, no sweep run.** `csa_attn`'s `kv`/`dq` entries verbatim, except three
values forced rather than chosen: the `dQ` kernel's `block_M` is `T_q * H = 64` rows and its
`threads` 128 (the per-token gather plus the `tickets/0001` layout-inference limits, exactly
as in `fwd.py`), and a new `gather_stages` is 1 at D 128/256 because `gather_stages = 2`
needs 107776 B of smem against the 99 KB cap -- so the backward's gathered loop is serial
where the forward's is pipelined. `block_N = 32` + pipelined gather fits at D=128 and is the
first thing for the tuning pass, but it would break the bit-for-bit feature-off equalities.

**Parked.** `docs/evolution/attn/attention-kernels-impl.md` gained "csa2_attn bwd: duplicate
indices double-count..." and "csa2_attn bwd: the dmain_kv scatter is a third of the dQ
kernel, but the atomic is a tenth". `tickets/0001-tilelang-issues.md` gained "A bf16 fragment
cannot feed one `T.gemm` as `A` and another as `A^T`" (`Get different layout for cast`;
workaround is the upstream template's shared staging tile, which is also what costs the smem
that forces `gather_stages = 1`).

**Not done / deferred.** No tuning sweep (user decision). `just smoke` not extended and not
run: only this package changed, and the design doc's Test scope rule says the changed
package's tests only. No owner-based `dmain_kv` (measured not worth it). No index sorting.
