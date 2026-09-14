# 0009 indexer_hier: two-level candidate selection (Reindex mode)

- status: todo
- depends on: 0006 (`indexer` forward); 0008 if the fp4 path is wanted here too
- package: `src/mzoo/layers/attn/indexer_hier/`

## Goal

Copy `indexer/` to `indexer_hier/` and add the two-level top-k: the candidate source block-max-pools
its scores and publishes a candidate list; later index sources ("Reindex") score only that list by
gather, making their cost constant in context length. Separate package because the scoring kernel
changes shape (gather over candidates, not a dense `[S, T]` sweep) and it has its own level-one
kernel.

## Semantics

Read `select_candidate_blocks` and `DeepseekV41Indexer.forward` step 4 in
`archs/dsv4/modeling_deepseek_v41.py`.

- Level one, on the candidate source's `index_scores [B, S, T]` (already `-inf`-masked for
  visibility): pad `T` up to a multiple of `candidate_block_size=8` with `-inf`, `amax` over each
  block, then **pin** the block holding the query's newest position
  (`(compress_lens-1)//block_size`) to `+inf` -- it is only partly filled and would otherwise be
  outscored by an older full block.
- `topk(candidate_topk_blocks=2048)` over the block scores, keep only picks with value `> -inf`
  (fewer reachable blocks than 2048 is normal early on), then `repeat_interleave(block_size)` and
  truncate to `T`. Result is a bool mask; a block score of `-inf` means "not reachable yet", never
  "low score".
- Level two: a later index source masks its own `index_scores` with `~candidates` to `-inf` before
  its own `index_topk=512`. The kernel form is the 0006 score kernel restricted to the gathered
  candidate positions (2048*8 = 16384 entries), so cost is O(candidates), not O(T).
- Everything else (relu, weighting, visibility, `-1` padding) is unchanged.
- Backward: same as 0007, with non-candidate entries getting exactly zero gradient.

## Deliverables

`candidates.py` (level-one pool + top-blocks kernel emitting a compact candidate *index list*, not a
bool mask, so level two can gather), `fwd.py` (candidate-gather score kernel), `bwd.py`, `attn.py`,
`ref.py`, `bench.py`, `*_test.py`, `README.md` (what/how, configs, bench, accuracy, decisions,
**Known issues**, next). Tests via `dense_attn/ref.py::assert_within_2x_torch`; extend `just smoke`.

## Acceptance

- Reference: `select_candidate_blocks` itself for level one (exact bool-mask equality, including the
  pinned partial block and the `> -inf` drop), and the torch indexer path for level two.
- Tie behaviour must be documented: `topk` on equal block scores may pick differently from torch;
  the test asserts set equality modulo ties.
- Cases: `T` not a multiple of 8, fewer reachable blocks than 2048, a query whose newest block is
  also its highest-scoring block.
- Bench vs `indexer/` at `T = 16384, 65536`: level-two time must be flat in `T` while level one
  stays O(T) but cheap (one pass, no GEMM).

## References

- `csa2_attn_design.md` step 6c; `examples/dsa_sparse_finetune/index.py` and
  `indexer_topk_reducesum.py`; `examples/dsa_hisa/` (block-sparse scoring shape).
- `fp_attn_survey.md` §5.

## Risks / open questions

- Materialising `[B, S, T]` scores to pool them defeats the purpose at large `T`; fuse the block-max
  into the level-one score kernel (decided in 0006's "emit top-k per query block" open question).
- 2048 blocks x 8 = 16384 candidates per query is 32x the final 512; check that the level-two gather
  is not itself the bottleneck.
- Reuse/Reindex scheduling across layers is a model-level concern -- the kernel only takes pointers,
  as in 0004.

## Learnings from earlier steps (2026-09-14)

- Decided in 0006: v1 materialises the full score matrix and runs `torch.topk` on it, which
  is 256 MB at S 4096 T 16384 B 1 and 2-4x the score kernel's time. This ticket only pays off
  if the full matrix is never materialised: emit per-query-block top-k (the
  `indexer_topk_reducesum.py` shape) and score only the candidate list.
- `select_candidate_blocks` pins the newest partially-filled block; replicate that exactly and
  test on a non-multiple-of-block `compressed_len`.
- Model top-k ties: compare chosen-set score values, not index sets.
- No tuning sweeps (user decision 2026-09-14).
