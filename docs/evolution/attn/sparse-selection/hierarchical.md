---
title: "Hierarchical selection"
order: 2
---

## Hierarchical selection

The previous section left one term growing with the context. The selected read is fixed at $k$
entries, but the indexer still scores every visible entry to find them, so its work per query is
$S / m$. At 1M-token context even the indexer's $S \times S/m$ score matrix is expensive. The remedy is the
one used in nearest-neighbour search, a coarse pass over groups followed by an exact pass inside the
surviving groups.

### Two levels

V4.1 adds a coarse level. The compressed entries are tiled into blocks of $b$ consecutive entries,
indexed by $b$. Each block is scored by its best member, and only the top $k_b$ blocks are passed to
the fine top-$k$:

$$
\beta_{i,b} = \max_{j \in \text{block } b} I_{i,j}, \qquad
\mathcal{B}_i = \operatorname{top}k_b\, \beta_{i,b}, \qquad
\mathcal{S}_i = \operatorname{top}k \{\, I_{i,j} : j \in \text{block} \in \mathcal{B}_i \,\} .
$$

Read that back in words. The block score $\beta_{i,b}$ is the largest entry score inside block $b$;
the coarse pass keeps the $k_b$ blocks whose best members scored highest; the fine pass then runs the
ordinary top-$k$ of the previous section, but only over the entries inside the surviving blocks.

The block score is a maximum and not a mean. A single strongly matching entry surrounded by
irrelevant ones must survive the coarse pass, and averaging would dilute it. The maximum also makes
the filter well behaved. A block's score is the largest score it contains, so if the $k$ best entries
of the whole table fall inside at most $k_b$ blocks, the coarse pass keeps all of them and the two-level
result equals the one-level result. That is the exactness condition. Blocks are contiguous runs of
positions, and the entries a query wants are usually clustered in a few passages, so the condition
holds often. The block containing the newest visible entry is always kept, so the coarse filter
never removes a query's immediate context.

### Where the saving is

Scoring a block requires the scores of its members, so the layer that builds the coarse level pays
the full $S / m$ scan. In the reference model that layer is 20, the first Full-mode layer of the
decoder. It computes the block candidates and publishes them. Later index layers — the Reindex
sources at 24, 28, 32 and 36, which score with their own weights — score only within those
candidates, inside the candidate mask. The saving is across layers rather than within one, and in
V4.1 it applies only in the decoder, whose Reindex layers would otherwise each rescore the whole
visible context.

The paper's sizes are $b = 8$ entries per block and $k_b = 2048$ blocks, so the candidate pool holds
up to $16384$ positions, from which the fine pass takes $k = 512$. The pool is the point: it is a
constant. At the extended context of 1{,}048{,}576 tokens the encoder table holds 524288 entries
and the decoder table 1048576, so a bounded pool is a factor of 32 below the one and 64 below the
other — and the gap widens as the context grows, because the pool does not. The per-query cost of
every deeper decoder indexer is bounded however long the conversation. In indexer bytes, the
candidate source layer's full scan at 1M costs $1048576 \cdot 128 \cdot 0.5625 = 72$ M, while each
later Reindex layer reads $16384 \cdot 128 \cdot 0.5625 = 1.125$ M. The three encoder index layers,
at 2, 8 and 14, are outside the scheme and keep their full scans of 36 M each at 1M.

The mechanism is introduced in post-training but is training-aware: the restriction is applied
identically while the model continues to train, so the deeper indexers are optimised under the same
candidate pool they will use at inference. That placement is consistent with what the mechanism
does. It changes which entries are considered, not what the model computes on them, and its
exactness condition is easiest to satisfy at the long contexts it is added for. In owlet1 the
candidate mask is computed but never consumed, because the six-layer schedule has no index layer
after the candidate source.

The mechanism is now complete at inference. What remains is how a model learns to use it, which is
the subject of the last section, because a hard top-$k$ passes no gradient to the scores that produced
it.
