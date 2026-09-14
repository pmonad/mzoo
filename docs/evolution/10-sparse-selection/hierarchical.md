# Hierarchical selection

The previous section left one term growing with the context. The selected read is fixed at $k$
entries, but the indexer still scores every entry to find them, so its work per query is $S / m$. At
million-token context even the indexer's $S \times S/m$ score matrix is expensive. The remedy is the
one used in nearest-neighbour search, a coarse pass over groups followed by an exact pass inside the
surviving groups.

## Two levels

V4.1 adds a coarse level. The compressed entries are tiled into blocks of $b$ consecutive entries.
Each block is scored by its best member, and only the top $k_b$ blocks are passed to the fine
top-$k$:

$$
\beta_{s,\ell} = \max_{j \in \text{block } \ell} I_{s,j}, \qquad
\mathcal{B}_s = \operatorname{top}k_b\, \beta_{s,\ell}, \qquad
\mathcal{S}_s = \operatorname{top}k \{\, I_{s,j} : j \in \text{block} \in \mathcal{B}_s \,\} .
$$

The block score is a maximum and not a mean. A single strongly matching entry surrounded by
irrelevant ones must survive the coarse pass, and averaging would dilute it. The maximum also makes
the filter well behaved. A block's score is the largest score it contains, so if the $k$ best entries
of the whole table fall inside at most $k_b$ blocks, the coarse pass keeps all of them and the two-level
result equals the one-level result. Blocks are contiguous runs of positions, and the entries a query
wants are usually clustered in a few passages, so that condition holds often. The block containing
the newest visible entry is always kept, so the coarse filter never removes a query's immediate
context.

## Where the saving is

Scoring a block requires the scores of its members, so the layer that builds the coarse level pays
the full $S / m$ scan. The first Full layer computes the block candidates and publishes them. Later
index layers score only within those candidates. The saving is across layers rather than within one,
and in V4.1 it applies only in the decoder, whose Reindex layers would otherwise each rescore the
whole visible context.

The paper's sizes are $b = 8$ entries per block and $k_b = 2048$ blocks, so the candidate pool holds
up to $16384$ positions, from which the fine pass takes $k = 512$. On the reference model at 131072
tokens with $m = 2$, the table is 65536 entries, so a layer after the first scores 16384 entries
instead of 65536, a factor of 4 — and, more importantly, a constant: the pool does not grow with
context, so the per-query cost of every deeper indexer is bounded however long the conversation.
The first layer's full scan remains, at $65536 \cdot 128 \cdot 0.5625 \approx 4.7$ MB of indexer
keys in FP4; each later layer reads a quarter of that.

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
