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
index layers score only within those candidates. The saving is across layers rather than within one.

Take the reference model at 131072 tokens with $m = 2$, so 65536 compressed entries, and take
$b = 128$ and $k_b = 32$ as an illustration of the sizes involved. The table is 512 blocks, and 32
surviving blocks leave 4096 candidate entries, from which the fine pass takes $k = 2048$. A layer
after the first therefore scores 4096 entries instead of 65536, a factor of 16. Across the 78 layers
of the reference model that read a global table, the indexer work per query falls from $78 \cdot
65536 \approx 5.1$ million scores to $65536 + 77 \cdot 4096 \approx 0.38$ million, a factor of 13.
With an indexer key of 64 dimensions in FP4, each later layer scans 147 KB instead of 2.4 MB.

The paper introduces this in post-training, as a way to run an already trained model at longer context,
rather than as part of pre-training. That placement is consistent with what the mechanism does. It
changes which entries are considered, not what the model computes on them, and its exactness
condition is easiest to satisfy at the long contexts it is added for. In owlet1 the candidate mask is
computed but never consumed, because the six-layer schedule has no index layer after the candidate
source.

The mechanism is now complete at inference. What remains is how a model learns to use it, which is
the subject of the last section, because a hard top-$k$ passes no gradient to the scores that produced
it.
