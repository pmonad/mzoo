# 9. Compressed shared attention

Chapter 8 made the recent past cheap. A sliding window of 128 tokens with a learned sink costs a
fixed 5.2 MB per sequence on the reference model and 0.34 GFLOPs per token, at any context length.
What it does not provide is a route to the distant past. Some layers must keep one, and for those
layers nothing has changed since chapter 7. Each stores one entry per token per layer and each query
reads all of them, so both the storage and the read grow as $O(S)$.

Two reductions are available, and they multiply. The first is to store fewer entries than there are
tokens. Consecutive tokens are highly redundant at the level a global read cares about, so a summary
of a group of tokens can stand in for the group. The second is to store the global table fewer times
than there are layers. Keys and values in nearby layers are computed from residual states that differ
by one or two sublayers, so one table can serve several layers that read it with their own queries.
V4.1's compressed sparse attention, CSA, does both. Chapter 10 adds a third reduction that acts on
reads rather than storage, by having each query touch only a subset of the entries that are stored.

The effect on the reference model is large. Suppose the global table is pooled at $m = 2$ tokens per
entry, is stored by 2 layer groups instead of 80 blocks, and holds a latent of width 512 in the FP4
format of chapter 12 at 0.5625 bytes per element. A per-block per-token latent in that format is
23 KB per token across the model, so the shared and pooled global part is
$23 \text{ KB} \times 2 / (80 \cdot 2) \approx 0.3$ KB per token. At 131072 tokens that is about
38 MB per sequence, next to a fixed 5.2 MB for the windows. Chapter 5's multi-head baseline at the
same length is 344 GB and chapter 7's MLA is 12 GB.

Compression is not free of consequences. A pooled entry cannot recover a single token exactly, so the
design depends on the distant past being needed at the level of a passage rather than a word, and on
the window covering everything that must be exact. Sharing ties layers together, so a group's later
layers can only ask questions the source layer's representation can answer. The two sections below
build the compressor and then the sharing scheme, and each states what it gives up.

Sections:

- [The compressor](compressor.md): pooling $m$ tokens into one latent.
- [Sharing across layers](sharing.md): source and reuse layers, and the resulting key set per layer.
