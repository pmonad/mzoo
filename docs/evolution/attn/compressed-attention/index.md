---
title: "9 Compressed attention"
order: 9
---

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

The effect on the model is large. Take the paper's own 40-layer configuration. The encoder's three
Full layers each store one FP4 latent of width 512 for every 2 tokens, and the decoder's five
sources (one Full, four Reindex) each store one for every token, so the model holds
$(3/2 + 5) \times 288\ \text{B} \approx 1.9$ KB per token of global cache, plus a fixed 2.6 MB of
windows. At 131072 tokens that is 245 MB per sequence. Without sharing or compression the same
40 layers would hold 11.5 KB per token in FP4 (1.5 GB), 40 KB in bf16 (5.2 GB), and a bf16
multi-head cache of the same depth would hold 172 GB.

Compression is not free of consequences. A pooled entry cannot recover a single token exactly, so the
design depends on the distant past being needed at the level of a passage rather than a word, and on
the window covering everything that must be exact. Sharing ties layers together, so a group's later
layers can only ask questions the source layer's representation can answer. The two sections below
build the compressor and then the sharing scheme, and each states what it gives up.

Sections:

- [The compressor](compressor.md): pooling $m$ tokens into one latent.
- [Sharing across layers](sharing.md): source and reuse layers, and the resulting key set per layer.
