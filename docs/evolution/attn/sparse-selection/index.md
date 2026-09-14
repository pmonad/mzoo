---
title: "10 Sparse selection"
order: 10
---

# 10. Sparse selection

Chapter 9 left the global table small in memory and still expensive to read. Compression and sharing
shrink what is stored, by $m$ and by the number of layer groups, but every query in every layer that
has a global path still touches every entry of that table. On the reference model at 131072 tokens
with $m = 2$ that is 65536 entries of 288 bytes per layer, 18.9 MB per layer and about 1.5 GB per
decode step across the model. The cost per query is $O(S / m)$, which is a smaller multiple of $S$
than chapter 5 had, but it is still a multiple of $S$.

Most of those entries are irrelevant to a given query. Chapter 9 cited the heavy-hitter result of
Zhang et al. 2023 and the retrieval-head result of Wu et al. 2024, and both say the same thing from
different directions. Attention mass concentrates on few keys, and a query that could name those keys
in advance could read them alone and produce nearly the same output. The cost per query would then be
a constant $k$, independent of the context length.

Naming them is the difficulty. The scores that identify the important keys are exactly what full
attention computes, so a selector must approximate them without doing the work it is meant to avoid.
This is possible because the selector may be much narrower than the attention it serves and may
produce a ranking rather than a distribution. It scores every entry with a small dot product against
a narrow key, then the main attention reads the full-width entries of the winners only. The fixed
patterns of chapter 8, from Sparse Transformer to Longformer and BigBird, chose their keys in advance
and identically for every input. A learned selector chooses per query and per content, which is what
makes a small $k$ sufficient.

DeepSeek-V3.2 introduced this as DeepSeek Sparse Attention, or DSA, with a lightning indexer that
selects 2048 keys per query over the per-token MLA cache. V4.1 keeps the mechanism and applies it to
the compressed table of chapter 9, so the two reductions compose. Selection changes the bytes read
per step, not the bytes stored. The whole table remains in memory, and a query that needs a distant
entry can still reach it, which is the property that eviction methods give up.

Sections:

- [The indexer](indexer.md): the scoring function and top-$k$.
- [Hierarchical selection](hierarchical.md): a coarse pass before the fine one.
- [Training the indexer](training.md): the problem with a hard top-$k$ and what the papers do about it.
