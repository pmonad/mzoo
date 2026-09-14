---
title: "10 Sparse selection"
order: 10
---

# 10. Sparse selection

Chapter 9 left the tables small in memory and still expensive to read. Compression and sharing
shrink what is stored, by the ratio $m$ and by the number of layer groups, but every query in every
layer that has a global path still touches every entry of its table. The reference model has two
such tables: an encoder table at $m = 2$ read by 18 layers and a decoder table at $m = 1$ read by 20
layers, 38 of the model's 40 layers in all. At the extended context of 1{,}048{,}576 tokens that
read is 2.6 G from the encoder table and 5.6 G from the decoder table, about 8.2 G per decode
step — the ledger of chapter 9. At the native 65536 it is 522 M, about half a G. The cost per query is $O(S / m)$,
which is a smaller multiple of $S$ than chapter 5 had, but it is still a multiple of $S$.

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
the compressed tables of chapter 9, so the two reductions compose. Selection changes the bytes read
per step, not the bytes stored. The whole table remains in memory, and a query that needs a distant
entry can still reach it, which is the property that eviction methods give up.

Sections:

1. [The indexer](indexer.md): the scoring function and top-$k$.
2. [Hierarchical selection](hierarchical.md): a coarse pass before the fine one.
3. [Training the indexer](training.md): the problem with a hard top-$k$ and what the papers do about it.
