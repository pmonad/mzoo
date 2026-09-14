---
title: "Indexer"
order: 1
---

## The indexer

A selector has to produce, for every query position and every compressed entry, a number that ranks
the entries by how useful they would be to that query. Three requirements shape it. It must be far
cheaper per entry than the attention it replaces, because it looks at all $S / m$ entries while the
attention will look at $k$ of them. It must be query-dependent, because a fixed ranking is the fixed
pattern of chapter 8. And it must be a ranking only, since nothing downstream uses its magnitudes.

### Building the score

Start from the cheapest query-dependent score, a dot product between a query vector and a key vector
per entry. Give it several narrow heads so that different kinds of relevance can be detected
independently, and let the model decide per position which head's opinion matters. The indexer is
therefore a small separate scorer built like an attention layer. It has its own queries with a
few narrow heads and one key per compressed entry. For query position $i$, indexer head $h$ and
entry $j$,

$$
a_{i,h,j} = \frac{\operatorname{ReLU}(q^{I}_{i,h} \cdot k^{I}_j)}{\sqrt{d_I}}, \qquad
I_{i,j} = \sum_{h=1}^{H_I} w_{i,h}\, a_{i,h,j}, \qquad w_i = \frac{W_w\, x_i}{\sqrt{H_I}} .
$$

Read that back in words. The first equation is one head's opinion of entry $j$: the query of head
$h$ at position $i$ dot-multiplied against the indexer key of entry $j$, passed through a ReLU and
scaled by $\sqrt{d_I}$ for the same reason the main attention scales by $\sqrt{d}$. The second
equation combines the $H_I$ opinions into one score per entry, and the third says the combining
weights are a plain linear map of the residual state $x_i$ at the querying position, normalised by
$\sqrt{H_I}$ so the sum stays on the same scale as a single head's score.

The ReLU keeps each head's score non-negative, so a head that finds nothing useful contributes zero
instead of a negative vote that could cancel another head's positive one, and the sum over heads
behaves as a vote rather than a subtraction. The per-query head weights $w_i$ set how much each
indexer head contributes at that position. A position that is looking for a rare name can weight a
retrieval-like head, and a position that is continuing a sentence can weight another. Unlike the
main attention there is no softmax over $j$, because only the order of the scores is used.

The indexer keys are derived from the same latents the compressor produces, before rotation, so a
source layer computes them once alongside the main table. Sharing follows the scheme of chapter 9. A
Reuse layer takes the source's selection unchanged and runs no indexer, and a Reindex layer runs its
own indexer over the source's keys. The reference model has eight index layers — 2, 8 and 14 in the
encoder, 20, 24, 28, 32 and 36 in the decoder — and the other thirty global layers reuse.

### From scores to a mask

Selection is a hard top-$k$: the $k$ entries with the highest scores are kept and every other entry
is dropped, with nothing in between. Top-$k$ selection means exactly this — sort the entries by
score, keep the top $k$, discard the rest — and it is "hard" because an entry is either fully in or
fully out. The result is turned into an additive mask:

$$
\mathcal{S}_i = \operatorname{top}k_j\, I_{i,j}, \qquad
M_{i,j} = \begin{cases} 0 & j \in \mathcal{S}_i \\ -\infty & \text{otherwise} \end{cases},
$$

and the main attention adds $M$ to its logits over the compressed part of the key set. Adding
$-\infty$ to a logit forces its softmax weight to exactly zero, because the softmax is a
normalised exponential and $e^{-\infty} = 0$; every unselected entry vanishes from the
distribution while the selected ones share the mass. That is the same operation as the causal mask
of chapter 1, which means a kernel can implement selection as a gather of $k$ entries followed by
ordinary attention over them. The mask applies to the compressed part only. The sliding window of
chapter 8 is always read in full, so the recent past is never at the mercy of the selector, and the
learned sink still absorbs mass when nothing selected is useful. V4.1 uses $k = 512$ with 32 indexer
heads of width 128; the smoke config uses $k = 64$ with 4 heads of width 32. The global read per
query is fixed at $k$ entries whatever the context length.

The whole path at decode time, with the scan where it actually runs:

$$
\begin{aligned}
&\ //\ \text{every decode step, every index layer, every query position } i: \\
&\ I_{i,\cdot} \leftarrow \text{score all visible entries against } x_i
\qquad \text{// the scan: } S/m \text{ FP4 keys, one narrow dot product each} \\
&\ \mathcal{S}_i \leftarrow \operatorname{topk}_j\, I_{i,j}
\qquad \text{// the } k \text{ winners; a ranking, no softmax} \\
&\ M_{i,j} \leftarrow \begin{cases} 0 & j \in \mathcal{S}_i \\ -\infty & \text{otherwise} \end{cases} \\[4pt]
&\ //\ \text{main attention: window read in full, selected entries gathered} \\
&\ o \leftarrow \operatorname{attn}\!\big(q_i,\ \mathcal{W}_i \cup \{\,c_j : j \in \mathcal{S}_i\,\},\ +M\big)
\end{aligned}
$$

### Cost

Work the example on the encoder table, the smaller of the two. At the extended context of
1{,}048{,}576 tokens with $m = 2$ it holds 524288 entries of width 512 stored in FP4 at 288 bytes
each. Reading the whole table costs 144 M per query per layer. Reading $k = 512$ selected entries
costs 144 K, a factor of 1024, and that factor grows with the context because $k$ does not. The
decoder table at $m = 1$ has twice the entries, 288 M per query per layer at the same context, and
the same selected read. Across the 38 layers with a global path, a decode step moves about 5.3 M of
selected entries instead of the 8.2 G of chapter 9's ledger.

The indexer must be paid for out of that saving. It reads one narrow key for every visible entry, so
its scan costs $S / m \cdot d_I \cdot 0.5625$ bytes per query per layer — one FP4 key of width
$d_I$ per entry. On the encoder table at 1M that is $524288 \cdot 128 \cdot 0.5625 = 36$ M;
at the native 65536, where the table holds 32768 entries, it is 2.25 M. The scan equals the 144 K
of the selected read only when $d_I = 0.5$, so for any usable indexer width the scan is by far the
larger of the two terms, and the total is dominated by a quantity that still grows with $S$.

When the indexer runs matters as much as what it reads. The block above executes once per index
layer per decode step, and its first line — the scan — is the one term in it that still grows with
$S$. That is the reason the hierarchy of the next section exists: the selected read is already
constant, and the hierarchy is what stops the scan growing with context too.

The indexer is small by construction. The smoke config gives it $H_I = 4$ heads of width
$d_I = 32$, against $H = 4$ heads of width 64 for the main attention, and at scale the gap is far
larger: 32 heads of width 128 against $H = 64$ heads of width 512. Its score matrix is still
$S \times S/m$. V4.1 therefore computes it in FP4, covered in chapter 12, and the next section
avoids computing all of it.

### Precision and stability

The indexer tolerates four-bit arithmetic in a way the main attention does not, and the reason is
its consumer. A score that is wrong by a few percent changes nothing, because only the order of the
scores is used: an entry is selected or not, and the boundary cases that flip are the ones the top-$k$
was indifferent between. The main attention consumes score *differences* through a softmax; the
indexer consumes score *ranks*. V4 reports the measurement: indexer queries and keys cached, loaded
and multiplied entirely in FP4 give a 2× faster top-$k$ at 99.7 percent KV-entry recall. The scale
format helps here too — a ue8m0 scale is a power of two, so applying it inside the matmul is an
exponent adjustment that introduces no rounding of its own. The scores themselves are accumulated in
fp32 and cast only to bf16 on the way out, which is the one precision the ranking does pay for.

Stability in training is a different question, and it is handled by construction rather than by
precision. The indexer is trained on a detached KL loss against the main attention's distribution.
"Detached" is an operational word: the loss is computed on the indexer's scores, but the stop is
placed so that no gradient flows back through the selection into the main path — the main
attention's weights learn from the language loss alone and never feel the indexer's objective.
The fake-quantisation of the indexer's queries and keys needs a straight-through estimator, a
trick that pretends a non-differentiable step is the identity for gradient purposes; chapter 12 and
the training section below define it properly. That arrangement, and its one known defect, are the
subject of the training section.
