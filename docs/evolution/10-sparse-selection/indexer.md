# The indexer

A selector has to produce, for every query position and every compressed entry, a number that ranks
the entries by how useful they would be to that query. Three requirements shape it. It must be far
cheaper per entry than the attention it replaces, because it looks at all $S / m$ entries while the
attention will look at $k$ of them. It must be query-dependent, because a fixed ranking is the fixed
pattern of chapter 8. And it must be a ranking only, since nothing downstream uses its magnitudes.

## Building the score

Start from the cheapest query-dependent score, a dot product between a query vector and a key vector
per entry. Give it several narrow heads so that different kinds of relevance can be detected
independently, and let the model decide per position which head's opinion matters. The indexer is
therefore a small separate scorer built like an attention layer. It has its own queries with a
few narrow heads and one key per compressed entry. For query position $s$ and entry $j$,

$$
a_{s,h,j} = \frac{\operatorname{ReLU}(q^{I}_{s,h} \cdot k^{I}_j)}{\sqrt{d_I}}, \qquad
I_{s,j} = \sum_{h=1}^{H_I} w_{s,h}\, a_{s,h,j}, \qquad w_s = \frac{W_w\, x_s}{\sqrt{H_I}} .
$$

The first term is one head's opinion of entry $j$, scaled by $\sqrt{d_I}$ for the same reason the
main attention scales by $\sqrt{d}$. The ReLU keeps scores non-negative, so a head that finds nothing
useful contributes zero instead of a negative vote that could cancel another head's positive one, and
the sum over heads behaves as a vote rather than a subtraction. The per-query head weights $w_s$ are a
linear function of the residual state at that position, so they set how much each indexer head
contributes at that position. A position that is looking for a rare name can weight a retrieval-like
head, and a position that is continuing a sentence can weight another. Unlike the main attention
there is no softmax over $j$, because only the order of the scores is used.

The indexer keys are derived from the same latents the compressor produces, before rotation, so a
source layer computes them once alongside the main table. Sharing follows the scheme of chapter 9. A
Reuse layer takes the source's selection unchanged and runs no indexer, and a Reindex layer runs its
own indexer over the source's keys.

## From scores to a mask

Selection is a hard top-$k$ turned into an additive mask:

$$
\mathcal{S}_s = \operatorname{top}k_j\, I_{s,j}, \qquad
M_{s,j} = \begin{cases} 0 & j \in \mathcal{S}_s \\ -\infty & \text{otherwise} \end{cases},
$$

and the main attention adds $M$ to its logits over the compressed part of the key set. The mask
applies to the compressed part only. The sliding window of chapter 8 is always read in full, so the
recent past is never at the mercy of the selector, and the learned sink still absorbs mass when
nothing selected is useful. Adding $-\infty$ before the softmax is the same operation as the causal
mask of chapter 1, which means a kernel can implement selection as a gather of $k$ entries followed
by ordinary attention over them. The paper uses $k = 2048$ and the smoke config uses 64. The global
read per query is fixed at $k$ entries whatever the context length.

## Cost

Take the reference model at 131072 tokens with $m = 2$, so 65536 compressed entries of width 512
stored in FP4 at 288 bytes each. Reading the whole table costs 18.9 MB per query per layer. Reading
$k = 2048$ selected entries costs 0.59 MB, a factor of 32, and that factor grows with the context
because $k$ does not. Across the 78 layers of the reference model that read a global table, a decode
step moves 46 MB instead of 1.5 GB.

The indexer must be paid for out of that saving. It reads one narrow key for all 65536 entries, so
its scan costs $65536 \cdot d_I \cdot 0.5625$ bytes per query per layer. This equals the 0.59 MB of
the selected read when $d_I = 16$, so for any usable indexer width the scan is the larger of the two
terms, and the total is dominated by a quantity that still grows with $S$. If the indexer key is 64
dimensions wide the scan is 2.4 MB, and the layer moves 3.0 MB in place of 18.9 MB.

The indexer is small by construction. The smoke config gives it $H_I = 4$ heads of width
$d_I = 32$, against $H = 4$ heads of width 64 for the main attention, and at scale the gap is far
larger. Its score matrix is still $S \times S/m$. V4.1 therefore computes it in FP4, covered in
chapter 12, and the next section avoids computing all of it.
