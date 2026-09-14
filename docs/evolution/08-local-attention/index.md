# 8. Local attention and sinks

Chapters 5 and 7 reduced what one token stores. Grouped-query attention removed most of the key and
value heads, and the latent of chapter 7 replaced the surviving per-head vectors with a single vector
of width 512 per token per block. Both changes attack the bytes the cache holds. Neither changes how
many entries of that cache a query touches, and neither changes how much arithmetic attention
performs. A query still compares itself against every position before it, so both quantities grow
with the length of the conversation. This chapter is the first to reduce the number of positions a
layer looks at, and chapters 9 and 10 continue that reduction.

## Storing less and reading less

It helps to separate two quantities that chapter 5 treated together.

The first is the bytes stored per token. This sets how many sequences fit in memory beside the
weights, and it is what chapters 5 and 7 reduced, from 2.6 MB per token for multi-head attention to
82 KB for the single shared latent.

The second is the bytes read per query and the operations performed per query. This sets the time of
a decode step and the arithmetic of a prefill. Reducing storage reduces reads in proportion when
every query reads everything, which is the case so far, because the two are then the same table seen
twice. They come apart as soon as a query is allowed to skip part of the table. A windowed layer
stores less and reads less. The sparse selection of chapter 10 stores everything and reads a fixed
subset. Chapter 9 does both at once. Keeping the two quantities distinct makes the ledger of the next
three chapters readable.

## What attention costs at 131072 tokens

Attention does two products per key. The query is compared against the key, and the value is added
into the output weighted by the resulting probability. Each is about $2d$ operations in one head, so
one token attending over $S$ keys with $H$ heads costs about

$$
4 \cdot S \cdot H \cdot d = 4 \cdot S \cdot D
$$

operations in one block. On the reference model, with $D = 8192$ and $L = 80$ blocks, that is 10.7
GFLOPs per token at the training length of $S = 4096$. The weights cost about two operations per
parameter, so about 130 GFLOPs per token for 65 billion parameters. At the training length attention
is under a tenth of the block's arithmetic, which is why the earlier chapters could count parameters
and treat that as the cost of the model.

At the long-context target of $S = 131072$ the same expression gives 344 GFLOPs per token. Attention
arithmetic is then about 2.6 times the arithmetic of every weight matrix in the model combined. The
memory side moves with it. Even with the single shared latent of chapter 7 at 82 KB per token, one
decode step at 131072 tokens reads 10.7 GB of cache, and 3.0 GB if that cache is held in the FP4
format of chapter 12.

The two phases of chapter 5 feel this differently. In prefill every position attends over every
earlier position, so the attention cost of a prompt of length $S$ is about $2 S^2 D$ operations per
block, which at $S = 131072$ over 80 blocks is about 22 PFLOPs against about 17 PFLOPs for the
weights. Prefill is limited by arithmetic, and attention is now the larger half of it. In decode each
step attends over the whole cache once, so the cost per step is linear in $S$ and, at about one
operation per byte read, limited by memory bandwidth. A change that removes keys helps prefill
quadratically and decode linearly.

Compressing what a token stores lowers the constant in front of $S$. It does not remove the $S$.

## What FlashAttention did and did not change

One well-known result should be placed here so that it is not mistaken for a solution to the problem
of this chapter. Dao et al. 2022, "FlashAttention: Fast and Memory-Efficient Exact Attention with
IO-Awareness", observed that a standard attention implementation writes the full $S \times S$ score
matrix to memory and reads it back twice, once for the softmax and once for the weighted sum. Their
kernel tiles the computation, keeps each tile of scores in on-chip memory and accumulates the softmax
with running normalisers, so the score matrix is never written out. Memory use falls from $O(S^2)$ to
$O(S)$ and prefill becomes several times faster. Dao 2023, "FlashAttention-2: Faster Attention with
Better Parallelism and Work Partitioning", improved the work partitioning further. Chapter 16 opens
with the mechanism — the online softmax and the tile schedule that make the removal exact.

FlashAttention is exact. It computes the same output as the naive implementation, and it performs the
same $4 S D$ operations per token per block. What it removes is traffic to and from memory for
intermediate scores, which is a prefill cost. It leaves the arithmetic of prefill unchanged and it
leaves the decode-time cache untouched, because the cache is a real store of keys and values, not an
intermediate. Every design in this chapter and the two that follow reduces the number of keys and is
therefore complementary to FlashAttention, and current sparse kernels are built on top of it.

## Fixed sparse patterns

Most of what a token needs is close to it. Adjacent words, the current sentence and the current
function body carry most of the information required to predict the next token. The distant past is
needed rarely and by few heads. Three designs from 2019 and 2020 turned this observation into fixed
patterns, chosen in advance and identical for every input.

Child et al. 2019, "Generating Long Sequences with Sparse Transformers", factorised the attention
matrix into two sparse patterns applied in alternating heads or layers, a local one covering the last
few positions and a strided one covering every $n$th position. The composition of the two connects
any pair of positions in two steps, and the cost per token falls from $O(S)$ to $O(\sqrt{S})$.

Beltagy et al. 2020, "Longformer: The Long-Document Transformer", used a sliding window, dilated
windows in some heads to widen the reach at the same cost, and a small set of positions marked as
global that attend to everything and are attended to by everything. The global positions are chosen
by the task, for example the question tokens in question answering.

Zaheer et al. 2020, "Big Bird: Transformers for Longer Sequences", combined a window, a few global
tokens and a few randomly chosen keys per query, and showed that the resulting sparse attention
retains the theoretical expressiveness of full attention, in the sense of universal approximation of
sequence functions and Turing completeness.

Three properties of this family carry into the modern designs. The window is the component that
carries most of the quality. Some direct route to distant positions is needed beside the window.
And a fixed pattern is chosen without looking at the content, which is what chapter 10 changes.

## Attending to a window of the recent past

Restrict each query to the last $W$ keys:

$$
o_i = \sum_{j = i - W + 1}^{i} p_{ij}\, v_j .
$$

The sum runs over a fixed number of positions rather than over the whole prefix. Per-token cost drops
from $O(S)$ to $O(W)$, and the cache for such a layer holds $W$ entries per sequence regardless of
length. On the reference model with $W = 128$ the arithmetic of one windowed layer is $4 \cdot 128
\cdot 8192$ operations per token, so 0.34 GFLOPs per token across 80 blocks in place of 344. The
storage is fixed too. Holding 128 latents of width 512 per block in FP8 costs $80 \cdot 128 \cdot 512
\cdot 1$ byte, about 5.2 MB per sequence, whether the conversation is a thousand tokens or a million.
The window is the one component in the whole book whose cost does not depend on the context length.

Information still travels further than $W$, because each layer moves it another $W$ positions
forward. After $\ell$ windowed layers a token can be influenced by one $\ell \cdot W$ positions back,
so the reference model's 80 layers with $W = 128$ have a nominal reach of 10240 tokens. That path is
indirect. It passes through the residual state of every intervening position, competing there with
everything else those positions carry, and a model with only windowed layers loses recall of specific
distant content such as a name or a number stated once. Jiang et al. 2023, "Mistral 7B", used
$W = 4096$ in every layer, which is the whole context at the reference training length and therefore
saves nothing below it. Later models interleave windowed and full layers so that some layers keep a
direct route to any position.

## Attention sinks

A softmax must place its mass somewhere. Clark et al. 2019, "What Does BERT Look At? An Analysis of
BERT's Attention", found heads placing most of their weight on the separator or the first token when
they had nothing useful to attend to. In decoder models the first token plays this role, as
characterised by Xiao et al. 2023, "Efficient Streaming Language Models with Attention Sinks", and
Sun et al. 2024, "Massive Activations in Large Language Models", tie it to a few very large
activations at those positions. When a sliding window evicts the first token the distribution of
every head shifts, which is why windowed inference degraded past $W$. StreamingLLM kept the first
tokens in the cache permanently.

The underlying difficulty is that the softmax has no null option. Its outputs are non-negative and
sum to one, so a head that finds nothing worth reading must still return a convex combination of
values. A model trained without a remedy learns one by allocating a position whose value vector is
close to useless and parking the mass there. That position must then stay in the cache forever.

gpt-oss and V4.1 make the sink explicit as a learned logit per head that enters the softmax
denominator but contributes no value:

$$
p_{h,ij} = \frac{e^{s_{h,ij}}}{\sum_{j'} e^{s_{h,ij'}} + e^{\sigma_h}}, \qquad
\sum_j p_{h,ij} = \frac{\sum_{j'} e^{s_{h,ij'}}}{\sum_{j'} e^{s_{h,ij'}} + e^{\sigma_h}} \le 1 .
$$

The extra term $e^{\sigma_h}$ is a single learned scalar per head. It absorbs probability mass
without contributing to the output, so the weights over real keys sum to less than one and the head
can scale its whole output down when no key matches. The head no longer needs a token to serve as the
sink, so no position has to be pinned in the cache, and the window can be made as small as the design
requires. V4.1 uses $W = 128$.

The same large activations reappear as a quantisation difficulty in chapter 12.

## Precision and stability

The sink earns its place in the denominator on numerical grounds as well. It puts a floor under
the softmax denominator, $\ell \ge e^{\sigma_h - m}$ whatever the keys contain, so a row whose
window is uninformative decays its output smoothly through the learned $\sigma_h$ instead of
concentrating on an arbitrary key. And because the sink mass absorbs whatever the real keys fail to
earn, errors in the keys — including quantisation errors — scale the real-key weights down
uniformly rather than redistributing them chaotically. A window without a sink has neither
property: every key error lands directly in the output.

The window is also where the cache is hardest to compress. V4.1 keeps the sliding-window cache in
FP8 and states the reason: sensitivity to quantisation. A recent token can dominate its row — for
the first token of a sequence the distribution is a single 1, and near the start of a document a
row's effective sample size $1/\sum_j p_{ij}^2$ is close to one — so the error of one quantised
entry moves the output by that error, with no averaging across keys to hide it. Distant keys, by
contrast, are read in their thousands and their independent errors cancel. The design follows the
statistics: the many-read distant table takes the aggressive format of chapter 12, and the
few-read window takes FP8. The compressed entries that sit between the two are normalised before
storage, which is the argument of the next chapter.

## Hybrid schedules

With a sink in place, a model can be mostly local and only occasionally global. The schedule then
becomes a design parameter with a simple ledger. If a fraction $f$ of layers sees the whole context
and the rest see a window, the long-context part of both the arithmetic and the cache reads is
multiplied by $f$, while the windowed layers contribute a fixed amount. One layer in four gives a
factor of four on everything that grows with $S$, and every layer still reaches any position after at
most a few hops through the nearest global layer below it.

V4.1's 40 layers — 20 encoder, 20 decoder — run the first two on the window alone and the rest on
the window plus a compressed view of the whole context, which chapter 9 describes. The window part
of every layer is the layer's own. Only the global part is shared. The choice to make the first
layers local is deliberate. Early layers work with token identity and local syntax, where a window
is sufficient, and the representations that a global read needs to compare are formed higher up.

## What the 2026 models do

gpt-oss (2025) and Gemma 3 (2025) alternate sliding-window layers with full layers. gpt-oss uses a
128-token window and learned sinks, the form given above. Kimi K2 (2025) takes the other position and
keeps full MLA attention at every layer at 128K context, which its latent of width 512 with 64
rotated dimensions makes affordable.

A second family replaces the windowed layer rather than narrowing it. Qwen3-Next (2025) and Qwen3.5
(2026) run three Gated DeltaNet linear-attention layers per softmax layer, and Kimi Linear (2025)
runs three Kimi Delta Attention layers per MLA layer. A linear layer keeps a state of fixed size per
sequence, so it costs the same per token at any length. It differs from a window in what it forgets.
A window has a hard boundary and perfect recall inside it, while a recurrent state has no boundary
and lossy recall throughout. In these hybrids the cache of the whole model is set by the one softmax
layer in four.

Two routes to long context are therefore in use. Recurrent hybrids shrink the cache by replacing most
softmax layers. Sparse and compressed attention keeps every layer softmax and reads a subset of the
past. Chapters 9 and 10 follow the second route, which is the one V4.1 takes.

## Where this leads

In owlet1 the window mask is built once in `decoder.py` and the sink is in `attention.py`. The window
handles the recent past at fixed cost, and the sink lets a head decline to use it. What remains is
the global path, which the non-windowed layers still pay $O(S)$ for in both storage and reads.
Chapter 9 attacks the storage in two ways at once, by keeping fewer entries than there are tokens and
by keeping them in fewer layers than there are layers. Chapter 10 then attacks the reads.
