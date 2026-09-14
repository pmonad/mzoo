---
title: "5 The KV cache"
order: 5
---

# 5. The KV cache

The previous three chapters changed how a block normalises, how it encodes position and how its
feed-forward layer is shaped. None of them changed the cost of running the model. This chapter is
the first that is about cost, and it introduces the quantity that drives most of the design
decisions in the rest of the book: the memory that attention needs at inference.

One anchor changes here. Chapters 1 to 6 do block arithmetic on a dense 65B reference model.
From this chapter on, the attention chapters use DeepSeek-V4.1-Flash as their reference model:
40 layers, 64 query heads of width 512, a residual stream of width $D = 5120$, a native context
of 65536 tokens extended to 1,048,576, and 552B backbone parameters — about 1 TB of weights in
bf16, with 8B active per token at prefill. Where this chapter needs a multi-head baseline that
V4.1 does not actually have, it uses a hypothetical full multi-head attention on V4.1's shape.

## Roadmap

1. **Why generation is different.** Training sees the whole sequence at once; generation cannot,
   and that gap creates the cache.
2. **How large it is.** One formula, evaluated on the reference model, produces the number the
   rest of the book attacks.
3. **Prefill and decode.** The two phases of serving a request are limited by different
   resources, and decode is the one that matters.
4. **Why the cache sets throughput.** Arithmetic intensity explains why the cache, not the
   arithmetic, sets the speed of decode.
5. **Where the cache lives.** PagedAttention fixed the waste in managing the cache without
   changing its size.
6. **Sharing keys and values across heads.** Multi-query and grouped-query attention shrink the
   cache by sharing, at a cost in stability.
7. **Precision and stability.** Sharing amplifies the hazard that QK-norm bounds.

## Why generation is different from training

During training the whole sequence is present at once. Attention for every position is computed in
one pass, and the keys and values of all positions are ordinary intermediate activations that are
used and discarded.

Generation is different. The model produces one token at a time, and each new token must attend to
every token before it. Take a three-token prompt, "the cat sat". The model computes the keys and
values of all three tokens, reads them with the last position's query, and produces a fourth token,
say "on". The fifth step must attend to all four tokens. The keys and values of "the cat sat" have
not changed — they are functions of those tokens alone — but the naive way to get them is to
recompute them, and reprocessing the whole prefix at every step would cost as much as reprocessing
the prompt at every step of generation. Instead the keys and values of every position, in every
layer, are kept in memory after they are first computed. This store is the key-value cache, or KV
cache. Each decode step computes the query, key and value for the one new token, appends the new
key and value to the cache, and attends over the cache.

The cache is therefore not an implementation detail. It is state that grows with the length of the
conversation and that must be resident for as long as the conversation is alive.

## How large it is

First, what is stored. Each attention head applies its own three small projection matrices to the
residual stream: a query projection, a key projection and a value projection, each producing a
vector of width $d$ per token. The query is used and discarded. The key and the value are what the
cache holds.

Every position, in every layer, stores one key and one value per head. With $H$ heads of width $d$
in each of $L$ layers,

$$
\text{KV bytes per token} = 2 \cdot L \cdot H \cdot d \cdot \text{bytes} .
$$

Read that back in words. The factor 2 counts the two stored objects, the key and the value. $L$
counts the layers, because every layer has its own attention with its own cache. $H$ counts the
heads, because in full multi-head attention every head stores its own key and value. $d$ is the
width of each, and "bytes" is the width of the number format — 2 for bf16.

The reference model has $L = 40$ layers and $H = 64$ heads of width $d = 512$. It does not store
per-head keys and values, but as a baseline suppose it did, in bf16:

$$
2 \cdot 40 \cdot 64 \cdot 512 \cdot 2 = 5{,}242{,}880 \text{ bytes} = 5.0 \text{ MB per token}.
$$

A single conversation at the native context of 65536 tokens then holds 320 GB of cache, and one at
the extended context of 1,048,576 tokens holds 5 TB. The weights of the model are about 1 TB
in bf16. One long conversation needs several × more memory for its cache than for the model,
and a server that wants to run many conversations at once needs that much for each of them.

## Prefill and decode

Serving a request has two phases with different character. In the prefill phase the prompt is
processed all at once, as in training. Every position is present, the matmuls have a large batch of
tokens on one side, and the accelerator's arithmetic units are the limit. The prefill phase also
writes the cache for the whole prompt. In the decode phase tokens are produced one at a time. Each
step has exactly one new token per sequence, and the matmuls have a single vector on one side. The
two phases are limited by different resources — V4.1 activates 8B parameters per token at prefill,
where the arithmetic is dense, and 16B at decode, where hyper-connections activate more — and the
design changes of this book are aimed mostly at the decode phase, because that is where a chat
model spends most of its time.

## Why the cache sets throughput

A useful measure of any computation on an accelerator is its arithmetic intensity: the number of
floating-point operations it performs divided by the number of bytes it moves from memory to do
them, in operations per byte. An accelerator has a fixed ratio of arithmetic throughput to memory
bandwidth, and a computation whose intensity is below that ratio cannot deliver enough work to the
arithmetic units per byte read. It is memory bound: its time is the time to move the bytes. An H100
performs about 990 TFLOPS of bf16 arithmetic and reads about 3.35 TB/s from its memory, a ratio
of roughly 300 operations per byte. A100 and the accelerators
before it have similar ratios.

Consider one decode step for one sequence in one block. Attention reads the whole cache of that
block, $S$ keys and $S$ values of width $H d$ each, and for each element read it performs about two
operations, one multiply and one add. Its intensity is therefore about one operation per byte, or
half that in bf16. Against a ratio of 300 the arithmetic units are idle more than 99 percent of the
time. The time of the step is the time to read the cache.

The weight matrices are also read once per step, and at about 1 TB they take longer to read than any
single cache. But the weights are the same for every sequence. If 64 sequences are decoded together
the weights are read once and used 64 times, an intensity of 64 operations per byte, and a larger
batch brings the weights close to the arithmetic limit. The cache does not batch. Each sequence has
its own, so 64 sequences read 64 caches, and the intensity of attention stays at one operation per
byte no matter how many sequences are in flight.

Two consequences follow. First, the number of sequences a server can decode at once is set by how
many caches fit in memory next to the weights. On a node with 1.5 TB of accelerator memory, the
reference model in bf16 leaves some 508 GB after the weights, which is enough for one sequence at
65536 tokens under the multi-head baseline — and would be enough for a dozen under grouped-query
attention, as the table below shows. Second, the time per step is set by the total cache bytes
of the batch, and shrinking the cache per token shortens every step in proportion.

This is why, from this chapter on, the size of the cache per token is the quantity that attention
designs are measured by. Every attention change in chapters 7 to 10 and the number formats of
chapter 12 can be read as an attack on the formula above, one factor at a time; chapter 6 attacks
the weights rather than the cache, which is why it sits outside that sequence. This chapter takes
the factor $H$.

## Where the cache lives

The cache also has to be managed. It grows by one entry per step, its final length is unknown when
the request starts, and requests start and finish at different times. Early serving systems
reserved a contiguous region for the maximum length per request and wasted most of it. Kwon et al.
2023, "Efficient Memory Management for Large Language Model Serving with PagedAttention" introduced
the scheme that current servers use: the cache is stored in fixed-size pages, a request holds a list
of pages rather than a region, and pages are allocated as the sequence grows and freed when it ends.
Attention kernels read through the page table. This removed most of the waste and let a server pack
several × as many sequences into the same memory, but it did not change the bytes per token. The
architectural changes below do.

## Sharing keys and values across heads

The sharing ladder — multi-query, then grouped-query — is the route that dense models climbed
historically, and it is worth walking in that order before seeing what the reference model does
instead. In multi-head attention every head has its own query, key and value projections. The query
side is where heads differ in what they look for, and there is no reason to give up that variety.
The key and value side is what is stored. If several heads could share one key and one value, the
cache would shrink by the sharing factor, and the query side could stay as it is.

### Multi-query attention

The extreme version keeps all $H$ query heads and gives them one shared key head and one shared
value head. To read its formula, two terms: each head scores a query against every key by a dot
product, and those raw scores are called logits; the softmax function converts a vector of logits
into positive weights that sum to one, which is how attention decides how much of each value to
read. Multi-query attention is then

$$
o_{h,i} = \sum_{j} \operatorname{softmax}_j\!\left(\frac{q_{h,i} \cdot k_j}{\sqrt{d}}\right) v_j .
$$

Compared with the multi-head formula of chapter 1, $k_j$ and $v_j$ have lost their head index. Each
head still computes its own attention pattern, because its query is its own, but all heads compare
against the same keys and read from the same values. The cache shrinks by a factor of $H$. On the
reference model's shape that is 80 KB per token and 5 GB at 64K tokens, in place of 320 GB.
Shazeer 2019, "Fast Transformer Decoding: One Write-Head is All You Need" introduced it.

The cost is quality and stability. All heads must find what they need in one shared key space, and
the value they read is the same vector weighted differently. Models trained this way lose a small
amount of quality against full multi-head attention, and training was reported to be less stable at
large scale.

### Grouped-query attention

Grouped-query attention sits between the two. It keeps $H_{kv}$ key and value heads, and each of
them serves a group of $H / H_{kv}$ query heads:

$$
o_{h,i} = \sum_{j} \operatorname{softmax}_j\!\left(\frac{q_{h,i} \cdot k_{g(h),j}}{\sqrt{d}}\right) v_{g(h),j},
\qquad g(h) = \left\lfloor \frac{h}{H / H_{kv}} \right\rfloor .
$$

The function $g$ maps a query head to its group. With $H = 64$ and $H_{kv} = 8$, heads 0 to 7 share
the first key-value pair, heads 8 to 15 the second, and so on. The cache shrinks by $H / H_{kv}$.
LLaMA-2 70B introduced $H_{kv} = 8$ with $H = 64$, and most dense models since have kept that
ratio. On the reference model's shape this is 640 KB per token and 40 GB at 64K tokens. Quality
is close to full multi-head attention, and the factor of eight is large enough to change what a
server can do. Ainslie et al. 2023, "GQA: Training Generalized Multi-Query Transformer Models from
Multi-Head Checkpoints" introduced it, together with the trick of converting an already-trained
multi-head checkpoint by averaging each group's heads.

The variants on the reference model's shape, at bf16:

| variant | $H_{kv}$ | per token | at 65536 tokens | at 1,048,576 tokens |
| --- | --- | --- | --- | --- |
| multi-head | 64 | 5.0 MB | 320 GB | 5 TB |
| grouped-query | 8 | 640 KB | 40 GB | 640 GB |
| multi-query | 1 | 80 KB | 5 GB | 80 GB |
| shared latent (ch 7) | — | 40 KB | 2.5 GB | 40 GB |

The last row is not a member of the sharing ladder. It is where the reference model actually sits:
DeepSeek-V4.1 stores one shared latent vector of width 512 per token per layer, 40 KB in bf16, and
chapter 7 shows how every head's key and value are reconstructed from it. It reaches the cache size
of multi-query attention without collapsing all heads into one key space.

## Precision and stability

Sharing amplifies one existing hazard. A key with an unusually large norm produces a large logit
against every query for reasons unrelated to its content, and with $H_{kv}$ shared key heads one
such key inflames all $H / H_{kv}$ query heads of its group at once instead of one head. This is
the instability reported for multi-query training, and it is why the sharing chapters of this book
pair every reduction with a bound on the logits. The standard bound is QK-norm, an RMSNorm applied
to $q$ and $k$ per head before the dot product, as in chapter 2:

$$
\hat q = \frac{q}{\operatorname{RMS}(q)}, \qquad \hat k = \frac{k}{\operatorname{RMS}(k)},
\qquad \big|\hat q \cdot \hat k\big| \le d \ \Rightarrow\ \Big|\frac{\hat q \cdot \hat k}{\sqrt d}\Big| \le \sqrt d .
$$

The Cauchy-Schwarz bound is the whole argument: after normalisation no input can push a logit past
$\sqrt{d}$ (23 for $d = 512$), whatever the residual stream did to $x$. A learnable scale per
dimension lets heads recover magnitude differences they actually want, and the normalisation is
removed from the cached object — it is applied where $q$ and $k$ are formed, so the cache stores
the normed key once and every use of it is bounded. Qwen3 and GLM-4.5 ship grouped-query attention
with exactly this pairing, and the same device returns in stronger form in chapters 7 and 9, where
the normed latent is what makes the low-precision cache of chapter 12 safe at all. The stabiliser
has a training-side counterpart in V4.1 itself, whose optimiser updates the query and key weight
matrices head by head — a per-head normalisation of the update rather than of the activation, for
the same reason: no head's outlier behaviour should set the scale of another's update.

## What the 2026 models do

Grouped-query attention is the default for dense and mid-size models. Qwen3 uses it with QK-norm,
a head width of 128, and 4 or 8 key-value heads depending on the model size. GLM-4.5 uses 96 query
heads with 8 key-value heads of width 128. Qwen3-Next goes further in its softmax layers, with 16
query heads and 2 key-value heads of width 256, together with an output gate on attention, and
relies on its linear-attention layers for the rest of the stack.

The largest mixture-of-experts models take a different route. DeepSeek-V3 and Kimi K2 use
multi-head latent attention, and DeepSeek-V4.1, this part's reference model, uses a single shared
latent per token. In those models the cache dominates serving cost even after grouped-query
sharing, and the low-rank methods of chapter 7 reduce it further while keeping every query head
distinct. The choice follows from where the cost sits: a dense 30B model is limited by its weights
and grouped-query attention is enough, while a 552B-parameter model with 8B active at prefill is
limited by its cache.

## Where this leads

Grouped-query attention reduces the number of key and value heads but leaves each remaining head
storing a full key and a full value of width $d$. The next step asks whether even that is necessary.
The keys and values of all heads are linear functions of the same input vector, so the information
they contain cannot exceed $D$ dimensions, and measured ranks are far lower. Chapter 7 stores one small
latent vector per token and reconstructs every head's key and value from it, which is how the
DeepSeek models reach the cache size of multi-query attention without its loss of quality — the
40 KB row of the table above.
