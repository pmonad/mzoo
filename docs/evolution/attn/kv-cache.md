---
title: "5 The KV cache"
order: 5
---

# 5. The KV cache

The previous three chapters changed how a block normalises, how it encodes position and how its
feed-forward layer is shaped. None of them changed the cost of running the model. This chapter is
the first that is about cost, and it introduces the quantity that drives most of the design
decisions in the rest of the book: the memory that attention needs at inference.

## Why generation is different from training

During training the whole sequence is present at once. Attention for every position is computed in
one pass, and the keys and values of all positions are ordinary intermediate activations that are
used and discarded.

Generation is different. The model produces one token at a time, and each new token must attend to
every token before it. Recomputing the keys and values of the whole prefix for every new token would
cost as much as reprocessing the prompt at every step. Instead the keys and values of every position,
in every layer, are kept in memory after they are first computed. This store is the key-value cache,
or KV cache. Each decode step computes the query, key and value for the one new token, appends the
new key and value to the cache, and attends over the cache.

The cache is therefore not an implementation detail. It is state that grows with the length of the
conversation and that must be resident for as long as the conversation is alive.

## How large it is

Every position, in every layer, stores one key and one value per head. With $H$ heads of width $d$
in each of $L$ layers,

$$
\text{KV bytes per token} = 2 \cdot L \cdot H \cdot d \cdot \text{bytes} .
$$

The factor 2 counts the key and the value. On the reference model in bf16 this is

$$
2 \cdot 80 \cdot 64 \cdot 128 \cdot 2 = 2.6 \text{ MB per token},
$$

so a single conversation at the training length of 4096 tokens holds 10.7 GB of cache, and one at
the long-context target of 131072 tokens holds 344 GB. The weights of the model, at two bytes per
parameter, are 140 GB. One long conversation needs more memory for its cache than for the model,
and a server that wants to run many conversations at once needs that much for each of them.

## Prefill and decode

Serving a request has two phases with different character. In the prefill phase the prompt is
processed all at once, as in training. Every position is present, the matmuls have a large batch of
tokens on one side, and the accelerator's arithmetic units are the limit. The prefill phase also
writes the cache for the whole prompt. In the decode phase tokens are produced one at a time. Each
step has exactly one new token per sequence, and the matmuls have a single vector on one side. The
two phases are limited by different resources, and the design changes of this book are aimed mostly
at the decode phase, because that is where a chat model spends most of its time.

## Why the cache sets throughput

A useful measure of any computation on an accelerator is its arithmetic intensity: the number of
floating-point operations it performs per byte it moves from memory. An accelerator has a fixed
ratio of arithmetic throughput to memory bandwidth, and a computation whose intensity is below that
ratio cannot keep the arithmetic units busy. It is memory bound. An H100 performs about 990 trillion
bf16 operations per second and reads about 3.35 trillion bytes per second from its memory, a ratio
of roughly 300 operations per byte. A100 and the accelerators before it have similar ratios.

Consider one decode step for one sequence in one block. Attention reads the whole cache of that
block, $S$ keys and $S$ values of width $H d$ each, and for each element read it performs about two
operations, one multiply and one add. Its intensity is therefore about one operation per byte, or
half that in bf16. Against a ratio of 300 the arithmetic units are idle more than 99 percent of the
time. The time of the step is the time to read the cache.

The weight matrices are also read once per step, and at 140 GB they take longer to read than any
single cache. But the weights are the same for every sequence. If 64 sequences are decoded together
the weights are read once and used 64 times, an intensity of 64 operations per byte, and a larger
batch brings the weights close to the arithmetic limit. The cache does not batch. Each sequence has
its own, so 64 sequences read 64 caches, and the intensity of attention stays at one operation per
byte no matter how many sequences are in flight.

Two consequences follow. First, the number of sequences a server can decode at once is set by how
many caches fit in memory next to the weights. On an eight-accelerator node with 640 GB, the
reference model in bf16 leaves 500 GB after the weights, which is enough for 47 sequences at 4096
tokens and one sequence at 131072 tokens. Second, the time per step is set by the total cache bytes
of the batch, and shrinking the cache per token shortens every step in proportion.

This is why, from this chapter on, the size of the cache per token is the quantity that attention
designs are measured by. Every attention change in chapters 7 to 10 and the number formats of
chapter 12 can be read as an attack on the formula above, one factor at a time. This chapter takes
the factor $H$.

## Where the cache lives

The cache also has to be managed. It grows by one entry per step, its final length is unknown when
the request starts, and requests start and finish at different times. Early serving systems
reserved a contiguous region for the maximum length per request and wasted most of it. Kwon et al.
2023, "Efficient Memory Management for Large Language Model Serving with PagedAttention" introduced
the scheme that current servers use: the cache is stored in fixed-size pages, a request holds a list
of pages rather than a region, and pages are allocated as the sequence grows and freed when it ends.
Attention kernels read through the page table. This removed most of the waste and let a server pack
several times as many sequences into the same memory, but it did not change the bytes per token. The
architectural changes below do.

## Sharing keys and values across heads

In multi-head attention every head has its own query, key and value projections. The query side is
where heads differ in what they look for, and there is no reason to give up that variety. The key
and value side is what is stored. If several heads could share one key and one value, the cache would
shrink by the sharing factor, and the query side could stay as it is.

### Multi-query attention

The extreme version keeps all $H$ query heads and gives them one shared key head and one shared
value head:

$$
o_{h,i} = \sum_{j} \operatorname{softmax}_j\!\left(\frac{q_{h,i} \cdot k_j}{\sqrt{d}}\right) v_j .
$$

Compared with the multi-head formula of chapter 1, $k_j$ and $v_j$ have lost their head index. Each
head still computes its own attention pattern, because its query is its own, but all heads compare
against the same keys and read from the same values. The cache shrinks by a factor of $H$. On the
reference model that is 41 KB per token and 5.4 GB at 131072 tokens, in place of 344 GB.

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

The function $g$ maps a query head to its group. Heads 0 to 7 share the first key-value pair, heads
8 to 15 the second, and so on. The cache shrinks by $H / H_{kv}$. LLaMA-2 70B introduced $H_{kv} = 8$
with $H = 64$, and most dense models since have kept that ratio. On the reference model this is
328 KB per token and 43 GB at 131072 tokens. Quality is close to full multi-head attention, and the
factor of eight is large enough to change what a server can do.

The three variants on the reference model, at bf16:

| variant | $H_{kv}$ | per token | at 4096 tokens | at 131072 tokens |
| --- | --- | --- | --- | --- |
| multi-head | 64 | 2.6 MB | 10.7 GB | 344 GB |
| grouped-query | 8 | 328 KB | 1.3 GB | 43 GB |
| multi-query | 1 | 41 KB | 168 MB | 5.4 GB |

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
$\sqrt{d}$ (11 for $d = 128$), whatever the residual stream did to $x$. A learnable scale per
dimension lets heads recover magnitude differences they actually want, and the normalisation is
removed from the cached object — it is applied where $q$ and $k$ are formed, so the cache stores
the normed key once and every use of it is bounded. Qwen3 and GLM-4.5 ship grouped-query attention
with exactly this pairing, and the same device returns in stronger form in chapters 7 and 9, where
the normed latent is what makes the low-precision cache of chapter 12 safe at all. The stabiliser
has a training-side counterpart in V4.1, whose optimiser updates the query and key weight matrices
head by head — a per-head normalisation of the update rather than of the activation, for the same
reason: no head's outlier behaviour should set the scale of another's update.

## What the 2026 models do

Grouped-query attention is the default for dense and mid-size models. Qwen3 uses it with QK-norm,
a head width of 128, and 4 or 8 key-value heads depending on the model size. GLM-4.5 uses 96 query
heads with 8 key-value heads of width 128. Qwen3-Next goes further in its softmax layers, with 16
query heads and 2 key-value heads of width 256, together with an output gate on attention, and
relies on its linear-attention layers for the rest of the stack.

The largest mixture-of-experts models take a different route. DeepSeek-V3 and Kimi K2 use
multi-head latent attention, and DeepSeek-V4.1 uses a single shared latent per token. In those
models the cache dominates serving cost even after grouped-query sharing, and the low-rank methods
of chapter 7 reduce it further while keeping every query head distinct. The choice follows from
where the cost sits: a dense 30B model is limited by its weights and grouped-query attention is
enough, while a trillion-parameter model with 32B active parameters is limited by its cache.

## Where this leads

Grouped-query attention reduces the number of key and value heads but leaves each remaining head
storing a full key and a full value of width $d$. The next step asks whether even that is necessary.
The keys and values of all heads are linear functions of the same input vector, so the information
they contain cannot exceed $D$ dimensions, and measured ranks are far lower. Chapter 7 stores one small
latent vector per token and reconstructs every head's key and value from it, which is how the
DeepSeek models reach the cache size of multi-query attention without its loss of quality.
