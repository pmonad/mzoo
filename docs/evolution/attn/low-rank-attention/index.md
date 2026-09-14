---
title: "7 Low-rank attention"
order: 7
---

# 7. Low-rank attention

Chapter 5 established the quantity that attention designs are measured by and took one factor out
of it. On the reference model — 40 layers, 64 heads of width 512 — full multi-head attention stores
$2 \cdot 40 \cdot 64 \cdot 512 \cdot 2 = 5{,}242{,}880$ bytes, or 5.0 MB, of cache per token.
Grouped-query attention with $H_{kv} = 8$ brings that to $2 \cdot 40 \cdot 8 \cdot 512 \cdot 2 =
655{,}360$ bytes, or 640 KB, a factor of eight. At the native context of 65536 tokens that is 40 GB
per sequence. Extending the context to the 1,048,576 the reference model is trained to reach makes
it 640 GB. The weights themselves are about 1 TB in bf16 — 552B parameters at two bytes
each — so two conversations at the extended context now cost more memory than the model itself, and
every one of those bytes is read again at every decode step for every sequence in flight.

Chapter 6 makes the imbalance worse rather than better. A mixture-of-experts model has most of its
parameters idle for any given token and reads only the active ones, and the weights it does read are
shared across the whole batch. The cache is not shared. The reference model has 552B
parameters of which 8B are active, so it does the arithmetic of a small model at every step
and carries the cache of a large one, and for the largest open-weight models the cache is the
dominant term in the cost of serving. The factor grouped-query attention left untouched is the width
of what each remaining head stores, and that is what this chapter attacks.

## The keys and values of one token are not independent

GQA, described in chapter 5, shrinks the cache by sharing key/value heads, but it still stores a
separate key and value vector per remaining head. DeepSeek-V2 observed that the per-head $k$ and
$v$ are all linear functions of the same input $x$, so the information they carry is at most
$D$-dimensional.
The cache can store one low-dimensional vector per token and reconstruct every head's key and value
from it on read.

The argument is worth stating exactly, because it bounds what is possible. Stack the key projections
of all heads into one matrix $W_K \in \mathbb{R}^{H d \times D}$, so that the concatenated keys of a
token are $W_K x$. Whatever $H$ and $d$ are, this vector lies in the column space of $W_K$, whose
dimension is at most $\min(H d, D)$. In plain terms, "the keys lie in a low-dimensional subspace"
means this: the concatenated key looks like 32768 independent numbers, but every one of them is a
fixed linear combination of the same 5120 entries of $x$, so only 5120 directions of content exist
to begin with, and if the matrix is biased to use fewer, fewer still. Whether it is so biased is
read off the singular values: the singular value decomposition writes $W_K$ as a sum of rank-one
pieces ordered from most to least contributory, and how fast the singular values decay says how few
pieces carry the content.

The reference model itself is over-wide in this sense: $H d = 64 \cdot 512 = 32768$ against
$D = 5120$, so its stored keys are redundant by construction and the redundancy is exact rather than
approximate. DeepSeek-V2 was the same shape of situation, $H d = 16384$ against $D = 5120$. When
$H d \le D$ the bound gives nothing on its own, and the case rests instead on the measured singular
values of $W_K$ and $W_V$, which decay fast enough that a few hundred directions carry most of the
content. The latent widths that the DeepSeek models use are far below the bound, so the design is
justified by training results and not by the rank argument alone. What the rank argument establishes
is that compression of this kind costs nothing in principle, which is not true of dropping heads.

This is a different move from the one in chapter 5. Grouped-query attention shrinks the cache by
removing heads, and what it removes is gone. Low-rank attention keeps all $H$ heads and stores a
shared summary that every head reads differently, so the head structure of the query side and the
output side survives intact. That is why the DeepSeek models report quality equal to or better than
full multi-head attention while caching less than multi-query attention does.

Sections:

1. [Multi-head latent attention](mla.md): the V2 and V3 form, decoupled RoPE, weight absorption.
2. [The single shared latent](shared-latent.md): the V4 form — which is the form the reference
   model itself takes — where the latent is the key and the value.
