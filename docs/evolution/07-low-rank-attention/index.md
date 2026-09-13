# 7. Low-rank attention

Chapter 5 established the quantity that attention designs are measured by and took one factor out of
it. Grouped-query attention with $H_{kv} = 8$ brings the reference model from 2.6 MB of cache per
token to 328 KB, and from 344 GB at 131072 tokens to 43 GB. That is a large improvement and it is
still not enough. The weights of the reference model are 140 GB, so three long conversations now cost
more memory than the model itself, and every one of those 43 GB is read again at every decode step
for every sequence in flight.

Chapter 6 makes the imbalance worse rather than better. A mixture-of-experts model has most of its
parameters idle for any given token and reads only the active ones, and the weights it does read are
shared across the whole batch. The cache is not shared. A model with 1 trillion parameters and 32
billion active does the arithmetic of a small model at every step and carries the cache of a large
one, so for the largest open-weight models the cache is the dominant term in the cost of serving. The
factor grouped-query attention left untouched is the width of what each remaining head stores, and
that is what this chapter attacks.

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
dimension is at most $\min(H d, D)$. When a model has more total head width than residual width, as
DeepSeek-V2 does with $H d = 16384$ against $D = 5120$, the stored keys are redundant by construction
and the redundancy is exact rather than approximate. When $H d = D$, as in the reference model, the
bound gives nothing on its own, and the case rests instead on the measured singular values of $W_K$
and $W_V$, which decay fast enough that a few hundred directions carry most of the content. The
latent widths that the DeepSeek models use are far below the bound, so the design is justified by
training results and not by the rank argument alone. What the rank argument establishes is that
compression of this kind costs nothing in principle, which is not true of dropping heads.

This is a different move from the one in chapter 5. Grouped-query attention shrinks the cache by
removing heads, and what it removes is gone. Low-rank attention keeps all $H$ heads and stores a
shared summary that every head reads differently, so the head structure of the query side and the
output side survives intact. That is why the DeepSeek models report quality equal to or better than
full multi-head attention while caching less than multi-query attention does.

Sections:

- [Multi-head latent attention](mla.md): the V2 and V3 form, decoupled RoPE, weight absorption.
- [The single shared latent](shared-latent.md): the V4 form, where the latent is the key and the value.
