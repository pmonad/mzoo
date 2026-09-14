---
title: "MLA"
order: 1
---

## Multi-head latent attention

Grouped-query attention left the reference model storing $2 \cdot 8 \cdot 128 = 2048$ elements per
token per layer, one key and one value of width 128 for each of eight key-value heads. Those 2048
numbers are produced from a single input vector of width $D = 8192$ by a fixed linear map, so they
are not 2048 independent quantities. Multi-head latent attention, introduced with DeepSeek-V2 in
2024, stores the small vector that generates them instead and rebuilds the rest on read. It is used
by DeepSeek-V3 and by Kimi K2, and it is the form that the single shared latent of the next section
simplifies.

### Compress, then expand

Project the input to a latent of width $d_c \ll H d$, cache the latent, and expand to per-head keys
and values on read:

$$
c_j = W_{DKV}\, x_j \in \mathbb{R}^{d_c}, \qquad
k_{h,j} = W_{UK,h}\, c_j, \qquad
v_{h,j} = W_{UV,h}\, c_j .
$$

$W_{DKV}$ is the down projection, shared by every head, and $W_{UK,h}$ and $W_{UV,h}$ are the up
projections of head $h$. Reading the three equations in order: one matrix squeezes the token state
into $d_c$ numbers, and $2H$ small matrices reopen those numbers into a distinct key and a distinct
value for every head. Every head still has its own key space and its own value space, since the up
projections are not shared. What is shared is the vector they all read from.

The cache holds $c_j$ only. In DeepSeek-V2, $d_c = 512$ against $H d = 128 \times 128 = 16384$, a
32-fold reduction relative to full multi-head attention and about 4-fold relative to GQA with
8 KV heads. Quality is equal to or better than the dense baseline because all $H = 128$ heads are
retained.

The factorisation also makes the attention parameters smaller, which is not the point of the design
but is worth recording. On the reference model shape, with $d_c = 512$, the key-value path holds
$W_{DKV}$ at $512 \cdot 8192 = 4.2$ million parameters and the two sets of up projections at
$8192 \cdot 512 = 4.2$ million each, about 12.6 million in all, against $2 D^2 = 134$ million for the
full-rank $W_K$ and $W_V$ of chapter 1. The same construction is the one Hu et al. 2021, "LoRA:
Low-Rank Adaptation of Large Language Models", uses to make a weight update cheap, applied here to
the weights themselves rather than to an update.

The query is compressed the same way. This saves no cache, but it saves parameters and activation
memory during training:

$$
q_{h,i} = W_{UQ,h}\, \operatorname{RMSNorm}(W_{DQ}\, x_i).
$$

The query of a position is needed only while that position is being processed, so there is nothing to
store, and the reason for compressing it is the parameter count and the size of the activation that
has to be kept for the backward pass. The RMSNorm between the two projections is the normalisation of
chapter 2 applied inside the factorisation, and it keeps the scale of the compressed query fixed
independently of the scale of $W_{DQ}$.

### Why RoPE has to be decoupled

RoPE rotates $k$ by an angle that depends on the key's position. Applying the rotation after the
expansion gives $R(j)\,W_{UK,h}\,c_j$. The rotation and the expansion do not commute, so the
expansion matrix cannot be folded away at inference, which the next section requires. Applying the
rotation to the latent before expansion leaves an expanded key that is no longer a rotation of
anything, and the relative-position property of chapter 3 is lost.

Both failures come from the same fact. $R(j)$ acts on pairs of adjacent dimensions of the key,
$W_{UK,h}$ mixes all dimensions of the latent into all dimensions of the key, and two matrices of that
kind do not commute. Take the two orders in turn. If the rotation is applied after the expansion, the
cached object must be $c_j$ and the per-token matrix $R(j) W_{UK,h}$ depends on $j$, so there is no
single matrix that can be moved to the query side once and reused for every cached position. If the
rotation is applied to the latent before the expansion, the object that reaches the score is
$W_{UK,h} R(j) c_j$, and the property of chapter 3 that the score between positions $i$ and $j$
depends only on $j - i$ requires the rotations to meet each other directly, which they no longer do
once an arbitrary matrix sits between them.

The solution is to split the key into a position-free part from the latent and a small rotated
part computed directly from $x$ and shared across heads:

$$
k_{h,j} = [\,W_{UK,h}\, c_j \,\|\, R(j)\, W_{KR}\, x_j\,], \qquad
q_{h,i} = [\,W_{UQ,h}\, c^q_i \,\|\, R(i)\, W_{QR,h}\, c^q_i\,].
$$

The cache stores $c_j$ and the rotated part $R(j) W_{KR} x_j$ of width $d_r = 64$, so the cost per
token is $d_c + d_r$.

The score splits along the concatenation into a sum of two dot products. The first is between the
compressed parts and carries content with no position in it. The second is between the rotated parts
and depends only on $i - j$, exactly as in chapter 3, because both sides are genuine rotations of
fixed vectors. The rotated key part is shared across heads, like the key of multi-query attention,
which is why it costs 64 elements and not $64 H$. The rotated query part is per head, because the
query side is not stored and can afford the width.

### Weight absorption

At inference the expansion matrices never need to be applied to the cache. The logit is

$$
q_{h,i}^{\top} k_{h,j} = \big(W_{UQ,h} c^q_i\big)^{\top} W_{UK,h}\, c_j
= \big(W_{UK,h}^{\top} W_{UQ,h} c^q_i\big)^{\top} c_j ,
$$

so $W_{UK,h}$ is absorbed into the query side and attention runs directly against the cached
latents. The same is done with $W_{UV,h}$ on the output side. Decode then reads $d_c$ values per
token per layer, which is the intended cache reduction.

Walking one decode step through shows what this means. The step has one new token $i$ and a cache of
$S$ latents. First the compressed query $c^q_i$ is formed, and for each head the matrix
$W_{UK,h}^{\top} W_{UQ,h}$ maps it to a vector of width $d_c$. That work is done once for the step and
does not touch the cache. Then the loop over the $S$ cached positions runs, and inside it each head
takes a dot product of length $d_c + d_r$ against $[\,c_j \,\|\, R(j) W_{KR} x_j\,]$ and, after the
softmax, accumulates $c_j$ into an output of width $d_c$. Finally, outside the loop again, each head's
accumulated $d_c$-vector is mapped to the residual stream by $W_{UV,h}$ composed with that head's
slice of $W_O$, which is one matrix per head from $d_c$ to $D$. Everything head-specific happens
before and after the loop. Inside the loop, where the cost is, the only object touched is the shared
latent, and that is why the bytes read per position do not grow with $H$.

The products $W_{UK}^\top W_{UQ}$ and $W_{UV}$ with the output projection are the QK and OV circuits
of Elhage et al. 2021, "A Mathematical Framework for Transformer Circuits", each a bilinear form of
rank at most $d$. Absorption computes attention directly in terms of these products, which is the
form circuit analysis already uses.

### The cache on the ledger

The cache holds $d_c + d_r = 512 + 64 = 576$ elements per token per layer. Over the reference model's
80 layers in bf16,

$$
80 \cdot 576 \cdot 2 = 92 \text{ KB per token},
$$

which is 12 GB at the long-context target of 131072 tokens. That is 3.6 times smaller than
grouped-query attention with eight key-value heads and within a factor of two of multi-query
attention, while keeping every query head separate.

| variant | elements per token per layer | per token | at 131072 tokens |
| --- | --- | --- | --- |
| multi-head | 16384 | 2.6 MB | 344 GB |
| grouped-query, $H_{kv} = 8$ | 2048 | 328 KB | 43 GB |
| multi-query | 256 | 41 KB | 5.4 GB |
| MLA, $d_c = 512$, $d_r = 64$ | 576 | 92 KB | 12 GB |

### What it costs

The saving is in bytes and it is paid for in arithmetic. Under the absorbed form each head takes a
dot product of length 576 against every cached position and accumulates a vector of width 512, rather
than working with a key and a value of width 128. Per cached position per layer the reference model
does about 139 thousand floating-point operations under MLA against about 33 thousand under
grouped-query attention, a factor of 4.2 more arithmetic to read 3.6 times fewer bytes.

That trade is favourable because of the ratio in chapter 5. The arithmetic intensity of decode
attention rises from $H / H_{kv} = 8$ operations per byte under grouped-query attention to about 121
under MLA, since each latent element is read once and used by all 64 heads in both the score and the
weighted sum. Against an accelerator ratio of roughly 300 operations per byte, MLA moves decode
attention from deeply memory bound to mildly memory bound, and the step time still falls in
proportion to the bytes.

Two further costs are structural rather than numerical. The absorbed form is right for decode and
wrong for prefill. When many query positions are processed at once, expanding each latent to per-head
keys and values once and reusing the expansion across all the queries is cheaper than giving every
query a 576-wide dot product, so implementations keep both paths and switch between them. And the
absorbed form has an effective head width of $d_c$ rather than $d$, which the standard fused
attention kernels are not written for, so MLA needs kernels of its own.

### What the 2026 models do

DeepSeek-V3 uses MLA with 128 heads, a latent of width 512 and 64 rotated dimensions, the same shape
as V2. Kimi K2 uses MLA with 64 heads and the same latent width of 512 and 64 rotated dimensions, and
keeps full attention over every position at 128K context, which is affordable only because the cache
per token is this small. GLM-5 moves to multi-head latent attention in 2026, combined with sparse
attention in the DeepSeek style of chapter 10, having used grouped-query attention in GLM-4.5.
DeepSeek-V4.1 takes the next step and replaces the latent and its expansions with one shared latent
of width 512 per token and 64 rotated dimensions.

The split described at the end of chapter 5 holds. Grouped-query attention remains the choice for
dense and mid-size models, where the weights dominate serving cost and a factor of eight on the cache
is enough. Multi-head latent attention and its successors are used by the largest
mixture-of-experts models, where the active weights are a small fraction of the total and the cache
is what limits how many requests a server can hold.

The next section removes the expansion matrices entirely.
