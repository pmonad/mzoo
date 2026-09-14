---
title: "MLA"
order: 1
---

## Multi-head latent attention

Grouped-query attention left the reference model storing $2 \cdot 8 \cdot 512 = 8192$ elements per
token per layer, one key and one value of width 512 for each of eight key-value heads. Those 8192
numbers are produced from a single input vector of width $D = 5120$ by a fixed linear map, so they
are not 8192 independent quantities. Multi-head latent attention, introduced with DeepSeek-V2 in
2024, stores the small vector that generates them instead and rebuilds the rest on read. It is used
by DeepSeek-V3 and by Kimi K2, and it is the form that the single shared latent of the next section
— the form the reference model itself takes — simplifies.

### Compress, then expand

Project the input to a latent of width $d_c \ll H d$, cache the latent, and expand to per-head keys
and values on read:

$$
c_j = W_{DKV}\, x_j \in \mathbb{R}^{d_c}, \qquad
k_{h,j} = W_{UK,h}\, c_j, \qquad
v_{h,j} = W_{UV,h}\, c_j .
$$

Read that back in words. One matrix squeezes the token state into $d_c$ numbers, and $2H$ small
matrices reopen those numbers into a distinct key and a distinct value for every head. Every head
still has its own key space and its own value space, since the up projections are not shared. What
is shared is the vector they all read from.

The cache holds $c_j$ only. In DeepSeek-V2, $d_c = 512$ against $H d = 128 \times 128 = 16384$, a
32× reduction relative to full multi-head attention and about 4× relative to GQA with
8 KV heads. Quality is equal to or better than the dense baseline because all $H = 128$ heads are
retained. V3 keeps the same latent shape, $d_c = 512$ with $d_r = 64$, as does Kimi K2 with
$H = 64$.

The factorisation also makes the attention parameters smaller, which is not the point of the design
but is worth recording. On the reference model shape, with $d_c = 512$, the key-value path holds
$W_{DKV}$ at $512 \cdot 5120 = 2.6\,\text{M}$ parameters and the two sets of up projections at
$512 \cdot 32768 = 16.8\,\text{M}$ each, about $36\,\text{M}$ in all, against
$2 \cdot 5120 \cdot 32768 = 336\,\text{M}$ for the full-rank $W_K$ and $W_V$ of chapter 1 — a
9.3× saving. The same construction is the one Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large
Language Models", uses to make a weight update cheap, applied here to the weights themselves rather
than to an update.

| scheme | matrices | parameters per layer | MACs per token | weight bytes (bf16) |
| --- | --- | --- | --- | --- |
| full-rank | $W_K, W_V$ each $Hd \times D = 32768 \times 5120$ | $335.5\,\text{M}$ | $335.5\,\text{M}$ | $640\,\text{MB}$ |
| MLA factorised | $W_{DKV}$ at $512 \times 5120$; $W_{UK}, W_{UV}$ each $32768 \times 512$ | $36.2\,\text{M}$ | $36.2\,\text{M}$ | $69\,\text{MB}$ |

Read that back in words. The factorised path costs 9.3× fewer parameters and 9.3× fewer
multiply-accumulates per token: the arithmetic saving and the parameter saving are the same saving,
because both are the rank of the bottleneck.

So yes, compute is saved, not only cache. The two-step path multiplies through the narrow
$d_c = 512$ bottleneck instead of the full 32768-wide concatenation of heads, so the key-value
projections cost about 9× less arithmetic per token. And the weight matrices are a cache of
a kind themselves: they are resident for the model's life and read at every decode step — shared
across the whole batch, unlike the KV cache — so 36M against 336M parameters per layer is 69 MB
against 640 MB of weights to store and to stream on every pass over the layers, a saving that holds
at any context length. At decode the saving is even cleaner than the table suggests: under the
absorption of the next section the up-projections migrate to the query side and touch nothing
cached, so the cache-reading loop is billed only the latent.

The query is compressed the same way. This saves no cache, but it saves parameters and activation
memory during training:

$$
q_{h,i} = W_{UQ,h}\, \operatorname{RMSNorm}(W_{DQ}\, x_i).
$$

Read that back in words. The query of a position is needed only while that position is being
processed, so there is nothing to store, and the reason for compressing it is the parameter count
and the size of the activation that has to be kept for the backward pass. The RMSNorm between the
two projections is the normalisation of chapter 2 applied inside the factorisation, and it keeps the
scale of the compressed query fixed independently of the scale of $W_{DQ}$. The reference model
compresses its queries to 1280 dimensions before the heads are formed.

### Why RoPE has to be decoupled

RoPE rotates $k$ by an angle that depends on the key's position. Applying the rotation after the
expansion gives $R(j)\,W_{UK,h}\,c_j$. The rotation and the expansion do not commute, so the
expansion matrix cannot be folded away at inference, which the next section requires. Applying the
rotation to the latent before expansion leaves an expanded key that is no longer a rotation of
anything, and the relative-position property of chapter 3 is lost.

Two square matrices commute when $AB = BA$. Most pairs do not, and the reason is easy to see on a
2-by-2 plane: take a rotation by 90 degrees and a shear along the $x$ axis. Rotate a point, then
shear it, and it lands somewhere; shear the same point first, then rotate, and it lands somewhere
else, because the shear pushes the point to a height the rotation then carries to a different
place. Order matters whenever the second matrix moves coordinates that the first matrix is about to
mix. A rotation acts on pairs of adjacent dimensions, and $W_{UK,h}$ mixes all dimensions of the
latent into all dimensions of the key, so each undoes the other's arrangement: they do not commute.

Both failures come from that fact. If the rotation is applied after the expansion, the cached object
must be $c_j$ and the per-token matrix $R(j) W_{UK,h}$ depends on $j$, so there is no single matrix
that can be moved to the query side once and reused for every cached position. If the rotation is
applied to the latent before the expansion, the object that reaches the score is
$W_{UK,h} R(j) c_j$, and the property of chapter 3 that the score between positions $i$ and $j$
depends only on $j - i$ requires the rotations to meet each other directly, which they no longer do
once an arbitrary matrix sits between them.

The solution is to split the key into a position-free part from the latent and a small rotated
part computed directly from $x$ and shared across heads:

$$
k_{h,j} = [\,W_{UK,h}\, c_j \,\|\, R(j)\, W_{KR}\, x_j\,], \qquad
q_{h,i} = [\,W_{UQ,h}\, c^q_i \,\|\, R(i)\, W_{QR,h}\, c^q_i\,].
$$

Read that back in words. Each key and query is two half-vectors glued together: a content half
from the latent, carrying no position, and a position half that is a genuine rotation of a fixed
vector. The cache stores $c_j$ and the rotated part $R(j) W_{KR} x_j$ of width $d_r = 64$, so the
cost per token is $d_c + d_r$.

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
latents. The same is done with $W_{UV,h}$ on the output side. "Absorbed" means, operationally:
never materialise the product $W_{UK,h} c_j$ at all. Instead, at model-load time, precompute
$W_{UK,h}^{\top} W_{UQ,h}$ once, fold it into the query projection, and let the decode loop take dot
products against the raw cached latent. The expansion matrix still exists in the model's definition;
it has simply been multiplied into a matrix that was going to be applied anyway. Decode then reads
$d_c$ values per token per layer, which is the intended cache reduction.

Walking one decode step through shows what this means. The step has one new token $i$ and a cache
of $S$ latents:

$$
\begin{array}{ll}
\textbf{decode}\,(i,\ \{c_j\}_{j<i}) & \\
\hline
\textit{// once per step, never touches the cache} & \\
c^q_i \leftarrow W_{DQ}\, x_i & \\
\bar q_h \leftarrow \big[\, W_{UK,h}^{\top} W_{UQ,h}\, c^q_i \,\|\, R(i)\, W_{QR,h}\, c^q_i \,\big] & \text{for each head } h \\
\hline
\textit{// the loop: runs } S \textit{ times, reads one latent per position} & \\
\textbf{for } j = 1, \dots, i-1: & \\
\quad s_{h,ij} \leftarrow \bar q_h \cdot \big[\, c_j \,\|\, R(j)\, W_{KR}\, x_j \,\big] & \text{for each head } h \\
\quad p_{h,i\cdot} \leftarrow \operatorname{softmax}_j\big(s_{h,i\cdot} / \sqrt{d}\,\big) & \\
\quad \bar o_h \leftarrow \bar o_h + \sum_j p_{h,ij}\; c_j & \\
\hline
\textit{// once per step again} & \\
y \leftarrow \sum_h W_{O,h}\, W_{UV,h}\; \bar o_h &
\end{array}
$$

Read the three blocks by their comments. Everything head-specific happens before and after the
loop: the absorbed query $\bar q_h$ is formed once for the step, and the per-head output map
$W_{O,h} W_{UV,h}$ is applied once to the accumulated $\bar o_h$. Inside the loop, where the cost
is, the only object touched is the shared latent — one read of $c_j$ serves every head's score and
every head's accumulation — and that is why the bytes read per position do not grow with $H$.

Do the matrices themselves change size when they are fused? The product
$W_{UK,h}^{\top} W_{UQ,h}$ has shape $d_c \times d_c^q$, and the pair it replaces holds
$d\,d_c^q + d\,d_c$ entries, so the answer depends on the head width. On the reference model's wide
heads the fused matrix is the smaller object: per head $512 \times 1280 = 640$ K entries against
$512 \times 1280 + 512 \times 512 = 896$ K for the pair. On V3's narrow 128-wide heads it is the
larger: $512 \times 512 = 256$ K against a pair of two $128 \times 512 = 64$ K matrices. Neither
direction matters much, for three reasons. The checkpoint is unchanged — absorption saves no
parameters, and the 9.3× saving of the table above belongs to the factorisation itself, with or
without absorption; the fused product is a derived object, computed at load time. The size
difference lands in weights, which are read once per step and shared across every sequence in the
batch, not in the cache, which each sequence pays for separately. And strictly the product need not
be materialised at all: applying $W_{UQ,h}$ and then $W_{UK,h}^{\top}$ to the one query vector of
the step produces the same absorbed query for a comparable handful of multiplies. What absorption
removes is not weight storage but the $S$-fold application of $W_{UK,h}$ to every cached key — the
saving lives in the loop, and that is the whole point.

The products $W_{UK}^\top W_{UQ}$ and $W_{UV}$ with the output projection are the QK and OV circuits
of Elhage et al. 2021, "A Mathematical Framework for Transformer Circuits", each a bilinear form of
rank at most $d$. Absorption computes attention directly in terms of these products, which is the
form circuit analysis already uses.

### The cache on the ledger

The cache holds $d_c + d_r = 512 + 64 = 576$ elements per token per layer. Over the reference
model's 40 layers in bf16 that is

$$
40 \cdot 576 \cdot 2 = 46{,}080 \text{ bytes} \approx 45 \text{ KB per token},
$$

which is 2.8 GB at the native context of 65536 tokens and 45 GB at the extended context of
1,048,576. That is 14.2× smaller than grouped-query attention with eight key-value heads and
within a factor of two of multi-query attention, while keeping every query head separate.

| variant | elements per token per layer | per token | at 65536 tokens | at 1M tokens |
| --- | --- | --- | --- | --- |
| multi-head | 65536 | 5.0 MB | 320 GB | 5 TB |
| grouped-query, $H_{kv} = 8$ | 8192 | 640 KB | 40 GB | 640 GB |
| multi-query | 1024 | 80 KB | 5 GB | 80 GB |
| MLA, $d_c = 512$, $d_r = 64$ | 576 | 45 KB | 2.8 GB | 45 GB |

### What it costs

The saving is in bytes, and on the reference model's 512-wide heads it is no longer paid for in
arithmetic. Count both sides per cached position per layer, counting a multiply-add as two
operations.

1. GQA-8: each of the 64 heads takes a 512-wide dot product for the score and a 512-wide
   multiply-accumulate for the value, so $64 \cdot (512 + 512) = 65{,}536$ multiply-adds, or
   131,072 operations. The bytes read are two vectors (key and value) of width 512 for each of 8
   heads in bf16, $2 \cdot 8 \cdot 512 \cdot 2 = 16{,}384$ bytes = 16 KB. Intensity:
   $131{,}072 / 16{,}384 = 8$ operations per byte, which is $H / H_{kv}$.
2. MLA: each head takes a 576-wide dot product for the score and a 512-wide multiply-accumulate for
   the value, so $64 \cdot (576 + 512) = 69{,}632$ multiply-adds, or 139,264 operations. The bytes
   read are the latent, $576 \cdot 2 = 1152$ bytes = 1.125 KB. Intensity:
   $139{,}264 / 1152 = 121$ operations per byte.

So MLA does 1.06× the arithmetic of GQA-8 per cached position — essentially the same — to read
14.2× fewer bytes. On the 128-wide heads of the models this chapter was written about
originally, MLA cost about 4× more arithmetic than GQA; widening the heads to 512 while
keeping the latent at 576 makes that penalty vanish, because the heads were already doing wide dot
products against short keys and now do the same work against the latent directly.

The reason the intensity is 121 rather than 128 is the rotated entry: the score uses all 576 cached
elements but the value accumulation uses only the 512 of the latent, so each byte serves
$64 \cdot (576 + 512) / 576$ operations. Against an accelerator ratio of roughly 300 operations per
byte, MLA moves decode attention from deeply memory bound to mildly memory bound, and the step time
still falls in proportion to the bytes.

Two further costs are structural rather than numerical. The absorbed form is right for decode and
wrong for prefill. When many query positions are processed at once, expanding each latent to per-head
keys and values once and reusing the expansion across all the queries is cheaper than giving every
query a 576-wide dot product, so implementations keep both paths and switch between them. And the
absorbed form has an effective head width of $d_c + d_r$ rather than $d$, which the standard fused
attention kernels are not written for, so MLA needs kernels of its own; chapter 16 covers that
kernel side and what it takes to make both paths fast.

### What the 2026 models do

DeepSeek-V3 uses MLA with 128 heads, a latent of width 512 and 64 rotated dimensions, the same shape
as V2. Kimi K2 uses MLA with 64 heads and the same latent width of 512 and 64 rotated dimensions, and
keeps full attention over every position at 128K context, which is affordable only because the cache
per token is this small. GLM-5 moves to multi-head latent attention in 2026, combined with sparse
attention in the DeepSeek style of chapter 10, having used grouped-query attention in GLM-4.5.
DeepSeek-V4.1 takes the next step and replaces the latent and its expansions with one shared latent
of width 512 per token and 64 rotated dimensions — and that step, described next, is the one the
reference model actually takes.

The split described at the end of chapter 5 holds. Grouped-query attention remains the choice for
dense and mid-size models, where the weights dominate serving cost and a factor of eight on the cache
is enough. Multi-head latent attention and its successors are used by the largest
mixture-of-experts models, where the active weights are a small fraction of the total and the cache
is what limits how many requests a server can hold.

The next section removes the expansion matrices entirely.
