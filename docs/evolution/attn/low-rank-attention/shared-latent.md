---
title: "Single shared latent"
order: 2
---

## The single shared latent

Weight absorption ended the previous section with an observation that is stronger than it first
appears. At decode time MLA never applies its expansion matrices to anything in the cache. The key
expansion is folded into the query side and the value expansion into the output side, so what
actually runs is a dot product between a per-head vector of width $d_c$ and the cached latent,
followed by a per-head linear map from the accumulated latent to the residual stream. The expansions
exist in the definition of the model and not in the computation that the model performs.

This section is therefore not another step away from the reference model. It is a description of
the form the reference model actually takes. DeepSeek-V4.1-Flash, the book's reference model from
here on, writes the model in the form the computation already had.

### From MLA to the V4 form

The reference model simplifies MLA further. The latent is no longer expanded into per-head keys and
values at all. One latent of width $d$ per token is both the key and the value for every head:

$$
\begin{aligned}
W_{KV} &\in \mathbb{R}^{d \times D}, \qquad x_j \in \mathbb{R}^{D},\\
c_j &= \operatorname{RMSNorm}\!\big(W_{KV}\, x_j\big) \in \mathbb{R}^{d},\\
k_j &= v_j = c_j .
\end{aligned}
$$

Read that back in words. One projection, one normalisation, one vector per token. There is no
per-head key, no per-head value, and nothing to expand.

Queries remain per head and low-rank, built exactly as the previous chapter built them — compressed
to $d_c^q$, then re-expanded to one vector of width $d$ per head:

$$
\begin{aligned}
W_{DQ} &\in \mathbb{R}^{d_c^q \times D}, \qquad W_{UQ,h} \in \mathbb{R}^{d \times d_c^q},\\
q_{h,i} &= W_{UQ,h}\,\operatorname{RMSNorm}\!\big(W_{DQ}\, x_i\big) \in \mathbb{R}^{d}.
\end{aligned}
$$

Attention for head $h$ is then a $d$-wide dot product, a softmax over positions, and a weighted sum
of latents:

$$
\begin{aligned}
s_{h,ij} &= \frac{q_{h,i} \cdot c_j}{\sqrt{d}} \in \mathbb{R},\\
p_{h,ij} &= \operatorname{softmax}_j\!\big(s_{h,ij}\big),\\
o_{h,i} &= \sum_j p_{h,ij}\; c_j \in \mathbb{R}^{d}.
\end{aligned}
$$

Read that back in words. Every head produces a different weighting over the same set of latents and
returns a different weighted sum of them. The per-head expansion matrices of MLA are gone. Their
role is taken by the query projection on one side and the output projection on the other, which
concatenates the $H$ head outputs of width $d$ and returns them to the residual stream:

$$
\begin{aligned}
o_i &= \big[\,o_{1,i};\, \dots;\, o_{H,i}\,\big] \in \mathbb{R}^{H d},\\
u_i &= W_O\, o_i \in \mathbb{R}^{D}.
\end{aligned}
$$

Here $d = 512$ is both the head width and the latent width, the residual width is $D = 5120$, the
query compression width is $d_c^q = 1280$, and there are $H = 64$ heads. The RMSNorm inside the
definition of $c_j$ fixes the scale of the cached object, which matters because that object is
stored in a low-precision format and reread thousands of times. The next two subsections add the
rotation and the sink; neither changes a shape.

Nothing is lost on the value side by this change. In MLA the accumulated output of a head was
$\sum_j p_{h,ij} W_{UV,h} c_j$, and since $W_{UV,h}$ does not depend on $j$ it can be moved outside
the sum and merged with the output projection, which is what absorption does. Caching $c_j$ and
applying one map afterwards is therefore exactly equivalent. On the key side the change is real but
small. MLA's absorbed query is $W_{UK,h}^{\top} W_{UQ,h} c^q_i$, a product of two matrices of rank
at most $d_c$, while the reference model produces $q_{h,i}$ from a single low-rank query projection
— compressed to width 1280 — whose rank is chosen independently. The score is a dot product against
the cached vector in both cases. What the V4 form removes is the constraint that the query-side map
be written as a product of two per-head expansions, the need to materialise those expansions at all
during prefill, and, as the next part shows, the separate rotated entry in the cache.

### Position with a shared latent

Since $k = v$, the value must carry the same rotation as the key. The stored latent is rotated at its
own position, and the attention output is rotated back at the query position:

$$
\tilde c_j = R(j)\, c_j, \qquad
o_{h,i} = R(-i) \sum_j p_{h,ij}\, \tilde c_j = \sum_j p_{h,ij}\, R(j - i)\, c_j .
$$

Read that back in words. Each cached latent is stored already rotated to its own position. The head
mixes them, and the result is un-rotated as if it had been sitting at the query's position. Only the
offset $j - i$ survives, and a single rotated buffer serves as both key and value. Only $d_r = 64$
of the $d$ dimensions rotate, on interleaved pairs at the trailing channels of the latent. The rest
pass through both rotations unchanged.

This is where the decoupling of the previous section becomes unnecessary. MLA had to keep a separate
rotated entry in the cache because an expansion matrix stood between the rotation and the score, and
two matrices of that kind do not commute. With no expansion matrix there is nothing standing between
them, so the latent can be rotated directly and the score of the rotated latent against the rotated
query already depends on $j - i$ alone. The cache holds 512 elements rather than $512 + 64$, and the
rotated part is no longer a separate object with its own projection. The rotation runs at a base
frequency of $\theta = 160000$, and the 1M-token context is reached by YaRN with a factor of 16
applied on this compressed branch only.

The second equation also shows what the shared latent does to the value side. In MLA the value a head
read was independent of position. Here each contribution arrives rotated by its own offset $j - i$,
so a head reading the same latent from two different distances receives two different vectors in the
$d_r$ rotated dimensions. The $d - d_r$ remaining dimensions are untouched by both rotations and
behave as a position-free value, so the two roles coexist in one vector, with the split between them
set by the choice of $d_r$.

### A per-head sink

The reference model adds one learned scalar $\sigma_h$ per head to the softmax denominator:

$$
p_{h,ij} = \frac{e^{s_{h,ij}}}{\sum_{j'} e^{s_{h,ij'}} + e^{\sigma_h}} .
$$

Read that back in words. Each head carries one extra logit that competes with all of the keys at
once and contributes no value to the output. The attention weights over the keys can then sum to
less than one, so a head need not spread probability mass over the keys when none of them is
relevant. Chapter 8 gives the origin of the idea.

The sink is one number per head per layer, it takes part in the normalisation, and it is not masked.
A row of the attention matrix in which every position is masked away therefore has a finite
denominator and produces a zero output rather than a division by zero, which matters once chapter 8
introduces windows that can leave a query with no visible keys. The mechanism is also load-bearing
for a shared latent specifically. With $k = v$ every unit of probability mass a head assigns is a
unit of some cached latent added to its output, so a head with nothing to retrieve cannot avoid
writing something into the residual stream unless the denominator gives it somewhere else to put the
mass.

### The grouped output projection

The output projection is itself low-rank and grouped: heads are split into $G$ groups, each group is
projected to a small rank, and one final matrix mixes the groups.

The reason it has to be factorised is the width of what the heads return. In chapter 1 each head
produced a value of width $d$ and the concatenation over $H$ heads had width $H d = D$, so $W_O$ was
a square $D \times D$ matrix. Under a shared latent each head returns a full latent of width 512, so
the concatenation over 64 heads — $64 \times 512 = 32768$ — is more than 6× the residual
width $D = 5120$. A dense output projection of that shape would hold
$32768 \cdot 5120 = 168\,\text{M}$ parameters per block, in one layer. Factorising it into $G$ per-group blocks of
$H d / G$ inputs to rank $r_o$, plus one mix of $G r_o$ inputs to $D$, brings the count from
$H d\, D$ down to $H d\, r_o + G\, r_o D$. Nothing crosses between groups in the first stage, and
the mix is what recombines them. The reference model uses $G = 8$ groups of rank $r_o = 1024$:
$32768 \cdot 1024 + 8 \cdot 1024 \cdot 5120 = 33.6 + 41.9 = 75.5\,\text{M}$ parameters, less than
half the dense count.

In the reference implementation this is one weight read as $G$ independent blocks and applied with
one batched matmul.

### Precision and stability

The latent concentrates the cache into one vector, which concentrates the numerics into one place.
Three properties follow.

1. The latent is normalised before use, by the same RMSNorm device as chapter 5. Every head's key
   and value is reconstructed from a vector of fixed RMS, so no head can inherit a magnitude
   blow-up from the residual stream, and the dot products that consume the latent have bounded
   inputs however long training runs. This is the property that later chapters spend: chapter 9
   states the magnitude bound that makes a four-bit cache safe, and the bound exists only because
   the norm is there.
2. The score is a dot product of width 512 rather than the 128 a chapter-1 head would have had.
   Rounding errors of independent dimensions add in quadrature, so the error of the sum grows as
   $\sqrt{512}$ while the sum itself can grow as 512; the relative error of a wide dot product is
   better, not worse, than a narrow one. The accumulation is fp32 in every kernel that matters,
   bf16 only in the stored operands.
3. The absorbed and unabsorbed forms of the chapter compute the same quantity in exact arithmetic
   and differ in floating point, because they associate the same product differently. DeepSeek keeps
   the attention operators themselves in bf16 through FP8 training — the V3 report lists attention
   among the components that "maintain the original precision" — which is the acknowledgement that
   the softmax is the wrong place to spend precision budget. The cache, by contrast, is written once
   and read many times, which is the subject of the next section and of chapter 12.

### Cache cost

Per token and per layer the cache holds one vector of width $d = 512$, against $d_c + d_r = 576$ for
V3's MLA and $2 \times 8 \times 512 = 8192$ for GQA with 8 heads on this shape. Chapter 12 then
stores this vector in FP8 or FP4, and chapters 8 and 9 reduce how many tokens need one at all.

The ledger, over the 40 layers of the reference model. bf16, two bytes per element:

$$
40 \cdot 512 \cdot 2 = 40{,}960 \text{ bytes} \approx 40 \text{ KB per token}.
$$

FP8, one byte per element, halves it to 20 KB. FP4 stores four bits per element with one e4m3
scale for every 16 elements, which is $4 + 8/16 = 4.5$ bits or 0.5625 bytes per element:

$$
40 \cdot 512 \cdot 0.5625 = 11{,}520 \text{ bytes} = 11.25 \text{ KB per token}.
$$

At the extended context of 1,048,576 tokens the three formats cost 40 GB, 20 GB and 11.25 GB per
sequence:

| format | bytes per element | per token | at 1M tokens |
| --- | --- | --- | --- |
| bf16 | 2 | 40 KB | 40 GB |
| FP8 e4m3 | 1 | 20 KB | 20 GB |
| FP4 e2m1 with one scale per 16 | 0.5625 | 11.25 KB | 11.25 GB |

The bottom row is 455× smaller than the 5.0 MB per token that multi-head attention started
with in chapter 5, and it is the number the rest of the book works from. Two chapters of
architecture and one of numerics have taken a full 1M-token conversation from 5 TB to 11.25 GB.

### Where this leads

Eleven and a quarter gigabytes per 1M-token sequence is small enough to hold many conversations at once, and
it is not small enough to be the end of the matter, because the cost of a decode step is set by
bytes read and not by bytes held. Every layer still reads every stored token at every step, so the
work per step still grows in proportion to the length of the conversation. A latent that is 455
× smaller has made the constant small and has not changed the growth.

The remaining chapters on attention attack the count of entries rather than the width of one.
Chapter 8 restricts most layers to a window of recent tokens, which makes their cost per step
independent of length, and gives the origin of the sink logit introduced above. Chapter 9 stores one
entry for several tokens and shares it across layers. Chapter 10 keeps every entry and reads only
the ones a query needs.
