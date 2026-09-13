# The single shared latent

Weight absorption ended the previous section with an observation that is stronger than it first
appears. At decode time MLA never applies its expansion matrices to anything in the cache. The key
expansion is folded into the query side and the value expansion into the output side, so what
actually runs is a dot product between a per-head vector of width $d_c$ and the cached latent,
followed by a per-head linear map from the accumulated latent to the residual stream. The expansions
exist in the definition of the model and not in the computation that the model performs. DeepSeek-V4.1
writes the model in the form the computation already had.

## From MLA to V4

DeepSeek-V4.1 simplifies MLA further. The latent is no longer expanded into per-head keys and values
at all. One latent of width $d$ per token is both the key and the value for every head:

$$
c_j = \operatorname{RMSNorm}(W_{KV}\, x_j) \in \mathbb{R}^{d}, \qquad
k_j = v_j = c_j .
$$

Queries remain per head and low-rank. Attention for head $h$ is then

$$
o_{h,i} = \sum_j p_{h,ij}\, c_j, \qquad
p_{h,ij} = \operatorname{softmax}_j\!\left(\frac{q_{h,i} \cdot c_j}{\sqrt{d}}\right).
$$

Every head produces a different weighting of the same set of latents. The per-head expansion
matrices of MLA are gone. Their role is taken by the query projection on one side and the output
projection on the other.

Here $d$ is both the head width and the latent width, and V4.1 sets it to 512, four times the head
width of the reference model. The RMSNorm inside the definition of $c_j$ fixes the scale of the
cached object, which matters because that object is stored in a low-precision format and reread
thousands of times.

Nothing is lost on the value side by this change. In MLA the accumulated output of a head was
$\sum_j p_{h,ij} W_{UV,h} c_j$, and since $W_{UV,h}$ does not depend on $j$ it can be moved outside
the sum and merged with the output projection, which is what absorption does. Caching $c_j$ and
applying one map afterwards is therefore exactly equivalent. On the key side the change is real but
small. MLA's absorbed query is $W_{UK,h}^{\top} W_{UQ,h} c^q_i$, a product of two matrices of rank at
most 128, while V4.1 produces $q_{h,i}$ from a single low-rank query projection whose rank is chosen
independently. The score is a dot product against the cached vector in both cases. What V4.1 removes
is the constraint that the query-side map be written as a product of two per-head expansions, the
need to materialise those expansions at all during prefill, and, as the next part shows, the separate
rotated entry in the cache.

## Position with a shared latent

Since $k = v$, the value must carry the same rotation as the key. The stored latent is rotated at its
own position, and the attention output is rotated back at the query position:

$$
\tilde c_j = R(j)\, c_j, \qquad
o_{h,i} = R(-i) \sum_j p_{h,ij}\, \tilde c_j = \sum_j p_{h,ij}\, R(j - i)\, c_j .
$$

The output depends on offsets only, and a single rotated buffer serves as both key and value. Only
$d_r$ of the $d$ dimensions rotate. The rest pass through both rotations unchanged.

This is where the decoupling of the previous section becomes unnecessary. MLA had to keep a separate
rotated entry in the cache because an expansion matrix stood between the rotation and the score, and
two matrices of that kind do not commute. With no expansion matrix there is nothing standing between
them, so the latent can be rotated directly and the score of the rotated latent against the rotated
query already depends on $j - i$ alone. The cache holds 512 elements rather than $512 + 64$, and the
rotated part is no longer a separate object with its own projection.

The second equation also shows what the shared latent does to the value side. In MLA the value a head
read was independent of position. Here each contribution arrives rotated by its own offset $j - i$,
so a head reading the same latent from two different distances receives two different vectors in the
$d_r$ rotated dimensions. The $d - d_r$ remaining dimensions are untouched by both rotations and
behave as a position-free value, so the two roles coexist in one vector, with the split between them
set by the choice of $d_r$.

## A per-head sink

V4.1 adds one learned scalar $\sigma_h$ per head to the softmax denominator:

$$
p_{h,ij} = \frac{e^{s_{h,ij}}}{\sum_{j'} e^{s_{h,ij'}} + e^{\sigma_h}} .
$$

The attention weights over the keys can then sum to less than one, so a head need not spread
probability mass over the keys when none of them is relevant. Chapter 8 gives the origin of the
idea.

The sink is one number per head per layer, it takes part in the normalisation and contributes no
value to the output, and it is not masked. A row of the attention matrix in which every position is
masked away therefore has a finite denominator and produces a zero output rather than a division by
zero, which matters once chapter 8 introduces windows that can leave a query with no visible keys.
The mechanism is also load-bearing for a shared latent specifically. With $k = v$ every unit of
probability mass a head assigns is a unit of some cached latent added to its output, so a head with
nothing to retrieve cannot avoid writing something into the residual stream unless the denominator
gives it somewhere else to put the mass.

## The grouped output projection

The output projection is itself low-rank and grouped: heads are split into
$G$ groups, each group is projected to a small rank, and one final matrix mixes the groups. At scale
this replaces a $32768 \times D$ dense matrix with $G$ blocks plus one mix.

The reason it has to be factorised is the width of what the heads return. In chapter 1 each head
produced a value of width $d$ and the concatenation over $H$ heads had width $H d = D$, so $W_O$ was
a square $D \times D$ matrix. Under a shared latent each head returns a full latent of width 512, so
the concatenation over 64 heads has width $H d = 32768$, four times the residual width. A dense
output projection of that shape would hold $32768 \cdot 8192 = 268$ million parameters per block,
which is as much as the entire attention block of chapter 1 and more than a routed expert layer
spends on any single token. Factorising it into $G$ per-group blocks of $H d / G$ inputs to rank
$r_o$, plus one mix of $G r_o$ inputs to $D$, brings the count from $H d D$ down to
$H d\, r_o + G\, r_o D$. Nothing crosses between groups in the first stage, and the mix is what
recombines them.

In owlet1 this is a `nn.Linear` subclass whose weight is read as $G$ independent blocks and applied
with one batched matmul. The shapes are in `src/mzoo/archs/owlet1/attention.md`.

## Cache cost

Per token and per layer the cache holds one vector of width $d = 512$, against $d_c + d_r = 576$ for
V3's MLA and $2 \times 8 \times 128 = 2048$ for GQA with 8 heads. Chapter 12 then stores this vector in
FP8 or FP4, and chapters 8 and 9 reduce how many tokens need one at all.

Over 80 layers in bf16 that is $80 \cdot 512 \cdot 2 = 82$ KB per token. Chapter 12 stores the same
vector in FP8 at one byte per element, giving 41 KB, and in FP4 at four bits per element with one
e4m3 scale for every 16 elements, which is $4 + 8/16 = 4.5$ bits or 0.5625 bytes per element and
gives 23 KB.

| format | bytes per element | per token | at 131072 tokens |
| --- | --- | --- | --- |
| bf16 | 2 | 82 KB | 10.7 GB |
| FP8 e4m3 | 1 | 41 KB | 5.4 GB |
| FP4 e2m1 with one scale per 16 | 0.5625 | 23 KB | 3.0 GB |

The bottom row is 114 times smaller than the 2.6 MB per token that multi-head attention started with
in chapter 5, and it is the number the rest of the book works from. Two chapters of architecture and
one of numerics have taken a full long-context conversation from 344 GB to 3 GB.

In owlet1 all of this is `attention.py`. The shapes table there gives the smoke-config sizes.

## Where this leads

Three gigabytes per sequence is small enough to hold many conversations at once, and it is not small
enough to be the end of the matter, because the cost of a decode step is set by bytes read and not by
bytes held. Every layer still reads every stored token at every step, so the work per step still
grows in proportion to the length of the conversation. A latent that is 114 times smaller has made
the constant small and has not changed the growth.

The remaining chapters on attention attack the count of entries rather than the width of one. Chapter
8 restricts most layers to a window of recent tokens, which makes their cost per step independent of
length, and gives the origin of the sink logit introduced above. Chapter 9 stores one entry for
several tokens and shares it across layers. Chapter 10 keeps every entry and reads only the ones a
query needs.
