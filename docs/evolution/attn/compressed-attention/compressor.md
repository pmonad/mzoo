---
title: "Compressor"
order: 1
---

## The compressor

The global path of a layer must summarise everything outside the sliding window. The design question
is what a token in the distant past contributes to a query issued much later. It is rarely the exact
word. It is the presence of a topic, a name, a definition or a piece of code somewhere back there.
That kind of content survives being represented at a coarser granularity than one entry per token,
and the redundancy between neighbouring tokens is high, so a summary of a small group of tokens can
stand in for the group.

Three ways of building such a summary are possible. The first is to evict, keeping some tokens whole
and discarding the rest, which is what the cache-eviction methods of the H2O line do at inference
time. Eviction is lossless for what it keeps and total for what it drops. The second is to average
each group of tokens, which keeps something of everything and loses whatever distinguishes one token
in the group from its neighbours. The third is to learn the summary, as Rae et al. 2019,
"Compressive Transformers for Long-Range Sequence Modelling", did by training a compression function
over blocks of old memories. V4.1 takes the third route and learns a pooling that can behave like
either of the first two.

### Pooling a group into one latent

A compressed layer keeps, in addition to its sliding window, one latent for every $m$ consecutive
tokens. Group $g$ covers tokens $mg, \dots, mg + m - 1$. Each token is first projected to a candidate
latent and a gate logit. The group is then pooled by a per-dimension softmax over the gate:

$$
\tilde k_t = W_{KV}\, x_t, \qquad g_t = W_{\text{gate}}\, x_t, \qquad
\alpha_t = \frac{e^{g_t}}{\sum_{t' \in g} e^{g_{t'}}} \ \text{(per dimension)}, \qquad
c_g = \operatorname{RMSNorm}\Big(\sum_{t \in g} \alpha_t \odot \tilde k_t\Big).
$$

Read the third term first. The gate logits $g_t$ have the same width as the latent, and the softmax
runs over the $m$ tokens of the group separately for each dimension, so $\alpha_t$ is a vector of
weights that sum to one across the group in every dimension. The pooled latent is then the
elementwise product of those weights with the candidates, summed over the group.

The per-dimension form is the point of the construction. A softmax over scalar gates would choose one
token per group and behave like eviction. A fixed uniform weight would average and behave like mean
pooling. With one weight per dimension the layer can take dimension 17 almost entirely from the third
token of the group and dimension 240 almost entirely from the first, so a latent can keep a
distinctive dimension from one token and a different dimension from another instead of averaging
them. Where the group is uniform the gate can stay flat and recover the average. The behaviour is
learned per dimension rather than fixed by the design.

The parameters this adds are two projections of width $D \times d_c$ per compressing layer, so
$2 \cdot 8192 \cdot 512 = 8.4$ M parameters on the reference model, against 805 M for a block. The
arithmetic is one extra projection per token. Neither is material.

### Why the normalisation is there

The pooled sum is a convex combination in every dimension, but its overall scale still varies. When
the candidates of a group agree, the sum has the magnitude of one candidate. When they disagree,
terms cancel and the magnitude falls. Groups near a sequence boundary may also pool fewer than $m$
tokens. Attention logits are dot products, so an entry with a larger norm receives a larger logit
against every query for reasons that have nothing to do with what the entry contains. RMSNorm, as
defined in chapter 2, sets every latent to a common scale and leaves only its direction to carry
information, so entries compete for attention on content. It also keeps the input to the query-key
product in the range the rest of the layer was trained for, which matters more once chapter 12 stores
these entries in four bits.

At $m = 1$ there is no pooling and $c_t = \operatorname{RMSNorm}(W_{KV} x_t)$. The gate softmax over a
group of one is the constant 1, so the expression above degenerates cleanly and no separate code path
is needed. Such a layer stores one entry per token, as in chapter 7, and still benefits from the
sharing of the next section.

#### The bound that makes four bits safe

The normalisation also carries a quantitative argument, which chapter 12 relies on. After RMSNorm a
latent of width $d_c = 512$ has norm exactly $\sqrt{512} \approx 22.6$, and RoPE, being a rotation,
preserves the norm. The largest element of the latent is therefore at most 22.6, and in training the
observed maxima are closer to 10. The storage format of chapter 12 — e2m1 values with one e4m3 scale
per 16 dimensions — represents values up to $448 \times 6 = 2688$. The ratio of headroom to worst
case is two orders of magnitude, which is why V4.1 can drop the second-level global scale that the
full NVFP4 format carries: the norm already did the global scaling, for free, before the cache
existed. A group-scaled format on an unnormalised latent would need that extra scale and the extra
pass to maintain it.

One ordering detail follows the same logic. The latent is rotated first and quantised after, so the
cache stores one object in one format for its rotated and unrotated parts alike. Quantising before
the rotation gains little accuracy and would force an extra dequantise-rotate-quantise round trip at
every decode step. The quantisation-aware training that keeps the cache accurate in this format is
introduced during post-training rather than pre-training, which the bound above makes plausible: the
values were already inside the format's range by construction, so the training that adapts them to
the grid is short.

### Positions and the key set

The latents are rotated with their own RoPE base at the positions of their groups, as in chapter 3.
A group's position index is the index of the group, not of its tokens, so the compressed table is a
sequence of length $S / m$ with its own position axis, and a separate base keeps the rotation
frequencies matched to that shorter axis. A query attends over the union of its window and the
compressed set:

$$
\text{keys}_i = \{\tilde c_j : j \in \text{window}(i)\} \ \cup\ \{c_g : g \text{ completed before } i\}.
$$

The two parts of the union are different objects. The window part holds the layer's own uncompressed
per-token latents for the last $W$ positions. The compressed part holds pooled group latents for
everything before that. Both enter one softmax, so a query weighs a precise recent token against a
coarse distant group in the same distribution, and the learned sink of chapter 8 sits in the same
denominator.

The global part of the cache is now $S / m$ entries instead of $S$. V4.1's 40 layers run the first
two on the window alone, the 18 encoder CSA2 layers at $m = 2$ and the 20 decoder CSA2 layers at
$m = 1$, so the finer view is available to the decoder, whose queries must select from the table
(see the next chapter). A group becomes visible only once its last token has been seen, which keeps
the model causal.
The tokens of the group still in progress are not missing from the key set, because they are inside
the sliding window, and with $W = 128$ and $m = 2$ the window covers the incomplete group many times
over. During decoding this means the compressed table grows by one entry every $m$ steps while the
window turns over every step.

### Why coarse entries are enough for most positions

Attention mass is concentrated, and for most queries a small set of keys receives most of the
weight, the heavy hitters of Zhang et al. 2023, "H2O: Heavy-Hitter Oracle for Efficient Generative
Inference of Large Language Models". A small set of heads also does most of the long-range
retrieval, as reported in Wu et al. 2024, "Retrieval Head Mechanistically Explains Long-Context
Factuality". Pooling $m$ tokens into one latent is therefore safe for the majority of positions that
never receive much attention, and chapter 10 selects the few that matter.
