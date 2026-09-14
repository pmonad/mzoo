---
title: "Sharing across layers"
order: 2
---

## Sharing across layers

The compressor divides the global table by $m$. The other factor in the cache formula of chapter 5 is
$L$, the number of layers, and it is the largest one. Every layer of every model in this book so far
has kept its own keys and values, on the assumption that each layer needs its own view of the past.

That assumption is only partly true. The keys of layer $\ell$ and layer $\ell + 1$ are linear
projections of residual states that differ by one attention output and one feed-forward output, so
they are strongly related. Several 2024 designs exploited this. Brandon et al. 2024, "Reducing
Transformer Key-Value Cache Size with Cross-Layer Attention", let pairs of adjacent layers share one
key-value cache and reported quality close to the unshared model at half the memory. Wu and Tu 2024,
"Layer-Condensed KV Cache for Efficient Inference of Large Language Models", went further and had
most layers attend to the keys and values of a single top layer. Sun et al. 2024, "You Only Cache
Once: Decoder-Decoder Architectures for Language Models", split the model into a first half that
produces one cache and a second half that only reads it. V4.1 applies the same idea to the compressed
table of the previous section.

### Source and reuse

Layers with the same compression ratio are grouped. The first layer of a group is the source. It
runs the compressor on its own input and publishes the latents. Later layers of the group do not
compress. They attend over the source's latents together with their own sliding window:

$$
\text{keys}^{(\ell)} = \mathcal{W}_\ell \ \oplus\ \hat C_{\text{src}(\ell)}, \qquad
\hat C_{\text{src}} = \operatorname{compress}_m\big(h_{\text{src}}\big).
$$

$\mathcal{W}_\ell$ is layer $\ell$'s own window and $\hat C_{\text{src}(\ell)}$ is the table published by the
source of its group. A reuse layer therefore performs no compression, stores no global entries and
allocates no global cache. It contributes queries.

The global cache is therefore stored once per group, not once per layer. The paper's schedule on
its 40 layers makes the trade concrete: the encoder's 18 CSA2 layers form three groups of six, each
led by one Full layer with five Reuse layers behind it, and the decoder's 20 CSA2 layers form five
groups of four, the first led by a Full layer and the rest by Reindex layers. Eight sources in
place of 38 storers is what the report means by a cache an order of magnitude smaller than V3's at
the same context.

The paper names three layer modes. *Full* compresses and indexes, which chapter 10 covers. *Reuse*
takes both the latents and the index selection from the most recent source. *Reindex* takes the
latents but runs its own indexer over them, which gives a layer a different subset of the same
table. The three modes trade a layer's independence against work. A Full layer is independent and
pays for a table and an index. A Reindex layer pays only for an index. A Reuse layer pays for
neither and reads exactly what the layer above it read, with different queries.

### What is and is not shared

The sliding window is never shared. Every layer computes its own $\mathcal{W}_\ell$ from its own input. What is
shared is the compressed table and, in Reuse layers, the selection mask over it. A reuse layer still
has its own queries and output projection, so it reads the shared table differently from the source.

The division follows what each part is for. The window carries precise local content, which changes
most from layer to layer and is cheap to store anyway. The global table carries coarse distant
content, which changes least from layer to layer and is expensive to store. Sharing is applied where
the redundancy is and avoided where it is not.

What sharing costs is expressive range. A reuse layer can only ask questions that the source layer's
representation can answer, because it never sees the distant past in any other form. Groups are
therefore kept short enough that the source is not far below the layers that read it, and a new
source is started when the representation has moved on.

### The ledger

Take the reference model's own configuration. One FP4 latent is $512 \cdot 0.5625 = 288$ bytes.
The encoder's three Full layers store one latent per $m = 2$ tokens, contributing
$3 \times 288/2 = 432$ bytes per token, and the decoder's five sources store one per token,
contributing $5 \times 288 = 1440$ bytes. The global cache is therefore about 1.8 KB per token:

$$
\Big(\frac{3}{2} + 5\Big) \times 288\ \text{B} \approx 1.8\ \text{KB per token},
$$

which is 117 MB at the native context of 65536 tokens and 1.8 GB at the YaRN-extended 1,048,576,
plus the fixed 2.5 MB of windows ($40 \times 128 \times 512$ in FP8). Against the same 40 layers
without sharing or compression that is 720 MB in FP4 at 65536 and 11.25 GB at 1M; in bf16 it is
40 KB per token; and a bf16 multi-head cache of the same shape (64 heads of width 512, keys and
values per layer) holds 5.0 MB per token, which is 5 TB at 1M. A server that could hold one
sequence of the multi-head model can now hold hundreds.

Sharing acts on storage, not on reads. Every CSA2 layer in a group still reads the whole shared
table on every decode step, so the bytes read per step fall only by the compression factor $m$.
The method is entries × bytes × reading layers, per table. At 1,048,576 tokens the
encoder's table is $1048576/2 = 524288$ entries of 288 B, or 144 MB, read by 18 encoder layers:
2.5 GB. The decoder's table is 1048576 entries of 288 B, or 288 MB, read by 20 decoder layers:
5.6 GB. The total is about 8.2 GB moved per decode step before selection. At the native 65536 the
same computation gives 9 MB by 18 and 18 MB by 20, about 522 MB — 0.5 GB — per step. Either
way that is far above what the windows cost, and storage is no longer the limit. Bandwidth is.

### In owlet1

The six-layer smoke config has ratios $[0, 0, 2, 2, 1, 1]$ with sources at layers 2 and 4. Layer 3
reuses layer 2's 256 latents, and layer 5 reuses layer 4's 512. The per-forward `shared` dictionary
carries them. See the forward-flow section of `src/mzoo/archs/owlet1/README.md`. In the training path
layer 4 sees layer 2's table as well as its own, which the paper does not describe.

### Where this leads

The global table is now small enough to store and still large enough that reading all of it dominates
a decode step — 8.2 GB of reads per step at 1M context. Chapter 10 gives each query a way to name
the few entries it needs, so that the read becomes a constant instead of $S / m$.
