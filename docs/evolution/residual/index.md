---
title: "11 Residual stream"
order: 11
---

# 11. The residual stream

Every chapter so far has changed a sublayer. Chapters 2 to 4 changed how a block normalises its
input, how it encodes position and how its feed-forward layer is shaped. Chapters 5 to 10 changed
what attention stores and what it reads, and chapter 6 changed how much of the feed-forward layer
runs for a given token. In all of them the frame around the sublayers is the frame GPT-2 used. The
connection between sublayers, $x \leftarrow x + f(x)$, is unchanged since 2015. This chapter changes
it.

## Where the residual came from

The residual connection was introduced for convolutional networks by He et al. 2015, "Deep Residual
Learning for Image Recognition". The problem it solved was optimisation rather than capacity. Plain
networks past a few tens of layers trained to a worse training loss than shallower ones, even though
a deeper network can represent everything a shallower one can by making the extra layers the
identity. Writing each layer as an identity path plus a learned correction makes that identity the
default rather than something the layer has to discover. It also gives the gradient a path from the
loss to every earlier layer that passes through no weight matrix, so the gradient reaching layer one
does not shrink with the number of layers above it. He et al. 2016, "Identity Mappings in Deep
Residual Networks", showed that this depends on the identity path being clean, and that placing any
scaling or gating on it degrades deep networks. Earlier, Srivastava et al. 2015, "Highway Networks",
had proposed gated skips of the same shape, and Veit et al. 2016, "Residual Networks Behave Like
Ensembles of Relatively Shallow Networks", showed that a residual network of $L$ layers behaves like
a collection of $2^L$ paths of varying length, most of them short.

Vaswani et al. 2017, "Attention Is All You Need", wrapped each transformer sublayer in the same
connection, with a LayerNorm after the addition. Chapter 2 covered the move of that norm to the
input of the sublayer, which Xiong et al. 2020, "On Layer Normalization in the Transformer
Architecture", justified by showing that the pre-norm arrangement keeps gradient magnitudes stable
across depth and removes the need for a learning-rate warmup. Every model in this book uses the
pre-norm form. Nothing after that changed the connection itself.

## Two weaknesses of a single residual

The first weakness is a consequence of the pre-norm arrangement. The normalisation sits on the
sublayer's input, so the sublayer's output has a size set by its own weights and not by the size of
the stream it is added to. The stream therefore accumulates. If successive sublayer outputs were
uncorrelated and of comparable size, the variance of the stream would grow in proportion to the
number of writes, and its norm in proportion to the square root of that number. Write number $\ell$
would then be about $1/\sqrt{\ell}$ of the stream it joins. On the reference model, with $L = 80$
blocks and two sublayers in each, the last of the 160 writes would be about 8 percent of the state
it is added to, and the first would be all of it. Deep layers have less influence on the final
prediction than shallow ones, and the effective depth of the network is smaller than its nominal
depth.

Several fixes for this have been proposed, all of which keep one stream and rescale the branch that
writes into it. Bachlechner et al. 2020, "ReZero is All You Need: Fast Convergence at Large Depth",
put a single learned scalar initialised at zero on each residual branch, so training begins from the
identity network. Touvron et al. 2021, "Going deeper with Image Transformers", used a learned
per-dimension scale initialised near zero for the same purpose. Wang et al. 2022, "DeepNet: Scaling
Transformers to 1,000 Layers", multiplies the stream rather than the branch by a constant chosen
from the depth, and reaches 1000 layers with a post-norm block. These control how large each write
is relative to the stream. None of them changes the fact that there is one stream.

The second weakness is that a layer has exactly one view of the past. It reads the vector it writes
to. A feature written near the input can reach a layer near the output only through a stream that
every intervening layer has also written into, and a layer has no way to read one combination of the
past while writing into another. Nothing forces an intervening layer to preserve what it does not
use.

## What several streams cost

The change in this chapter is to carry $n$ residual vectors instead of one. Widening the residual is
not free, but what it costs is activation memory rather than arithmetic. With $n = 4$ streams the
residual state of one token grows from $D = 8192$ elements to $n D = 32768$ elements, which is 16 KB
against 64 KB in bf16. Training keeps that state at every site for the backward pass, so across the
80 blocks the residual activations of one token grow from about 1.3 MB to about 5.2 MB, and one
training sequence of 4096 tokens holds about 21 GB of them in place of about 5.4 GB. At decode only
one state per sequence is live, so 64 KB is small next to the cache figures of chapter 5, and this
chapter does not change the cache at all.

The sublayers themselves are untouched. Attention and the feed-forward layer still receive one
$D$-vector and still return one $D$-vector, so their parameter counts and their FLOPs per token are
exactly those of the previous chapters. The added arithmetic is the read, the write and the mixing
of the streams, all of which are $O(nD)$ per site. That is about 32768 multiply-accumulate
operations against the 268 million of the attention projections in the same block.

## How the three sections fit

The first section builds the mechanism, with $n$ streams, learned read weights, learned write
weights and a learned mixing matrix. The second constrains the mixing matrix, because the
unconstrained form amplifies or extinguishes streams over 80 blocks and was found to be fragile at
scale. The third removes a serial dependency that the constrained form introduces between the
coefficient computation and the sublayer, which is what makes it usable in a production training
run.

- [Hyper-connections](hyper-connections.md): several residual streams with learned read and write weights.
- [Manifold-constrained hyper-connections](mhc.md): constraining the mixing so the streams stay bounded.
- [Single-pass mHC](single-pass.md): removing the serial dependency that mHC adds.
