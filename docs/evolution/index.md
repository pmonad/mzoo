# From GPT-2 to DeepSeek-V4.1

This book explains how the transformer block changed between GPT-2 in 2019 and the open-weight
models of 2026, ending with DeepSeek-V4.1 as the worked example. It is written for someone who
already knows what a transformer is and can read an equation, but who has not followed the design
changes of the last few years closely and wants to understand a current model well enough to read
its code.

## How the book is organised

Almost every change since GPT-2 is a local edit to one part of the block. The book is therefore
organised by the part of the block that changes, not by date. Chapter 1 sets out the GPT-2 block and
fixes the notation. Chapters 2 to 5 cover the small changes to the dense block that every modern
model adopted: the normalisation, the position encoding, the feed-forward layer and the key-value
cache. Chapters 6 to 10 cover the two big ideas of the period, sparsity in the feed-forward layer
through mixture of experts and sparsity in attention through low-rank, local, compressed and
selected attention. Chapter 11 changes the residual connection itself, chapter 12 covers number
formats, and chapters 13 and 14 cover components outside the block. Chapter 15 assembles the
complete DeepSeek-V4.1 block from the pieces, and chapters 16 and 17 drop to the kernel level: how
the attention of chapters 7 to 10 is tiled and computed on one GPU, and which of its operands can
leave full precision.

Each chapter follows the same course. It first says what the block looks like at that point and
what has become the limiting cost or the limiting weakness. It then gives the change in words, then
as an equation, then works the effect out on a reference model so the numbers are concrete. It ends
by saying what the change costs and where the next chapter picks up. Chapters that cover several
concepts are split into sections, listed at the start of the chapter.

## The reference model

To keep the numbers comparable the book carries one reference model through every chapter. It is a
dense model of the LLaMA-2 70B shape, chosen because it is the last widely used design before the
changes of chapters 6 to 12 and because its numbers are round.

| symbol | meaning | value |
| --- | --- | --- |
| $L$ | number of blocks | 80 |
| $D$ | width of the residual stream | 8192 |
| $H$ | number of attention heads | 64 |
| $d$ | width of one head, $D = H d$ | 128 |
| $V$ | vocabulary size | 32000 |
| $S$ | context length | 4096 in training, 131072 as the long-context target |
| bytes | bytes per element | 2 for bf16 |

With these values and the GPT-2 block of chapter 1 the model has about 65 billion parameters.
LLaMA-2 70B reaches 70 billion with a wider feed-forward layer. Where a chapter changes one of these
numbers, it says so and works out the consequence.

## Notation

Equations are written for one token unless a sequence index appears. $x \in \mathbb{R}^D$ is the
residual state of a token, and one coordinate of it is called a dimension. Vectors are column
vectors. $W x$ is a matrix acting on a vector. $a \cdot b$ is the dot product, $a \odot b$ the
elementwise product, and $[\,a \,\|\, b\,]$ the concatenation of two vectors into a longer one. Each
of these is also introduced in the chapter where it first matters.

## Contents

| # | chapter | modifies |
|---|---|---|
| 1 | [The reference block: GPT-2](01-gpt2/index.md) | baseline and notation |
| 2 | [Normalisation](02-normalisation/index.md) | LayerNorm to RMSNorm, placement |
| 3 | [Positions](03-positions/index.md) | learned embeddings to RoPE |
| 4 | [Feed-forward](04-feed-forward/index.md) | GELU MLP to SwiGLU |
| 5 | [The KV cache](05-kv-cache/index.md) | MHA to MQA and GQA |
| 6 | [Mixture of experts](06-moe/index.md) | dense FFN to routed experts |
| 7 | [Low-rank attention](07-low-rank-attention/index.md) | MLA and the single shared latent |
| 8 | [Local attention and sinks](08-local-attention/index.md) | sliding window, sink, hybrid schedules |
| 9 | [Compressed shared attention](09-compressed-attention/index.md) | compressor, cross-layer KV sharing |
| 10 | [Sparse selection](10-sparse-selection/index.md) | the indexer and top-k |
| 11 | [The residual stream](11-residual/index.md) | hyper-connections and mHC |
| 12 | [Numerics](12-numerics/index.md) | bf16, FP8, FP4 |
| 13 | [Engram memory](13-engram/index.md) | n-gram hash lookup |
| 14 | [Beyond the block](14-beyond-the-block/index.md) | MTP, DSpark, encoder-decoder |
| 15 | [Assembly: DeepSeek-V4.1](15-assembly/index.md) | the complete block |
| 16 | [Attention kernels](16-attention-kernels/index.md) | FlashAttention-2 tiling, the CSA2 kernel |
| 17 | [Low-precision attention](17-low-precision-attention/index.md) | FP8/FP4 attention variants, the merged recipe |

Where a component exists in this repository the chapter points to the implementation under
`src/mzoo/archs/owlet1/`, a small trainable version of the DeepSeek-V4.1 block.
