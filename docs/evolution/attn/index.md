---
title: "Attention"
order: 7
---

# Attention

This part follows the cost of attention at inference. The KV cache sets the bytes, the bytes set
the memory traffic, the traffic sets what the kernels must do, and precision sets how cheap each
byte can be; every chapter in the arc attacks one link of that chain.

1. [Chapter 5, The KV cache](kv-cache.md) — why generation needs a cache, how big it is, and how
   sharing across heads (MQA, GQA) shrank it in dense models.
2. [Chapter 7, Low-rank attention](low-rank-attention/index.md) — one shared latent per token,
   from which every head's key and value are reconstructed.
3. [Chapter 8, Local attention](local-attention.md) — the sliding window that caps what decode
   must read at all.
4. [Chapter 9, Compressed attention](compressed-attention/index.md) — the compressor that shrinks
   what is stored and shared across layers.
5. [Chapter 10, Sparse selection](sparse-selection/index.md) — the indexer that picks the few
   entries each token actually attends to.
6. [Chapter 16, Attention kernels](attention-kernels/index.md) — how the whole design is made to
   run fast on real hardware.
7. [Chapter 17, Low-precision attention](low-precision-attention/index.md) — the number formats
   that make each cached byte cheaper.

A note on reference models. Chapters 1 to 6 use a dense 65B model shaped like LLaMA-2 70B for
their block arithmetic. This part instead uses DeepSeek-V4.1-Flash as its reference model — 40
layers, 64 query heads of width 512, one shared KV latent per token, native context 65536 extended
to 1,048,576, 552B backbone parameters — because its attention is the design this part explains.
