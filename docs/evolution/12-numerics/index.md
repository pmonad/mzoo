# 12. Numerics

Chapters 5 to 10 reduced the number of elements the model stores and reads. Grouped-query attention
cut the number of key and value heads, low-rank attention replaced them with one latent per token,
sharing removed the per-block copy, and selection reduced how much of the cache each query touches.
Chapter 11 changed the residual and left the cache alone. This chapter attacks the same quantities
from the only remaining direction, the number of bits used to store one element. The two are
multiplicative. A cache eight times smaller in elements and four times smaller in bits per element is
thirty-two times smaller in bytes.

Every reduction in this book is measured in bytes and FLOPs. Both scale with the number of bits
per element. Halving precision halves memory traffic, halves cache size, and on recent hardware
doubles matmul throughput. The obstacle is that training is sensitive to rounding. Gradients span
many orders of magnitude, and a format with too few exponent bits underflows them.

## The ledger line

Take the single shared latent of chapter 7, which stores 512 elements per token per block for each of
the 80 blocks. The bytes it occupies depend only on the format.

| format | bytes per element | per token | at 131072 tokens |
| --- | --- | --- | --- |
| bf16 | 2 | 82 KB | 10.7 GB |
| FP8 E4M3 | 1 | 41 KB | 5.4 GB |
| FP4 E2M1 with one E4M3 scale per 16 elements | 0.5625 | 23 KB | 3.0 GB |

FP4 is not four times smaller than bf16 but about 3.6 times smaller, because four-bit elements cannot
be interpreted without a shared scale and that scale has to be stored with them. The section on FP4
works out where the extra 0.0625 bytes per element goes and why the block cannot be made much larger.

## Where numbers of different kinds live

Three populations of numbers pass through a training step, and they do not have the same
requirements. Weights change by small increments and must accumulate those increments over hundreds
of thousands of steps, so they need precision. Gradients span many orders of magnitude across layers
and across the run, and their small values carry signal, so they need range. Activations are produced
and consumed within one step and are the largest of the three by volume, so they are where reduced
precision saves the most. The key-value cache is a fourth population with the easiest requirements of
all, because it is written once, read many times, and consumed by a softmax.

The three sections follow that order. The first covers the formats used for training and the block
scaling that makes eight bits workable. The second covers the four-bit format used for the cache and
the indexer, where the requirements are loosest. The third covers how a gradient passes through a
rounding operation at all, which is what makes it possible to train a model that knows it will be
quantised.

- [bf16 and FP8](fp8.md): the training formats and block scaling.
- [FP4 for the cache and indexer](fp4.md): where four bits are enough.
- [Gradients through quantisation](ste.md): the straight-through estimator and what QAT requires.
