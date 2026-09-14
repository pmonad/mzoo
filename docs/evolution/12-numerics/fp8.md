# bf16 and FP8

The formats a model trains in are chosen before anything else in this chapter, because they decide
what the weights, the gradients and the optimiser state cost and how fast the matmuls run. This
section builds the choice from the layout of a floating-point number, follows the move from fp32 to
fp16 to bf16, and then works out what has to be added to make eight bits usable.

## How a floating-point format is laid out

A floating-point number is a sign bit, an exponent field of $E$ bits and a mantissa field of $M$ bits.
For all values except the smallest, the stored mantissa is the fractional part of a number whose
leading bit is an implicit one:

$$
x = (-1)^{s} \cdot 2^{\,e - b} \cdot \left(1 + \frac{m}{2^{M}}\right),
\qquad b = 2^{E-1} - 1 .
$$

The exponent field $e$ selects a power of two and the mantissa interpolates within it. When $e$ is
zero the implicit leading one is dropped and the value is subnormal, which extends the range downward
at reduced precision.

Two properties follow, and they are the only two that matter for choosing a format. The number of
exponent bits sets the dynamic range, the ratio between the largest and the smallest value the format
can express, and that ratio doubles in span with each exponent bit. The number of mantissa bits sets
the relative precision, which is the same everywhere. Within any interval $[2^k, 2^{k+1})$ there are
exactly $2^M$ representable values, evenly spaced, so the gap between neighbours is about $2^{-M}$ of
the value and rounding costs at most half of that. A format cannot trade one for the other after the
fact. The split is fixed when the bits are allocated.

| format | bits | exponent | mantissa | largest finite | smallest subnormal | relative step | decades of range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp32 | 32 | 8 | 23 | $3.4 \times 10^{38}$ | $1.4 \times 10^{-45}$ | $1.2 \times 10^{-7}$ | 83 |
| fp16 | 16 | 5 | 10 | 65504 | $6.0 \times 10^{-8}$ | $9.8 \times 10^{-4}$ | 12 |
| bf16 | 16 | 8 | 7 | $3.4 \times 10^{38}$ | $9.2 \times 10^{-41}$ | $7.8 \times 10^{-3}$ | 78 |
| E4M3 | 8 | 4 | 3 | 448 | $2.0 \times 10^{-3}$ | $1.3 \times 10^{-1}$ | 5.4 |
| E5M2 | 8 | 5 | 2 | 57344 | $1.5 \times 10^{-5}$ | $2.5 \times 10^{-1}$ | 9.6 |
| E2M1 | 4 | 2 | 1 | 6 | 0.5 | $5.0 \times 10^{-1}$ | 1.1 |

The last column is the useful summary. fp32 and bf16 can hold a tensor whose values span 78 orders of
magnitude. E4M3 can hold one that spans five. That single number is why eight-bit training needs the
scaling machinery described below and why four bits are confined to the places described in the next
section.

## bf16

GPT-2 trained in fp32. fp16 halves the bytes but has 5 exponent bits, so small gradients underflow
and loss scaling is required. bf16 keeps fp32's 8 exponent bits and drops mantissa bits instead. For
training that is the better trade, because dynamic range matters more than precision. Since about
2021 the optimiser keeps weights in fp32 and the computation runs in bf16.

Loss scaling, which bf16 removes the need for, is worth understanding because it shows what the
exponent bits are for. Micikevicius et al. 2018, "Mixed Precision Training", observed that in fp16
training a large fraction of gradient values fall below the smallest representable magnitude and
become exactly zero, while the top of the fp16 range goes unused. The remedy is to multiply the loss
by a constant $S$ before the backward pass, which multiplies every gradient by $S$ and moves the
whole distribution into the representable range, then divide by $S$ before the optimiser update.
Choosing $S$ is itself a problem, since too large a value overflows, so implementations raise $S$
while no overflow occurs and halve it when one does. bf16, described by Kalamkar et al. 2019, "A
Study of BFLOAT16 for Deep Learning Training", has the same exponent field as fp32 and therefore the
same underflow threshold, so none of this is needed. A bf16 tensor can be produced by truncating an
fp32 tensor and read back with no change of range.

What bf16 gives up is precision, and this is why the master copy of the weights stays in fp32. bf16
has seven mantissa bits, so adding to a weight anything smaller than about $2^{-8}$ of that weight
leaves it unchanged. Late in training, updates are routinely that small relative to the weight, and
in a bf16 accumulator they would be discarded one after another. Keeping the master weights in fp32
and casting a bf16 copy for the forward and backward passes avoids this. Stochastic rounding, which
rounds up with a probability equal to the discarded remainder, is the alternative that keeps the
expected update correct without the fp32 copy.

The fp32 master copy is not free. With Adam the optimiser holds an fp32 weight, an fp32 first moment
and an fp32 second moment, which is 12 bytes per parameter, plus the 2-byte bf16 copy used for
compute. On the reference model's 65 billion parameters that is about 780 GB of optimiser state
against 130 GB of bf16 weights. Reduced precision in the forward and backward passes does not touch
this, which is why the optimiser state, not the weights, dominates the memory of a training run.

## What a tensor core accelerates

The throughput claim needs one hardware fact. Since the Volta generation of 2017, accelerators carry
units that compute a small matrix multiply-accumulate as a single instruction, taking tiles of two
low-precision input matrices and accumulating the product into a wider-precision tile. Peak
throughput roughly doubles each time the input width halves, so fp16 and bf16 inputs are about twice
fp32, FP8 inputs about twice bf16, and FP4 about twice FP8. The accumulator stays wide, because
summing hundreds of products in the input format would lose more than the rounding of the inputs
does.

Only matmul-shaped work benefits. Normalisation, softmax, the elementwise parts of SwiGLU, the
routing of chapter 6 and the optimiser update are not matmuls, gain nothing from narrower inputs, and
are limited by memory bandwidth rather than arithmetic. Narrower formats still help them, but through
the bytes moved rather than through the tensor cores. This is why halving the matmul precision does
not halve the time of a training step.

## FP8

DeepSeek-V3 was the first large open model trained with FP8 matmuls. Two 8-bit formats exist:

| format | exponent | mantissa | max | use |
|---|---|---|---|---|
| E4M3 | 4 | 3 | 448 | activations, weights |
| E5M2 | 5 | 2 | 57344 | gradients |

Both are defined by Micikevicius et al. 2022, "FP8 Formats for Deep Learning". The split of roles
follows the argument above. Activations and weights within one layer occupy a narrow band of
magnitudes once scaled, so they can afford to spend a bit on the mantissa. Gradients span a much
wider band, so they take the extra exponent bit and accept two mantissa bits.

Three mantissa bits cannot represent a tensor whose values span several orders of magnitude, so the
tensor is split into blocks and each block carries its own scale:

$$
\hat x_j = \operatorname{round}_{\text{E4M3}}\!\left(\frac{x_j}{s}\right) \cdot s, \qquad
s = \frac{\max_{j \in \text{block}} |x_j|}{448} .
$$

Dividing by $s$ puts the largest element of the block at the top of the E4M3 range, so the format's
five decades of range are spent on the block's own distribution rather than on the distribution of
the whole tensor. The smaller the block, the narrower the spread inside it and the less of the range
is wasted. The scale is kept alongside and reapplied when the block is read, so the stored numbers
are four-bit or eight-bit codes and the arithmetic that consumes them is exact in the accumulator.

V3 uses $1 \times 128$ blocks for activations and $128 \times 128$ for weights. It accumulates matmul
partial sums in fp32 rather than in the tensor cores' lower-precision accumulator. The result was a
training run whose loss matched a bf16 baseline within noise at half the matmul cost.

The block shapes are chosen to match the matmul. A $1 \times 128$ activation block lies along the
reduction dimension of a row, so one scale multiplies a contiguous run of the dot product and can be
applied once to the partial sum instead of element by element. The storage overhead follows the block
size directly. One four-byte scale for every 128 one-byte elements is 3.1 percent, and one one-byte
scale for the same block is 0.8 percent. For weights a $128 \times 128$ block carries one scale for
16384 elements, which is negligible, and weights can afford the coarser granularity because they
change slowly and are rescaled at every step anyway.

Transformers develop a small number of dimensions whose values are far larger than the rest.
Dettmers et al. 2022, "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", found
that such outlier features emerge as models grow and concentrate in a few dimensions, and Sun et al.
2024, "Massive Activations in Large Language Models", report a handful of activations orders of
magnitude larger than the others, tied to attention sinks. A per-tensor scale is set by those
values, so the format's range is spent on them and everything else rounds to zero. Block scaling
with small blocks confines each outlier's effect to its own block, which is why fine-grained scales
rather than per-tensor scales are used for training in FP8 and for the FP4 caches.

Block scaling is not the only response to outliers. Xiao et al. 2023, "SmoothQuant: Accurate and
Efficient Post-Training Quantization for Large Language Models", divides the activations of the
offending dimensions by a per-dimension constant and multiplies the corresponding weight rows by the
same constant, which leaves the product unchanged and moves the difficulty from the activations,
where it is hard to quantise, into the weights, where it is easier. Fine-grained block scaling
achieves a similar effect without altering the weights, and it is what the training recipes in this
chapter use.

## Scale formats

The scale $s$ itself must be stored. Two conventions appear in V4.1. A ue8m0 scale is a power of two
with 8 exponent bits and no mantissa, obtained by rounding $\max |x| / 448$ up. Multiplying by it is
exact and the clamp never clips, at up to 2x coarser resolution. An E4M3 scale is finer but not
exact. V4.1 stores the sliding-window key/value cache in FP8 E4M3 with ue8m0 scales per 32 dimensions,
because the report found the window cache too sensitive for anything smaller.

The trade between the two is worth stating plainly. A power-of-two scale changes only the exponent
field of every element in the block, so applying it and removing it introduce no rounding at all, and
rounding the scale upward guarantees that every element lands inside the format's range. What it
costs is that the true maximum may sit anywhere between half and all of the format's top value, so up
to one bit of the mantissa is unused. An E4M3 scale places the maximum at the top of the range much
more closely, recovering that bit, but multiplying by it is itself a rounding operation. Where the
stored values are already coarse, as they are in FP4, the finer scale is worth its inexactness. Where
they are less coarse and the tensor is more sensitive, as in the window cache, the exact scale is
preferred.

The next section extracts the general construction behind this recipe — quantised operands, scales
applied to partial sums — which is the same construction the FP8 and FP4 attention kernels of
chapter 17 instantiate, and the section after that takes it down to four bits, where the format
holds about one decade of range and every one of these choices becomes tighter.
