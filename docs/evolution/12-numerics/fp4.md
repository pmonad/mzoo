# FP4 for the cache and indexer

Eight bits are the floor for the tensors that participate in training. The cache is not one of them.
It is written once, read many times, and consumed by a softmax, and those three properties are enough
to take it to four bits. This section gives the four-bit format, states the conditions under which it
is safe, and works out what it does to the ledger.

## The grid

The FP4 format E2M1 has 16 codes and 8 magnitudes: $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ with a sign.
Nothing can be trained in it directly. A tensor can be stored in it when the tensor is written once
and read many times and its consumer is a softmax that tolerates small errors.

Four bits leave one for the sign, two for the exponent and one for the mantissa, which is why the
magnitudes come in pairs within each power of two. The whole range spans a factor of 12 from the
smallest non-zero magnitude to the largest, which is about one decade against the five decades of
E4M3 and the 78 of bf16. The spacing is correspondingly coarse. The largest gap is between 4 and 6,
so a value of 5 is stored with a 20 percent error. Decoding is not worth an arithmetic circuit at
this size, and the format is handled as a lookup into a table of 16 entries, which is what the
$\operatorname{LUT}$ below denotes.

That coarseness is what rules out training in the format. A weight update smaller than the gap to the
next grid point leaves the weight where it is, and the gaps here are a third to a half of the value.
Four-bit weights have been used for inference and for adaptation, as in Frantar et al. 2023, "GPTQ:
Accurate Post-Training Quantization for Generative Pre-trained Transformers", which quantises a
trained model's weights after the fact, and Dettmers et al. 2023, "QLoRA: Efficient Finetuning of
Quantized LLMs", which freezes four-bit weights and trains a small higher-precision adapter beside
them. In both the four-bit tensor is a constant. V4.1 applies the format to tensors that are produced
at run time instead, which is a different setting and needs a different argument.

## Why written-once tensors tolerate four bits

Three conditions hold for the tensors V4.1 stores in FP4, and all three are needed.

The rounding is paid once. A cache entry is written when its token is first processed and is then
read at every later step without being rewritten. There is no accumulation, so the error stays at the
one-step level however long the sequence gets. A weight in an optimiser loop has the opposite
property, which is the subject of the previous section.

The consumer is a sum over many dimensions. A dot product over a latent of width 512 adds 512
rounded terms whose errors are largely independent, so they grow as $\sqrt{512}$ while the sum can
grow as 512. The relative error of the wide product is better than that of a narrow one.

The consumer only needs the result approximately. A softmax turns scores into weights through
differences, and the selection of chapter 10 needs the ordering of scores rather than their values.
Both absorb small perturbations that would be unacceptable in a weight.

## The two places V4.1 uses it

V4.1 uses it in two places, both as instances of the quantised matmul of the previous section. The
compressed key/value latents from chapter 9 are stored as E2M1 with one E4M3 scale per 16
dimensions:

$$
\hat c_j = \operatorname{LUT}\!\left[\operatorname{round}_{\text{E2M1}}\!\left(\operatorname{clamp}\!\left(\frac{c_j}{s}, -6, 6\right)\right)\right] \cdot s, \qquad
s = \operatorname{E4M3}\!\left(\frac{\max |c|}{6}\right).
$$

The scale is set so that the largest element of the block lands on the top grid point, and the clamp
catches the case where the rounded scale is slightly too small. The block of 16 is small for a
reason. With one decade of range, any element more than about twelve times smaller than the block
maximum rounds to zero, so blocks have to be short enough that the spread inside them stays under
that ratio. A block of 16 elements with an eight-bit scale costs one byte for every eight bytes of
data, which is 0.5625 bytes per element rather than 0.5. Why a 16-wide block on a normed latent is
safe at all is chapter 9's magnitude bound, not this section's.

The indexer queries and keys from chapter 10 use the OCP MXFP4 convention: E2M1 with a ue8m0 scale
per 32 dimensions. The indexer score matrix is the largest single computation at long context, and
running it in four bits is a large part of what makes million-token selection affordable.

The two conventions differ in exactly the way the previous section described. The cache uses an E4M3
scale, finer but inexact, because its stored values are what a later softmax will read and every bit
of resolution is worth having. The indexer uses a power-of-two scale over a block of 32, which costs
0.53125 bytes per element, because the scale there is applied inside a matmul on the fly and an exact
power of two folds into the accumulation without a rounding step. The indexer is also the case where
FP4 buys throughput rather than storage, since its scores are computed and discarded rather than
kept.

## Effect on the cache

Per token, V4.1 stores one 512-wide latent from chapter 7, once per group rather than once per layer
by the sharing of chapter 9, pooled by the compressor of chapter 9, in FP4 with a scale every 16
dimensions. Measured against a bf16 GQA cache of $2 \times 8 \times 128$ per layer, these steps
together give the order-of-magnitude reduction the report claims.

The arithmetic runs as follows, on the paper's own 40-layer configuration. One 512-wide latent per
layer per token would be 40 KB in bf16 and 11.5 KB in FP4, which is 5.2 GB against 1.5 GB at the
long-context target of 131072 tokens. The real schedule stores 8 sources instead of 38 layers, with
the encoder's three pooling at $m = 2$ and the decoder's five at $m = 1$, which leaves
$(3/2 + 5) \times 288\ \text{B} \approx 1.9$ KB per token, or 245 MB across the whole 131072-token
context. The per-layer sliding window of chapter 8 is held in FP8 and is a fixed 2.6 MB per sequence
whatever the length. The total is about 248 MB, against the 21 GB a bf16 grouped-query cache of the
same depth would need for the same context and the 172 GB of a multi-head cache of chapter 5's
design.

How the four-bit entries are read back — dequantised into a bf16 tile on load rather than
multiplied in place — is a kernel decision, and is the subject of chapters 16 and 17.

## In owlet1

`quant.py` reproduces the rounding in fp32. This is fake quantisation: training sees the same values
the real kernels would produce, and nothing is stored in four bits.

Fake quantisation is the standard way to develop against a format before the kernels exist, and it
leaves one problem open. A rounding step now sits inside the forward pass during training, so the
parameters that feed it must receive a gradient through it, and rounding has no gradient. The next
section covers what is done about that.
