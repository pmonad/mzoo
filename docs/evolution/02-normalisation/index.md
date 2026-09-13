# 2. Normalisation

Chapter 1 left a LayerNorm at the entry of each sublayer and said only that it maps the $D$
dimensions of a token to zero mean and unit variance. This chapter takes that component apart. It is
the smallest thing in the block. Its parameters are $2D$ against the $12D^2$ of the block, it moves
none of the three cost formulas of chapter 1, and its arithmetic is a rounding error next to the
matrix products around it. It is also the component that every model after GPT-2 changed, because
the choice of statistic and the placement of the norm decide whether a stack of 80 blocks trains at
all.

Three changes are covered. The statistic was simplified from mean and variance to root mean square.
The placement was settled on pre-norm. And a norm was added inside attention, on the queries and
keys, to stop the logits from growing during training. The chapter ends with the predecessor these
forms displaced, batch normalisation, because the reasons it does not suit a language model explain
what the surviving forms are for.

## Why normalise

A language model is a stack of transformer blocks, and each block acts on the vector that represents
one token. The activations of a deep stack drift in scale from block to block. The gradient of a
layer then depends on where the layer sits in the stack, so one learning rate cannot suit all layers,
and training either needs a very small learning rate or diverges. Normalising the activations to a
fixed scale at the entry of each block removes the drift and makes the layers comparable.

The drift is not an accident of initialisation. In the residual arrangement of chapter 1 every
sublayer adds its output to the running sum, so the sum grows as depth increases even when every
sublayer is well behaved. A sublayer near the top therefore receives an input several times larger
than one near the bottom, and it must learn weights several times smaller to produce an output of
the same size. Two layers with identical roles then need different learning rates.

Normalisation also changes the geometry of the optimisation. A norm placed before a linear layer
makes the composition invariant to the scale of its input, so multiplying $x$ by any positive
constant leaves the output unchanged. Two facts follow. The gradient with respect to $x$ has no
component along $x$, because moving along $x$ does not change the output, so an update can only
rotate the representation rather than inflate it. And the gradient with respect to the weights of
the layer that produced $x$ shrinks as those weights grow, which bounds a runaway in which large
weights produce large activations that produce large gradients.

The historical claim, from Ioffe and Szegedy 2015, "Batch Normalization: Accelerating Deep Network
Training by Reducing Internal Covariate Shift", was that normalisation removes a shift in the
distribution of each layer's inputs during training. Santurkar et al. 2018, "How Does Batch
Normalization Help Optimization?", showed that the shift is not the mechanism and that the benefit
comes from a smoother loss surface with better bounded gradients. The practical conclusion is the
same either way. Normalise the input of each sublayer, and the stack becomes trainable at a
learning rate that does not depend on depth.

## Layer normalisation

Layer normalisation is the form GPT-2 uses. It comes from Ba et al. 2016, "Layer Normalization",
which adapted batch normalisation to recurrent networks by taking the statistics over the dimensions
of one example instead of over the batch.
LayerNorm takes its statistics per token over the $D$
dimensions. It subtracts the mean, divides by the standard deviation, and applies a learned gain and
bias:

$$
\operatorname{LN}(x) = g \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + b .
$$

The mean and the variance are those of the $D$ dimensions of that one token, so
$\mu = \tfrac{1}{D}\sum_i x_i$ and $\sigma^2 = \tfrac{1}{D}\sum_i (x_i - \mu)^2$.

Here $g, b \in \mathbb{R}^D$ are learned vectors and $\odot$ is the elementwise product, so dimension $i$ of the
normalised vector is multiplied by its own gain $g_i$ and shifted by its own bias $b_i$. The same symbol
is used for elementwise products throughout the book. The constant $\epsilon$ is a small positive
number, typically $10^{-5}$ or $10^{-6}$, added so that a token whose dimensions are all equal does
not divide by zero.

Read geometrically, the operation maps $x$ onto a sphere. Subtracting the mean removes the component
of $x$ along the all-ones direction, and dividing by the standard deviation fixes the length of what
remains at $\sqrt{D}$. Every token entering a sublayer therefore has the same length and lies in the
same $D - 1$ dimensional subspace, and only its direction carries information. The gain and bias then
restore the freedom to give dimensions different scales and offsets, but as learned constants rather
than as something the previous layers can inflate.

The statistics belong to a single token, so the result is independent of the batch, is the same for
one token as for a full batch, and needs no running averages. The cost is two reductions over $D$,
one for the mean and one for the variance, followed by an elementwise gain and bias.

### What a norm costs on the reference model

On the reference model one norm acts on a vector of $D = 8192$ numbers, which is 16 KB in bf16.
LayerNorm makes two passes over that vector, one to accumulate the mean and the variance and one to
subtract, scale, and apply the gain and the bias, and it performs a few operations per dimension, at
most about $4D = 33$ thousand multiply-adds. The block that surrounds it performs 805 million
multiply-adds on the same token, one for each parameter of chapter 1. The norm is therefore about
one twenty-thousandth of the arithmetic of the block, and its parameters are $2D = 16$ thousand
against $12D^2 = 805$ million. Over the whole model, 161 norms at two vectors each come to 2.6
million parameters out of 65 billion.

The cost of a norm is not arithmetic. It is memory traffic and serialisation. The vector must be
read, reduced to two scalars, and read again, and the matrix product that follows cannot begin until
the reduction has finished. As chapter 5 sets out for the cache, a current accelerator performs
several hundred floating-point operations in the time it takes to read one byte from memory, so an
operation that moves 16 KB in order to do 33 thousand multiply-adds runs at a small fraction of the
machine's rate and is charged for its bytes rather than its operations. This is why the useful
saving in the next section is a pass over the activation removed, and not a multiply saved.

## RMSNorm

Two of the operations in LayerNorm are unnecessary. Removing the mean was found not to hurt quality,
and the bias $b$ is redundant with the bias of the following linear layer. Circuit analyses
make the same simplification when they fold the LayerNorm gain and bias into the next linear
layer, as in Elhage et al. 2021, "A Mathematical Framework for Transformer Circuits". Dropping
both gives

$$
\operatorname{RMSNorm}(x) = g \odot \frac{x}{\sqrt{\tfrac{1}{D}\sum_i x_i^2 + \epsilon}} .
$$

The form is due to Zhang and Sennrich 2019, "Root Mean Square Layer Normalization". The denominator
is the root mean square of the $D$ dimensions, so the operation fixes the length of $x$ at $\sqrt{D}$
and leaves its direction alone. Compared with LayerNorm it keeps the scale invariance, which is the
property that made normalisation useful, and gives up the shift invariance, which was doing no work
in a network whose next operation is an affine map.

This is one reduction instead of two, and one fewer parameter vector. Each removed operation was a
separate pass over the activation, and on accelerators such small memory-bound operations take a
noticeable share of the non-matmul time. All LLaMA-family models and every DeepSeek model use this
form. The statistic is computed in fp32 even when activations are bf16, because the sum of squares
overflows or loses precision in half precision. On the reference model that sum runs over 8192
squared values, and bf16 carries eight bits of mantissa, so an accumulation in bf16 would stop
adding small terms long before the end of the vector.

In owlet1 this is `DeepseekV41RMSNorm` in `src/mzoo/archs/owlet1/norm_rope.py`, which casts to fp32,
takes the mean of squares, and casts back. An unweighted variant in the same file drops $g$ as well,
and is used where the following layer supplies the scale.

## Placement

GPT-2 already used pre-norm. Pre-norm normalises the sublayer input and adds the raw sublayer
output to the residual. The alternative is post-norm from the original transformer, which
normalises the sum:

$$
\text{post-norm:}\quad x \leftarrow \operatorname{Norm}\big(x + f(x)\big), \qquad
\text{pre-norm:}\quad x \leftarrow x + f\big(\operatorname{Norm}(x)\big).
$$

Pre-norm keeps an unnormalised path from input to output, so gradients reach early layers without
passing through a norm at every block. It trains stably at depth without learning-rate warm-up,
at the cost of the residual growing in magnitude with depth. Xiong et al. 2020, "On Layer
Normalization in the Transformer Architecture", made the difference precise. Under post-norm the
expected gradient at the output layer grows with depth at initialisation, which is why the original
recipe needs a warm-up schedule, and under pre-norm it does not.

The growth of the pre-norm residual can be estimated. Each sublayer writes a vector that is roughly
independent of what is already there, so the variances add and the length of the residual grows
about as the square root of the number of writes. After the 160 writes of the reference model the
residual is about thirteen times the length of the embedding it started from. The norm at the entry
of each sublayer divides that length out, so a sublayer high in the stack sees the same input scale
as one low in the stack, and what changes with depth is that each new write is a smaller fraction of
the sum. Deep pre-norm stacks therefore tend to make small relative edits near the top.

Post-norm bounds the output of every
block and gives a better conditioned final representation, and it was used in the original
transformer and in BERT, but at depth it needs warm-up and careful initialisation. Wang et al. 2022,
"DeepNet: Scaling Transformers to 1,000 Layers", showed that post-norm can be pushed to very large
depth if the residual branch is scaled down by a factor derived from the depth, which confirms that
the difficulty is the magnitude of the residual branch rather than the placement as such. Some
models add a
second norm after the sublayer output, a form called sandwich norm and used in Gemma 2, to limit
that growth. Every model in this document is pre-norm. Chapter 11 returns to the residual itself.

Pre-norm has one consequence at the top of the stack. The last sublayer output is added to the
residual without ever being normalised, so the final state has no fixed scale. That is why the model
of chapter 1 ends with a separate normalisation before the output projection. In owlet1 the two
per-sublayer norms are `attn_norm` and `ffn_norm` in `src/mzoo/archs/owlet1/decoder.py`, and the
final norm is at the end of the same file.

## Normalising inside attention

Attention logits are $q \cdot k / \sqrt{d}$. If the projections grow during training, logits grow
with them, the softmax saturates, and training destabilises. Applying RMSNorm to $q$ and $k$ per head
bounds the logit magnitude:

$$
s_{ij} = \frac{\operatorname{RMSNorm}(q_i) \cdot \operatorname{RMSNorm}(k_j)}{\sqrt{d}} .
$$

Chapter 1 divided by $\sqrt{d}$ on the assumption that the dimensions of $q$ and $k$ have unit
variance. Nothing enforces that assumption. The norms of $W_Q$ and $W_K$ are free to grow, and a
model can reduce its loss by growing them, because larger logits give a sharper softmax and a more
confident copy of whatever the head has found. The failure mode is that the softmax reaches the
region where one weight is one and the rest are zero, where its gradient is zero, and the head stops
learning at a pattern it can no longer adjust. Zhai et al. 2023, "Stabilizing Transformer Training
by Preventing Attention Entropy Collapse", describes this collapse and shows that controlling the
logit scale prevents it. With the norms in place, $\operatorname{RMSNorm}(q)$ and
$\operatorname{RMSNorm}(k)$ both have length $\sqrt{d}$, so the logit is at most $\sqrt{d}$
regardless of what the projections do, and a learned per-head gain remains the only way for the
model to sharpen the distribution.

The form is from Henry et al. 2020, "Query-Key Normalization for Transformers", and it entered large
models through Dehghani et al. 2023, "Scaling Vision Transformers to 22 Billion Parameters", which
needed it to train at that size. Qwen3 in 2025 uses grouped-query attention with QK-norm in every
attention layer, and GLM-4.5 in 2025 uses it as well.

DeepSeek applies a norm to the low-rank query and key/value latents of chapter 7 rather than to the
per-head vectors. This has the same effect at lower cost. The latents are smaller than the
concatenated per-head vectors, and normalising them before they are expanded bounds every head at
once. In owlet1 these are `q_norm` and `kv_norm` in `src/mzoo/archs/owlet1/attention.py`.

## Predecessor: batch normalisation

Batch normalisation is the earlier method, and it suits convolutional networks on fixed-size images.
It takes its statistics per dimension over the batch. For dimension $c$, let $\mu_c$
and $\sigma_c^2$ be the mean and variance of that dimension over all tokens in the batch:

$$
\operatorname{BN}(x)_c = g_c \frac{x_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + b_c .
$$

The difference from the two forms above is which axis is reduced. LayerNorm and RMSNorm reduce over
the $D$ dimensions of one token and treat each token separately. Batch normalisation reduces over
the tokens and treats each dimension separately, so its output for one token depends on every other
token in the batch.

It does not suit language models, for three reasons. The statistics depend on the composition of the
batch, so the output for one sequence changes when the other sequences change. Sequences have
variable length, so the number of tokens contributing to a dimension is not fixed and padding positions
distort the statistics. Inference often processes one sequence at a time, where there is no batch to
average over. Batch normalisation therefore keeps running averages of $\mu_c$ and $\sigma_c^2$
collected during training and uses them at inference, which makes the function at test time different
from the function that was trained.

A fourth reason is distributed training. The statistics of a batch that is split across accelerators
require a reduction across all of them at every normalisation, in the forward pass and again in the
backward pass. A per-token statistic needs no communication at all, which is what makes it usable at
the scale of the models in this book.

## Summary

| | Statistics over | Parameters | Reductions per token | Batch dependent | Used in |
| --- | --- | --- | --- | --- | --- |
| LayerNorm | $D$ dimensions of one token | gain and bias, $2D$ | two | no | GPT-2 and BERT |
| RMSNorm | $D$ dimensions of one token | gain, $D$ | one | no | LLaMA and DeepSeek |
| BatchNorm | batch, per dimension | gain and bias per dimension | two, over the batch | yes | CNNs |

Every model from here on uses pre-norm RMSNorm at the entry of both sublayers, a final RMSNorm
before the output projection, and a norm on the query and key side of attention. On the reference
model that choice costs 1.3 million parameters out of 65 billion and one reduction per norm per
token, and it is what allows the remaining chapters to change the contents of the sublayers without
revisiting whether the stack still trains.

Chapter 3 turns to the other component chapter 1 left provisional. Position in GPT-2 is a learned
table added to the embedding, which caps the context and encodes absolute rather than relative
position. The models that follow remove it from the residual stream entirely and put position inside
the attention dot product.
