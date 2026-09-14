---
title: "4 Feed-forward"
order: 4
---

# 4. Feed-forward

Chapters 2 and 3 changed the normalisation and the position encoding, neither of which holds
parameters worth counting. The feed-forward layer is the other half of the block and it holds two
thirds of its parameters, 537 million of 805 million on the reference model. It is also the simplest
part. Attention decides which positions a token reads from, and the feed-forward layer is what the
token then computes on its own, one position at a time, with no reference to any other. Every change
in this chapter is a change to the shape of that computation.

The change that every model adopted is a gate. GPT-2 passes the hidden layer through an activation
function and projects it back. Modern models compute the hidden layer twice from different weights
and multiply the two results together before projecting back. The chapter works out what that costs
in parameters on the reference model, why the standard width of $\tfrac{8}{3}D$ follows from holding
the parameter count fixed, and why the same layer is the object that chapter 6 replicates hundreds
of times.

## The ungated MLP

The GPT-2 MLP is a single ungated hidden layer, $W_2\,\operatorname{GELU}(W_1 x)$. The
activation scales each hidden unit, and that scale comes from the same projection that produces
the value. Shazeer's 2020 comparison of gated variants found that a separate gate gives a
consistent and cheap quality gain.

Consider what one hidden unit does. Row $r$ of $W_1$ gives a score $w_r \cdot x$, the activation maps
that score to a number, and column $r$ of $W_2$ is multiplied by that number and added into the
output. The score therefore has two jobs at once. It decides whether the unit contributes, through
the shape of the activation near zero, and it decides how much the unit contributes, through its own
magnitude. One vector of parameters cannot be tuned for both. A unit that should fire strongly
whenever a weak pattern is present, or weakly whenever a strong pattern is present, has no way to
express that with a single row.

There is a second limitation. The output of an ungated layer is a sum of terms each of which is a
function of one linear projection of $x$. Products between two different projections of the input
cannot be formed. Gating supplies exactly that missing product.

The idea comes from Dauphin et al. 2017, "Language Modeling with Gated Convolutional Networks",
which introduced the gated linear unit as one projection multiplied elementwise by a sigmoid of
another. Shazeer 2020, "GLU Variants Improve Transformer", tried the same construction inside the
transformer feed-forward layer with several activation functions and found the variants using SiLU
and GELU consistently ahead of the ungated layer at equal parameter count.

## SwiGLU

Two projections up, one multiplicative gate, one projection down:

$$
\operatorname{FFN}(x) = W_2\big(\operatorname{SiLU}(W_1 x) \odot W_3 x\big), \qquad
\operatorname{SiLU}(z) = z\,\sigma(z).
$$

$W_1 x$ sets how much of each hidden unit passes through, and $W_3 x$ supplies the value. To keep the
parameter count equal to the two-matrix MLP, the hidden width drops from $4D$ to about $\tfrac{8}{3}D$.
Three matrices of $\tfrac{8}{3}D \times D$ have the same size as two of $4D \times D$. Compute per
token is unchanged.

The activation $\operatorname{SiLU}(z) = z\,\sigma(z)$, with $\sigma$ the logistic function, is from
Elfwing et al. 2018, "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in
Reinforcement Learning", and was found again by the search of Ramachandran et al. 2017, "Searching
for Activation Functions", under the name Swish. It is close to the identity for large positive $z$,
close to zero for large negative $z$, and slightly negative near $z = -1$, so it is smooth and
non-monotone rather than a hard switch. The name SwiGLU is the composition of that activation with
the gated linear unit.

Two properties distinguish the gated form from the ungated one. Each hidden unit now has two rows of
parameters, one that decides whether it passes and one that decides what it carries, so the two jobs
described above are separated. And the output is bilinear in $x$ before the down-projection, which
means the layer can represent products of two different features of the input rather than only sums
of functions of single features.

### The parameter match on the reference model

The width $\tfrac{8}{3}D$ is not a tuned constant. It is what comes out of requiring the same
parameter count as the GPT-2 layer. An ungated layer of hidden width $F$ holds $2FD$ parameters and
a gated one holds $3FD$, so matching $3FD = 8D^2$ gives $F = \tfrac{8}{3}D$. On the reference model
with $D = 8192$ that is 21845 after rounding down, and the two counts are

$$
2 \cdot 32768 \cdot 8192 = 536\,870\,912, \qquad 3 \cdot 21845 \cdot 8192 = 536\,862\,720 .
$$

The difference is 8192 parameters out of 537 million, one part in sixty-five thousand, so the two
layers are the same size to any accuracy that matters. Compute follows the parameters. Three matrix
products of $21845 \times 8192$ against a vector are 537 million multiply-adds, which is 1.07 GFLOPs
per token per block, the figure chapter 6 compares its routed experts against.

Real models do not use 21845. LLaMA-2 70B has the depth and width of the reference model and a
feed-forward hidden width of 28672, which is $3.5D$ rather than $\tfrac{8}{3}D \approx 2.67D$. Its
three matrices hold $3 \cdot 28672 \cdot 8192 = 705$ million parameters per block instead of 537
million, and 1.41 GFLOPs per token per block instead of 1.07. That widening, at 80 blocks, is most
of the difference between the 65 billion parameters of the reference model and the 70 billion of the
model it is drawn from. Parameter matching against the GPT-2 layer is a convention for comparison,
not a constraint on the design. Widths are also rounded to a multiple of a power of two so the
matrix products tile cleanly on the accelerator, which is another reason no model uses 21845 exactly.

What the change costs is small and real. There are three matrix products where there were two, so
the layer launches more work of smaller size, and the elementwise product needs both $W_1 x$ and
$W_3 x$ resident at the hidden width before the down-projection can start, which is more activation
memory during training than the ungated layer at the same parameter count. Against that, the layer
is now the standard one. Every model in this book from here on uses it.

## The MLP as a memory

Geva et al. 2021, "Transformer Feed-Forward Layers Are Key-Value Memories" describe the MLP as a set
of key-value memories. Each row of the first projection acts as a key that is matched against the
input by a dot product, and the corresponding column of the second projection is the value written
to the residual stream. In SwiGLU that matching is multiplicative, because the gate
$\operatorname{SiLU}(W_1 x)$ scales the value supplied by $W_3 x$ before $W_2$ writes it out. Meng
et al. 2022, "Locating and Editing Factual Associations in GPT" locate factual associations in
mid-layer MLPs and edit them by changing the weights there. Individual hidden units do not
correspond to one feature each, because a model stores more features than it has dimensions and
represents them in superposition, as described by Elhage et al. 2022, "Toy Models of Superposition".
Read this way, the mixture of experts of chapter 6 adds memory capacity without adding compute per
token.

The reading also explains why the hidden width is the natural place to add capacity. The number of
patterns the layer can hold scales with the number of rows of $W_1$, while the width $D$ of the
residual stream is fixed by the rest of the block. On the reference model the layer holds 21845 such
entries per block and 1.7 million across the 80 blocks, and every one of them is read for every
token. Chapter 6 keeps the read cost fixed and raises the entry count by a factor of twenty-four.

## In DeepSeek

Every expert in the MoE layers of chapter 6 is a SwiGLU block of this form. V4.1 adds a clamp on the
pre-activations to bound the product when running in low precision:

$$
g = \min(W_1 x,\ c), \qquad u = \operatorname{clip}(W_3 x,\ -c,\ c), \qquad c = 10 .
$$

SiLU is already bounded below, so the gate needs only an upper clamp. This is a numerical guard, not a
modelling change.

The reason a guard is needed here rather than elsewhere in the block is the multiplication. Two
projections that are each within the range of the number format can produce a product that is not,
and chapter 12 runs these matrices in FP8 and FP4, where the representable range is narrow. Clamping
the two factors at 10 bounds the product at 100 before the down-projection sees it. In owlet1 this is
`DeepseekV41Expert` in `src/mzoo/archs/owlet1/moe.py`, which computes the gate and the up-projection
in fp32, clamps both, multiplies, and casts back before the down-projection.

The dense block is now complete. It normalises with RMSNorm, encodes position by rotating queries and
keys, attends with $H$ heads, and computes a gated feed-forward layer of width $\tfrac{8}{3}D$ or
wider. Nothing in chapters 2 to 4 changed what the block costs to run, only what it computes. Chapter
5 turns to cost, and to the quantity that drives every remaining chapter of the book, the memory that
attention needs when the model generates one token at a time.
