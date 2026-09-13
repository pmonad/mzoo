# 1. The reference block: GPT-2

GPT-2 comes first because almost every later change is a local edit to it. The
model is an embedding, a stack of identical blocks, and a projection back to the vocabulary. Every
design in this book keeps that outer shape and rewrites one part of one block, so the GPT-2 version
of each part is the baseline the rest of the book measures against. This chapter sets out that
version in full and fixes the notation.

The chapter also works the arithmetic of the reference model of the preface through once. Three
quantities come out of it: the parameters in one block, the floating-point operations per token, and
the bytes of key-value cache per token. Every chapter from here to chapter 12 changes one factor of
one of those three, and each quotes the values computed here rather than recomputing them. The last
section is a table of which chapter moves which factor.

## Where GPT-2 sits

The transformer is due to Vaswani et al. 2017, "Attention Is All You Need", as an encoder-decoder
model for translation. GPT-2, from Radford et al. 2019, "Language Models are Unsupervised Multitask
Learners", keeps only the decoder half. It has no encoder and no cross-attention, every block is
identical, and the single training objective is to predict the next token of ordinary text. That
shape has not changed since. DeepSeek-V4.1 in 2026 is still an embedding, a stack of identical
blocks and an unembedding, trained to predict the next token.

Two smaller decisions of GPT-2 also survived. It placed the normalisation at the input of each
sublayer rather than after it, the arrangement chapter 2 calls pre-norm, and it added a final
normalisation before the output projection. Both are in every model in this book.

The largest GPT-2 has 48 blocks, a residual width of 1600, a context of 1024 positions, a
byte-pair vocabulary of 50257 entries and about 1.5 billion parameters. The reference model of this
book is a hundred times larger in nothing but its dimensions. It has 80 blocks, a width of 8192 and a
context of 4096, and its block is the GPT-2 block unchanged.

## Input

Text is first cut into tokens by a byte-pair encoder, so a token is a common byte string rather than
a word, and the vocabulary $V$ is a fixed list of such strings. The model never sees characters. It
sees an index into that list.

Each token gets a learned vector and a learned position vector, added together:

$$
x_0 = E[\text{id}] + P[\text{pos}], \qquad E \in \mathbb{R}^{V \times D},\ P \in \mathbb{R}^{S_{\max} \times D}.
$$

$E$ holds one row of width $D$ per vocabulary entry and $P$ one row of width $D$ per position. The
two rows are added rather than concatenated, so from the first layer onward position occupies the
same $D$ dimensions as content and no part of the vector is reserved for either.

Positions come from a lookup table. The model cannot run past $S_{\max}$. It has no
representation of relative distance beyond what the table learns. Both defects follow from the table
being indexed by absolute position. Row 4097 does not exist, and rows 500 and 501 are learned
independently of rows 1500 and 1501 even though the offset between the members of each pair is the
same. On the reference model with $S_{\max} = 4096$ the table holds
$4096 \cdot 8192 = 33.6$ million parameters, small against the 65 billion of the whole model, so the
table is not expensive. It is inflexible. Chapter 3 removes it.

## The block

Two sublayers, each wrapped in a residual connection with a LayerNorm applied before it:

$$
\begin{aligned}
x &\leftarrow x + \operatorname{Attn}\big(\operatorname{LN}(x)\big) \\
x &\leftarrow x + \operatorname{MLP}\big(\operatorname{LN}(x)\big)
\end{aligned}
$$

The two arrows are applied in order, and the block is this pair repeated $L$ times with separate
weights. The additions are residual connections in the sense of He et al. 2015, "Deep Residual
Learning for Image Recognition". Each sublayer computes a correction to its input rather than a
replacement for it, so a sublayer that has learned nothing yet is the zero function and leaves the
stack intact, and the gradient of the loss reaches every depth through the chain of additions
without being multiplied by a weight matrix at each step.

Here $\operatorname{LN}$ is layer normalisation. It normalises the $D$ dimensions of one token to
zero mean and unit variance, then applies a learned gain and bias. Chapter 2 defines it and its
successors.

### Attention

Attention is the only operation in the block that moves information between positions. Each position
forms a query, every position offers a key and a value, and the position reads a weighted average of
the values of the positions whose keys match its query.

Attention has $H$ heads. For head $h$ and position $i$, with queries, keys and values projected from
the normed input,

$$
o_{h,i} = \sum_{j \le i} \operatorname{softmax}_j\!\left(\frac{q_{h,i} \cdot k_{h,j}}{\sqrt{d}}\right) v_{h,j},
\qquad
\operatorname{Attn}(x) = W_O\,[\,o_1 \,\|\, \dots \,\|\, o_H\,].
$$

Read this from the inside out. The vectors $q_{h,i}$, $k_{h,j}$ and $v_{h,j}$ each have width $d$
and come from the $D \times D$ projections $W_Q$, $W_K$ and $W_V$ applied to $\operatorname{LN}(x)$
and then cut into $H$ pieces of width $d$. The dot product $q_{h,i} \cdot k_{h,j}$ scores position
$j$ as a source for position $i$. The softmax over $j$ turns those scores into weights that are
positive and sum to one, so the output is an average of the values and stays at the scale of a
single value. The restriction $j \le i$ is the causal mask. Without it a position could read from
its own future and next-token prediction would be trivial. With it, one forward pass over a sequence
of length $S$ produces $S$ separate predictions, each conditioned only on its own prefix.

The division by $\sqrt{d}$ has a reason. If the dimensions of $q$ and $k$ are independent with unit
variance, the dot product of two $d$-dimensional vectors has variance $d$ and therefore a typical
magnitude of $\sqrt{d}$. Dividing by $\sqrt{d}$ returns the score to unit scale, which keeps the
softmax away from the flat region where all weights are equal and away from the saturated region
where one weight is one and the gradient vanishes. Chapter 2 returns to this point, because the
assumption of unit-variance dimensions is exactly what fails as training proceeds.

There are $H$ heads rather than one because a position usually needs to gather several unrelated
things at once, and one softmax can only concentrate on one set of positions. Splitting the width
$D$ into $H$ pieces of width $d = D/H$ buys $H$ independent attention patterns at no extra
parameter cost, since the projections are $D \times D$ either way. The head outputs, each of width
$d$, are concatenated into one vector of width $D$ and mixed by $W_O$. Four matrices of
$D \times D$ make up the whole sublayer.

In an implementation the per-head sum is written as three matrix products over the whole sequence,

$$
\operatorname{Attn}_h(X) = \operatorname{softmax}\!\left(\frac{Q_h K_h^{\top}}{\sqrt{d}} + M\right) V_h ,
$$

where $Q_h$, $K_h$ and $V_h$ stack the per-position vectors as rows and $M$ is zero where $j \le i$
and $-\infty$ elsewhere. The $S \times S$ score matrix in the middle is the term that makes training
cost grow with the square of the sequence length.

### The feed-forward sublayer

GPT-2 uses an MLP with one hidden layer of width $4D$. The ratio 4 is a convention taken
from the original transformer rather than a derived quantity, and later models use other
ratios.

$$
\operatorname{MLP}(x) = W_2\, \operatorname{GELU}(W_1 x), \qquad W_1 \in \mathbb{R}^{4D \times D}.
$$

$W_1$ lifts the token from $D$ to $4D$, the activation applies a nonlinearity to each of the $4D$
hidden units, and $W_2 \in \mathbb{R}^{D \times 4D}$ projects the result back to width $D$. The
activation is the Gaussian error linear unit of Hendrycks and Gimpel 2016, "Gaussian Error Linear
Units (GELUs)", which is

$$
\operatorname{GELU}(z) = z\,\Phi(z),
$$

with $\Phi$ the standard normal cumulative distribution function. It behaves like the identity for
large positive $z$ and decays smoothly to zero for negative $z$, and unlike a rectified linear unit
it has a nonzero gradient everywhere.

This sublayer acts on one token at a time and never compares two positions. Attention decides what
to read, and the feed-forward layer decides what to do with it. Chapter 4 changes the activation
and the shape of the MLP, and chapter 6 replaces the single MLP with many copies of it.

## The residual stream

The two sublayers add into a running sum, so the state at any depth is the embedding plus everything
written into it so far. Elhage et al. 2021, "A Mathematical Framework for Transformer Circuits",
describe this sum as a shared medium that every layer reads from and writes to. On the reference
model that medium is 8192 numbers wide, its width is the same at every depth, and it is written into
160 times, twice in each of 80 blocks. The logit lens of
nostalgebraist 2020, "interpreting GPT: the logit lens", decodes an intermediate state with the
unembedding $E$ and shows the prediction being refined from layer to layer. This picture of the
residual as a shared medium of fixed width is used in chapter 11.

One consequence matters for the chapters in between. Because every sublayer reads the whole
residual and writes the whole residual, a change that alters what one sublayer computes does not
alter the interface between sublayers. That is why the changes in this book compose. A model can
take the normalisation of chapter 2, the positions of chapter 3, the feed-forward layer of chapter 4
and the attention of chapter 7 independently of each other, and chapter 15 does exactly that.

## Output

A final LayerNorm, then the transposed embedding matrix as the output projection, then
cross-entropy on the next token:

$$
z = E\, \operatorname{LN}(x_L), \qquad
\mathcal{L} = -\sum_t \log \operatorname{softmax}(z_t)_{\text{id}_{t+1}}.
$$

The vector $z$ has one entry per vocabulary item, called a logit, and the softmax turns the logits
into a distribution over the next token. The loss is the negative log probability the model assigned
to the token that actually followed. Using the same matrix $E$ for the input embedding and the
output projection is weight tying, from Press and Wolf 2017, "Using the Output Embedding to Improve
Language Models". It saves $V \cdot D$ parameters and forces a token's input vector and the
direction that predicts it to coincide.

The loss is summed over positions, and the causal mask means every position of a training sequence
supplies one prediction, so a batch of $B$ sequences of length $S$ yields $BS$ predictions from one
forward pass. Generation reverses the arrangement. The model emits one token, that token is fed back
as the next input, and the pass is repeated. Chapter 5 is about what that reversal costs.

## What it costs

Three quantities are used in the later chapters. All three are worked out here on the reference
model of the preface, with $L = 80$ blocks, $D = 8192$, $H = 64$ heads of width $d = 128$,
$V = 32000$ and two bytes per element.

### Parameters per block

Per token and per layer, the parameter count is dominated by the four attention matrices
at $4D^2$ and the two MLP matrices at $8D^2$:

$$
4D^2 = 4 \cdot 8192^2 = 268\,\text{M}, \qquad 8D^2 = 537\,\text{M}, \qquad 12D^2 = 805\,\text{M}.
$$

The two norms contribute $4D = 32768$ parameters per block, one part in twenty thousand of the
block, and are left out of every count in this book. Over 80 blocks the stack holds
$80 \cdot 805\,\text{M} = 64.4$ billion parameters, the embedding adds
$V \cdot D = 32000 \cdot 8192 = 262$ million, and the total is about 65 billion. LLaMA-2 70B, from
Touvron et al. 2023, "Llama 2: Open Foundation and Fine-Tuned Chat Models", reaches 70 billion at
the same depth and width with a wider feed-forward layer, worked out in chapter 4. At two bytes per
parameter its weights occupy 140 GB, the figure the cache is compared against in chapter 5.

Two thirds of a block, 537 M of 805 M, is the feed-forward layer, and the ratio is fixed by the
constants 8 and 4 in $8D^2$ and $4D^2$ rather than by the width. That ratio is the reason chapter 6
changes the feed-forward layer rather than attention when it separates parameter count from compute.

### Arithmetic per token

A matrix of $m$ rows and $n$ columns applied to a vector costs $mn$ multiply-adds, counted as $2mn$
floating-point operations. Every weight is used exactly once per token, so
compute per token is about twice the parameter count in
FLOPs, which is about 130 GFLOPs on the reference model. The backward pass costs about twice the
forward pass, one product for the gradient with respect to the input and one for the gradient with
respect to the weights, which gives the usual estimate of six FLOPs per parameter per token of
training. At the twenty tokens per parameter of Hoffmann et al. 2022, "Training Compute-Optimal
Large Language Models", training the reference model on 1.3 trillion tokens costs about
$6 \cdot 65 \times 10^9 \cdot 1.3 \times 10^{12} \approx 5 \times 10^{23}$ FLOPs.

The attention scores and the weighted sum are not accounted for by any parameter, because they
involve the sequence rather than the weights, and they grow with sequence length:

$$
\text{FLOPs}_{\text{attn scores}} \propto S \cdot H d = S D \quad \text{per token}.
$$

For one head, comparing the query against $S$ keys of width $d$ is $Sd$ multiply-adds and the
weighted sum over $S$ values of width $d$ is another $Sd$. Over $H$ heads that is $2SHd = 2SD$
multiply-adds, or $4SD$ FLOPs per token per block. At the training length $S = 4096$ this is
$4 \cdot 4096 \cdot 8192 = 134$ MFLOPs per block and 10.7 GFLOPs over the 80 blocks, about eight
percent of the 130 GFLOPs from the weights. At the long-context target $S = 131072$ it is 344
GFLOPs, more than twice the weights, so at that length the sequence term dominates the model term.
The term is linear in $S$ for one token and therefore quadratic in $S$ over a whole sequence, which
is why chapters 8 and 10 reduce the number of keys a query reads.

### Cache bytes per token

At inference every generated token must attend to all previous keys and values, so they are kept in a
cache. Its size per token is

$$
\text{KV bytes per token} = 2 \cdot L \cdot H \cdot d \cdot \text{bytes per element},
$$

where the factor 2 counts one key and one value. In bf16 on the reference model this is
$2 \cdot 80 \cdot 64 \cdot 128 \cdot 2 = 2.6$ MB for every token held in context, so a sequence at
the training length carries 10.7 GB of cache and one at the long-context target carries 344 GB
against 140 GB of weights.
For long contexts this cache, not the weights,
becomes the limit on batch size and therefore on throughput. Chapters 5, 7, 8, 9 and 12 each
reduce a different term in this formula.

## What the later chapters change

Each of the next eleven chapters edits one factor of one of the three quantities above.

| chapter | change | factor it moves |
| --- | --- | --- |
| 2 | LayerNorm to RMSNorm, pre-norm placement | none of the three, one reduction per norm |
| 3 | learned table to RoPE | removes $P$ and the bound $S_{\max}$ |
| 4 | GELU MLP to SwiGLU | the shape of the $8D^2$ term |
| 5 | multi-head to grouped-query attention | $H$ in the cache formula |
| 6 | dense feed-forward to routed experts | separates its parameters from its FLOPs |
| 7 | per-head keys and values to one latent | $H d$ in the cache formula |
| 8 | full attention to a sliding window | the positions each layer keeps and reads |
| 9 | compression and cross-layer sharing | $L$, and the tokens per stored entry |
| 10 | sparse selection | the bytes read per step rather than stored |
| 11 | one residual stream to several | the shared medium of the residual section above |
| 12 | bf16 to FP8 and FP4 | bytes per element |

Chapters 13 and 14 add components outside the block, and chapter 15 assembles the whole of
DeepSeek-V4.1 from these pieces.

The first change is the smallest. Normalisation moves no term in any of the three formulas and adds
parameters that round to nothing against 65 billion, and every model after GPT-2 still changed it.
Chapter 2 sets out what the GPT-2 form does, what of it is unnecessary, and where in the block the
norm belongs.
