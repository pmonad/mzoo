# 3. Positions

Attention as chapter 1 defined it is a sum over positions with no reference to the order of those
positions. Permute the keys and values of a prefix and every output is unchanged. Order has to be
supplied from outside the sum, and GPT-2 supplies it by adding a learned position vector to the
embedding before the first block. Chapter 2 changed how the block normalises without touching that
arrangement. This chapter replaces it.

The replacement is worth the space because it is the one change of the four small ones that alters
what the model can do rather than what it costs. A learned table caps the context at its own length.
Rotary position embedding removes the cap, adds no parameters, and gives attention a dependence on
the offset between two positions rather than on their absolute indices, which is what makes the
long-context designs of chapters 8 to 10 possible at all.

### Roadmap

The chapter moves from the failure of the old scheme to the details that later chapters depend on:

1. **Why a table of positions fails** — a learned table is absolute and finite, and attention needs
   relative offsets over unbounded length, so the parameterisation fights the task.
2. **Rotary position embedding** — replacing the added table with a rotation of queries and keys makes
   the score depend on the offset alone, at zero parameters, which is the design the rest of the book
   uses.
3. **Partial rotation** — rotating only a slice of the head dimensions, because the rotated part
   cannot be compressed or absorbed and chapter 7's low-rank attention depends on keeping it small.
4. **Extending the context** — rescaling frequencies so a model trained short runs long, which is what
   chapters 8 to 10 build on.
5. **Two bases in one model** — compressed latent streams need a larger rotation base than per-token
   keys, the arrangement chapter 9 uses.
6. **Position in recent models** — a survey of how four model families configure rotation width and
   base, showing which choices have settled.

## Why a table of positions fails

A learned position table $P$ has two defects. It caps the context at the table length. It also
encodes absolute position, while the quantity relevant to attention is the offset $i - j$ between a
query and a key. A model trained with absolute positions must learn every offset separately at
every position.

The second defect is the one that limits quality. Attention wants to express relations of the form
"the token three back" or "the nearest preceding open bracket", and these are statements about
$i - j$. With an absolute table the model can only reach them indirectly, by learning a query at
position $i$ that matches a key at position $i - 3$, and then learning the same relation again at
position $i + 1$, and again at every position where it applies. Nothing in the parameterisation ties
those cases together. Rows of $P$ that are never reached during training, because no training
sequence was long enough, are never learned at all, so the model does not merely lack an
extrapolation rule beyond $S_{\max}$. It has untrained parameters there.

### The fixes that preceded the one that won

Several fixes preceded the one that won. They are worth a closer look, because each one moves
position one step further from the residual stream and closer to the score, and the winner is the
step that finishes the move.

**Sinusoidal table.** Vaswani et al. 2017, "Attention Is All You Need", replaced the learned table
with a fixed one. Position $p$ and dimension pair $2m, 2m+1$ carry

$$
\text{PE}(p, 2m) = \sin(p\,\omega_m), \qquad \text{PE}(p, 2m+1) = \cos(p\,\omega_m), \qquad
\omega_m = 10000^{-2m/d},
$$

so a value is defined at every position and the table can no longer be exhausted. But the vector is
still added to the embedding before the blocks. The attention score between two tokens then expands
to four terms,

$$
(x_i + p_i)^{\top} (x_j + p_j) = \underbrace{x_i^{\top} x_j}_{\text{content}}
+ \underbrace{x_i^{\top} p_j + p_i^{\top} x_j}_{\text{mixed}}
+ \underbrace{p_i^{\top} p_j}_{\text{position}},
$$

and only the last term is a function of the offset. The mixed terms entangle content with absolute
position, so the score is not cleanly relative, and the position information has to survive every
residual addition and every normalisation between the embedding and the dot product.

**Learned relative vectors.** Shaw et al. 2018, "Self-Attention with Relative Position
Representations", moved position out of the residual and into the score. The logit becomes

$$
e_{ij} = x_i^{\top} W^Q {}^{\top} \big( x_j W^K + r^K_{i-j} \big) + r^Q_{i-j},
$$

where $r^K_{i-j}$ and $r^Q_{i-j}$ are learned vectors indexed by the offset. This is genuinely
relative: the same offset contributes the same term at every position. The cost is a table of
offsets again, now two vectors per offset per layer, usually clipped to a window of reachable
offsets, and extra work per score to fetch and combine them. The extrapolation defect returns at the
edge of the window.

**Bucketed scalar bias.** Raffel et al. 2020, "Exploring the Limits of Transfer Learning with a
Unified Text-to-Text Transformer", reduced the two learned vectors to a single learned scalar per
bucket of offsets, shared across all positions, layers and heads:

$$
e_{ij} = q_i^{\top} k_j + b\big(\beta(i - j)\big),
$$

where $\beta$ maps an offset to one of a small number of logarithmically spaced buckets. This is
cheap and relative, but it is still a table, now of scalars. Offsets beyond the last bucket share
one value, and a bucket wider than one token cannot tell adjacent offsets apart.

**Fixed linear penalty.** Press et al. 2021, "Train Short, Test Long: Attention with Linear Biases
Enables Input Length Extrapolation", removed learning altogether and gave each head a fixed slope
$m_h$:

$$
e_{ij} = q_i^{\top} k_j - m_h \,(i - j), \qquad i \ge j .
$$

Nothing is learned, so nothing fails to extrapolate, and models trained short do run long. But the
penalty is monotone in the offset, so every head is forced to prefer nearby positions, and the
slopes must be chosen by hand. Position has become a bias on the score that the content cannot fully
override.

The common shape of these fixes is that position is a term added to the score. Rotary position
embedding takes the other route. Position becomes a transformation of the query and key, and the
relative dependence falls out of the geometry rather than being tabulated.

## Rotary position embedding

### The core idea in two dimensions

Start with two dimensions. A vector in a plane can be rotated by an angle without changing its
length, and the dot product of two rotated vectors depends only on the difference of the two angles,
because rotating both by the same amount changes nothing. That single fact is the whole idea. If the
angle is made proportional to position, then the dot product of a query at position $i$ with a key
at position $j$ depends on $i - j$ and not on $i$ or $j$ separately.

### The rotation

RoPE removes the table and instead rotates queries and keys by an angle proportional to their
position. Dimensions are taken in pairs, and pair $m$ has its own frequency $\omega_m$:

$$
\begin{pmatrix} y_{2m} \\ y_{2m+1} \end{pmatrix} =
\begin{pmatrix} \cos p\,\omega_m & -\sin p\,\omega_m \\ \sin p\,\omega_m & \cos p\,\omega_m \end{pmatrix}
\begin{pmatrix} x_{2m} \\ x_{2m+1} \end{pmatrix},
\qquad
\omega_m = \theta^{-2m/d} .
$$

A head of width $d$ has $d/2$ such pairs, each rotated by its own angle $p\,\omega_m$, so the
transformation of the whole head is block diagonal with $d/2$ blocks of size two. The form is from
Su et al. 2021, "RoFormer: Enhanced Transformer with Rotary Position Embedding". It is applied to
$q$ and $k$ after the projections $W_Q$ and $W_K$ and before the dot product, and it is not applied
to $v$, because the value is what is read rather than what is matched.

Write the rotation at position $p$ as $R(p)$. Rotations compose by adding angles, so the attention
logit depends only on the offset:

$$
\big(R(i)\, q\big) \cdot \big(R(j)\, k\big) = q^{\top} R(j - i)\, k .
$$

Read that back in words. The score between a query at position $i$ and a key at position $j$ is the
score the same content vectors would have had at positions $0$ and $j - i$. Absolute position has
cancelled. A head that has learned to attend three tokens back has learned one relation, expressed
once in the geometry of $q$ and $k$, and that relation holds at every position in the sequence
including positions longer than any seen in training. The equation also shows why the rotation is
free of parameters. It rearranges the content the projections already produced, and it introduces no
new weights to learn.

Nothing is added to the residual. Position enters only through the dot product. Low-index pairs
rotate fast and resolve nearby offsets. High-index pairs rotate slowly and resolve distant ones. The
base $\theta$ sets the slowest frequency and therefore the largest offset the model can distinguish.
GPT-NeoX, Black et al. 2022, "GPT-NeoX-20B: An Open Source Autoregressive Language Model", and
LLaMA, Touvron et al. 2023, "LLaMA: Open and Efficient Foundation Language Models", used
$\theta = 10^4$. Later long-context models use $10^5$ to $10^6$ or higher.

The implementation never builds the $d \times d$ matrix. A two-dimensional rotation is a linear
combination of the pair with a swap, so the whole transformation is two elementwise products:

$$
y = x \odot \big[\cos p\,\omega_0, \dots, \cos p\,\omega_{d/2-1}\big]^{\oplus 2}
\;+\; \text{swap}(x) \odot \big[\sin p\,\omega_0, \dots, \sin p\,\omega_{d/2-1}\big]^{\oplus 2},
$$

where $\text{swap}(x)$ replaces each pair $(x_{2m}, x_{2m+1})$ with $(-x_{2m+1}, x_{2m})$ and
$[\,\cdot\,]^{\oplus 2}$ repeats each frequency twice, once per dimension of the pair. The cosines
and sines depend only on the position and the frequency, so they are computed once per position and
reused by every head and every layer.

### The frequency spectrum on the reference model

The reference model has heads of width $d = 128$, so a head carries 64 pairs and 64 frequencies. With
$\theta = 10^4$ the fastest pair, $m = 0$, has $\omega_0 = 1$ and completes a turn every $2\pi
\approx 6.3$ positions, which resolves immediate neighbours and tells adjacent tokens apart. The
slowest pair, $m = 63$, has $\omega_{63} = 10^{-4 \cdot 63/64} \approx 1.15 \times 10^{-4}$ and
completes a turn every 54 thousand positions. Between them the frequencies are spaced geometrically,
so every scale from a few tokens to tens of thousands has some pair that changes appreciably over
it.

Two numbers follow from this. Over a training sequence of 4096 tokens the slowest pair sweeps
$4096 \cdot 1.15 \times 10^{-4} \approx 0.47$ radians, under eight percent of a turn, so the model
has seen only a small arc of its slowest angles. And at the long-context target of 131072 tokens the
slowest pair would sweep 2.4 turns, so two offsets 54 thousand apart would produce the same rotation
in every pair. Both facts are the subject of the section on extending the context.

The arithmetic cost is negligible. Each rotated dimension needs one multiply by a cosine, one by a
sine and one add, and the reference model rotates $2 H d = 16384$ numbers per token per block for the
queries and keys together, about 50 thousand multiply-adds against the block's 805 million.

### Long-range decay

There is one further property. Averaged over random content, the magnitude of $q^{\top} R(j-i)\, k$
falls as the offset grows, because the many pairs rotate at different rates and their contributions
lose alignment. Su et al. 2021 present this decay as a feature, a mild default preference for nearby
tokens that a head can override with content. It is a tendency of the average and not a bound, which
is why RoPE models can still attend to a single distant token.

## Partial rotation

Rotating every dimension forces all of $q$ and $k$ to carry position. Later models rotate only a
subset of dimensions and leave the rest position-free:

$$
q = [\,q_{\text{nope}} \,\|\, R(p)\, q_{\text{rope}}\,], \qquad d = d_{\text{nope}} + d_r .
$$

Here $q_{\text{rope}}$ and $k_{\text{rope}}$ are the $d_r$ dimensions of the head that are rotated,
and $q_{\text{nope}}$ and $k_{\text{nope}}$ are the remaining $d_{\text{nope}} = d - d_r$ dimensions
that are not rotated. The name "nope" stands for no positional encoding. Both names come from the
DeepSeek-V2 code and paper, which introduced the scheme as "decoupled" RoPE. The symbol
$[\,a \,\|\, b\,]$ is concatenation along the dimension axis, so a vector of length $d_{\text{nope}}$
followed by one of length $d_r$ gives a vector of length $d$. That is the meaning of $\|$ throughout
the book.

With this split the attention dot product separates into two independent terms:

$$
q_i \cdot k_j = q_{\text{nope},i} \cdot k_{\text{nope},j} + q_{\text{rope},i}^{\top} R(j - i)\, k_{\text{rope},j} .
$$

The first term compares content and ignores position. The second term depends on the offset only.
The two sets of dimensions can therefore specialise, and a head that needs no position at all can
put its weight in the first term. Fully rotated attention forces every dimension to carry both
content and position.

Fewer dimensions to rotate is a small saving in compute, and in cache when the rotated part is
stored separately. The larger point is compatibility with the low-rank key and value compression of
chapter 7. The unrotated part can be compressed to a latent and expanded again, or absorbed into the
query, because no position-dependent rotation sits between the latent and the key. The rotated part
cannot be absorbed, so it is kept small and shared across heads.

> **Why the rotated part cannot be absorbed (a chapter 7 preview).** The reason absorption fails for
> the rotated part is worth stating, because it is why every low-rank attention design in this book
> carries a small rotated appendix. If the key of head $h$ is $W_{K,h} c$ for a latent $c$, then the
> score $q^{\top} W_{K,h} c$ can be computed by first folding $W_{K,h}$ into the query, once per
> token, and then taking a dot product against the stored latent. Insert a rotation and the score
> becomes $q^{\top} R(j-i) W_{K,h} c$. The rotation sits between the two matrices and depends on the
> pair of positions, so the product cannot be precomputed. Keeping $d_r$ small keeps the part that
> must be stored and rotated per token small.

Rotated and unrotated widths in recent models. Here $d$ is the width of one attention head and
$d_r$ the number of its dimensions that are rotated:

| model | attention | $d$ | $d_r$ | fraction rotated |
| --- | --- | --- | --- | --- |
| Qwen3 | GQA | 128 | 128 | 1 |
| GLM-4.5 | GQA | 128 | 64 | 1/2 |
| Qwen3-Next, softmax layers | GQA | 256 | 64 | 1/4 |
| DeepSeek-V3, Kimi K2 | MLA | 192 | 64 | 1/3 |
| DeepSeek-V4.1 | shared latent | 512 | 64 | 1/8 |
| owlet1 smoke config | shared latent | 64 | 16 | 1/4 |

The fixed value 64 in most rows is the point of the table. As heads and latents grow wider, the
rotated part does not grow with them, because 64 dimensions carry 32 frequencies and that is already
enough to separate offsets across a long context. Everything above that width is spent on content.

## Extending the context

A model trained at length $S$ has never seen angles $p\,\omega_m$ for $p > S$ in the slow pairs. To
run longer, the frequencies are rescaled so the slow pairs stay in the range seen during training,
while fast pairs are left alone. NTK-aware scaling and YaRN do this. The change affects $\omega_m$
only and needs a short fine-tune rather than retraining.

The three ways of doing the rescaling differ in what they preserve:

| scheme | what changes | what is preserved | cost |
| --- | --- | --- | --- |
| Position interpolation (PI), Chen et al. 2023 | divides every position by the extension factor $s$: $p \mapsto p/s$ | every angle stays in the range seen in training | fast pairs are compressed, losing the resolution that told adjacent tokens apart |
| NTK-aware scaling | raises the base $\theta \mapsto \theta'$ | $\omega_0 = 1$ untouched, local resolution kept | only the long-range pairs are altered; exact split of fast and slow depends on $s$ |
| YaRN, Peng et al. 2023 | per-pair choice: interpolate slow pairs, leave fast pairs, ramp between | pairs that completed full turns in training are untouched, and local resolution | needs the ramp and an attention-logit scale factor to correct average score magnitude |

Position interpolation, from Chen et al. 2023, "Extending Context Window of Large Language Models
via Positional Interpolation", divides every position by the extension factor, so a context of $4S$
is mapped onto the angles the model already knows. Every angle stays in range, at the cost of
compressing the fast pairs, which are the ones that told adjacent tokens apart. NTK-aware scaling
raises the base $\theta$ instead. A larger base leaves $\omega_0 = 1$ untouched and stretches the
slow end, so local resolution is kept and only the long-range pairs are altered. YaRN, from Peng et
al. 2023, "YaRN: Efficient Context Window Extension of Large Language Models", makes the choice per
pair. Pairs whose period is shorter than the training length have been seen through complete turns
and are left alone, pairs whose period exceeds the extended length are interpolated, and a ramp
covers the pairs in between. It also scales the attention logits by a constant to compensate for the
change in average score magnitude at the longer length.

On the reference model the split is easy to locate. At $\theta = 10^4$ a pair has a period shorter
than the 4096 training positions when $2\pi / \omega_m < 4096$, which holds for roughly the fastest
two thirds of the 64 pairs. Those are left as they are. The remaining slow pairs are the ones that
have only ever seen a fraction of a turn, and they are the ones any extension scheme has to treat.

## Two bases in one model

This section previews the arrangement chapter 9 uses. V4.1 rotates two kinds of vector: ordinary
per-token keys, and compressed latents that each stand for $m$ tokens as described in chapter 9.
Latents sit $m$ positions apart, so they use a separate, larger base of $1.6 \times 10^5$ against
$10^4$. Their slow pairs then still resolve the whole context.

The numbers make the necessity clear. The rotated part is 64 dimensions, so 32 pairs, and at
$\theta = 10^4$ the slowest of them repeats after roughly 50 thousand steps of its own index. A
latent stream at $m = 2$ over a context of 131072 tokens contains 65536 latents, more than that
period, so two latents far apart in the sequence would receive indistinguishable rotations. The
slowest period grows very nearly in proportion to the base, so raising $\theta$ by a factor of 16
raises the range of distinguishable offsets by about the same factor and puts the whole latent
stream inside one turn. The per-token keys are unaffected and keep the smaller base, because they
are consumed by the sliding-window path of chapter 8, which never looks further than its window.

## Position in recent models

Qwen3 in 2025 rotates the full head, with grouped-query attention, a RoPE base of $10^6$, and YaRN
scaling for context extension beyond the training length. Qwen3-Next in 2025 is a hybrid in which
three of every four layers are Gated DeltaNet, a linear-attention recurrence, and the fourth is
softmax attention. Only the softmax layers use RoPE, rotating one quarter of the head dimensions,
64 of 256. The linear-attention layers carry no positional encoding, and order comes from the
recurrence. Qwen3.5 in 2026 continues this design.

GLM-4.5 and GLM-4.6 in 2025 use grouped-query attention with partial rotation, rotating half of the
128 head dimensions, 64 of 128, together with QK-norm. GLM-5 in 2026 moves to a DeepSeek-style
design with multi-head latent attention, decoupled partial RoPE, and sparse attention.

Kimi K2 in 2025 and K2.5 in 2026 use the DeepSeek-V3 design, multi-head latent attention with
decoupled RoPE, 64 rotated of 192 dimensions per head. Kimi Linear in 2025 is a hybrid of three
Kimi Delta Attention layers, a linear-attention recurrence, per one multi-head latent attention
layer. The latent layers use no positional encoding, and position comes entirely from the
recurrent layers.

DeepSeek-V3 in 2024 and V3.2 in 2025 use multi-head latent attention with decoupled RoPE, 64
rotated of 192 dimensions per head, and V3.2 adds sparse attention over the same keys. DeepSeek-V4.1
in 2026 uses one shared latent per token of width 512, with 64 dimensions rotated, and two RoPE
bases, $10^4$ for per-token keys and $1.6 \times 10^5$ for compressed latents.

Llama 4 in 2025 uses global attention layers with no positional encoding and local layers with
RoPE, called iRoPE. Kazemnejad et al. 2023, "The Impact of Positional Encoding on Length
Generalization in Transformers", explains why a causal mask alone can supply order in the global
layers. Gemma 3 in 2025 uses a RoPE base of $10^4$ in local sliding-window layers and $10^6$ in
global layers.

Three patterns recur across the four families. Partial rotation is now common, at one quarter, one
half, or 64 fixed dimensions of the head. Hybrids that mix a linear-attention recurrence with a few
softmax layers put RoPE only in the softmax layers, or in none of them, letting the recurrence
carry position. The base is chosen per layer type, larger where the layer sees more distant keys.
Full-head RoPE with one base, the Qwen3 configuration, is now the exception.

## Takeaways

1. Attention itself is order-blind; position must be injected from outside the weighted sum, and
   where it is injected — into the residual stream or into the score — is the design decision.
2. A learned absolute table both caps the context and stores untrained parameters at every position
   beyond the training length. It also forces the model to relearn each relative offset at every
   absolute position, because nothing in the parameterisation ties the cases together.
3. The historical fixes (sinusoidal tables, learned relative vectors, bucketed biases, linear
   penalties) are all terms added to the score or the residual. Each improves on the last, but each
   remains a table or a fixed bias.
4. Rotary position embedding makes position a rotation of queries and keys. Because rotations
   compose by adding angles, the score $q^{\top} R(j-i)\, k$ depends on the offset alone, holds at
   every position including unseen ones, and costs zero parameters.
5. The rotation is applied after $W_Q$ and $W_K$ and before the dot product, never to $v$, and never
   added to the residual stream. It is implemented as two elementwise products against cached cosine
   and sine vectors.
6. Frequencies are geometrically spaced: fast pairs resolve adjacent tokens, slow pairs resolve
   distant ones, and the base $\theta$ sets the largest distinguishable offset. On the reference
   model the fastest pair turns every $\approx 6.3$ positions and the slowest every 54 thousand.
7. Averaged over random content, scores decay with offset as the pairs lose alignment. This is a
   tendency of the average, not a bound — heads can still attend far.
8. Partial rotation splits the head into an unrotated content part and a rotated position part. The
   score separates into a content term plus a relative term, letting heads specialise.
9. The rotated part cannot be absorbed into the query, because the rotation depends on the pair of
   positions and sits between the two matrices. That is why low-rank attention designs (chapter 7)
   keep the rotated slice small — typically a fixed 64 dimensions regardless of head width.
10. Context extension rescales frequencies rather than retraining: position interpolation compresses
    everything, NTK-aware scaling stretches only the slow end, and YaRN chooses per pair and rescales
    the logits.
11. Compressed latent streams sit $m$ positions apart and need a proportionally larger base than
    per-token keys; V4.1 uses $10^4$ for per-token keys and $1.6 \times 10^5$ for latents (chapter 9).
12. Hybrid models put RoPE only in their softmax layers, or in none, letting linear-attention
    recurrence carry order; a causal mask alone can suffice in global layers (iRoPE, NoPE).
13. Full-head RoPE with a single base, the Qwen3 configuration, is now the exception. Partial
    rotation and per-layer bases are the settled practice.

Position is now settled. It costs no parameters, it lives inside the attention dot product rather
than in the residual stream, and its only free choices are how many dimensions to rotate and at what
base. Chapter 4 turns to the last of the small changes, the feed-forward layer, which holds two
thirds of the parameters of the block and is the one place where a change to the block moves the
parameter count by hundreds of millions.
