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

Several fixes preceded the one that won. Vaswani et al. 2017, "Attention Is All You Need", used a
fixed sinusoidal table rather than a learned one, which at least defines a value at every position,
but the vectors are still added to the residual and the dot product between two of them is not a
clean function of the offset. Shaw et al. 2018, "Self-Attention with Relative Position
Representations", added a learned vector per offset inside the attention score, which is genuinely
relative but needs a table of offsets and extra work per score. Raffel et al. 2020, "Exploring the
Limits of Transfer Learning with a Unified Text-to-Text Transformer", reduced that to a learned
scalar bias per bucket of offsets, and Press et al. 2021, "Train Short, Test Long: Attention with
Linear Biases Enables Input Length Extrapolation", reduced it further to a fixed linear penalty in
the offset, which extrapolates but forces every head to prefer nearby positions.

The common shape of these fixes is that position is a term added to the score. Rotary position
embedding takes the other route. Position becomes a transformation of the query and key, and the
relative dependence falls out of the geometry rather than being tabulated.

## Rotary position embedding

Start with two dimensions. A vector in a plane can be rotated by an angle without changing its
length, and the dot product of two rotated vectors depends only on the difference of the two angles,
because rotating both by the same amount changes nothing. That single fact is the whole idea. If the
angle is made proportional to position, then the dot product of a query at position $i$ with a key
at position $j$ depends on $i - j$ and not on $i$ or $j$ separately.

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
GPT-NeoX and LLaMA used $\theta = 10^4$. Later long-context models use $10^5$ to $10^6$ or higher.

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
queries and keys together, about 50 thousand multiply-adds against the block's 805 million. The
cosines and sines depend only on the position and the frequency, so they are computed once per
position and reused by every head and every layer. In owlet1 the table is built by
`DeepseekV41RotaryEmbedding` and applied by `apply_rotary_pos_emb`, both in
`src/mzoo/archs/owlet1/norm_rope.py`.

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

The reason absorption fails for the rotated part is worth stating, because it is why every low-rank
attention design in this book carries a small rotated appendix. If the key of head $h$ is
$W_{K,h} c$ for a latent $c$, then the score $q^{\top} W_{K,h} c$ can be computed by first folding
$W_{K,h}$ into the query, once per token, and then taking a dot product against the stored latent.
Insert a rotation and the score becomes $q^{\top} R(j-i) W_{K,h} c$. The rotation sits between the
two matrices and depends on the pair of positions, so the product cannot be precomputed. Keeping
$d_r$ small keeps the part that must be stored and rotated per token small.

Rotated and unrotated widths in recent models:

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

The three ways of doing the rescaling differ in what they preserve. Position interpolation, from
Chen et al. 2023, "Extending Context Window of Large Language Models via Positional Interpolation",
divides every position by the extension factor, so a context of $4S$ is mapped onto the angles the
model already knows. Every angle stays in range, at the cost of compressing the fast pairs, which
are the ones that told adjacent tokens apart. NTK-aware scaling raises the base $\theta$ instead. A
larger base leaves $\omega_0 = 1$ untouched and stretches the slow end, so local resolution is kept
and only the long-range pairs are altered. YaRN, from Peng et al. 2023, "YaRN: Efficient Context
Window Extension of Large Language Models", makes the choice per pair. Pairs whose period is shorter
than the training length have been seen through complete turns and are left alone, pairs whose period
exceeds the extended length are interpolated, and a ramp covers the pairs in between. It also scales
the attention logits by a constant to compensate for the change in average score magnitude at the
longer length.

On the reference model the split is easy to locate. At $\theta = 10^4$ a pair has a period shorter
than the 4096 training positions when $2\pi / \omega_m < 4096$, which holds for roughly the fastest
two thirds of the 64 pairs. Those are left as they are. The remaining slow pairs are the ones that
have only ever seen a fraction of a turn, and they are the ones any extension scheme has to treat.

## Two bases in one model

V4.1 rotates two kinds of vector: ordinary per-token keys, and compressed latents that each stand for
$m$ tokens as described in chapter 9. Latents sit $m$ positions apart, so they use a separate, larger
base of $1.6 \times 10^5$ against $10^4$. Their slow pairs then still resolve the whole context.

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

Position is now settled. It costs no parameters, it lives inside the attention dot product rather
than in the residual stream, and its only free choices are how many dimensions to rotate and at what
base. Chapter 4 turns to the last of the small changes, the feed-forward layer, which holds two
thirds of the parameters of the block and is the one place where a change to the block moves the
parameter count by hundreds of millions.
