# 13. Engram memory

Attention is a computed lookup. To recall that a phrase was seen, the model has to attend back to
it. A literal table suits some knowledge better. Which token tends to follow a given short sequence
of tokens is a fact about the training distribution. It can be stored as a lookup keyed on the
sequence itself, at no attention cost.

## Computed recall against stored recall

The distinction is worth drawing carefully, because the whole component follows from it. A
transformer has two ways of knowing that a short sequence of tokens is usually followed by a
particular token. It can attend to an earlier occurrence of the same sequence in the current context
and copy what came next, which is a computation over the context and works for sequences the model
has never seen before. Or the fact can be baked into its weights during training, in which case it is
available without any context at all but has to be reconstructed by the arithmetic of a
feed-forward layer.

Neither is a table. Statistical language models before the neural period were tables. Systems built
on the estimators of Kneser and Ney 1995, "Improved backing-off for M-gram language modeling", stored
counts for every $n$-gram they had seen and read them by direct indexing, and they were good at
exactly the thing a transformer spends parameters and arithmetic on. A table has two properties that
neither mechanism above has. Reading it costs the same whatever the context length, and its capacity
is set by the number of rows rather than by the width of a layer.

The capacity comparison is stark on the reference model. The feed-forward layers of chapter 1 have
$4D = 32768$ hidden units per block, so all 80 blocks together hold about 2.6 million of them, and
those units also have to implement everything else the feed-forward layers do. A hash table with 384
million rows has more than a hundred times as many addressable slots and does nothing else. Explicit
memories of this kind have been attached to language models before, by Lample et al. 2019, "Large
Memory Layers with Product Keys", which inserts a large learned key-value store in place of a
feed-forward layer, by Khandelwal et al. 2020, "Generalization through Memorization: Nearest
Neighbor Language Models", which interpolates the model's distribution with one read from a datastore
of training contexts, and by Borgeaud et al. 2022, "Improving Language Models by Retrieving from
Trillions of Tokens", which retrieves whole passages. What distinguishes the engram layer from all of
them is that its key is not learned and not searched. It is computed from the token identifiers by
hashing, so a read is an array index.

## Hashed n-gram lookup

At each position, take the last $n$ tokens, hash them, and look up a row in a large embedding table.
Several independent hash functions per $n$-gram order give several rows:

$$
h_{m,j} = \big(\operatorname{hash}(t_{s-m+1}, \dots, t_s) \bmod p_{m,j}\big) + \text{offset}_{m,j},
\qquad m \in \{2, 3, 4\},\ j = 1, \dots, 8 .
$$

Each $(m, j)$ pair has a disjoint range of the table, sized by a distinct prime $p_{m,j}$. The ranges
tile the table, and different hash heads collide differently. Tokens are first mapped to a smaller
vocabulary in which case and accent variants coincide, so the lookup is insensitive to surface form.

Three orders and eight hash functions give 24 rows per position. The construction is the hashing trick
of Weinberger et al. 2009, "Feature Hashing for Large Scale Multitask Learning", applied to $n$-grams.
Its logic is worth following one step at a time. A table with a row for every possible 4-gram is
impossible, since the count is the vocabulary size raised to the fourth power. Hashing into a table of
fixed size makes the memory bounded, at the cost that unrelated $n$-grams land on the same row and
their gradients are added together. A single hash would therefore corrupt a row for both of its
occupants with no way to tell them apart. Eight independent hashes per order fix this, because two
$n$-grams that collide under one hash almost certainly do not collide under the other seven, so the
24-row set fetched for one $n$-gram is unique to it even though individual rows are not. The
projection that consumes the 24 rows can then treat the collided components as noise and the
consistent components as signal. Distinct primes for the moduli keep the ranges disjoint by
construction and make the collision patterns of the different hash heads independent of each other.

Reducing the vocabulary before hashing is a separate decision with the same purpose. Case variants,
accented forms and leading-space variants of a word are different tokens but usually predict the same
continuation, so folding them together before the hash means one row serves all of them and receives
the counts of all of them.

## Reading the table into the residual

The fetched rows are concatenated and projected to one key per residual stream and one value. The
value is added to each stream through a gate that measures agreement between the stream and its key:

$$
a_c = \frac{\langle x_c \odot \omega_c,\ k_c\rangle}{\sqrt{D}\ \operatorname{rms}(x_c)\ \operatorname{rms}(k_c)}, \qquad
x_c \leftarrow x_c + \sigma\!\big(\operatorname{sign}(a_c)\sqrt{|a_c|}\big)\, v .
$$

The value enters a stream only where the key matches the content the stream already holds.

The numerator is an inner product between the stream, reweighted per dimension by a learned vector
$\omega_c$, and the key that the table produced for that stream. Dividing by $\sqrt{D}$ and by the two
root-mean-square norms makes $a_c$ a scale-free measure of agreement rather than a quantity that grows
with the size of either vector, in the same way and for the same reason as the $1/\sqrt{d}$ of the
attention score in chapter 1. The sign-preserving square root expands the region near zero, where
agreements of this kind concentrate, so the gate is sensitive there. The sigmoid is monotone, so a
stream whose content agrees with the key receives more of the value and one that disagrees receives
less.

This is the one component in the book that is aware of the multiple residual streams of chapter 11.
The table produces a separate key for each of the four streams and one shared value, so the same
retrieved fact can be admitted into one stream and held out of another.

## Cost

A lookup is $O(1)$ per position regardless of context length. The table need not be on the
accelerator, because it is read once per token per engram layer. The released V4.1-Flash has two
engram layers with about 384 million rows each, stored in FP8. In owlet1 the module exists but is
disabled by `engram_layer_ids=[]`. See `src/mzoo/archs/owlet1/engram.md`.

The traffic is what makes the placement possible. Twenty-four rows are fetched per token per engram
layer, so a decode step touches 48 rows in total, and the address of each is known as soon as the
token identifiers are, which is before the forward pass begins. A table that is read this sparsely
and this predictably can be held in host memory and prefetched, so its size is limited by what the
machine has rather than by what fits beside the weights on the accelerator. Against that, the rows
are trained parameters and take part in the backward pass, and the count of them is large enough that
the memory holding them is a real cost of the design rather than a rounding error.

## Run-time and stored n-gram statistics

Induction heads implement in-context copying. Olsson et al. 2022, "In-context Learning and Induction
Heads", describe the pattern: given that the current token was followed by some token
earlier in the context, the head attends to that earlier occurrence and predicts what followed it.
This is an $n$-gram statistic computed at run time by attention. Static $n$-gram statistics learned
from the training corpus are instead stored in MLP weights, which chapter 4 discussed, following
Geva et al. 2021, "Transformer Feed-Forward Layers Are Key-Value Memories". Engram is designed to
move the static part into an explicit hash table with far more rows than an MLP has neurons, leaving
attention and the MLPs to do the parts that need computation.

The engram layer is the last component that sits inside the block. The next chapter covers three
parts of V4.1 that change the training objective or the shape of the network as a whole.
