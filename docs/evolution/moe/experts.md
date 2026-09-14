---
title: "Fine-grained and shared experts"
order: 2
---

## Fine-grained and shared experts

Routing fixed how a token picks its experts. It said nothing about how many there are, how wide each
one is, or whether some should run for every token. Those choices are separate from the router and
they are almost free to make, because the two quantities that matter for cost are already fixed by
something else. Total parameters are the number of experts times the width of one, and active
parameters are the number chosen times the width of one. Any pair of values with the same two
products costs the same. The shape of the expert set is therefore a design choice made at constant
budget, and the current answer is very different from the first one.

Mixtral-style MoE uses a few large experts: $E = 8$, $K = 2$, and each expert is a full-width FFN.
DeepSeekMoE in 2024 made two changes that every DeepSeek model since has kept.

### Fine-grained experts

Split each expert into $m$ smaller ones and pick $mK$ of them. Parameter count and compute per
token are unchanged. The number of possible expert combinations rises from $\binom{E}{K}$ to
$\binom{mE}{mK}$:

$$
E = 8,\ K = 2:\ \ \binom{8}{2} = 28 \qquad\text{against}\qquad
E = 64,\ K = 16:\ \ \binom{64}{16} \approx 4.9 \times 10^{14}.
$$

The two sides of that comparison hold the same parameters and do the same arithmetic. Splitting each
of the 8 experts into 8 pieces of one eighth the hidden width gives 64 experts, and taking 16 of them
uses the same total hidden width as taking 2 of the originals. Only the number of distinct functions
the layer can produce has changed, and it has changed by thirteen orders of magnitude.

Each token composes its FFN from many small pieces instead of choosing between a few large ones.
DeepSeek-V3 uses 256 experts of width $2048$ with $K = 8$, against a hidden size of $7168$. On the
reference model that layout gives $\binom{256}{8} \approx 4.1 \times 10^{14}$ combinations, and each
expert is a little over a quarter of the residual width, so a token assembles a hidden layer of width
$8 \cdot 2048 = 16384$ from 256 available blocks.

The reason this matters is the memory reading of chapter 4. The feed-forward layer is a store of
key-value memories, and $K$ out of $E$ is the granularity at which the model can retrieve from that
store. With $E = 8$ and $K = 2$ a token that needs two unrelated pieces of knowledge must find them in
two full-width experts, and each of those experts must therefore hold everything that any token
selecting it might need. The expert is forced to be a general-purpose FFN. With 256 narrow experts
the same token selects eight specific regions of the store and pays for eight, and the content that
sits between two topics no longer has to be replicated in the two experts that cover them.

The cost of fine-graining is on the systems side, and it grows steadily as $E$ rises. The router
matrix is $E \times D$, so it grows linearly with the expert count, though from a base so small that
it does not matter until $E$ is very large. The all-to-all dispatch of the chapter introduction sends
$K$ messages per token rather than $K/m$, so its message count grows in proportion to $mK$ even
though its total volume does not. And the matmuls get thinner. If a device holds a batch of $N$
tokens, the average number of rows given to one expert is $N K / E$. With $N = 8192$, $E = 256$ and
$K = 8$ that is 256 rows, which multiplied against an $8192 \times 2048$ matrix is still a matmul
large enough to use an accelerator well. Shrink the experts by another factor of eight and the
average group falls to 32 rows, which is not. The granularity of an expert is bounded below by the
shape at which a matmul stops being efficient, and that bound depends on the batch size a server
runs, not on the model.

### Shared experts

Some knowledge is needed by every token. If it lives in routed experts it is duplicated across them.
DeepSeekMoE sets aside $E_s$ experts that every token passes through, outside the router:

$$
\operatorname{MoE}(x) = \sum_{i \in \mathcal{I}(x)} w_i\, \operatorname{FFN}_i(x) + \sum_{s=1}^{E_s} \operatorname{FFN}_s(x).
$$

The routed experts then hold only specialised knowledge. V3 and V4.1 use one shared expert.

The duplication argument is worth making concrete. Suppose a feature is needed by most tokens,
whatever their subject. Under pure routing it must be present in enough of the 256 experts that a
token is likely to reach one, so it occupies a slice of many experts, and every copy is trained on a
fraction of the data that needs it. Moving it to one expert that every token visits stores it once
and trains it on everything. Parameters that would have gone into copies are released for content
that genuinely differs between tokens.

The shared expert also changes the character of the layer during training. The routed branch is a
discontinuous function of $x$, because a small change in the token state can swap one expert for
another and change the output by the difference between two independently trained functions. The
shared branch is continuous and is updated by every token in every batch, so the layer always has one
smooth, fully trained path from input to output. That path is present from the first step, before the
router has learned anything, which removes the cold-start problem in which a randomly initialised
router sends tokens to randomly initialised experts.

What it costs is a fixed slice of the compute budget. On the reference layer one shared expert of
width 2048 is one of the nine experts a token runs, so it takes 11 percent of the layer's arithmetic
and cannot be avoided for any token, including tokens for which its content is irrelevant. The
scaling factor $\gamma = 1.5$ from [Routing](routing.md) is the counterweight, setting the routed
branch to contribute half again as much as the shared one at equal expert magnitudes.

### Which layers are mixtures

One more choice in the layout is how many blocks use a mixture at all. The early blocks of a model
operate on token states that are still close to the embedding, and DeepSeek-V3 keeps its first three
blocks dense for that reason, switching to routed layers for the remainder. V4.1 removes the
exception and makes every backbone block a mixture, which is why the configuration in this repository
has no dense-layer schedule at all. See `src/mzoo/archs/owlet1/moe.md` for the fields that are absent.

### What experts specialise on

Routing analysis in Mixtral found that experts specialise by syntax and token position more than by
topic, with consecutive tokens often routed to the same expert. This is reported in Jiang et al.
2024, "Mixtral of Experts". Fine-grained experts and a shared expert follow from that observation.
If specialisation is at the level of token types rather than domains, many small experts compose
better than a few large ones, and features that every token needs belong in an expert that every
token visits. Dai et al. 2024, "DeepSeekMoE: Towards Ultimate Expert Specialization in
Mixture-of-Experts Language Models", gives this as the design rationale. In the terms of chapter 4, more
experts means more storage for features without more compute per token.

The finding also sets expectations for what a routed model can be expected to do. An expert is not a
domain module and cannot be extracted as one, because the tokens that reach it are selected by
surface properties as much as by content, and because the shared expert and the attention layers
carry whatever the routed branch does not. The same conclusion follows from the superposition
argument of chapter 4: a model represents more features than it has dimensions, so no unit of the
architecture, hidden dimension or whole expert, corresponds to one interpretable thing.

Granularity and sharing settle the shape of the expert set. Neither of them addresses whether the
router will use that set evenly, and by default it will not, which is the subject of
[Load balancing](balancing.md).
