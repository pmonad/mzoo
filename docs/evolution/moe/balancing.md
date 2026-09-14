---
title: "Load balancing"
order: 3
---

## Load balancing

A router that is trained only by the language-modelling loss does not use its experts evenly. It
concentrates on a few of them and leaves the rest untouched, and it does so from the first steps of
training. The reason is in the gradient rule of [Routing](routing.md), and the effect defeats the
purpose of the layer, because parameters that are never selected are parameters that were bought and
not used. Every mixture-of-experts model therefore carries a mechanism whose only job is to keep the
load spread out.

### Why routers collapse

Routing is self-reinforcing. An expert that is chosen slightly more often trains slightly more,
becomes slightly better, and is chosen more still. Without a correction, a router sends most tokens
to a few experts and the rest stay idle. This wastes parameters. Experts are also spread across
devices, so it makes some devices wait for others.

The loop runs as follows. At initialisation the rows of $W_g$ are random, so for any token some
expert has the highest logit by chance alone. That expert is selected, runs, and receives a gradient,
so its matrices move toward being useful for the tokens it saw. Every other expert receives nothing,
because a token outside its selection produces no gradient in its parameters. At the next step the
selected expert is genuinely better than the untouched ones, so it earns a larger weight $w_i$, which
raises its logit, which widens its lead. There is no term anywhere in the loss that pushes in the
other direction. An unselected expert has no path by which its quality could improve and no path by
which its logit could rise, so once it is behind it stays behind.

The endpoint is a layer with $K$ or so effective experts and the rest frozen at initialisation. On
the reference layer, if 16 of the 256 routed experts carry all the load, the block's 12.9 billion
feed-forward parameters deliver the capacity of about 850 million, and the design has bought a
24-fold parameter increase and kept a factor of 1.6.

### What imbalance costs at run time

The quality loss above is only half of it. Under the expert parallelism of the chapter introduction
the experts of a layer sit on different devices, and each device runs its own group of tokens. The
layer is not finished until the last device finishes, so the time of the step is set by the largest
group, not the average one. A layer in which one device receives four times its share takes four
times as long as a balanced one, while the other devices wait. Balance is therefore a throughput
property with a direct multiplier on training cost, independent of whether the unused experts would
have helped the loss.

Because the groups have to be allocated somewhere, early implementations bounded them. GShard and
Switch give each expert a fixed capacity, computed as the average group size times a capacity factor
slightly above one. Tokens beyond that bound are dropped, meaning the expert is skipped and the token
passes through the layer on its residual connection alone. A capacity factor of 1.25 wastes a quarter
of the expert compute when load is uniform and still drops tokens when it is not, so the factor
trades wasted arithmetic against lost tokens and neither end of the trade is good. Gale et al. 2023,
"MegaBlocks: Efficient Sparse Training with Mixture-of-Experts", removed the choice by expressing the
whole layer as one block-sparse matmul with variable-size blocks, so groups of any size are handled
and no token is dropped. Dropless routing of this kind is what current implementations use, which
means imbalance now shows up purely as time rather than as silently skipped tokens.

### The auxiliary loss

The original fix, from Shazeer 2017 and Switch, adds a loss that is minimised when load is uniform.
Let $f_i$ be the fraction of tokens routed to expert $i$ and $P_i$ the mean router probability for
it.

$$
\mathcal{L}_{\text{bal}} = \alpha\, E \sum_{i=1}^{E} f_i\, P_i .
$$

$f_i$ is not differentiable but $P_i$ is, so the gradient lowers the probability of overloaded
experts. The coefficient $\alpha$ must be tuned. Too small and the router collapses. Too large and
the balancing term outweighs the language-modelling loss and hurts quality.

The product form is what makes it work. Both vectors sum to one over the experts, so the sum equals
$1/E$ when the load is uniform and the factor $E$ makes the whole term equal to 1 at that point for
any expert count. The two vectors rise together, so concentrating the load raises the sum above its
uniform value. Multiplying the hard count by the soft probability gives each expert a gradient
proportional to how overloaded it is: $\partial \mathcal{L}_{\text{bal}} / \partial P_i = \alpha E f_i$,
so the expert that took the most tokens has its probability pushed down the hardest, and an expert
with no tokens is not pushed at all.

Two weaknesses remain. The first is the tuning, which is a real cost because the right $\alpha$
depends on the expert count, the batch size and the stage of training. The second is subtler. The
gradient of this term is a gradient on the router that is not a gradient on the loss the model is
being trained for, so at every step the router is pulled away from the assignment it considers best.
The balance is bought with a distortion of the model's own objective, and the size of the distortion
is exactly $\alpha$.

The fractions $f_i$ also have to be counted over some set of tokens, and the choice matters. Counted
over one micro-batch on one device, the target is noisy and the term penalises a router for
specialising on a batch that happens to be homogeneous, which is common when a batch is one long
document. Qwen3-235B-A22B computes the term over the global batch instead, so a router may be
unbalanced on any one device as long as the load evens out across the step.

### Auxiliary-loss-free balancing

DeepSeek-V3 removed the loss term. Each expert instead carries a bias $b_i$ that is added to its
score during selection only:

$$
\mathcal{I}(x) = \operatorname{top}K_i\big(s_i + b_i\big), \qquad
w_i = \frac{s_i}{\sum_{j \in \mathcal{I}} s_j} \quad \text{(no } b_i \text{)} .
$$

After each step, $b_i$ is decreased by a fixed amount $u$ if expert $i$ received more than its share
of tokens and increased by $u$ otherwise. No gradient flows through $b_i$. The bias enters only the
argmax and not the weights, so the output of the layer is exactly what the unbiased scores give.
Only the set of consulted experts changes. V3 and V4.1 keep a very small sequence-level balance
loss alongside this as a safeguard, with weight $10^{-4}$.

Written as an update, with $n_i$ the number of tokens that chose expert $i$ during the step and $N$
the number of tokens,

$$
b_i \leftarrow b_i + u \cdot \operatorname{sign}(\bar n - n_i), \qquad \bar n = \frac{N K}{E} .
$$

This is a controller and not a learning rule. It observes an error, the difference between an
expert's load and its target, and moves the bias one fixed step against it. An expert that is
starved drifts upward until it crosses into some token's top-$K$, at which point it starts receiving
tokens and gradient and can improve. An expert that is overloaded drifts down until it loses its
marginal tokens to its neighbours. The step size $u$ sets how fast the loop reacts, and it trades
tracking speed against oscillation around the target in the usual way for a fixed-step controller.

The reason this is better than the auxiliary loss is that it never touches the objective. The
gradient the model sees is the gradient of the language-modelling loss and nothing else. The bias
changes which experts are consulted, and given that set the layer computes exactly what the unbiased
scores prescribe, so there is no scaling distortion in the output and no term competing with the
loss. Wang et al. 2024, "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts",
introduced the method and DeepSeek-V3 adopted it.

The remaining cost is that balance is now enforced only in the aggregate over a step. A single
sequence can still send most of its tokens to a few experts, which is fine for device utilisation
during training and less fine for inference on one request, and this is what the very small
sequence-level term is there to limit. The method also depends on the load statistics being visible
where the update is applied, so the counts must be reduced across devices before the bias moves.

### The gap in owlet1

In owlet1 the bias exists as `gate.bias`, but the update rule is not implemented and there is no
balance loss, so the smoke config runs unbalanced. See the open issues in
`src/mzoo/archs/owlet1/README.md`.

The state is worth being precise about, because the failure is silent. `gate.bias` is
zero-initialised, kept in fp32, and fed only into the non-differentiable top-$K$, so it receives no
gradient and, with no update rule, never leaves zero. Selection reduces to a plain top-$K$ on the
scores. No auxiliary loss is computed either, and because the model returns no `aux_loss` field the
trainer's expert-load logging never fires, so the imbalance is not only uncorrected but unmeasured.
`src/mzoo/archs/owlet1/moe.md` documents this under its notes. At the smoke scale of 8 experts and
top-2 the consequence is small, and at any real scale it would be the first thing to fix.

### What the 2026 models do

DeepSeek-V3 uses bias-based, auxiliary-loss-free balancing with a small sequence-level safeguard.
GLM-4.5 follows DeepSeek-V3, with sigmoid gating and loss-free balancing. Kimi K2 uses the
DeepSeek-V3 layer design unchanged and therefore the same method. Qwen3-235B-A22B is the exception
among the large open-weight families, keeping a softmax router with a load-balancing loss computed
over the global batch. Bias-based balancing is the widespread choice, and the auxiliary loss survives
mainly in a reduced form as a safeguard.

That completes the mixture-of-experts layer. Parameters and compute per token have been separated by
a factor of $E/K$, and the feed-forward side of the block is no longer what limits the size of a
model. The attention side is, and chapter 7 returns to the cache formula of chapter 5 with the factor
that grouped-query attention left untouched.
