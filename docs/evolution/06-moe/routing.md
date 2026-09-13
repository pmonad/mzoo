# Routing

The chapter introduction left the layer as a sum over a chosen set of experts without saying how the
set is chosen. That is the router. It is the only genuinely new component of a mixture-of-experts
layer, it holds a negligible share of the parameters, and it decides everything about how well the
layer works. This section builds it from the dense feed-forward layer, explains why its output is
hard to train, and follows the sequence of score functions that DeepSeek arrived at.

## What the router has to be

Start from the dense layer. It applies one function to the token, which is the degenerate case
$E = 1$, $K = 1$, $w_1 = 1$. To make the choice non-trivial, something must produce $K$ indices and
$K$ weights from the token state $x$ alone.

Three constraints fix the form almost completely. The choice must be made before any expert runs,
because the point of the design is to avoid running them all, so it cannot depend on expert outputs.
It must be cheap, because any cost it adds is paid on every token and would eat the saving. And it
must be trainable, because which expert suits which token is not known in advance. The cheapest
trainable function from $\mathbb{R}^D$ to $E$ numbers is a single linear layer, and that is what
every model uses.

$$
\ell = W_g\, x \in \mathbb{R}^{E}, \qquad
\mathcal{I}(x) = \operatorname{top}K\big(s(\ell)\big), \qquad
w_i = \frac{s(\ell_i)}{\sum_{j \in \mathcal{I}} s(\ell_j)} \quad (i \in \mathcal{I}).
$$

$W_g$ has one row per expert, so $\ell_i$ is the dot product of the token with expert $i$'s row.
Read together with the memory picture of chapter 4, the router row is a key for a whole expert in the
same sense that a row of $W_1$ is a key for one hidden unit. The scores $s(\ell)$ are made positive
by the function $s$, the top $K$ of them are kept, and the weights of the kept experts are
renormalised so they sum to one.

On the reference layer $W_g$ is $256 \times 8192$, which is 2.1 million parameters against 12.9
billion in the experts of the same block, less than one part in six thousand. Its arithmetic is
$2 \cdot 256 \cdot 8192 = 4.2$ MFLOPs per token against 906 MFLOPs for the eight experts. The router
is free in every accounting sense and expensive in every other.

## Why the gradient only reaches the chosen experts

The top-$K$ is a hard, non-differentiable choice. Gradient reaches the router only through the
weights $w_i$ of the chosen experts. If expert $i$ lowers the loss, $w_i$ is pushed up, which pushes
$\ell_i$ up, which makes $i$ more likely to be chosen next time.

Written out, the derivative of the layer output with respect to the weight of a chosen expert is
$\partial \operatorname{MoE}(x) / \partial w_i = \operatorname{FFN}_i(x)$, so the gradient arriving at
$w_i$ is the dot product of the incoming gradient with that expert's output. An expert whose output
points in a useful direction gets its weight raised. Everything else about the choice is a step
function of $\ell$, whose derivative is zero wherever it is defined, so it contributes nothing.

The consequence is that an expert outside $\mathcal{I}(x)$ receives no gradient at all for this token,
neither in its own matrices nor in its row of $W_g$. It is not merely disfavoured. It is invisible to
the update. A router therefore has no mechanism that pulls an unused expert back into use, which is
the whole reason [Load balancing](balancing.md) needs a separate device.

Two families of alternatives have been tried and are not what current models use. One is to make the
selection itself differentiable, by sampling the experts and estimating the gradient with a policy
method, which Shazeer et al. 2017 discussed and rejected as high variance for this setting. The other
is to change who chooses. Zhou et al. 2022, "Mixture-of-Experts with Expert Choice Routing", has each
expert select its top tokens rather than each token selecting its experts, which makes load exactly
uniform by construction. It also lets a token be picked by any number of experts including none, and
it looks at the whole batch at once, which is not available when generating one token at a time.
Current decoder models keep token choice and correct the load separately.

## Choosing the score function

The function $s$ turns logits into positive scores. Three choices have been used in sequence, and
each replaced the last for a reason that is visible in the formula.

### Softmax over all experts

The original choice for $s$ was a softmax over all $E$ logits, as in Switch, Mixtral and
DeepSeek-V2. A softmax couples the experts: raising one score lowers the others. The normalisation
over $E$ experts is also wasted when only $K$ are kept.

Both objections grow with $E$. The coupling means that the gradient which raises a good expert also
lowers every other expert for that token, including experts that were never consulted and whose
suitability was never tested, so the update carries information it does not have. The waste means
that with $E = 256$ the scores are pushed toward $1/256$ before being renormalised over the eight
survivors, and that first normalisation is then discarded. What survives it is only the ratios among
the chosen $K$, which the softmax preserved but which it also compressed, since a large logit spends
most of its effect suppressing the 248 experts that are about to be dropped.

### A sigmoid per expert

DeepSeek-V3 switched to a per-expert sigmoid, $s(\ell_i) = \sigma(\ell_i)$, then renormalised over
the chosen $K$. Each expert is now scored on its own merits, the score of expert $i$ depends on
$\ell_i$ alone, and raising one expert does not lower the rest. The scale of the whole logit vector
no longer matters to the ranking, only to how sharp the weights are.

The remaining weakness of the sigmoid is saturation. It is bounded above by one, so once a logit is
comfortably positive the score stops responding and its derivative goes to zero. When several chosen
experts are all in that regime their scores are all near one, and after renormalisation the token
receives eight nearly equal contributions. The router can rank the experts but cannot express how
much more it prefers the first to the eighth, and it receives almost no gradient to change the
preference.

### Square-root softplus

V4.1 uses

$$
s(\ell_i) = \sqrt{\operatorname{softplus}(\ell_i)} = \sqrt{\log(1 + e^{\ell_i})} .
$$

This is positive, monotone and unbounded above, so a strongly preferred expert can carry a large
weight. It grows only like $\sqrt{\ell}$ for large logits, which keeps the weights from growing too
fast.

The three properties answer the three problems in order. Positive and per-expert, so nothing is
coupled and no normalisation over $E$ is wasted. Unbounded, so the ratio between the first and the
eighth chosen expert stays expressive and the derivative never vanishes, which is what the sigmoid
gave up. And the square root damps the growth to a rate slow enough that one large logit cannot make
the layer effectively top-1. For large negative $\ell$ the softplus decays like $e^{\ell}$, so the
score decays like $e^{\ell/2}$ and a clearly unsuitable expert is scored near zero without ever
reaching it exactly.

The router is evaluated in fp32 in all of these models, and its input is normalised. Zoph et al.
2022, "ST-MoE: Designing Stable and Transferable Sparse Expert Models", traced a large share of
mixture-of-experts training instabilities to the exponentials in the router and added a router
z-loss that penalises large logits, on top of computing the gate in higher precision. The score
functions above are partly a response to the same problem: the softplus branch and the square root
bound how fast a score can grow out of range.

## The scaling factor

After renormalising, the $K$ weights sum to one. DeepSeek multiplies them by a constant
$\gamma$, equal to 1.5 in V4.1, so that the routed contribution is larger than the shared expert's.
The next section covers the shared expert.

$$
\operatorname{MoE}(x) = \gamma \sum_{i \in \mathcal{I}(x)} w_i\, \operatorname{FFN}_i(x) + \operatorname{FFN}_{\text{shared}}(x).
$$

The renormalisation removed all information about the absolute size of the scores, so without
$\gamma$ the routed branch would always contribute a convex combination of unit total weight and the
shared branch a fixed extra term of weight one, splitting the layer output evenly between them at
initialisation. Setting $\gamma = 1.5$ fixes the ratio at three to two in favour of the routed side.
It is a constant of the architecture and not a learned parameter, and it is applied after
normalisation, so it changes the magnitude the layer writes into the residual stream and nothing
about which experts are chosen.

In owlet1 this is `moe.py` with `routed_scaling_factor=1.5`, top-2 of 8 experts.

## What the 2026 models do

DeepSeek-V3 routes to 8 of 256 experts with sigmoid scores, one shared expert alongside, and the
bias-based balancing of the next section. It has 671 billion parameters with 37 billion active.

Qwen3-235B-A22B keeps a softmax router over 128 experts with top-8 and no shared expert, and
balances with an auxiliary loss computed over the global batch rather than the local one.
Qwen3-Next-80B-A3B moves in the other direction on granularity, with 512 routed experts, top-10 and
one shared expert, for 3 billion active parameters.

GLM-4.5 has 355 billion parameters with 32 billion active, 160 routed experts with top-8 and one
shared expert, and follows DeepSeek-V3 in using sigmoid gating with loss-free balancing.

Kimi K2 has 1 trillion parameters with 32 billion active and 384 routed experts with top-8 plus one
shared expert, using the DeepSeek-V3 layer design unchanged.

The common shape is 128 to 512 routed experts with 8 to 10 active, a shared expert in most designs,
bias-based balancing rather than a large auxiliary loss, and an active fraction that has fallen to 3
to 6 percent of total parameters. The router itself has converged on a per-expert positive score
followed by renormalisation over the chosen set.

The number of experts and the presence of the shared one are the subject of the next section.
