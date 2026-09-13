# Training the indexer

Everything in this chapter so far describes a model that already works. The indexer's parameters are
what make selection useful rather than random, and they have to be learned. This section is about the
one part of the design that ordinary backpropagation cannot supply.

## Why a hard top-k has no gradient

The mask $M$ is $0$ or $-\infty$. It has no gradient with respect to the scores $I_{s,j}$. The main
attention loss therefore carries no information about which entries the indexer should have chosen.
Without a separate objective the indexer parameters never move and selection stays random.

The failure is worth stating precisely, because it is not the usual difficulty of a small or noisy
gradient. A small change in $I_{s,j}$ leaves the selected set unchanged, so the derivative of the loss
with respect to every indexer parameter is exactly zero almost everywhere, and at the points where
the set does change the loss jumps. Nor can the loss report on entries that were not selected, since
those entries were never read and the model has no evidence about what they would have contributed.

The router of chapter 6 faces the same discrete choice and escapes it. There the chosen experts'
outputs are multiplied by their gate weights, so the loss differentiates through the weight of a
chosen expert even though the choice itself is discrete, and the balancing terms handle the rest. The
indexer has no such multiplicative path. Its output enters the model only through a mask that is
constant on either side of the threshold.

The general remedies for discrete choices are known and none of them is free. A straight-through
estimator, in the sense of Bengio et al. 2013, "Estimating or Propagating Gradients Through
Stochastic Neurons for Conditional Computation", passes the gradient of a continuous surrogate in
place of the true one, which is what chapter 12 does for quantisation. A continuous relaxation such
as Jang et al. 2017, "Categorical Reparameterization with Gumbel-Softmax", or Maddison et al. 2017,
"The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables", replaces the hard
choice with a soft one during training and anneals it, which for a top-$k$ over 65536 entries means
computing the dense attention the design exists to avoid. The third option is to give the selector
its own supervised objective, and that is the route the DeepSeek models take.

## Distilling the indexer from dense attention

DeepSeek-V3.2 started from a trained dense model and trained the indexer to imitate the dense
attention pattern. With $p_{s,j}$ the dense attention probability summed over heads and
$\hat I_{s,j} = \operatorname{softmax}_j I_{s,j}$,

$$
\mathcal{L}_{I} = \sum_s \operatorname{KL}\big(p_{s,\cdot} \,\|\, \hat I_{s,\cdot}\big),
$$

first with dense attention still active, the warm-up phase, then with the sparse mask in place while
the KL term continues to supervise the indexer. The indexer thus has its own objective and does not
depend on gradient through the mask.

The target costs nothing extra during the warm-up phase, because the dense model computes $p_{s,j}$
anyway. The objective asks the indexer to reproduce the ranking that full attention would have
produced, which is exactly the ranking the top-$k$ needs, and it supplies a signal for every entry
including the ones a sparse model would not have read. Summing over heads before the KL is what makes
a single indexer serve all of them. An entry that any head wants keeps a high target probability. Once
the mask is switched on the KL term is still computed over the entries the model can see, so the
indexer keeps being corrected while the rest of the model adapts to sparse reads.

A different route exists. Yuan et al. 2025, "Native Sparse Attention: Hardware-Aligned and Natively
Trainable Sparse Attention", derives its block selection scores from the attention weights of a
parallel compression branch, so the scores are part of the model's own computation and are trained
with it from the beginning rather than supervised against a dense teacher.

## What the V4.1 report says

The V4.1 report states that sparse attention is trained from scratch, without a dense warm-up, and
that indexer parameters are optimised with gradient aggregated across replicas. It does not state
the objective that produces those gradients. Whether a KL target, a differentiable relaxation of the
top-$k$, or something else is used is not documented. Training from scratch removes the dense teacher
that V3.2 relied on, so some other source of signal must exist, and the report does not identify it.
In owlet1 the indexer currently receives no gradient at all, both because of the hard mask and
because the FP4 fake-quantisation of its queries and keys detaches them from the graph, as described
in chapter 12. This is the largest open issue in the port.

## What the 2026 models do

DeepSeek-V3.2 (2025) is the first production use of the design. Its lightning indexer selects 2048
keys per query over the per-token MLA cache of chapter 7, so it reduces reads while leaving the cache
as V3 left it. DeepSeek-V4.1 (2026) combines the two chapters, adding the compressed shared tables of
chapter 9 under the same indexer, so storage and reads are reduced together. GLM-5 (2026) uses
multi-head latent attention with sparse attention in the DeepSeek style, which makes this the second
family to adopt it rather than a single vendor's choice.

The contrast with the other route to long context is now sharp. Qwen3-Next (2025), Qwen3.5 (2026) and
Kimi Linear (2025) keep only one softmax layer in four and give the rest a recurrent state of fixed
size, so the cache is small because most layers do not have one. The DeepSeek line keeps every layer
softmax and reduces what each layer stores and reads. Kimi K2 (2025) shows that the unmodified path
is still viable at 128K context with a compact MLA cache. All three positions were held by
strong open-weight models in 2026, and no consensus of the kind that formed around RMSNorm, RoPE and
SwiGLU has formed here.

## Where this leads

Attention is now as cheap as this book will make it. Chapters 5 to 10 reduced the cache from 2.6 MB
per token to a fraction of a kilobyte and the read per query from $S$ to a constant. The remaining
chapters leave the sublayers and change what surrounds them. Chapter 11 changes the residual
connection that carries their outputs.
