# Gradients through quantisation

The previous two sections put rounding operations inside the model. Weights and activations are
rounded to FP8 before every matmul, cache entries are rounded to FP4 before being stored, and the
indexer's queries and keys are rounded before their scores are computed. During inference this is
only a loss of precision. During training it breaks the backward pass, because the operation that was
inserted has a derivative of zero.

## Rounding has no useful derivative

Rounding has zero gradient almost everywhere. When quantisation is applied inside the forward pass
during training, the parameters that feed it receive no signal unless the backward pass ignores the
rounding.

A quantiser is a step function. It is constant on each interval between two grid points, so its
derivative is zero on the interior of every interval and undefined at the boundaries. Differentiating
the chain rule through it multiplies the gradient of everything upstream by zero. The projection that
produces the indexer's queries would receive exactly no gradient, and it would stay at its
initialisation for the whole run. The problem is not that the gradient is small or noisy. It is that
the true gradient of the composed function is zero and is correct in saying so, because moving a
parameter by an infinitesimal amount really does leave the rounded output unchanged.

## The straight-through estimator

The straight-through estimator does this:

$$
\text{forward:}\ \ \hat x = Q(x), \qquad
\text{backward:}\ \ \frac{\partial \mathcal{L}}{\partial x} := \frac{\partial \mathcal{L}}{\partial \hat x} .
$$

The forward pass uses the quantised value and the backward pass pretends the quantiser was the
identity. The idea goes back to Bengio et al. 2013, "Estimating or Propagating Gradients Through
Stochastic Neurons for Conditional Computation", and was used to train networks with binary weights
by Courbariaux et al. 2015, "BinaryConnect: Training Deep Neural Networks with binary weights during
propagations".

The substituted gradient is deliberately wrong, and the size of the error is bounded by how far $Q$
is from the identity. A quantiser moves its input by at most half a grid step, so the estimator is
close to correct when that step is small compared with the scale over which the loss varies, and it
degrades as the grid coarsens. This is one reason four-bit quantisation-aware training is harder than
eight-bit. It is also why the estimator is normally combined with a clamp, with the gradient set to
zero for inputs pushed outside the representable range, since for those inputs the forward value does
not follow the input at all.

The implementation is one line in any framework with automatic differentiation. Computing
$x + \operatorname{detach}(Q(x) - x)$ returns the value $Q(x)$ in the forward pass, while the only term
that carries a gradient is $x$ itself. Anything that detaches the whole of $Q(x)$ instead breaks the
path, which is the failure described at the end of this section.

## Quantisation-aware training

Training with $Q$ in the forward pass and the identity in the backward pass is quantisation-aware
training, or QAT. The parameters learn to produce values that survive rounding, and there is no gap
between training and the quantised model at inference.

The alternative is to train in full precision and quantise afterwards, which is what GPTQ and the
other post-training methods named in the previous section do. Post-training quantisation is cheap and
needs no access to the training run, but the parameters it quantises were never asked to be robust to
rounding, so some quality is lost and has to be recovered by careful choice of scales. QAT pays the
cost during training instead. Jacob et al. 2018, "Quantization and Training of Neural Networks for
Efficient Integer-Arithmetic-Only Inference", established the recipe of simulating the target format
in the forward pass and using the straight-through estimator in the backward pass, and it is the same
recipe the FP4 indexer uses.

## The owlet1 gap

The V4.1 report says the FP4 indexer queries and keys are trained with QAT. That requires the
identity gradient above. In owlet1 the FP8 path behaves this way, but the FP4 lookup-table path returns
a tensor detached from the graph, so the indexer projections receive no gradient. This detachment,
rather than a missing loss term, leaves the smoke-config indexer at its random initialisation.

That closes the numerics of the block. Chapters 5 to 12 have changed what attention stores, how much
of the feed-forward layer runs, how the residual carries state and how many bits every element takes.
The next two chapters cover the parts of V4.1 that are not inside the block at all, starting with a
memory that is read by hashing rather than by attending.
