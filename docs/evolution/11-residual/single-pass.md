# Single-pass mHC

The constraint of the previous section makes the multi-stream residual stable, but it leaves the
block with a chain of dependent small operations in front of every sublayer. This section removes
the part of that chain that matters most, at the cost of letting one set of coefficients lag the
state by one site.

## The problem mHC introduced

In dynamic hyper-connections the read weights $\mathbf{p}$ are computed from $X$, and the sublayer
input $u = \mathbf{p}^{\top} X$ depends on them. Each site therefore needs three dependent steps before
the sublayer can start: compute $\operatorname{mix}(X)$, then the read, then $f$. In a pipelined or
communication-overlapped training setup that serial chain is on the critical path of every layer.

The chain is short in arithmetic and long in latency. The coefficient projection is a matrix-vector
product of 786 thousand parameters, the Sinkhorn projection is twenty passes over a $4 \times 4$
matrix, and the read is a weighted sum of four rows. None of them keeps an accelerator busy. Two of
them also touch the whole residual state. The coefficient projection consumes all $nD = 32768$
elements, and the read consumes them again, so the state of every token is fetched twice per site
where a single-stream residual would fetch it once. On the reference model that is 128 KB of traffic
per token per site instead of the 64 KB the read alone needs. The chain repeats at all 160 sites, and
in a training run that overlaps expert communication with computation it is exactly the kind of
sequential dependency that leaves the overlap with nothing to hide behind.

## The shift

V4.1 uses the read weights computed by the previous site rather than the current one. Site $\ell$
computes $(\mathbf{p}_\ell, \mathbf{q}_\ell, C_\ell)$ from $X_\ell$ but reads with $\mathbf{p}_{\ell - 1}$:

$$
u_\ell = \mathbf{p}_{\ell-1}^{\top} X_\ell, \qquad
X_{\ell+1} = \mathbf{q}_\ell\, f(u_\ell)^{\top} + C_\ell^{\top} X_\ell .
$$

Since $\mathbf{p}_{\ell-1}$ is already known when $X_\ell$ arrives, the read and the mix projection are
independent and can run in one pass over $X_\ell$. The very first read uses a fixed one-hot vector, and
the final read after the last layer uses the last site's $\mathbf{p}$. Read weights then lag the state
by one site. The paper applies the shift per block. owlet1 applies it across both sites of each
block, attention and MoE.

Reading the change back. Nothing is removed and nothing is approximated. The same coefficients are
computed from the same states. Only the pairing changes, so that the vector a site reads with was
produced one site earlier. The write weights $\mathbf{q}_\ell$ and the mixing $C_\ell$ are still the
ones the current site computed from the current state, so the write and the mixing are not delayed.
Only the read is.

The boundaries close the recursion. Before the first site there is no previous set of coefficients,
so the read is the fixed one-hot vector $(1, 0, \dots, 0)$. Since every stream starts as a copy of the
token embedding, that first read returns the embedding, and the first sublayer sees exactly what it
would see in a single-stream network. After the last site there is one set of coefficients left over,
and it is used for the final read that reduces the $n$ streams to the one vector the unembedding
consumes. No coefficient is computed and discarded.

## What the lag costs

A site can no longer choose what to read on the basis of what the site immediately before it wrote.
The read weights it uses were derived from the state as it stood one site earlier. The information is
not lost, because the state itself is current and the write from the previous site is present in it.
What is lost is the ability to react to that write when deciding the proportions in which to read.
Reaction is delayed by one site rather than prevented, since the coefficients this site computes will
be applied at the next one.

The shift also changes what the coefficient projection is trained to do. Its output is used one site
later, so it must predict a useful read for the sublayer that follows rather than for itself.
Applying the shift per site, as owlet1 does, gives a lag of one sublayer. Applying it per block, as
the report describes, gives the attention site and the feed-forward site of a block the same
inherited read weights and a lag of one block.

With the residual settled, the remaining reductions in the block are not structural. Chapter 12 takes
the last factor in every byte count in this book, the number of bits used to store an element.
