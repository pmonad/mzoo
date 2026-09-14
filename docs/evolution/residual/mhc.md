---
title: "mHC"
order: 2
---

## Manifold-constrained hyper-connections

The previous section left one free matrix in the residual update. $C$ is predicted per token, applied
at every site, and bounded by nothing. Over the 160 sites of the reference model its repeated
application is what decides whether a stream survives. The fix is to keep the mixing and restrict the
matrices it can produce to a set whose repeated application can neither amplify nor extinguish
anything.

DeepSeek's mHC, published in 2025, keeps the dynamic hyper-connection but projects $C$ onto the doubly stochastic
matrices before use: every row and every column sums to one and all entries are non-negative.

$$
C = \operatorname{Sinkhorn}_T\big(\operatorname{softmax}(\text{logits})\big), \qquad
\sum_j C_{jk} = 1,\ \ \sum_k C_{jk} = 1,\ \ C_{jk} \ge 0 .
$$

### The set the matrix is projected onto

The $n \times n$ doubly stochastic matrices form a compact convex set, the Birkhoff polytope. By the
Birkhoff and von Neumann theorem every member of it is a convex combination of permutation matrices.
The mixing step is therefore always an average of permutations of the streams, weighted by
coefficients the network chooses per token. At one extreme the network picks a single permutation and
rotates the streams without altering their contents. At the other it picks the uniform matrix and
replaces every stream by the average of all of them. Nothing outside that range is reachable, which
is what the name refers to.

### Sinkhorn's algorithm in words

The projection is computed iteratively. Start from a matrix with strictly positive entries, which the
softmax over the predicted logits guarantees. Divide every row by its sum, so the rows sum to one.
Then divide every column by its sum, so the columns sum to one and the rows no longer do. Repeat.
Sinkhorn 1964, "A Relationship Between Arbitrary Positive Matrices and Doubly Stochastic Matrices",
proved that this alternation converges to a doubly stochastic matrix for any strictly positive
starting matrix, and that the limit is unique up to the diagonal scalings used to reach it. Cuturi
2013, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport", brought the same iteration
into machine learning as a differentiable relaxation of assignment problems, which is the property
that matters here. Every step is a division, so the whole projection is differentiable and the
logits receive gradient through it.

Convergence is geometric, and the rate improves the smaller the matrix and the closer the starting
matrix is to balanced. $T = 20$ iterations is enough for a
$4 \times 4$ matrix. The consequence for the write step is that

$$
x'_k = q_k\, y + \sum_j C_{jk}\, x_j
$$

gives each output stream a convex combination of the input streams. The residual part of the update
neither amplifies nor shrinks the total, however many layers are stacked. The unconstrained version
gives no such guarantee.

### What the constraint guarantees

Three statements follow from the two sum conditions, and together they are the reason for the
constraint.

Each output stream is an average. The coefficients $C_{jk}$ for fixed $k$ are non-negative and sum to
one, so $\sum_j C_{jk} x_j$ lies in the convex hull of the input streams and its norm is at most the
largest input norm. One stream cannot grow by drawing repeatedly on the others.

The total is conserved exactly. Summing the residual part over $k$ and using the other sum condition
gives $\sum_k \sum_j C_{jk} x_j = \sum_j x_j$. Whatever the network does with the mixing, the sum of
the four streams passes through it unchanged, so there is always one linear combination of the
streams that behaves exactly like the ordinary residual of chapter 1 and carries the clean gradient
path with it.

The backward pass is non-expansive too. A doubly stochastic matrix has both its maximum row sum and
its maximum column sum equal to one, so its spectral norm is at most one. Multiplying by $C$ or by
its transpose can never increase the Euclidean norm of a vector. Applying it 160 times can never
increase it either, which removes the geometric drift described at the end of the previous section.

The read and write coefficients are left with ranges instead of constraints, because growth from the
write side is the ordinary residual growth that pre-norm already produces. The read weights are kept positive with a sigmoid, and the write gates
$q_k = 2\sigma(\cdot)$ range over $(0, 2)$, so a layer can write more than one copy of its output.
A write gate near zero lets a layer leave a stream untouched, which is what preserves an early
feature for a distant reader.

### What it costs

The arithmetic of the projection is negligible. Twenty iterations on a $4 \times 4$ matrix is about
640 operations per token per site, against the 537 million floating-point operations of the attention
projections in the same block. What it costs is not FLOPs but shape. The iterations are serial, each
one is a tiny reduction, and they sit between the coefficient projection and the write. On an
accelerator this is a chain of small latency-bound kernels rather than a large matmul, and the chain
is on the path of every one of the 160 sites.

V4.1 uses $n = 4$ streams, which multiplies activation memory for the residual by four but leaves the
sublayer compute unchanged. The paper reports improved training stability and quality at 40 layers
against a single residual. The full derivation for the owlet1 implementation is in
`src/mzoo/archs/owlet1/mHC.md`.

The projection is not the only serial chain the design adds. The read weights are predicted from the
state and the sublayer input depends on them, so the sublayer cannot start until the coefficients are
ready. The next section removes that dependency.
