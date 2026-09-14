---
title: "Hyper-connections"
order: 1
---

## Hyper-connections

Replace the single residual vector with $n$ parallel streams stacked as $X \in \mathbb{R}^{n \times D}$.
This is the construction of Zhu et al. 2024, "Hyper-Connections". It is the smallest change that
addresses both weaknesses named in the chapter introduction at once. Depth dilution is addressed
because a sublayer chooses how much of its output goes into each stream, so a stream can be left
alone by most layers and stay legible to a distant one. The single view of the past is addressed
because the combination a sublayer reads is separate from the combination it writes.

The residual stream has limited bandwidth. Elhage et al. 2021, "A Mathematical Framework for
Transformer Circuits", describe it as a shared medium that every layer reads from and writes to,
with the same $D$ dimensions shared by all of them, and Elhage et al. 2022, "Toy Models of
Superposition", show that networks represent more features than they have dimensions by storing them
in superposition. A later layer that needs a feature written many layers earlier must recover it
from a stream that the intervening layers have kept writing into. Several streams with learned read
and write weights give the network a way to keep a feature where the intermediate layers do not
overwrite it.

### The state and its boundaries

The residual state of a token is now a matrix of $n$ rows, each of width $D$. The rows are the
streams. At the input every stream receives a copy of the token embedding, so the network starts
from $n$ identical rows and the streams differentiate only through what the layers write into them.
At the output the rows are reduced back to a single $D$-vector by a read of the same form as the one
used inside the block, and that vector goes to the final normalisation and the unembedding. Between
those two boundaries the number of rows is constant, and every site of every block performs the same
three operations.

### Read, compute, write

A sublayer reads a weighted combination of the streams. Its output is written back into each stream
with its own weight, and the streams are also mixed among themselves:

$$
u = \mathbf{p}^{\top} X, \qquad
y = f\big(\operatorname{RMSNorm}(u)\big), \qquad
X' = \mathbf{q}\, y^{\top} + C^{\top} X .
$$

Here $\mathbf{p} \in \mathbb{R}^n$ are read weights, $\mathbf{q} \in \mathbb{R}^n$ write weights, and
$C \in \mathbb{R}^{n \times n}$ mixes the streams. With $n = 1$, $\mathbf{p} = \mathbf{q} = C = 1$ this is
the ordinary residual. The sublayer $f$ is unchanged and sees one $D$-vector, so the extra cost is the
small read and write, not the sublayer.

Reading the three parts back. The read $\mathbf{p}^{\top} X$ collapses the $n$ rows into one row of
width $D$ by a weighted sum, so the sublayer never learns that there is more than one stream. The
term $\mathbf{q}\, y^{\top}$ is an outer product of an $n$-vector with a $D$-vector, which broadcasts
the single output row into all $n$ rows with a per-stream weight. The term $C^{\top} X$ is the only
part that moves content between streams, and it acts on the stream index alone. It applies the same
$n \times n$ mixing to every one of the $D$ dimensions, so no dimension is mixed with another and the
whole construction is a reordering of information across streams rather than a new linear layer of
width $nD$.

Setting $C = I$ and $\mathbf{p} = \mathbf{q} = e_k$ for a fixed $k$ recovers $n$ independent residual
streams, only one of which is ever used. Setting $C = I$ and letting $\mathbf{p}$ and $\mathbf{q}$
vary recovers $n$ independent residual streams that every sublayer reads from and writes to in
learned proportions. The full form with a general $C$ adds a fixed rotation of content between the
streams at every site. All three are reachable by the same parameters, so what the network uses is
learned rather than chosen by the designer.

### Coefficients predicted per token

The coefficients may be static parameters. In the dynamic variant they are predicted per token from
the current state:

$$
(\mathbf{p}, \mathbf{q}, C) = \operatorname{mix}(X) = W \operatorname{RMSNorm}(\operatorname{vec} X) .
$$

Different layers can then read different mixes. One layer may read mostly what the previous layer
wrote. Another may read a stream that has not been written for several layers. The streams act as
$n$ residuals with learned routing between them.

The dynamic variant makes the routing depend on the token, so the same layer can read one
combination for one token and another for the next. The cost is one small projection per site. Its
input is the flattened state of $nD = 32768$ elements and its output is the $n + n + n^2 = 24$
coefficients needed at $n = 4$, so $W$ holds about 786 thousand parameters. Over two sites in each of
80 blocks that is about 126 million parameters, which is 0.2 percent of the reference model's 65
billion, and about 3 MFLOPs per token per block against the 537 MFLOPs of the attention projections.
The routing is therefore cheap in both parameters and arithmetic. What it is not cheap in is
activation memory, for the reason given in the chapter introduction.

### Why the unconstrained form is fragile

The weakness is stability. Nothing constrains $C$. Repeated application over $L$ layers can amplify
or extinguish a stream, and the design was found to be fragile at scale.

The mechanism is worth being precise about, because it explains what the next section constrains.
Ignore the write term for a moment and follow only the residual part of the update. After $m$ sites
the state has been multiplied by a product of $m$ mixing matrices. If those matrices are similar to
each other, that product behaves like a power, and the state grows or decays geometrically in the
spectral radius of $C$. A radius of 1.01 gives a factor of about 4.9 over the 160 sites of the
reference model, and a radius of 0.99 gives a factor of about 0.2. Neither is fatal on its own, but
the radius is a learned quantity that drifts during training, and there is no term in the loss that
pushes it towards one. A stream whose content decays towards zero also loses the gradient path that
made the residual connection useful in the first place, which is the property He et al. 2016
identified as depending on the identity path being clean.

The fix is not to remove the mixing but to restrict it to matrices whose repeated application is
harmless. The next section gives the restriction DeepSeek adopted.
