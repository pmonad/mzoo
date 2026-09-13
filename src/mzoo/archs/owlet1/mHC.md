# mHC: the multi-stream residual

Every block in owlet1 is a standard pre-norm transformer block except for the residual connection.
Instead of one vector per token, the residual is $hc = 4$ vectors, and each sublayer reads a weighted
combination of them and writes back into all of them. This file describes those read and write
operations and why the read weights are delayed by one site. Exact formulas, including the Sinkhorn
iteration and the $\epsilon$ terms, are in [decoder.md](decoder.md).

## State

Per token, the residual state is

$$
X = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix} \in \mathbb{R}^{hc \times D},
$$

stored as `[B, S, hc, D]`. The rows are called streams. At the input every row is the token embedding;
the rows diverge only because the write step (below) writes different amounts into each of them.

## One site

A *site* is one sublayer $f$ (attention or MoE) together with its read and write. Given the current
state $X$ and a read vector $\mathbf{p}_{\text{in}} \in \mathbb{R}^{hc}$:

$$
\begin{aligned}
(\mathbf{p}, \mathbf{q}, C) &= \operatorname{mix}(X)
  && \mathbf{p}, \mathbf{q} \in \mathbb{R}^{hc},\ \ C \in \mathbb{R}^{hc \times hc} \\
u &= \mathbf{p}_{\text{in}}^{\top} X
  && \text{read: one } D\text{-vector} \\
y &= f\big(\operatorname{RMSNorm}(u)\big)
  && \text{the sublayer, unchanged from a single-stream model} \\
X' &= \mathbf{q}\, y^{\top} + C^{\top} X
  && \text{write}
\end{aligned}
$$

The site returns $X'$ and $\mathbf{p}$. In index form the write is

$$
x'_k = q_k\, y + \sum_{j=1}^{hc} C_{jk}\, x_j ,
$$

so stream $k$ receives the sublayer output scaled by its gate $q_k$, plus a mixture of all incoming
streams weighted by column $k$ of $C$.

Setting $hc = 1$, $\mathbf{q} = 1$, $C = 1$ recovers the ordinary block $x' = x + f(\operatorname{RMSNorm}(x))$.

## The coefficients

$\operatorname{mix}$ is one linear map of the flattened, RMS-normalised state, followed by a different
squashing for each of the three outputs:

$$
m = W\,\operatorname{RMSNorm}\big(\operatorname{vec}(X)\big) \in \mathbb{R}^{(2+hc)\,hc}, \qquad
\mathbf{p} = \sigma(m_{[0:hc]}) + \epsilon, \qquad
\mathbf{q} = 2\,\sigma(m_{[hc:2hc]}), \qquad
C = \operatorname{Sinkhorn}\big(\operatorname{softmax}(m_{[2hc:]})\big).
$$

Each slice has its own learned scale and bias, omitted here. The consequences of the three ranges:

- $\mathbf{p} \in (\epsilon, 1)^{hc}$: read weights are positive; no stream can be read with weight exactly zero.
- $\mathbf{q} \in (0, 2)^{hc}$: each stream's write gate; the network can write more than one copy of $y$ into a stream.
- $C$ doubly stochastic (20 Sinkhorn iterations): every row and every column sums to 1. Each output stream
  is therefore a convex combination of the input streams, and the total mass across streams is preserved.
  This is what keeps four parallel residuals bounded without applying a norm to $X$ itself.

Because $m$ is computed per token, all three coefficient sets vary from position to position.

## The one-site delay

$\operatorname{mix}(X)$ is evaluated from the state at the start of the site, and its $\mathbf{q}$ and
$C$ are used by that site's write. Its $\mathbf{p}$, however, is not used by that site's read. The read
uses $\mathbf{p}_{\text{in}}$ from the previous site, and the current $\mathbf{p}$ is handed forward.

In a block with sites $a$ (attention) and $f$ (MoE), and blocks indexed by $\ell$:

$$
\text{read}_a^{(\ell)} \text{ uses } \mathbf{p}_f^{(\ell-1)}, \qquad
\text{read}_f^{(\ell)} \text{ uses } \mathbf{p}_a^{(\ell)}, \qquad
\text{read}_a^{(\ell+1)} \text{ uses } \mathbf{p}_f^{(\ell)}.
$$

The first read, at block 0, uses the fixed vector $(1, 0, 0, 0)$. The final read after the last block
uses $\mathbf{p}_f^{(L-1)}$.

The reason is scheduling. Without the delay, a site would have to compute $\operatorname{mix}(X)$, then
the read, then the sublayer: three dependent steps before $f$ can start. With the delay, the read vector
is already known when $X$ arrives, so the read and $\operatorname{mix}(X)$ are independent and the
sublayer input is available after a single pass over $X$. This is the "single-pass" in Single-Pass mHC
(§2.4.1 eq. 6 of the report). The paper states the shift for one site per block; owlet1 applies the same
rule across both sites of a block.

## Cost

Per token and per site: one $(2+hc)\,hc \times hc\,D$ projection ($24 \times 1024$), 20 Sinkhorn
normalisations of a $4 \times 4$ matrix, one $hc \times D$ read, and one rank-1 write plus a
$hc \times hc$ mixing of the streams. The sublayer itself receives a single $D$-vector, so attention and
MoE cost the same as in a single-stream model.
