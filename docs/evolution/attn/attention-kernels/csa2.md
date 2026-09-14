---
title: "One kernel for CSA2"
order: 2
---

## One kernel for CSA2

After chapters 8 to 10, one query's attention in a CSA2 layer reads three sources: its window, the
selected entries of the compressed main table, and the sink. Mathematically, for query token $t$ in
head $h$ with latent scale $\text{scale} = d^{-1/2}$:

$$
z = \big[\, q \cdot c^{win}_{t-W+1..t}\ \|\ q \cdot c^{main}_{\mathcal{S}_t}\ \|\ s_h \,\big]\cdot\text{scale},
\qquad
p = \operatorname{softmax}(z), \qquad
o = \textstyle\sum p_j\, c_j ,
$$

where the sink column contributes to the denominator only, $W = 128$, and $\mathcal{S}_t$ is the
top-$k$ index set from chapter 10. A modular implementation would run this as three attentions and
merge. The kernel runs it as one, and the merge is the interesting part.

### One running softmax across two sources

The online softmax of the previous section needs one running $(m, \ell, o)$ per query row. If the
window walk and the main-table walk were separate kernels, each would emit its own
$\operatorname{LSE}$ and unnormalised output, and the caller would combine them:

$$
o = \frac{e^{m_1 - m} o_1 + e^{m_2 - m} o_2}{e^{m_1 - m}\ell_1 + e^{m_2 - m}\ell_2},
\qquad m = \max(m_1, m_2),
$$

an extra pass over both outputs' worth of traffic. Instead the kernel walks the window tiles, then
the selected main tiles, updating the same statistics — window first, since it is contiguous and
its tiles prefetch while the gather addresses resolve. No merge pass exists. The sink is the third
source, folded in at the end as in the previous section.

The window walk is bounded: with $W = 128$ and $B_N = 64$, it is at most two key tiles regardless
of context length, masked by the band $t - 127 \le j \le t$. The main walk is a gather:

$$
\texttt{KV\_shared}[i, :] = \texttt{main\_kv}\big[\,\texttt{Indices}[t, i]\,\big][:], \qquad i < k,
$$

$k$ tiles of 64 entries ($k = 512$ at scale and in the design scope, 64 in the smoke config), with
out-of-range or $-1$ indices masked to $-\infty$ logits. This is exactly the shape
DeepSeek's public DSA sparse kernel uses, which is why it is the porting template rather than
something derived from the dense FA2 loop. One gather serves all 64 heads because the index set is
per token, shared across heads — the payoff of the 1-token × 64-head block layout.

The main table has a second mask the window does not: compression at ratio $m$ means entry $j$
summarises tokens $[jm, jm + m)$, and it is visible to query $t$ only when $jm + m - 1 \le t$,
i.e. $j < (t+1)/m$ (group-causal). The indexer respects the same boundary, so the gather rarely
returns a masked entry, but the kernel enforces it independently.

### Concrete shapes

For one sequence, one layer group, at the V4.1-Flash scale ($H = 64$, $D = 512$, $W = 128$,
$m = 2$, $k = 512$):

| tensor | shape | format |
| --- | --- | --- |
| $Q$ (per layer) | $[S, 64, 512]$ | bf16 |
| window cache (per block) | $[S, 512]$, last 128 live | fp8 + ue8m0 scale per 32 ch |
| main cache (per group) | $[S/m, 512]$ | fp4 e2m1 + e4m3 scale per 16 ch |
| indices | $[S, k]$ | int32 |
| output | $[S, 64, 512]$ | bf16 |

The two cache rows are the point of the table: the window is per block, short and FP8; the main
table is per group, half as long, and FP4. Everything else in the kernel is bf16.

At the repository's smoke scale these are $[S, 4, 64]$, a window of 128, $m = 4$ and $k = 64$, and
the kernel is tested layer-by-layer against the eager reference in `archs/dsv4/`.

### What the precision buys, on paper

The formats are chapter 12's; the reason they are *dequantised on load* rather than multiplied in
place is bandwidth arithmetic, and it opens the question the next chapter answers in full — which
operands, if any, deserve four bits in the arithmetic itself. A decode step for one query reads,
per block,

$$
\underbrace{128 \times 512 \times 1}_{\text{window, fp8}} + \underbrace{512 \times 512 \times
0.5625}_{\text{main, fp4}} \approx 65\ \text{KB} + 147\ \text{KB},
$$

against 0.64 MB if both were bf16 — a 3× reduction in bytes moved, on a part where the step is
bounded by 273 GB/s. The fp4 *multiply*, by contrast, runs at roughly fp8 speed on SM120 (measured
within ~1.2× on this class of hardware; see the low-precision section), so computing in bf16 after
dequantising into the shared-memory tile loses almost nothing in arithmetic and keeps the softmax
and both accumulations in full precision. This is the same judgement FlashInfer and FlashMLA make
in production, and it is why the design grows the fp8/fp4 loaders as version-2 steps after each
correct bf16 version-1 exists.

### What stays outside

The kernel takes pointers. Which main cache and which indices it reads — the Full, Reindex and
Reuse modes of chapter 9's sharing scheme — is a scheduling decision made by the model code; the
same kernel serves all three. RoPE (and its inverse on the output), the compressor, the indexer
score and the top-$k$ are separate passes: the score is a matmul in its own format (next chapter),
and top-$k$ is left to `torch.topk` until profiling says otherwise. Decode-specific machinery —
split-KV over the gathered entries, a paged main cache — is deferred until inference matters; the
training kernels only need the prefill-shaped forward and, per step, its backward.
