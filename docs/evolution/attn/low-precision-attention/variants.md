---
title: "The variants"
order: 1
---

## The variants

Six designs are worth reading together, because each answers the same three questions — which
operands are quantised, at what granularity, and what the scales cost — and the answers move in one
direction: toward finer scales in cheaper places, and toward arithmetic that stays bf16 while the
*values* it consumes are quantisation-aware. The running example, hardware anchors and evidence
paths are those stated in chapter 17's introduction and are not repeated here.

Two measures recur throughout and are worth defining once. The root-mean-square error (RMSE)
between a quantised output and the bf16 one is the square root of the mean squared difference
between them — one number for "how far off is the result", where smaller is better and 0 is exact.
The cosine similarity of the two outputs is the cosine of the angle between them as vectors — 1.0
means identical up to length, and it degrades gracefully where RMSE scales with the output
magnitude.

### 1. FlashAttention-3 FP8: block scales and incoherent processing

FA3 (H100) quantises $Q$, $K$, $V$ per tile — one scale each for the $B_M \times d$ query block
and the $B_N \times d$ key block — and corrects the product by the scale product, which is free
because it can be folded into the softmax shift:

$$
\hat X = \operatorname{round}_{e4m3}\!\Big(\frac{X}{s_X}\Big), \quad s_Q = \max|Q_{\text{tile}}|/448,
\qquad \hat Q \hat K^\top = \frac{1}{s_Q s_K} Q K^\top \ \text{(up to rounding)} .
$$

The format names encode their bit split: e4m3 spends 4 bits on exponent and 3 on mantissa (more
resolution, less range — largest finite value 448), e5m2 spends 5 on exponent and 2 on mantissa
(more range, less resolution), and e2m1 spends 2 on exponent and 1 on mantissa (six nonzero
magnitudes: 0.5 to 6). One bit is always the sign.

Per-tensor scaling (one scale for the whole tensor) gives RMSE $2.4\times10^{-2}$ against bf16;
block scaling alone gives $9.3\times10^{-3}$ — a 2.6× improvement from granularity alone. FA3 also
rotates $Q \leftarrow QM$, $K \leftarrow KM$ with $M$ a random-sign Hadamard ("incoherent
processing"; $(QM)(KM)^\top = QK^\top$ exactly), which spreads outliers across dimensions; it adds
only $9.1 \times 10^{-3}$, i.e. ~2% on top of block scales. The lesson everyone inherited:
**granularity does the work; rotation is a refinement**. The other lesson: FA3's FP8 path is
forward-only, and its backward rejects FP8 — a pattern that repeats.

### 2. SageAttention 1 and 2: integer scores, float probabilities

SageAttention keeps the structure "scores cheap, second product accurate" and pushes it to integer.
Version 1: $Q, K$ in INT8 per tile, $PV$ in fp16 with an fp16 accumulator. Its key exactness trick
is smoothing: softmax is shift-invariant in $K$, so subtracting the per-head token mean
$K \leftarrow K - \bar k$ changes nothing in the exact arithmetic while centring the distribution
on the INT8 grid.

Version 2 attacks granularity: scales aligned to *thread fragment ownership* rather than tiles. An
`mma.m16n8k64` C-fragment gives each thread 2 of the 16 output rows; giving each thread one Q
scale for its 8 query rows and one K scale for its 4 key-pair columns means the correction multiply
happens on values already in that thread's registers, with no cross-lane traffic — 32× finer than
per-block for $Q$ at zero cost. The $P \cdot V$ product uses two ideas worth keeping:

1. **Static P scale.** $P$ is a probability, so every entry lies in $[0, 1]$, and under the online
   softmax the row max is exactly 1 at normalisation time — the largest value the format must
   represent is known before the kernel runs, so the scale can be a compile-time constant instead
   of a per-row reduction: quantise with a fixed $\delta_P = 1/448$. A *global constant* dequant
   means the PV mma accumulates directly into the output accumulator — no per-row rescale pass, no
   second buffer.
2. **Two-level accumulation.** The fp8 mma accumulates in "FP22" (1 sign, 8 exponent, 13 mantissa
   bits — only 22 effective bits per partial); that is fine for $B_N = 64$ keys, and the partials
   are promoted to fp32 when summed into $O$. This split (small-k low-precision inner, fp32 outer)
   is what DeepSeek's FP8 GEMMs also do with their every-128-element promotion.

### 3. SageAttention3: NVFP4 everywhere and the two-level P scale

On SM120, SageAttention3 runs $Q$, $K$, $P$ and $V$ all in NVFP4 (e2m1 values, one e4m3 scale per
16 elements) on the block-scaled mma, one fused atom for QK and PV. The problem specific to FP4 is
the scale encoding of $P$: an e4m3 block scale for values in $[0, 1/6]$-ish sits in the bottom of
e4m3's range and wastes it (cos-sim 93.3%). The fix is a second, per-row level:

$$
s_{P1} = \frac{\operatorname{rowmax}(P)}{448 \cdot 6} = \frac{1}{2688}, \qquad
\hat P_2 = \operatorname{round}_{e2m1}\!\Big(\frac{P}{s_{P1}}\Big), \qquad
O = s_{P1} \cdot \big(\hat P_2 \hat V\big),
$$

with $s_{P1}$ folded into the $\exp_2$ argument of the online softmax — the same fold as FA3's
scale correction, one level deeper. Read the two levels back in words. The first level is per
*row* and static — because the row max of $P$ is exactly 1, the fixed divisor $s_{P1} = 1/2688$
puts the row maximum at $448 \cdot 6$, the top of what the second level can encode. The second
level is the ordinary NVFP4 block scale, one e4m3 factor per 16 elements, which maps each block's
largest value onto the top of e2m1's six-magnitude range. Without the first level, a row of
probabilities in $[0, 1/6]$-ish sits at the bottom of the encodable range and the block scale
wastes it; with it, cosine similarity rises from 93.3% to 99.5%. The kernel
also keeps smooth-K and smooth-Q, with the rank-1 correction $Q\bar k^\top$-term added back into
the score accumulator.

### 4. Attn-QAT: train in the loop, compute in bf16

The variants so far are inference-time approximations of a bf16-trained model. Attn-QAT instead
*trains* with quantisation in the loop so the weights adapt to the grid, and its findings are the
ones that matter for a training repository:

- The forward **fake-quantises** ($\hat X = \phi^{-1}(\phi(X))$: round each value to the nearest
  point of the low-precision grid and immediately round back to high precision, so the forward pass
  *sees* quantised values without the GEMM computing in the format) and computes the GEMMs in
  **bf16** — the FP4 tensor-core path is not used for training at all on SM120.
- Naive FP4 forward + standard FA backward explodes, because FA's backward uses the identity
  $P^\top dP = dO^\top O$, which assumes high-precision $O$. Two fixes: fake-quantise $P$ the same
  way in the backward recompute, and carry a second forward output
  $O' = \sum_j \tilde P_{ij} \hat V_j$ (high-precision $P$, quantised $V$) used *only* for the
  gradient correction $D = \operatorname{rowsum}(dO \odot O')$.
- The STE — the *straight-through estimator* — lets gradients pass the non-differentiable rounding
  step as if it were the identity, so the backward uses the values it needs without a derivative
  of $\operatorname{round}$ existing. It is asymmetric by design:
  $dS = P \odot (dP - D)$ uses the **unquantised** $P$ while
  $dV = (\hat P)^\top dO$ uses the quantised one. Scales are statistics and receive no gradient.

For CSA2 this validates the repository's plan directly: fake-quantise the caches with
straight-through gradients in the model code, dequantise into bf16 tiles in the kernel, and keep
every training GEMM bf16. It also supplies the one subtlety our backward must copy: the same
fake-quant applied in both passes, and $D$ from a high-precision-$P$ output.

### 5. FP4 FlashAttention-4: what does not survive training

The FA4-FP4 study runs real FP4 tensor-core training on data-center Blackwell and reports the
sharpest negative result in the field: **every MXFP4 $P$/$V$ training trajectory diverged** (one run
tracks FP8 to update 300, then loss 7.01 → 16.25 in 25 updates), while FP8 $P/V$ trains. Two
further precision observations: gradients in e4m3 but $dO$ in e5m2, because e4m3 rounded ~97% of
observed $dO$ values to zero (e5m2: 14%) — gradients need range, not resolution; and NVFP4
$Q,K$ (block-16) trains fine, confirming that the $QK^\top$ product tolerates FP4 while the
probability-weighted sum does not.

### 6. Dequantise-on-load: FP4 as bandwidth, not arithmetic

The recurring hardware fact, stated once for GB10: FP4 mma throughput on SM120-class parts
measures within ~1.2× of FP8 (ffpa-attn's FP4 attention kernel is only ~23% faster than its FP8
one; the win "comes mainly from bandwidth"), and end-to-end FP4 gains over FP8 are 1.2–1.4×, not
2–4×, because 99 K of shared memory forces small tiles and 273 GB/s caps everything. So the
optimal use of four bits on this hardware is *storage*: keep the cache at 0.5625 bytes/element,
dequantise into the bf16 shared-memory tile on load, and run the mma in bf16. Production agrees —
FlashInfer repacks fp8/fp4 KV tiles into a 16-bit staging buffer then does bf16 `ldmatrix` (the
warp-level instruction that loads a 16×16 tile of 16-bit elements from shared memory into
registers, pre-arranged in the fragment layout the `mma` expects) followed by the mma;
FlashMLA dequantises in registers before a bf16 UMMA; and the reference model's own released
kernels dequantise the main cache rather than multiply it in FP4. The arithmetic-intensity
version of the argument: prefill is compute-bound and FP4 compute is ~FP8, so nothing is lost by
bf16 math; decode is bandwidth-bound and FP4 storage is a 3.6× byte reduction (§ one-kernel),
which then transfers almost fully to wall-clock.

The one place FP4 *math* survives in CSA2 is the indexer score of chapter 10: it is a matmul
consumed only by a top-$k$ (ordering, not magnitude), computed at FP8-or-better effective speed on
the block-scaled path, and its keys are stored MXFP4 — quantise, multiply in the format, discard.

### A merged recipe

Reading the six together, the convergent design for a *training* CSA2 kernel on SM120:

| operand | format | why |
| --- | --- | --- |
| main + window KV storage | fp8/fp4 + block scales | bandwidth (variants 1, 6) |
| KV on load | dequantise → bf16 tile | FP4 mma ≈ FP8 mma; exact softmax (6) |
| $QK^\top$, $PV$ and all backward GEMMs | bf16, fp32 accumulate | training stability (4, 5) |
| $P$ | unquantised; $D$ from a high-precision-$P$ output | the FA identity (4) |
| indexer $Q, K$ | MXFP4 native mma | ordering-only consumer, real FP4 win (6) |
| fake-quant in training | forward-visible, STE backward, scales detached | QAT (4), chapter 12 |

Everything in this table exists in a shipped kernel somewhere; the contribution of the survey was
checking that no combination other than this one has a public backward that trains. The format
column carries chapter 12's argument, the compute column carries this chapter's, and they meet
exactly at the shared-memory tile: four bits in DRAM, bf16 in SRAM.

### Takeaways

These close the attention thread of chapters 16 and 17.

1. A naive attention forward at the native context length of 65536 writes a 16 G fp32 score
   matrix per block per sequence — 128× the useful traffic — and reads it twice, 384× in total;
   the factor grows with the sequence length. Every kernel decision in this
   chapter is a memory decision first.
2. The online softmax makes tiling exact: running $(m, \ell, o)$ per query row, one rescale per
   key block, and never a score matrix in DRAM. FlashAttention changes what attention *moves*, not
   what it computes.
3. On SM120 the tile shape is derived, not chosen: a $64 \times 256$ fp32 output accumulator is
   128 registers/thread, which fixes $B_M = 64$ and leaves room for at most two 32 K KV stages in
   99 K of shared memory. $D = 512$ forces a split-$D$ scheme.
4. Chapter 7's latent makes $K = V$: one shared-memory tile serves both products, the value load
   disappears, and packing 64 heads of one token into the block rows is what lets chapter 10's
   per-token selection gather serve every head at once.
5. The learned sink is a denominator-only column: one $\operatorname{exp2}$ per row against a
   per-head parameter, never loaded from the cache.
6. CSA2's window, selected main entries and sink run in *one* online-softmax loop — window tiles
   first, then the gather over $k$ indices — so no LSE merge pass exists and the group-causal
   bound $j < (t+1)/m$ is the only extra mask.
7. Decode for one query moves $\approx 128 \times 512 \times 1 + k \times 288$ bytes per block;
   at the reference $k = 512$ that is 208 K — a window read of 64 K plus a selected read of
   144 K — against 0.625 M if the caches were bf16.
8. Scale granularity does the accuracy work in low-precision attention: FA3's own ablation is
   2.4e-2 → 9.3e-3 RMSE from block scales alone, with incoherent processing adding ~2% on top.
9. The cheapest correct place for a scale correction is a constant folded into the $\exp_2$
   argument — FA3's block scales, SageAttention2's static $\delta_P = 1/448$ and SageAttention3's
   two-level $1/2688$ are the same trick at increasing depth.
10. $P$ is a scale-encoding problem, not a range problem: every family lands it differently
    (static $1/448$, fixed $2^8$, two-level $1/2688$, affine "Direct-P", or nothing under QAT),
    and all that matters is landing the row in the format's sweet spot.
11. Training evidence is one-directional: MXFP4 $P/V$ training runs diverge (FA4-FP4), FP8 $P/V$
    trains, and the sensitive gradient products — $dP = dO V^\top$, the FA identity $P^\top dP =
    dO^\top O$ — are kept in higher precision by every design that trains.
12. Quantisation that a model is trained *through* (fake-quant, STE, scales detached as
    statistics) recovers accuracy that post-training quantisation loses; Attn-QAT's two backward
    fixes — re-quantise $P$ in the recompute and take $D$ from a high-precision-$P$ output — are
    the load-bearing details.
13. On a part where FP4 mma runs at ≈FP8 speed and 273 GB/s caps everything, four bits are a
    storage format: dequantise into the bf16 tile on load, compute in bf16. Production kernels
    (FlashInfer, FlashMLA, the reference model's own) agree; the indexer score, whose consumer
    only ranks, is the one place FP4 math pays.

The kernel side is settled for this repository: bf16 tiles, one loop, fake-quant at the model
level. What remains open is the backward for the gather and the group-causal mask, and the
long-context accuracy of the whole stack — nobody has published a RULER-style evaluation of FP4
attention, so the repository's own check is unavoidable.
