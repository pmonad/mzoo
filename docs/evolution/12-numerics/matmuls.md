# Low-precision matmuls

The FP8 recipe of the previous section is one instance of a general construction: run a matmul on
quantised operands and correct for the quantisation afterwards. The construction is the same for a
Linear layer, an attention score product, or an indexer scoring pass, and every variant of it in
this book — DeepSeek's GEMMs, the FP8 attention kernels, the FP4 indexer — is the same three
decisions. State them once here; chapter 17 lists the attention-specific instances.

## The anatomy

Quantise each operand by blocks, multiply the codes on the tensor cores, and put the scales back:

$$
\hat A_{ib} = \operatorname{round}\!\Big(\frac{A_{ib}}{s^A_{i,\cdot}}\Big), \quad
\hat B_{bj} = \operatorname{round}\!\Big(\frac{B_{bj}}{s^B_{\cdot,j}}\Big), \quad
AB \approx \operatorname{diag}\big(s^A\big)\, \hat A \hat B\, \operatorname{diag}\big(s^B\big),
$$

where $s^A$ and $s^B$ are one scale per block, the block shape is a free parameter, and the only
exactness in the whole pipeline lives in the correction: the scales are applied to *partial sums*
in the wide accumulator, so they introduce no rounding of their own. Everything downstream of the
three decisions below is engineering.

## Decision one: the scale granularity

The scale exists to spend the format's range on the block's own spread instead of the tensor's.
The hierarchy, coarse to fine:

1. **Per-tensor** — one scale for the whole operand. Free to store, and defeated by outliers: one
   large value sets the scale and everything else rounds to zero. This is why LLM.int8() found
   eight-bit matmuls failing at scale, and why nobody trains with it.
2. **Per-block along the reduction** — V3's $1 \times 128$ activations and $128 \times 128$
   weights; V4.1's caches at 16 and 32 elements. The block lies along the dot product, so one
   scale covers a contiguous run of the accumulation and is applied once to a partial sum.
3. **Per-tile of the kernel** — FlashAttention-3 gives one scale to each $B_M \times d$ query tile
   and $B_N \times d$ key tile it loads. Moving from per-tensor to per-tile took FA3's FP8 error
   from $2.4 \times 10^{-2}$ to $9.3 \times 10^{-3}$, and the further step of rotating the operands
   (incoherent processing, a Hadamard with random signs that leaves $QK^\top$ unchanged) added only
   $9.1 \times 10^{-3}$. Granularity does the work; rotation is a refinement.
4. **Per-thread / per-mma-fragment** — the finest that exists. A warp-level mma instruction gives
   each thread ownership of specific rows and columns of the output tile, so a scale per fragment
   can be applied to values already in that thread's registers, with no cross-lane traffic.
   SageAttention2 uses this for its INT4 attention scores: 32× finer than per-tile at zero cost.

The rule the hierarchy encodes: make the scale as fine as the place where the correction multiply
already happens. Finer than that costs traffic; coarser wastes range.

## Decision two: where the correction goes

The scale product $s^A s^B$ must be multiplied back in somewhere. Three placements cover every
implementation:

1. **Into the accumulator.** Applied to the fp32 partial sum per block, as V3 does every $N_C = 128$
   elements along the reduction. The natural placement for GEMMs.
2. **Folded into a constant of the surrounding arithmetic.** If one of the scales is constant over
   a row — as a probability scale is under the online softmax, where the row maximum is 1 — it can
   be folded into the exponent of the $\exp_2$: one addition per element instead of one multiply.
   FA3's block scales, SageAttention2's static $1/448$ and SageAttention3's two-level $1/2688$ are
   this trick at increasing depth.
3. **Skipped entirely: dequantise on load.** When the operand is a *stored* tensor read many
   times, the cheapest correction is not to quantise the arithmetic at all — expand the codes to
   bf16 in registers or shared memory on load and run a full-precision matmul. FlashInfer and
   FlashMLA both serve FP8/FP4 caches this way, and V4.1's main cache is dequantised before
   attention for exactly this reason. On a part where FP4 matmul throughput is barely above FP8,
   the bytes saved in storage are collected either way, and the arithmetic stays exact.

## Decision three: the accumulation width

The tensor-core accumulator is wider than the operands but not necessarily fp32. An FP8 mma
partial has around 22 effective mantissa-plus-exponent bits, which is acceptable over a small run
of the reduction and must be promoted before the run gets long. The promotion interval is a real
parameter — V3 promotes every $N_C = 128$ elements, four mma instructions, the minimum its
pipeline allows — and getting it wrong shows up as a slow drift in the loss rather than as an
obvious failure. FP4's two-level scheme in chapter 17 is the same split, one level deeper.

## Training uses the construction, not the kernel

Every public kernel that quantises the matmul operands natively — FA3's FP8, both SageAttention
families, the FP4 kernels — is forward-only. Training runs the same construction as *fake
quantisation*: quantise and dequantise the operands in the forward pass, run bf16 matmuls on the
rounded values, and let the straight-through estimator of the next section carry the gradient. The
matmul the model trains against is bit-identical to the one the inference kernel will compute only
if the scales and grids match exactly, which is why chapter 17 insists on reproducing the exact
rounding rules of the target format rather than calling a stock quantiser.
