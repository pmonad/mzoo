---
title: "The FA2 template on SM120"
order: 1
---

## The FlashAttention-2 template on SM120

Chapter 8 stated what FlashAttention removes: a score matrix the size of the sequence length
squared never reaches memory. This section shows the mechanism, because every later kernel in this
chapter is a modification of it, and because on SM120 the template's tile sizes are set by
register arithmetic that is worth doing once.

### The four CUDA facts the rest of the chapter uses

A GPU is a set of streaming multiprocessors (SMs) — the reference part has 48. A program launches
a grid of thread blocks; the CUDA documentation calls a thread block a *CTA* (cooperative thread
array). Each SM runs one or more CTAs, and each CTA is cut into *warps* of 32 threads that execute
in lockstep: one instruction, issued once, for all 32 lanes.

Three levels of memory matter, and their sizes run inverse to their speed. DRAM is off-chip: the
part moves 273 GB/s, and every byte the kernel touches there is a byte of runtime. Shared memory
is on-chip SRAM private to a CTA: 99 K on this part, fast enough that the kernel's inner loop
should never leave it. Registers are per-thread: at most 255 of them (64 K per SM), the fastest
storage of all, and the scarce resource that ends up choosing the tile sizes below.

The arithmetic runs on tensor cores. A warp-level `mma` (matrix-multiply-accumulate) instruction
takes small, fixed-shape fragments of two operand matrices and an accumulator, all held across the
warp's registers, and performs one step of a matrix product in one instruction. Tensor cores exist
because nearly all of a transformer's arithmetic is multiply-add; dedicating silicon to that one
operation buys roughly an order of magnitude over doing it with scalar instructions. On SM120 the
`mma` is warp-level (`mma.sync`); the warpgroup and tensor-memory forms of Hopper and SM100 are
absent, which is why this chapter builds on the FlashAttention-2 template rather than the newer
ones.

Finally, pipelines. Computing on a tile takes time, and so does fetching the next tile from DRAM;
doing them one after the other wastes half the chip. A *pipeline stage* is one shared-memory buffer
holding one in-flight tile; *double buffering* is two stages, so that the `mma` reads the full
buffer while the loads fill the empty one. With 99 K of shared memory, at most two to three
stages of any reasonable tile fit — the constraint the introduction's table lists as a
consequence.

### Why the naive kernel loses

A naive attention forward materialises $P = \operatorname{softmax}(Q K^\top / \sqrt{d})$ and then
computes $O = P V$. At the native context length of 65536 with the chapter 7 latent, the score
matrix for one sequence, per block, is 65536² fp32 values: $65536^2 \times 4$ bytes is 16 G,
written once and read twice — once by the softmax, once by the second product. The useful work —
the two products — moves $2 \cdot 65536 \cdot 512 \cdot 2 = 128$ M of $Q$, $K$ and $V$. The
write alone is 128× the useful traffic; the two reads double it again — 48 G of intermediate
traffic against 128 M of useful, 384× in all. Counted exactly, the ratio is $3 S / D$, so it grows
linearly in the sequence length — at the
extended target of 1,048,576 the score matrix alone is 4 T per block per sequence, which no part
holds. On a part with 273 GB/s of bandwidth, streaming the 48 G of intermediates takes 0.19 s per
block per sequence — 7.6 s for the model's 40 blocks, before any useful arithmetic runs. This is
the entire runtime.

### Tiling and the online softmax

FlashAttention splits $Q$ into row blocks of $B_M$ queries and $K, V$ into column blocks of $B_N$
keys, and walks the key blocks keeping one block of scores in registers or shared memory. The
obstacle is the softmax denominator: it needs a max and a sum over *all* keys before any output
element is final. The online softmax removes it by keeping running statistics per query row.
Writing $S_{ij}$ for the scores of query block $i$ against key block $j$, $m^{(j)}$ for the
running maximum of a row after key block $j$, $\ell$ for the running softmax denominator (the sum
the softmax will divide by), and $o$ for the unnormalised output of one query row:

$$
m^{(j)} = \max\!\big(m^{(j-1)},\ \max_n S_{i,n}\big), \qquad
\ell^{(j)} = e^{m^{(j-1)} - m^{(j)}} \ell^{(j-1)} + \sum_n e^{S_{i,n} - m^{(j)}},
$$

$$
o^{(j)} = e^{m^{(j-1)} - m^{(j)}}\, o^{(j-1)} + \sum_n e^{S_{i,n} - m^{(j)}}\, v_n,
\qquad
o = \frac{o^{(J)}}{\ell^{(J)}} .
$$

The same update as the kernel writes it — one running $(m, \ell, o)$ per query row of the resident
query block:

$$
\begin{aligned}
&\ //\ \text{init: one running triple per query row of block } i \\
&\ m \leftarrow -\infty, \qquad \ell \leftarrow 0, \qquad o \leftarrow 0 \\[4pt]
&\ //\ \text{inner loop: walk the key blocks, one } B_N \text{-wide tile at a time} \\
&\ \text{for key block } j: \\
&\ \quad T \leftarrow Q_i\, K_j^\top \cdot \text{scale}
\qquad \text{// the } B_M \times B_N \text{ score tile, in registers} \\
&\ \quad m' \leftarrow \max\!\big(m,\ \operatorname{rowmax}(T)\big) \\
&\ \quad \ell \leftarrow e^{\,m - m'}\,\ell + \operatorname{rowsum}\!\big(e^{\,T - m'}\big)
\qquad \text{// rescale old sum, add this block's} \\
&\ \quad o \leftarrow e^{\,m - m'}\, o + e^{\,T - m'}\, V_j
\qquad \text{// rescale old output, add this block's} \\
&\ \quad m \leftarrow m' \\[4pt]
&\ //\ \text{finalise} \\
&\ O_i \leftarrow o\, /\, \ell
\end{aligned}
$$

Read that back in words. Each key block rescales everything the row has accumulated so far by the
factor its new maximum demands, then adds its own contribution at the new scale. Nothing larger
than a $B_M \times B_N$ tile of scores ever exists. The output is exact — the rescaling algebra is
identical to computing the full softmax — and memory for intermediates falls from quadratic to
linear in the sequence length. Chapter 8 stated the effect; this is the mechanism that produces it.

FlashAttention-2's contribution over version 1 is the work partitioning: outer loop over query
blocks (each held resident for the whole key walk), inner loop over key blocks, and enough thread
blocks in flight that the key walk of one block overlaps the loads of another. On a warp-level-mma
part this is still the right shape; the Hopper and SM100 versions move the same algebra onto
wgmma/tcgen05 with warpgroups, which SM120 does not have.

### Choosing the tile sizes by arithmetic

The output accumulator fixes everything else. A block computing $B_M$ queries at latent width $D$
holds an fp32 accumulator of $B_M \times D$ values distributed across threads. With 128 threads and
$D = 256$:

$$
\frac{64 \times 256 \times 4\ \text{bytes}}{128\ \text{threads}} = 512\ \text{bytes/thread}
= 128\ \text{registers/thread},
$$

half of a thread's 255-register budget before scores, pointers and pipeline bookkeeping. So
$B_M = 64$ at $D = 256$, and the $B_N \times D$ key and value tiles in shared memory are
$64 \times 256 \times 2 = 32$ K each in bf16. Two stages of one tile plus the resident $Q$ block
(32 K) is 96 K against the 99 K limit — one stage, or two if $Q$ and one buffer alias. This is
why the design document fixes $B_M = B_N = 64$, 128–256 threads, and head dims $\{64, 96, 128,
256\}$: at $D = 512$ the accumulator alone wants 256 registers/thread and the kernel must split the
latent across blocks (a split-$D$ scheme), which is exactly what the reference model's own released
kernels do and what this repository's kernels defer by keeping $D = 512$ out of the smoke scope.

### The shared latent: $K = V$ and heads in the row dimension

Chapter 7's latent makes one change to the template that halves the shared-memory traffic: the key
and the value are the *same vector*. There is one KV head for 64 query heads (multi-query
attention), and the attention reads the latent once and uses it in both products. The natural
layout packs the 64 query heads of one token into the $B_M$ rows — a block of $1 \text{ token}
\times 64 \text{ heads}$ — so

$$
S = Q\, c^\top_{[\,t-127..\,t\,]}, \qquad O = P\, c_{[\,t-127..\,t\,]},
$$

consume one shared-memory tile of latents. Read the pair back in words: the same tile is
$K$ in the first product and $V$ in the second, and the value load disappears from the kernel. The
causal mask becomes a per-row property derived from the row's token rather than a per-block one,
which is free in registers. It also sets up chapter 10: when the selected key set is shared across
heads (one top-$k$ per token), the 1-token block is the only layout in which a gather — loading
list-indexed rows instead of a contiguous range — serves all 64 heads at once.

RoPE needs a remark. MLA-style kernels split the latent into a no-RoPE part and a RoPE part and run
two GEMMs per tile to avoid rotating keys on the fly. The reference model rotates the latent in
place, so the kernel takes a plain slice of $D$ and there is one GEMM per product — simpler, and
the rotation stays outside the kernel in any case (applied to $Q$ and to the cache on write).

### The sink inside the softmax

The learned sink of chapter 8 is one extra column in the softmax that is present for every query
and never loaded from the cache. Its probability only enters the denominator, since there is no
value to weight:

$$
\ell \leftarrow \ell + e^{\sigma_h \cdot \text{scale} - m}, \qquad h = 1..H ,
$$

added once per row after the key walk (or at any point, since the running max handles ordering).
In the kernel it is one extra $\operatorname{exp2}$ per row with the per-head sink value
$\sigma_h$; the reference implementation is `v002_fa2_fwd.py` in `src/mzoo/layers/attn/dense_attn/`,
against which the windowed and selected kernels of the next section are tested.

The template is now settled: one query block resident, one key-value stream, one set of running
statistics, and a mask. Nothing in the rest of the chapter changes it — the next section only
changes what the stream contains.
