# The FlashAttention-2 template on SM120

Chapter 8 stated what FlashAttention removes: the $S \times S$ score matrix never reaches memory.
This section shows the mechanism, because every later kernel in this chapter is a modification of
it, and because on SM120 the template's tile sizes are set by register arithmetic that is worth
doing once.

## Why the naive kernel loses

A naive attention forward materialises $P = \operatorname{softmax}(Q K^\top / \sqrt{d})$ and then
computes $O = P V$. At the reference training length with the chapter 7 latent, the score matrix
for one sequence is $S \times S$ per block: $4096^2 \times 4$ bytes in fp32 is 64 MB, written once
and read twice, per block, per sequence. The useful work — the two products — moves
$2 \cdot S \cdot 512 \cdot 2$ bytes of $Q$, $K$ and $V$; at $S = 4096$ that is 8 MB. The
intermediates dominate traffic by a factor of eight already at training length, and the factor
grows linearly in $S$. On a part with 273 GB/s of bandwidth this is the entire runtime.

## Tiling and the online softmax

FlashAttention splits $Q$ into row blocks of $B_M$ queries and $K, V$ into column blocks of $B_N$
keys, and walks the key blocks keeping one block of scores in registers or shared memory. The
obstacle is the softmax denominator: it needs a max and a sum over *all* keys before any output
element is final. The online softmax removes it by keeping running statistics per query row.
Writing $S_{ij}$ for the scores of query block $i$ against key block $j$, and $m$, $\ell$, $o$ for
the running max, denominator and unnormalised output of one query row:

$$
m^{(j)} = \max\!\big(m^{(j-1)},\ \max_n S_{i,n}\big), \qquad
\ell^{(j)} = e^{m^{(j-1)} - m^{(j)}} \ell^{(j-1)} + \sum_n e^{S_{i,n} - m^{(j)}},
$$

$$
o^{(j)} = e^{m^{(j-1)} - m^{(j)}}\, o^{(j-1)} + \sum_n e^{S_{i,n} - m^{(j)}}\, v_n,
\qquad
o = \frac{o^{(J)}}{\ell^{(J)}} .
$$

Read that back in words. Each key block rescales everything the row has accumulated so far by the
factor its new maximum demands, then adds its own contribution at the new scale. Nothing larger
than a $B_M \times B_N$ tile of scores ever exists. The output is exact — the rescaling algebra is
identical to computing the full softmax — and memory for intermediates falls from $O(S^2)$ to
$O(S)$. Chapter 8 stated the effect; this is the mechanism that produces it.

FlashAttention-2's contribution over version 1 is the work partitioning: outer loop over query
blocks (each held resident for the whole key walk), inner loop over key blocks, and enough thread
blocks in flight that the key walk of one block overlaps the loads of another. On a warp-level-mma
part this is still the right shape; the Hopper and SM100 versions move the same algebra onto
wgmma/tcgen05 with warpgroups, which SM120 does not have.

## Choosing the tile sizes by arithmetic

The output accumulator fixes everything else. A block computing $B_M$ queries at latent width $D$
holds an fp32 accumulator of $B_M \times D$ values distributed across threads. With 128 threads and
$D = 256$:

$$
\frac{64 \times 256 \times 4\ \text{bytes}}{128\ \text{threads}} = 512\ \text{bytes/thread}
= 128\ \text{registers/thread},
$$

half of a thread's 255-register budget before scores, pointers and pipeline bookkeeping. So
$B_M = 64$ at $D = 256$, and the $B_N \times D$ key and value tiles in shared memory are
$64 \times 256 \times 2 = 32$ KB each in bf16. Two stages of one tile plus the resident $Q$ block
(32 KB) is 96 KB against the 99 KB limit — one stage, or two if $Q$ and one buffer alias. This is
why the design document fixes $B_M = B_N = 64$, 128–256 threads, and head dims $\{64, 96, 128,
256\}$: at $D = 512$ the accumulator alone wants 256 registers/thread and the kernel must split the
latent across blocks (a split-$D$ scheme), which is exactly what the released DSV4.1-Flash kernels
do and what this repository's kernels defer by declaring $D = 512$ out of scope.

## The shared latent: $K = V$ and heads in the row dimension

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
heads (one top-$k$ per token), the 1-token block is the only layout in which a gather serves all
64 heads at once.

RoPE needs a remark. MLA-style kernels split the latent into a no-RoPE part and a RoPE part and run
two GEMMs per tile to avoid rotating keys on the fly. V4.1 rotates the latent in place, so the
kernel takes a plain slice of $D$ and there is one GEMM per product — simpler, and the rotation
stays outside the kernel in any case (applied to $Q$ and to the cache on write).

## The sink inside the softmax

The learned sink of chapter 8 is one extra column in the softmax that is present for every query
and never loaded from the cache. Its probability only enters the denominator, since there is no
value to weight:

$$
\ell \leftarrow \ell + e^{s_h \cdot \text{scale} - m}, \qquad h = 1..H ,
$$

added once per row after the key walk (or at any point, since the running max handles ordering).
In the kernel it is one extra $\operatorname{exp2}$ per row with the per-head sink value
$s_h$; the reference implementation is `v002_fa2_fwd.py` in `src/mzoo/layers/attn/dense_attn/`,
against which the windowed and selected kernels of the next section are tested.

The template is now settled: one query block resident, one key-value stream, one set of running
statistics, and a mask. Nothing in the rest of the chapter changes it — the next section only
changes what the stream contains.
