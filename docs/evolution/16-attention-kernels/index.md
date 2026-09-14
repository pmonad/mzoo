# 16. Attention kernels and low precision

Chapters 7 to 10 settled what a long-context attention computes: a shared latent, a window, a
compressed table, a selected subset of it. Chapter 12 settled the formats it is stored in. What
neither chapter says is how any of it runs on a GPU, and the gap is not cosmetic. Two
implementations of the same mathematics can differ by an order of magnitude in time, and the
difference is made entirely of memory decisions. This chapter covers the kernel: the tile schedule
every design inherits, and the one kernel that CSA2's three attention sources collapse into. The
next chapter asks which of the operands may leave bf16, and answers it as a family of variants.

The chapter assumes the reader knows CUDA: thread blocks, warp-level `mma` instructions, shared
memory and registers. The reference GPU is the one this repository develops on, GB10 (compute
capability 12.1, the SM120 family), because its constraints are sharp enough to force the design
and are worth stating once. Where a claim about the model side carries a number, the reference
model of chapter 1 supplies it; where a claim about the chip does, this table does:

| property | value | consequence |
| --- | --- | --- |
| tensor cores | warp-level `mma.sync` only (no wgmma/tcgen05, no tensor memory, no TMA multicast) | the FlashAttention-2 template, not the Hopper or SM100 ones |
| shared memory | 99 KB per block | small tiles; at most 2–3 pipeline stages |
| registers | 64 K per SM, 255 per thread | the output accumulator sets the tile shape |
| FP4 arithmetic | ≈ FP8 throughput in practice (measured within ~1.2×) | FP4 is a *storage* format here, not a compute format |
| DRAM bandwidth | 273 GB/s | decode and cache reads are bandwidth-bound |

The worked kernels live in `src/mzoo/layers/attn/`: the design document `csa2_attn_design.md`
grows them step by step, each step a numbered version file with a reference and a test beside it,
and `fp_attn_survey.md` with `fp_attn_notes.md` hold the low-precision evidence with sources.

### Roadmap

1. **Why the naive kernel is memory-bound** — the score matrix, not the arithmetic, sets the
   runtime, which is the defect FlashAttention removes and the reason every later choice is a
   memory decision first.
2. **The online softmax and the tile sizes** — running statistics make the removal exact, and on
   SM120 the register file, not preference, picks the tile shape.
3. **The shared latent makes $K = V$** — chapter 7's latent halves the kernel's shared-memory
   traffic and fixes the block layout that chapter 10's shared selection depends on.
4. **One kernel for three sources** — window, selected entries and sink share one running softmax,
   which is why CSA2 needs no merge pass and why the gather serves all 64 heads at once.

Sections:

1. [The FlashAttention-2 template on SM120](fa2.md): roadmap items 1–3.
2. [One kernel for CSA2](csa2.md): roadmap item 4, closing with the precision question that
   chapter 17 takes up.
