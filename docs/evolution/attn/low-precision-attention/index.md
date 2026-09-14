---
title: "17 Low-precision attention"
order: 17
---

# 17. Low-precision attention

Chapters 5 to 10 settled what a long-context attention computes and chapter 16 settled how the
bf16 kernel tiles it. Neither chapter lets four bits into the arithmetic. A parallel line of work
does, and its variants are easy to confuse because they all quantise the same three tensors — the
queries and keys that produce the scores, the probabilities, and the values. They differ in where
the scales live, and the differences are exactly where the hardware is. This chapter presents each
variant with its equation and its motivation, and ends with the per-operand recipe the CSA2 kernels
of this repository follow, which is also the closing takeaways of the attention thread.

The running example is one attention head: $S = QK^\top/\sqrt{d}$, $P = \operatorname{softmax}(S)$,
$O = PV$, with tiles of $B_M$ queries by $B_N$ keys of width $d$. The hardware anchors are the
chapter 16 table: on SM120, FP4 matrix multiplication runs at roughly FP8 speed, and 273 GB/s caps
everything, so the chapter's question — which operands deserve four bits in *arithmetic* — has a
different answer than the storage question of chapter 12. The evidence, with sources and file
paths, is in `src/mzoo/layers/attn/fp_attn_survey.md` and the longer `fp_attn_notes.md`.

### Roadmap

1. **Block scales before rotation** — FlashAttention-3's FP8 established that granularity does the
   accuracy work and that rotating the operands adds little, which fixes the vocabulary for
   everything after.
2. **Integer scores, float probabilities** — SageAttention carries the split "cheap scores,
   accurate second product" into INT8/INT4 and contributes the static probability scale and
   two-level accumulation.
3. **Four bits everywhere and the two-level P scale** — SageAttention3 shows both what NVFP4 buys
   and the one scale-encoding problem FP4 has that FP8 does not.
4. **Training through the quantisation** — Attn-QAT keeps the arithmetic bf16, trains against the
   fake-quantised values, and supplies the two backward fixes without which gradients explode.
5. **What does not survive training** — the FP4 FlashAttention-4 divergence result and the e5m2
   gradient lesson.
6. **Dequantise on load** — the merged reading: on a bandwidth-bound part, four bits are a storage
   format; the recipe table and the takeaways close the attention thread.

Sections:

1. [The variants](variants.md): roadmap items 1–6, ending with the recipe table and the takeaways
   for chapters 16 and 17.
