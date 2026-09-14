---
title: "Attention kernels: implementation notes"
---

# Attention kernels: implementation notes

Parking lot for problems met while implementing the kernels of this chapter
(`src/mzoo/layers/attn/`). One short section per problem: what went wrong,
what the measurement showed, how it was fixed. No expansion here; the full
write-up lands in the chapter later.

## latent_attn backward: 2x slower than SDPA

**Problem.** `latent_attn` (shared K==V latent, heads packed into the tile
rows) matches or beats SDPA in the forward (D128 H64 S4096 causal: 2.90 ms vs
3.35 ms), but fwd+bwd runs 33.2 ms against SDPA's 15.0 ms with the latent
expanded to 64 heads (D64: 14.9 vs 7.3 ms). `dense_attn`'s backward on the
same template is only ~1.2x behind SDPA, so the gap is specific to the MQA
layout. First suspects: the fp32 `atomic_add` scatter of dQ (about a third of
the main kernel in a quick ablation), a KV-owner grid that is tiny because all
H heads are rows of one query tile (32 CTAs on 48 SMs before the query-loop
split was added), and a bwd config table keyed on head dim only and tuned at
H=64.
