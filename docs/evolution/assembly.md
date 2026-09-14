---
title: "15 Assembly"
order: 15
---

# 15. Assembly: DeepSeek-V4.1

Every chapter of this book changed one part of the block and left the rest as it was. This chapter
puts the changes together. The result is one block of DeepSeek-V4.1, written so that each line can be
traced to the chapter that introduced it, followed by a ledger that totals the effect of all of them
on the reference model.

One V4.1 block for a token with residual state $X \in \mathbb{R}^{4 \times D}$ and incoming read
vector $\mathbf{p}_{\text{in}}$:

$$
\begin{aligned}
X &\leftarrow \operatorname{Engram}(X) &&\text{selected layers only (13)} \\
(\mathbf{p}_a, \mathbf{q}_a, C_a) &= \operatorname{mix}_a(X) &&\text{mHC coefficients (11)} \\
u &= \mathbf{p}_{\text{in}}^{\top} X &&\text{read, one-site delay (11)} \\
y &= \operatorname{CSA}_\ell\big(\operatorname{RMSNorm}(u)\big) &&\text{window + shared compressed + top-}k\ (7, 8, 9, 10) \\
X &\leftarrow \mathbf{q}_a\, y^{\top} + C_a^{\top} X &&\text{write (11)} \\
(\mathbf{p}_f, \mathbf{q}_f, C_f) &= \operatorname{mix}_f(X) \\
u &= \mathbf{p}_a^{\top} X \\
y &= \operatorname{MoE}_\ell\big(\operatorname{RMSNorm}(u)\big) &&\text{shared + routed SwiGLU experts (4, 6)} \\
X &\leftarrow \mathbf{q}_f\, y^{\top} + C_f^{\top} X \\
\mathbf{p}_{\text{out}} &= \mathbf{p}_f
\end{aligned}
$$

$\operatorname{CSA}_\ell$ contains a low-rank query and a single shared latent that is both key and
value, from chapter 7. It applies partial RoPE with two bases from chapter 3 and a per-head sink
from chapter 8. It reads the layer's own 128-token window plus the group's compressed table from
chapter 9, masked to the indexer's top-$k$ from chapter 10. The window cache is held in FP8 and the
compressed cache in FP4, both from chapter 12.

## Reading the block

The shape of the whole is the GPT-2 block of chapter 1. There are two sites, attention first and a
feed-forward layer second, each normalising its input and adding its output back. Every line beyond
that is one of the changes of chapters 2 to 13.

The residual state is a matrix of four rows rather than a vector, which is chapter 11. Each site
computes its coefficients from the current state but reads with the coefficients the previous site
computed, which is why the attention site reads with $\mathbf{p}_{\text{in}}$ and the feed-forward
site reads with $\mathbf{p}_a$. The block passes $\mathbf{p}_f$ on as $\mathbf{p}_{\text{out}}$, so
the chain continues into the next block, and only the first block and the final read need special
handling.

The engram lookup is placed before the coefficients are computed, because it writes into the streams
and its gate is a function of what the streams already hold. Only selected layers have it.

The two sublayers see a single $D$-vector each, normalised by RMSNorm from chapter 2 with no
re-centring and no bias. Neither knows that the residual has four rows. This is what allowed
chapters 2 to 10 and chapter 11 to be developed independently of each other, and it is why the block
can be read one line at a time.

## What changed since GPT-2

| GPT-2 | V4.1 | chapter |
|---|---|---|
| LayerNorm | RMSNorm | 2 |
| learned positions | partial RoPE, two bases | 3 |
| GELU MLP | SwiGLU experts, 1 shared + top-$k$ routed | 4, 6 |
| per-head $k$, $v$ in bf16/fp32 | one shared latent per token, FP8 / FP4 | 5, 7, 12 |
| full causal attention | 128-token window + compressed shared table + top-$k$ | 8, 9, 10 |
| single residual | 4 streams, doubly stochastic mixing, one-site delay | 11 |
| none | hashed n-gram memory | 13 |
| next-token CE | next-token CE, plus post-training draft heads | 14 |

## The running ledger

The reference model is the dense 80-block, $D = 8192$ model of the preface. Each row applies the
change named in it to the row above, so the last row is the whole book applied at once. Cache figures
are per token over all 80 blocks, and the long-context column is at $S = 131072$.

| after | chapter | parameters | FLOPs per token from weights | cache per token | cache at 131072 |
| --- | --- | --- | --- | --- | --- |
| dense block, multi-head attention, bf16 | 1, 5 | 65B | 130 G | 2.6 MB | 344 GB |
| grouped-query, $H_{kv} = 8$ | 5 | 65B | 130 G | 328 KB | 43 GB |
| routed experts, 256 + 1, top-8 | 6 | about 1T, 36B active | about 130 G | 328 KB | 43 GB |
| multi-head latent attention, $d_c = 512$, $d_r = 64$ | 7 | unchanged | unchanged | 92 KB | 12 GB |
| one shared latent of width 512 | 7 | unchanged | unchanged | 82 KB | 10.7 GB |
| FP4 latents with a scale per 16 elements | 12 | unchanged | unchanged | 23 KB | 3.0 GB |
| compressed and shared, the paper's schedule | 9 | unchanged | unchanged | 1.9 KB plus a fixed window | about 248 MB |

Three rows need a word of explanation. The mixture-of-experts row multiplies the total parameter
count by about fifteen while leaving the arithmetic where it was, because only 9 of the 257 experts
in a block run for a given token. It is the one change in the book that improves quality by spending
memory rather than by saving it. The attention rows are marked unchanged in the parameter and FLOP
columns because the projections of the different attention variants differ from each other by a few
percent of the model total, which is below the resolution of this ledger. The last row is not a pure
per-token figure, and it moves to the paper's own 40-layer configuration rather than the 80-block
reference: the encoder's three Full layers pool at $m = 2$ and the decoder's five sources (one Full,
four Reindex) store per token, so the global table is $(3/2 + 5) \times 288\ \text{B} \approx 1.9$
KB per token. Without sharing or compression the same 40 layers would hold 11.5 KB per token in
FP4. Beside the global table every layer keeps a 128-token sliding window in FP8, which is a fixed
2.6 MB for the whole sequence whatever its length. At 131072 tokens the global part is about
245 MB, the windows 2.6 MB.

The FLOP column tracks only the weights. Attention arithmetic is separate and it is what makes long
context expensive. As the preface sets out, the scores and the weighted sum cost about $4 S D$ per
block, which is 10.7 GFLOPs per token at the training length of 4096 and 344 GFLOPs per token at
131072, at which point attention costs more than every weight in the model. The selection of chapter
10 is what removes this. Reading $k = 512$ entries per query in place of the whole context replaces
$S$ by $k$ in that formula and gives about 1.3 GFLOPs per token at any length. The compressor of
chapter 9 reduces the number of entries the indexer has to score in the first place.

Read together, the ledger says that the cache per token fell by about four orders of magnitude and
the attention arithmetic at the long-context target by about two, while the arithmetic per token from
the weights stayed near where it was and the parameter count rose by a factor of fifteen. That is the
shape of the period this book covers. Compute per token held roughly constant, memory per token
attacked from every direction, and the parameters freed by both spent on capacity that only some
tokens use.

## Where each piece lives in owlet1

| chapter | file under `src/mzoo/archs/owlet1/` |
|---|---|
| 2, 3 | `norm_rope.py` |
| 4, 6 | `moe.py` |
| 7, 8 | `attention.py` |
| 9 | `cache.py` (compressor), `attention.py` (sharing) |
| 10 | `indexer.py` |
| 11 | `decoder.py`, `mHC.md` |
| 12 | `quant.py` |
| 13 | `engram.py` |
| all | `config.py` for the layer schedule, `model.py` for the smoke config |

The known gaps between this code and the report are listed in the owlet1 `README.md` under
"Paper vs. this code" and "Open issues".

## A reading order for the code

Every module has a sibling `.md` file carrying the mathematics for that module, so the two can be
read side by side. A useful order is the following.

Start with `README.md`, whose module map and forward-flow section walk one token through the smoke
configuration of 6 layers at $D = 256$ with the shapes written out. Then read `model.py`, which is
the entry point and is short, and the layer-schedule part of `config.py`, which decides which layers
are which. With the schedule in mind, read `decoder.py` alongside `mHC.md` for the residual and the
layer body, which is the outermost structure and corresponds to the equation at the top of this
chapter.

The attention path is next and is the largest part. Read `attention.py` with `attention.md` for the
low-rank query, the shared latent and the sink, then `cache.py` for the window ring and the
compressor of chapter 9, then `indexer.py` for the selection of chapter 10. After that the remaining
modules can be read in any order, since none depends on the others: `moe.py` for chapters 4 and 6,
`norm_rope.py` for chapters 2 and 3, `quant.py` for chapter 12 and `engram.py` for chapter 13.

Finish with `GAPS.md` and `TRAINING.md`. The first records where the code departs from the report and
why, which is the honest version of the correspondence table above. The second covers running it.

## What this book did not cover

The book followed one thread, the arithmetic and memory of the transformer block, and left several
others alone. It said nothing about tokenisation, about the data a model is trained on, or about the
optimiser, the learning-rate schedule and the initialisation, all of which matter as much for the
final quality as anything in these chapters. It said nothing about how a model of this size is
distributed across many accelerators, and the parallelism strategy is often what decides whether a
design is trainable at all. Chapter 12 described number formats but not the kernels that implement
them. Nothing here covers post-training, instruction tuning or alignment, which is where a
pre-trained model becomes a usable one, and nothing covers evaluation.

Within the architecture itself, one branch was treated only in outline. Chapter 8 named the two
routes to long context and this book followed one of them, the sparse and compressed softmax
attention of chapters 9 and 10, because it is the route DeepSeek-V4.1 takes. The other route replaces
most softmax layers with recurrent linear-attention layers that keep a fixed-size state per sequence,
as Qwen3-Next and Kimi Linear do, and it reaches the same goal by a different argument. A reader who
has followed the cache ledger through this book has what is needed to work out why.
