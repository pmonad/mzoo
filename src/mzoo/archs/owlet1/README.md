# owlet1

A split-for-study fork of the frozen `dsv4` baseline (vendored from the unmerged HF PR
[transformers#48721](https://github.com/huggingface/transformers/pull/48721), tracking
`deepseek-ai/DeepSeek-V4.1-Flash`). One concern per file, each with a sibling `.md` carrying the maths.
Class names stay `DeepseekV41*` so `diff -r ../dsv4 .` stays clean. A ~28.5M-param smoke config
(6 layers, `D=256`) trains on TinyStories; paper references below are to
`../dsv4/DeepSeek_V41_Tech_Report.pdf`.

## Module map

| file | covers | doc |
|---|---|---|
| `__init__.py` | re-exports (`build`, configs, model classes, cache, engram) | [__init__.md](__init__.md) |
| `model.py` | `build(tok, seq_len, ...)` — the harness entry point, smoke-config knobs | [model.md](model.md) |
| `config.py` | `DeepseekV41TextConfig` / composite config, layer-schedule auto-derivation | [config.md](config.md) |
| `norm_rope.py` | RMSNorm (weighted / unweighted), two-base interleaved-pair RoPE | [norm_rope.md](norm_rope.md) |
| `quant.py` | fake FP4 (E2M1) / FP8 (E4M3) block quant, ue8m0 + e4m3 scales | [quant.md](quant.md) |
| `cache.py` | CSA2 cache (window ring + compressed branch) and the compressor | [cache.md](cache.md) |
| `indexer.py` | sparse selector: scores, candidate blocks, `topk_bias` | [indexer.md](indexer.md) |
| `attention.py` | CSA2 block: low-rank Q, shared K=V latent, per-head sink, grouped output | [attention.md](attention.md) |
| `moe.py` | `sqrtsoftplus` / `noaux_tc` router, SwiGLU expert, shared+routed block | [moe.md](moe.md) |
| `engram.py` | n-gram hash conditional memory (disabled: `engram_layer_ids=[]`) | [engram.md](engram.md) |
| `decoder.py` | mHC residual, decoder layer, text backbone, causal-LM head + loss | [decoder.md](decoder.md), [mHC.md](mHC.md) |

## Forward flow

Notation: $B=8$, $S=512$, $D=256$, $hc=4$ streams, $H=4$ heads of width $d=64$, $V=32000$. Equations are
written for one token position; the batch and sequence axes are omitted. The residual state of a token
is a matrix $X \in \mathbb{R}^{hc \times D}$ whose rows $x_1, \dots, x_4$ are the four hyper-connection
streams, stored as `[B, S, hc, D]`. The components (attention, MoE, norms, quantisation) are documented
in their own `.md` files; this section follows the data through them. The read and write operations
used below are defined in [mHC.md](mHC.md).

### 1. Input

$$
e = E[\text{input\_id}] \in \mathbb{R}^{D}, \qquad
X^{(0)} = \begin{bmatrix} e \\ e \\ e \\ e \end{bmatrix} \in \mathbb{R}^{hc \times D}, \qquad
\mathbf{p}^{(0)} = (1, 0, 0, 0).
$$

Two objects enter block 0: the residual state $X^{(0)}$ and a read vector $\mathbf{p}^{(0)}$. A sublayer
never receives $X$ directly. It receives the weighted sum $\sum_k p_k\, x_k \in \mathbb{R}^{D}$, and
$\mathbf{p}$ is the weight vector for that sum. Every later read vector is computed by the network; the
first one is fixed to select stream 0. Since all four rows equal $e$ at this point, the value read is $e$
regardless of the choice.

### 2. One block

Block $\ell$ receives $(X, \mathbf{p}_{\text{in}})$ and returns $(X, \mathbf{p}_{\text{out}})$. It contains
two sublayers, attention then MoE, each wrapped in the same read / sublayer / write pattern. The
coefficients $(\mathbf{p}, \mathbf{q}, C)$ of a site are computed from $X$ at the start of the site;
$\mathbf{q}$ and $C$ are used by that site's write, while $\mathbf{p}$ is passed on as the read vector of
the *next* site.

$$
\begin{aligned}
&\textbf{attention site} \\
(\mathbf{p}_a, \mathbf{q}_a, C_a) &= \operatorname{mix}_{a}(X) \\
u &= \mathbf{p}_{\text{in}}^{\top} X &&\in \mathbb{R}^{D} \quad \text{read with the incoming vector} \\
y &= \operatorname{Attn}_\ell\!\big(\operatorname{RMSNorm}(u)\big) &&\in \mathbb{R}^{D} \\
X &\leftarrow \mathbf{q}_a\, y^{\top} + C_a^{\top} X &&\in \mathbb{R}^{hc \times D} \quad \text{write} \\[6pt]
&\textbf{MoE site} \\
(\mathbf{p}_f, \mathbf{q}_f, C_f) &= \operatorname{mix}_{f}(X) \\
u &= \mathbf{p}_a^{\top} X &&\quad \text{read with the attention site's vector} \\
y &= \operatorname{MoE}_\ell\!\big(\operatorname{RMSNorm}(u)\big) \\
X &\leftarrow \mathbf{q}_f\, y^{\top} + C_f^{\top} X \\[6pt]
\mathbf{p}_{\text{out}} &= \mathbf{p}_f &&\quad \text{read vector for block } \ell + 1
\end{aligned}
$$

The read vector used by a sublayer is always the one produced one site earlier. This is the
single-pass form of mHC (§2.4.1 eq. 6); the reason for the delay and the properties of
$\mathbf{q}$ and $C$ are covered in [mHC.md](mHC.md). $\operatorname{Attn}_\ell$ is a CSA2 layer whose
key/value set depends on $\ell$ (next section). $\operatorname{MoE}_\ell$ is one shared expert plus
two of eight routed experts, with no auxiliary loss ([moe.md](moe.md)). The engram lookup that would
precede the attention site is disabled (`engram_layer_ids=[]`).

### 3. Across the six blocks

Per token, $(X, \mathbf{p})$ is the only state carried from block to block. In addition, a dictionary
`shared`, created once per forward pass, carries key/value tables between attention layers. This is
how CSA2 groups layers: a *source* layer builds a compressed key/value table from its own input and stores it;
the layers after it in the group read the table instead of building their own.

Let $h_\ell \in \mathbb{R}^{S \times D}$ be the normed input of attention layer $\ell$. Every layer
projects $h_\ell$ to its own key latent ($K = V$ in CSA2) and attends over it inside a 128-token window,
$W_\ell$. Source layers additionally compress $h_\ell$ into a table $\hat C_\ell$ and store it in `shared`;
every layer from the source onward appends the stored tables to its key set:

$$
\begin{aligned}
\ell = 0, 1:\quad & \text{keys} = W_\ell \\
\ell = 2:\quad & \hat C_2 = \operatorname{compress}_{2}(h_2) \in \mathbb{R}^{256 \times d}
  \;\;\rightarrow\; \texttt{shared}, \qquad \text{keys} = W_2 \oplus \hat C_2 \\
\ell = 3:\quad & \text{keys} = W_3 \oplus \hat C_2 \\
\ell = 4:\quad & \hat C_4 = \operatorname{compress}_{1}(h_4) \in \mathbb{R}^{512 \times d}
  \;\;\rightarrow\; \texttt{shared}, \qquad \text{keys} = W_4 \oplus \hat C_2 \oplus \hat C_4 \\
\ell = 5:\quad & \text{keys} = W_5 \oplus \hat C_2 \oplus \hat C_4
\end{aligned}
$$

$\operatorname{compress}_m$ pools $m$ consecutive positions into one latent, so ratio 2 yields 256
entries for $S = 512$ and ratio 1 yields 512. The compressed part of the key set is not attended in
full: the source layer also runs the indexer, which scores each query against the compressed entries
and publishes a mask selecting the top 64 (`shared["topk_bias"]`); the consuming layers reuse that mask.
The window part $W_\ell$ is always the layer's own and is never shared. That layer 4 sees both $\hat C_2$
and $\hat C_4$ is a property of this code path, not of the paper; see [Open issues](#open-issues).

### 4. Output

After block 5 the streams are read one final time, with the read vector produced by the last MoE site,
then normed and projected with the tied embedding matrix:

$$
\begin{aligned}
h &= \operatorname{RMSNorm}\big(\mathbf{p}_{\text{out}}^{(5)\top} X^{(6)}\big) &&\in \mathbb{R}^{D} \\
z &= E\, h &&\in \mathbb{R}^{V} \quad \text{(fp32)} \\
\mathcal{L} &= -\frac{1}{N} \sum_{t:\ y_{t+1} \neq -100} \log \operatorname{softmax}(z_t)_{y_{t+1}}
\end{aligned}
$$

The loss is next-token cross-entropy only. There is no MoE balance term, no indexer loss and no MTP
head in this configuration; the gaps are listed under [Paper vs. this code](#paper-vs-this-code).

### CSA2 layer schedule (6-layer smoke config)

`compress_ratios=[0,0,2,2,1,1]`, `kv_source_layer_ids=[2,4]`, `index_source_layer_ids=[2,4]`,
`sliding_window=128`, `index_topk=64`.

| layer | ratio | role | paper mode (§2.3.1) | KV axis seen by attention |
|---|---|---|---|---|
| 0, 1 | 0 | window only, `main` rope | SWA | `[B,1,512,64]` (window mask) |
| 2 | 2 | KV **source** + index source; publishes 256 latents | Full | `512 + 256` |
| 3 | 2 | consumer, reads `shared["compress_kv"]` / `["topk_bias"]` | Reuse | `512 + 256` |
| 4 | 1 | KV **source** + index source; publishes 512 latents | Full | `512 + 768` ⚠ (see open issues) |
| 5 | 1 | consumer | Reuse | same as layer 4 |

Matches the released 40-layer shape scaled down: 2 SWA layers, then a ratio-2 group, then a ratio-1
group ([fig. 3](figs/fig3-overall-architecture.png)). A **Reindex** layer (index source that is not a
KV source) never occurs at `n=6` because the auto-derivation sets `index_source_layer_ids ==
kv_source_layer_ids`.

![CSA2 modes](figs/fig4-csa2-modes.png)

## Paper vs. this code

| feature | paper | this code | status |
|---|---|---|---|
| CSA2 modes | §2.3.1 Full / Reindex / Reuse; a layer reuses "*the most recent available main KV from a preceding layer*" — one table | Full + Reuse only at `n=6`; Reindex path exists but unreachable | matches (subset) |
| Multi-ratio KV | §4.2.1 encoder all `m=2`, decoder all `m=1`; never mixed in one layer | no-cache path **concatenates** ratio-2 and ratio-1 tables at layer 4 | **diverges** |
| CED | §2.2 decoder global KV is `C_l = H_{L/2} W^{KV}_l`, from the encoder's last hidden state | compressor reads the *current* layer's hidden state | **not implemented** |
| Hierarchical Sparse Indexer | §2.3.2 first Full layer builds a shared candidate pool; "*introduced in post-training*" | `select_candidate_blocks` implemented, never consumed (`candidate_source_layer_id=4` is last) | inert |
| Indexer training | §3.1.2 indexer params have a "*single logical owner*… responsible for optimization", with "*gradient aggregation*" | zero gradient (hard top-k + detaching fp4) | **diverges** |
| FP4 indexer q/k | §2.4.4 "*QAT … for FP4 indexer queries and keys*", OCP MXFP4 (E2M1, ue8m0 / 32) | format matches; gradient **fully detached**, so it is not QAT | **diverges** |
| FP4 main KV | §2.4.4 "*E2M1 with one E4M3 scale per 16 channels*", quantised *after* RoPE | exactly this; grad leaks only via the scale (16/256 elems) | format ✓, STE ✗ |
| FP8 SWA KV | §2.4.4 "*We retain FP8 for the SWA KV cache due to its sensitivity to quantization*" | fp8 E4M3, full STE (grad = 1) | matches |
| Single-Pass mHC | §2.4.1 eq. (6) shifts mixing coeffs by one block; §4.2.1 expansion factor 4, 20 Sinkhorn-Knopp iters | `hc_mult=4`, `hc_sinkhorn_iters=20`, `pre` shifted by one site | matches |
| MoE load balancing | §2.1.1 per-modality correction biases, "*updated independently according to their respective expert loads*"; §4.2.2 bias update speed 0.001 + sequence-level balance loss weight 1e-4 | `gate.bias` / `bias_vl` exist, frozen at 0; no update rule, no balance loss | **not implemented** |
| Router scoring fn | paper does not specify (only "*retain the shared and fine-grained routed experts of DeepSeekMoE*", §2.1) | `sqrtsoftplus`, `gate_temp=1`, `routed_scaling_factor=1.5` | unverifiable from paper |
| MTP | §2.1 "*We omit the MTP module during backbone pre-training*" | absent | matches |
| DSpark | §2.4.3 3 blocks, SWA 128, 5 draft positions, Markov + confidence heads; "*a dedicated stage after pre-training … backbone frozen*" | config fields only (`num_nextn_predict_layers=3`, `dspark_block_size=5`), no modules | matches for pre-training |
| Engram | §2.4.2 orders {2,3,4}, 8 hash heads, distinct primes, FP8 tables, no causal conv | same design, `engram_layer_ids=[]` so disabled | matches (off) |
| Sparse-attn warmup | §1 "*Sparse attention is trained from scratch at a sequence length of 64K, without any dense attention warmup stages*" | trained from scratch, no warmup | matches |

The paper **never describes an auxiliary or KL distillation loss for the indexer** — searching the
full text for "KL" / "distill" / "auxiliary" turns up only MoE load balancing and post-training OPD.
§3.1.2 establishes that indexer parameters *are* optimized, but the objective producing those
gradients is not stated in this report. Treat any claim of a specific indexer loss as unsourced.

![Hierarchical Sparse Indexer](figs/fig5-hierarchical-sparse-indexer.png)

## Open issues

- **Indexer is frozen at random init** — hard top-k gives no gradient, and `_fake_quant_fp4_block(x, 32)`
  (ue8m0) returns a tensor fully detached from the graph, so even the scale path is dead. → [indexer.md](indexer.md), [quant.md](quant.md)
- **FP4 vs FP8 STE asymmetry** — fp8 passes grad = 1, fp4/e4m3 leaks 16/256 elements, fp4/ue8m0 passes
  nothing; §2.4.4 calls all of these QAT, which requires gradient flow. → [quant.md](quant.md)
- **Multi-source `compress_kv` append/replace** — no-cache path `cat`s, cache path assigns; layer 4 ends up
  attending `[B,1,768,64]` across two compression ratios under a single `ratio=1` visibility rule. → [attention.md](attention.md)
- **No MoE aux loss, expert load unmonitored** — `gate.bias` never updates, `aux_loss` is never populated so
  the trainer's `moe/*` logs never fire. → [moe.md](moe.md)
- **No DSpark / MTP** — correct for backbone pre-training (§2.1), but the config fields are dead weight. → [config.md](config.md)
- **No CED** — the decoder half computes its own KV instead of projecting `H_{L/2}`. → [attention.md](attention.md)
- **`use_cache=False`, inference broken** — transformers 5.17 builds cache layers without `config`. → [cache.md](cache.md)
- **Candidate pool computed but unused** at `n=6`; needs an index source after the candidate source. → [indexer.md](indexer.md)

## Running it

```
just src/mzoo/ pt --arch=owlet1 --proj=owlet1 --exp=smoke --max_steps=20
```

`--arch=dsv4` runs the frozen single-file baseline on the same harness for comparison; the only
intentional code difference is `shift_labels` handling in `decoder.py` (see [decoder.md](decoder.md)).
