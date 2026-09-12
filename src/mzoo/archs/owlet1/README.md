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
| `decoder.py` | mHC residual, decoder layer, text backbone, causal-LM head + loss | [decoder.md](decoder.md) |

## Forward flow

`[B, S, hc, D]` = `[8, 512, 4, 256]`; `H=4`, `head_dim d=64`, `V=32000`.

```
input_ids [B,S] ──embed──> e [B,S,D] ──broadcast──> streams [B,S,4,256]   (all 4 copies = e)
                                                     pre_mix = (1,0,0,0)
 for layer 0..5:
   ┌ engram(streams)                       # None here: engram_layer_ids=[]
   │ hc_mixes -> attn_pre/post/comb        # comb: Sinkhorn(20 it) doubly-stochastic [B,S,4,4]
   │ collapsed = Σ_k pre_mix_k·stream_k    # [B,S,256]   <- previous SITE's pre
   │ attn_out  = attention(attn_norm(collapsed))        # [B,S,256]
   │ streams   = post·attn_out + comb ᵀ· residual       # expand, [B,S,4,256]
   │ hc_mixes -> ffn_pre/post/comb
   │ collapsed = Σ_k attn_pre_k·stream_k
   │ ffn_out   = MoE(ffn_norm(collapsed))               # 1 shared + top-2-of-8 routed
   └ streams   = post·ffn_out + combᵀ·residual;  carry ffn_pre to the next layer
 last_hidden = RMSNorm(Σ_k ffn_pre_k·stream_k)          # [B,S,256]
 logits = lm_head(last_hidden).float()                  # [B,S,32000], weight tied to embed
 loss   = cross_entropy(shift(logits), labels)          # plain CE, nothing else
```

The `pre` a site computes is consumed by the **next** site — the one-block shift of Single-Pass mHC
(§2.4.1 eq. 6), generalised to two mHC sites per block. `post`/`comb` are used in place.

### mHC streams

```
        stream 0  stream 1  stream 2  stream 3
          │         │         │         │
  collapse└───── Σ pre_k ─────┴─────────┘  ──> [B,S,256] one sublayer input
                        f(·)
  expand  ┌─────────────┴───────────────┐   out_k = post_k·f(x) + Σ_j comb[j,k]·residual_j
          ▼         ▼         ▼         ▼   (comb columns sum to 1 ⇒ convex mix of streams)
```

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
