# owlet1 — known gaps

What is broken, how we know, what the paper says, the fix, the risk. Companion to [README.md](README.md)
(flow + divergence table) and the per-module docs — maths lives there. Paper =
`../dsv4/DeepSeek_V41_Tech_Report.pdf`, extracted per section under `../dsv4/paper/`.

Measurements re-taken **2026-09-12** on the 6-layer smoke config (`build(tok, 512, layers=6)`:
`compress_ratios=[0,0,2,2,1,1]`, `kv_source_layer_ids=[2,4]`, `index_topk=64`, `sliding_window=128`,
8 routed experts top-2), torch 2.14.0+cu130 / transformers 5.17, CUDA.

## Triage

| # | gap | severity | effort | ours/upstream | status |
|---|---|---|---|---|---|
| 1a | fp8 **backward** underflows → every `wkv`/`kv_norm` frozen | **critical** | low | upstream | open |
| 1b | fp4 ue8m0 returns a detached tensor (indexer q/k) | high | low | upstream | open |
| 1c | hard top-k severs the indexer — **STE alone does not fix it** | high | high (design) | upstream | open |
| 2 | multi-source `compress_kv` concatenates instead of replacing | medium | low | upstream | open |
| 3a | expert load unmonitored; the router log would be garbage | medium | low | ours (harness) | open |
| 3b | `noaux_tc` correction biases never updated | medium | low | upstream | open |
| 3c | no 1e-4 sequence-level balance loss | low | low | upstream | open |
| 4 | CED absent (decoder KV not projected from `H_{L/2}`) | medium | high | upstream | open |
| 5 | DSpark / MTP absent | none | — | — | **not a gap** |
| 6 | `use_cache=True` raises; generation falls back to O(n²) | low | low | upstream (transformers) | open |
| 7 | inert config / untested paths | low | low | mixed | open |
| 9 | weight decay hits 30 params the paper excludes | medium | low | **ours (harness)** | open |
| 8 | `eval_loss` silently skipped | — | — | ours | **fixed** |

Net today: **34 params get zero or no gradient** — 14 KV-projection (1a), 8 indexer (1b/1c), 12 router
biases (3b). Everything else trains.

---

## 1a. The fp8 backward underflows — the entire shared-KV projection is frozen

*Not on the original list; found while checking 1b. Biggest gap here.*

**Symptom.** `attn.wkv.weight` and `attn.kv_norm.weight` get **exactly zero** gradient in all six layers, plus `layers.4.attn.compressor.{wkv,norm}`. K=V is one shared latent ([attention.md](attention.md)), so the model's whole KV path sits at random init.

**Evidence.** `quant.py:85` — `dequantized = quantized.to(torch.float8_e4m3fn).float() * scale`. Autograd routes the incoming gradient *through* the float8 tensor, so the **gradient is itself rounded to e4m3**. `float8_e4m3fn`'s smallest subnormal is 2⁻⁹ ≈ 0.00195; anything below flushes to zero. Measured on `[2,1,256,64]`: upstream grad 1.0 → 100 % of elements survive; 1e-2 → 0 %; 1e-5 → 0 %. Real gradients at the call site (`attention.py:154`) are ~3e-6, so **100 % underflows**. Ablation: replacing `_fake_quant_fp8_block` with the identity takes `layers.0.attn.wkv.weight.grad.abs().sum()` from `0.0` to `75.6`. The earlier probe reporting "fp8: full STE, 256/256" used an upstream gradient of exactly 1.0 — the one magnitude that survives. Survivors are also distorted (1e-3 → 1.95e-3, ~2×).

**Paper.** §2.4.4 p.14: *"We retain FP8 for the SWA KV cache due to its sensitivity to quantization."* QAT is asserted for these tensors and QAT needs gradient flow. No estimator is named — **0 hits for "straight-through"/"STE"** in 51 pages — so the choice is ours.

**Fix.** One wrapper at each of the three quantizer call sites; forward bit-identical, backward = identity: `x + (quant(x) - x).detach()`. Measured: revives **14 of the 34 dead params**; `torch.equal(ste(x), quant(x))` holds.

**Risk.** Low — forward numerics provably unchanged, so any curve movement is real. Diverges from `dsv4`; keep it owlet1-only and A/B. Caveat: STE is a coarse approximation for fp4's wide bins.

## 1b. `_fake_quant_fp4_block(x, 32)` (ue8m0) returns a fully detached tensor

**Symptom.** Indexer q (`indexer.py:129`) and k (`indexer.py:106`) leave the graph entirely.

**Evidence.** `quant.py:66` returns `values * scale`; `values` is an integer LUT gather and `scale` comes from `_pow2_ceil_scale`'s int32 bit-twiddling — both non-differentiable. Measured (CPU and CUDA agree):

| path | call site | backward |
|---|---|---|
| fp4 ue8m0 blk 32 | indexer q/k | **detached**, no graph |
| fp4 e4m3 blk 16 | compressed KV `attention.py:183` | grad via the `amax` path only, 16/256 elements |
| fp8 e4m3 blk 32 | window KV `attention.py:154` | connected but underflows to 0 — see 1a |

**Paper.** §2.4.4 p.14: *"DeepSeek-V4 already uses quantization-aware training (QAT) (Jacob et al., 2018) for FP4 indexer queries and keys"* — Jacob et al. 2018 is the canonical STE reference. (Nuance: that credits the prior model V4; V4.1's new contribution is *"we extend QAT to the main KV cache"*.) Formats match ours: MXFP4/OCP for the indexer, *"E2M1 with one E4M3 scale per 16 channels"* for main KV, quantized after RoPE.

**Fix / risk.** Same wrapper as 1a; low risk. On its own it changes nothing measurable — see 1c.

## 1c. The hard top-k severs the indexer regardless — STE alone does **not** fix it

*Corrects the original framing.*

**Symptom.** All 8 indexer params (`wq_b`, `weights_proj`, `wk`, `k_norm` on layers 2 and 4) have `grad is None` **both before and after** the STE fix. Measured: STE takes dead params 34 → 20; every indexer param is still among the 20.

**Evidence.** `indexer.py:159-165`: `block_bias` is a fresh `new_full(-inf)` into which the *constant* `0.0` is scattered at integer `topk.indices`, so `index_scores` never enters its graph. Measured: `shared["topk_bias"].requires_grad == False`. The selector is a pure argmax feeding a 0/−inf mask, and masks carry no gradient.

**Paper.** §3.1.2 p.18 confirms indexer params *are* ordinary trained parameters (*"The owner remains responsible for optimization… parameter synchronization and gradient aggregation keep the shadow replicas consistent"*). §1 p.6: *"Sparse attention is trained from scratch … without any dense attention warmup stages."* But **no objective is specified** that would produce those gradients: a full-PDF sweep gives 0 hits for `KL`, `Kullback`, `divergence`, `straight-through`, "indexer loss"; every `auxiliary` hit is MoE load balancing, every `distill` hit is post-training OPD. The V3.2 DSA KL-warmup recipe is **not in this report**. The mechanism by which DeepSeek's indexer learns is genuinely **unstated** — treat any specific indexer loss as unsourced.

**Fix — a decision, not a patch.** (1) Soft bias: add the STE'd selected `index_scores` into `block_bias` instead of a constant `0.0`, so chosen entries carry gradient — cheapest, slightly changes attention semantics. (2) Reintroduce a V3.2-style auxiliary KL against the dense attention distribution — well-trodden but off-paper, needs a dense reference pass. (3) Leave it frozen and document it as a random-projection selector baseline — legitimate.

**Risk.** High; design choice. Do 1a/1b first, confirm the curve improves, then pick, and record the choice in [README.md](README.md)'s divergence table.

## 2. Multi-source `compress_kv` concatenates instead of replacing

**Symptom.** A layer whose group has an earlier KV source attends over a table mixing two compression ratios under a single ratio's visibility rule.

**Evidence.** `attention.py:188-191`, the **no-cache path** — the training path, since `build()` sets `use_cache=False`: `shared["compress_kv"] = rotated if None else torch.cat([...], dim=2)`. Nine lines later (`:197`) the cache path *replaces*. Measured at `seq_len=512`: layer 2 publishes 256 ratio-2 latents, layer 4 appends its own 512 ratio-1 latents → `compress_kv = [B,1,768,64]`. The indexer then applies layer 4's single `ratio=1` visibility mask (`compress_lens = (pos+1)//ratio`) to the merged table, so layer 2's stale entries at indices 0–255 are treated as ratio-1 positions. **Byte-identical in `dsv4`** (`modeling_deepseek_v41.py:747-750`) → upstream bug, not ours.

**Paper.** §2.3.1 p.11 is consistently singular: *"The layer reuses **the most recent available** main KV from **a** preceding layer together with its corresponding indexer K."* §4.2.1 p.22 keeps ratios strictly disjoint — 18 encoder CSA2 layers at m=2 in *"three identically configured groups of six"*, 20 decoder layers at m=1 in *"five groups of four"*; no layer mixes ratios. §2.2/§2.3.1: the Full-Mode decoder layer *"computes its own global KV from the hidden state of the (L/2)-th layer"* — fresh, not appended. **Honest caveat:** the paper never *forbids* concatenation, it just never describes it; a sweep for `concat|merg` finds only intra-layer concat of selected main KV with the layer's own SWA KV (fig. 4, p.10).

**Fix / risk.** Make the no-cache branch assign, matching `:197` — one line. Low mechanically, but it changes what layers 4/5 see, so expect a curve step. Guard with `assert compress_kv.shape[2] == seq_len // ratio` at each consumer.

## 3. MoE load balancing: nothing implemented, nothing even measured

**3a — monitoring (do first).** `moe/*` never appears in any log. `trainer.py:30` gates `_track` on `outputs.aux_loss is not None`, and the model returns `MoeCausalLMOutputWithPast` with `aux_loss=None` (measured), so it never fires; `_log_router_load` (`trainer.py:58`) needs a `load_counts` buffer owlet1 routers lack. **Correction to the original note:** `decoder.py:255` records `OutputRecorder(DeepseekV41TopKRouter)`, and `router_logits[i]` is **not** a `(weights, indices)` tuple — it is one `float32` tensor of shape `[tokens, top_k]`, the routing *weights* (measured `(512, 2)`, values in [0.62, 0.88]). So `_track`'s `r.topk(k).indices` would index the top-k **axis** (0/1) and `minlength=r.shape[-1]=2` would bin into 2 buckets, not 8 experts — confident nonsense rather than a crash. *Fix:* record `indices` instead, or keep a `load_counts` buffer on the router and let the existing path pick it up. *Risk:* none, read-only.

**3b — correction biases never update.** `gate.bias`/`gate.bias_vl` (12 params, zero-init) have `grad is None` forever, so `(scores + bias).topk(...)` at `moe.py:36` is plain top-K. They are `nn.Parameter`s (`moe.py:26`) but only integer top-k indices are downstream, and no update rule exists. *Paper* §2.1.1 p.8: *"After each training step, the two sets of biases are updated independently according to their respective expert loads."* §4.2.2 p.22: *"we set the bias update speed to 0.001 for both image and text tokens"*. (The paper says *"auxiliary-loss-free load balancing"*; it never uses the token `noaux_tc`.) *Fix:* post-step callback `bias -= 0.001 * sign(load - load.mean())` using 3a's counts; `bias_vl` stays inert (text-only). *Risk:* low — needs 3a, and the biases must stay out of the optimizer/weight decay.

**3c — no balance loss.** Total loss is plain CE (`decoder.py:399-403`); `router_aux_loss_coef` is never read. *Paper* §4.2.2 p.22: *"retaining a small sequence-level balance loss with a loss weight of 0.0001 to avoid extreme imbalance within single sequences."* *Risk:* low, but at 6 layers × 8 experts likely unmeasurable. Do last.

## 4. CED not implemented

**Symptom.** Every KV-source layer compresses its *own* hidden state.

**Evidence.** `cache.py:142-176`, `DeepseekV41Compressor.forward(hidden_states, ...)` — the argument is the caller's `attn_norm(collapsed)`, i.e. the current layer's stream (`attention.py:170`). Nothing plumbs an encoder hidden state anywhere; no layer ever receives `H_{L/2}`. Verified by reading both call paths.

**Paper.** §2.2 p.9 eq. (1): for *"the upper half layers (i.e., the decoder, l > L/2), the KV entries are not derived from their respective hidden states H_l. Instead, they are projected directly from the hidden state of the (L/2)-th layer"* — `C_l = H_{L/2} W_l^{KV}`, `Z_l = H_{L/2} W_l^Z`. SWA KV is explicitly **excluded** (*"the local keys and values are derived directly from the current layer's hidden state H_l"*). §2.3.1 p.11: under CSA2 only Full-Mode decoder layers materialize it.

**Fix / risk.** Stash layer `L/2`'s output in the per-forward `shared` dict and feed it to the compressor of every `kv_source_layer_id > L/2`. One extra `shared` key, but it rewires the gradient graph (layer `L/2` then carries gradient from every decoder KV) and interacts with 2 — fixing CED makes 2 moot for decoder layers. Medium-high risk; own branch, own A/B.

## 5. DSpark / MTP — **not a gap**

Config fields exist (`num_nextn_predict_layers=3`, `dspark_block_size=5`) and nothing reads them; `model.py:48` sets `num_nextn_predict_layers=0`. **Correct for our use case.** §2.1 p.8: *"We omit the MTP module during backbone pre-training and use DSpark for speculative decoding. We train DSpark separately after the backbone pre-training stage."* §2.4.3 p.14: *"we train only DSpark while keeping the backbone frozen."* Recorded so nobody re-investigates; the only cost is dead config surface.

## 6. `use_cache=True` raises; generation still works, slowly

**Symptom.** `TypeError: DeepseekV41CSACache.__init__() missing 1 required positional argument: 'config'`, reproduced on both `forward(use_cache=True)` and `generate(use_cache=True)`.

**Evidence.** transformers 5.17 builds dynamic cache layers via `DYNAMIC_LAYER_TYPE_MAPPING[layer_type](**layer_kwargs)` without passing `config`; `cache.py:40` requires it. **Correction to the original note:** this does **not** block sampling. `build()` sets `use_cache=False` and `m.generate(...)` runs fine under it, just re-forwarding the whole prefix each step (measured: `compress_kv` grows 12 → 13 entries as `seq_len` goes 8 → 9, i.e. no reuse). Eval-by-sampling is *available but O(n²)*. Training unaffected.

**Fix / risk.** Default `config`, or read `sliding_window` from `layer_kwargs` — one line. But the cached decode path is then wholly untested and has real group-boundary logic (partial-group buffers, `entry_count`); it wants a prefill-vs-decode equivalence test before being trusted. Low to unbreak, medium to trust; not worth it before 1a/2.

## 7. Smaller things (each re-verified)

- **`n_shared_experts` is never read** by `moe.py` — `moe.py:83` unconditionally builds exactly one `shared_experts` MLP, so `build(shared=…)` is inert. Silent wrong-config hazard. See [moe.md](moe.md).
- **Two-level candidate top-k computed, never consumed.** `candidate_source_layer_id` auto-derives to `kv_sources[-1] = 4` (`config.py:390-391`) and `uses_candidates` needs a *later* index source (`indexer.py:63`) — layer 4 is the last. Measured: `shared["candidates"]` is published (`[2,256,384]` at `seq_len=256`) and nothing reads it. **Correction:** `candidate_topk_blocks=64` is *not* a no-op today — at `seq_len=512` layer 4 sees 768 entries / `candidate_block_size=8` = **96 blocks**, so top-64 would bite; it becomes a no-op only once gap 2 is fixed and the table drops to 512/8 = 64. Either way untested at `n=6`; exercising it needs a Reindex layer, i.e. ≥8 layers. §2.3.2 p.12 calls the mechanism *"introduced in post-training"*, so absence is defensible.
- **Dead config fields:** `router_aux_loss_coef`, `output_router_logits`, `router_jitter_noise` (`config.py:304-306`) are read by nothing. `output_router_logits=True` does work, but via HF's generic `_can_record_outputs` machinery, not that field.
- **Engram disabled** — `engram_layer_ids=[]` (`config.py:409-410`), so `decoder.py:58` takes the `None` branch. Matches the released non-196B checkpoints; `engram.py` is 283 untested lines. See [engram.md](engram.md).

## 9. Weight decay is applied to 30 parameters the paper explicitly excludes

*The one gap where our **harness** diverges from the paper rather than the vendored model code.*

**Symptom.** `hc_*_scale`, `hc_*_base` and `attn_sink` all land in the optimizer's weight-decay group. A naming accident, not a decision.

**Evidence.** HF `Trainer.create_optimizer` builds the no-decay set as `get_parameter_names(model, ALL_LAYERNORM_LAYERS)` filtered by `"bias" not in name`. Our mHC tensors are raw `nn.Parameter` attributes (`hc_attn_scale`, `hc_ffn_base`, `attn_sink`) — no `bias` substring, and not inside an `nn.LayerNorm` (`ALL_LAYERNORM_LAYERS == [LayerNorm]`; `DeepseekV41RMSNorm` is not in it). Reproduced by running HF's exact selection logic on the smoke model:

| param class | n | in decay group | paper §2.5 | verdict |
|---|---|---|---|---|
| `hc_*_scale` | 12 | all 12 | scaling factors: no decay | **hazard** |
| `hc_*_base` | 12 | all 12 | biases: no decay | **hazard** |
| `attn_sink` | 6 | all 6 | scaling factor: no decay | **hazard** |
| `hc_*_fn` | 12 | all 12 | weight matrices: decay | correct |
| `RMSNorm.weight` | 29 | all 29 | norm weights: decay **yes** | correct (by luck) |
| `gate.bias`/`bias_vl` | 12 | none | biases: no decay | correct |

**Paper.** §2.5 p.16: *"normalization-layer weights are also subject to weight decay, whereas biases and scaling factors are not."* (§2.5 also splits AdamW / Muon / Sinkhorn-balanced updates by parameter class; our harness uses plain fused AdamW for everything — a separate, larger divergence, noted here only so it is not forgotten.)

**Impact.** `_init_weights` gives `hc_*_scale = 1`, `hc_*_base = 0`, `attn_sink = 0` (verified). Decoupled AdamW applies `p -= lr*wd*p` per step; at the harness defaults (`pt.py:48` `lr=6e-4`, `pt.py:51` `weight_decay=0.1`) the per-step factor is `1 - 6e-5` → ×0.994 over 100 steps (negligible), **×0.55 over 10k steps**. So `hc_*_scale` drifts 1.0 → ~0.55 on a real run. `scale` multiplies the mix logits before the sigmoid/softmax in `hc_mixes`, so shrinking it flattens `pre`/`post` toward their midpoints and pushes `comb` toward uniform — a slow, silent damping of the hyper-connection routing that the loss curve would not obviously reveal. The zero-init params are harmless at init but get fought once they learn an offset.

**Fix / risk.** A no-decay param group in `mzoo/train/pt.py` (or an owlet1-side `create_optimizer` override) matching `hc_*_scale` / `hc_*_base` / `attn_sink` explicitly rather than relying on the `bias` substring. The fix lives in the harness, not this package. Affects `dsv4` identically (same names, same harness). Invisible at our 20-step smoke scale; real over thousands of steps.

## 8. Already fixed — do not re-investigate

`eval_loss` was silently skipped: the vendored forward exposed both `labels` and `shift_labels`, so HF's `find_labels` returned both, and `Trainer.prediction_step` requires *all* declared label names present — our collator supplies only `labels`, so it concluded there were none. Fixed at `decoder.py:387` by popping `shift_labels` out of `**kwargs` rather than declaring it. Verified: `find_labels(...) == ['labels']`, and `260911-0002-evalfix` logs `eval_loss` 9.023 (step 10) → 7.843 (step 20). **Still the only intentional code divergence from `dsv4`.**

## Suggested order

1. **3a (monitoring)** — read-only, no semantics change, and you need load numbers before judging 3b/3c.
2. **1a + 1b (the STE wrapper)** — one helper, three call sites, forward bit-identical, revives 14 params including the model's entire KV projection. Best value per line here. Baseline the curve first.
3. **2 (`cat` → assign)** — one line, removes a ratio-mixing pathology. Land after step 2 so the two curve changes stay separable.
4. **3b (bias update)** and **9 (no-decay param group)** — both small, both paper-numbered (0.001 / §2.5); 9 is harness-side so it lands independently of any owlet1 change.
5. **1c (indexer objective)** — stop and decide; the paper does not decide for you, and option 3 ("leave it frozen, document it as a baseline") is a legitimate answer.
6. **4 (CED)**, **6 (cache)** — both real, both invasive; own branch, once the cheap wins are in.
7. **3c, 7** — cleanup.


## Suspicions — unverified, check before trusting

Things that *smelled off* during a broad read but were **not** verified or diagnosed — hunches to check, not claims. Nothing here duplicates gaps 1–9.

### numerics / precision

- Explicit `.float()` may not survive bf16 autocast: `F.linear(flat, fn.float())` (`decoder.py:77`), `hc_expand`'s einsum, and `cache.py:155` all return **bf16** under autocast in isolation — the "fp32 from the start" comments may not hold in a real `bf16=True` run.
- Same suspicion for the indexer score einsum (`indexer.py:134`): `.float()` on both operands, matmul probably still bf16.
- `keep_fp32()` (`precision.py:24`) matches only class names ending in `Router`; the mHC mix projection and the compressor pooling look like the other two sensitive spots and may want the same wrapper.
- `rms_norm_eps = 1e-20` (`config.py:296`) is the eps for every RMSNorm, `hc_input_norm` and the engram gate — effectively no epsilon. Maybe faithful to the checkpoint, suspect for from-scratch.
- `logits = lm_head(...).float()` (`decoder.py:397`) materializes `[B,S,V]` fp32 before the loss (~2 GB at bs32/seq512) — suspect this is the memory ceiling, not the model.

### quantization (beyond 1a/1b)

- `_fake_quant_*` silently returns `x` **unquantized** when `n % block_size != 0` (`quant.py:56`, `:80`), no warning — a `head_dim`/`index_head_dim` change could disable QAT invisibly.
- `layers.2.attn.compressor.wgate.weight` is also zero-grad (measured **37** dead params on the smoke config, not 34) — probably the same fp4 root cause as 1b, but absent from gap 1's accounting.
- fp8 floors amax at `1e-4` (`quant.py:82`) while fp4 floors at `6·2⁻¹²⁶` — asymmetric; worth checking both against the reference kernel.
- `_e2m1_codes` rebuilds its `boundaries`/`ties_up` tensors on every call (`quant.py:40-41`), unlike the cached LUT — a per-forward alloc on the hot path.

### attention / CSA2

- `attn_sink` enters the softmax **unscaled and unmasked** (`attention.py:63-65`) — the only logit not multiplied by `scaling` nor offset by the mask; at init it contributes `exp(0)=1`.
- `block_bias.to(attention_mask.dtype)` (`attention.py:207`): if a **bool** mask ever reaches here, `-inf → True` inverts the bias. Eager-only today, so possibly moot.
- Nothing asserts `block_bias.shape[-1] == compressed_kv.shape[2]` before the cat — a source/index-source mismatch could mis-align silently rather than crash.
- The guard is `isinstance(attention_mask, torch.Tensor)`; if the mask is ever `None` the appended compressed entries get **no mask at all**. Worth confirming `create_sliding_window_causal_mask` cannot return `None`.
- The output inverse-RoPE uses the **query's** position (`attention.py:233`) on a value average mixing window KV (token positions) and compressed latents (group positions) — exact only under an assumption; check intent.
- `rope_layer_type` is per **layer**, not per branch (`attention.py:100`), so a compressed layer's *sliding-window* KV is also rotated with `compress_rope_theta` (160k) rather than `rope_theta`. May be intended.
- `num_key_value_heads` (`config.py:229`) and `attention_bias` (`:238`) are never read — another silent wrong-config surface like `n_shared_experts`.

### indexer

- `shared["index_k"]` uses the **same multi-source `torch.cat`** pattern as gap 2, on a separate line (`indexer.py:111-113`) — if gap 2 is fixed this probably needs the identical fix; easy to miss.
- Indexer RoPE width comes from the *attention* head_dim: `partial_rotary_factor = qk_rope_head_dim/head_dim` (`config.py:428`) is reused for `index_head_dim`, so the indexer rotates 16 of 32 channels (50%) vs attention's 25%. Suspect accidental.
- `scores.relu_()` runs **before** the per-head weighting (`indexer.py:131`), so a negative `weights_proj` head can only subtract, never rescue a suppressed score — worth checking the reference ordering.
- `select_candidate_blocks` pins the newest block via `last = (compress_lens-1)//block_size`; at `compress_lens == 0` that is `-1` and nothing is pinned. Probably fine, unverified.

### MoE

- `routed_scaling_factor = 1.5` makes the top-k weights sum to **1.5**, not 1 (`moe.py:41`), and the shared expert adds at weight 1.0 on top (`moe.py:97`) — ~2.5× residual gain per block. Check vs paper.
- `swiglu_limit` clamps the **gate** branch from above only but the **up** branch on both sides (`moe.py:63-65`) — asymmetric; may be deliberate fp8 range control, may be a slip.
- Every layer is MoE — no `first_k_dense_replace` equivalent, while DeepSeek keeps early layers dense. Worth checking whether it matters at 6 layers.
- `trainer.py:46` reads `cfg.mlp_only_layers`, which owlet1's config does not define — a latent `AttributeError` that fires the moment `aux_loss` becomes non-`None`, i.e. right after gap 3a is fixed.
- `counts[i] == 0` in the dispatch loop (`moe.py:94`) is a GPU→host sync per expert, per layer, per step — tolerable at 8 experts, suspect it dominates at 384.
- `gate_temp` divides before `sqrtsoftplus` (`moe.py:34-35`), which is unbounded above; check the ordering, and whether `topk_method="noaux_tc"` (`config.py:266`, never read) implies something different.

### mHC

- The Sinkhorn loop ends on a `dim=-2` pass (`decoder.py:85-88`), so `comb` is exactly **column**-stochastic and only approximately row-stochastic. May or may not be intended.
- `hc_eps` is added *after* each normalization, so every pass re-breaks the stochasticity it just established — likely negligible at 1e-6 × 20 iters, unverified.
- Initial `pre_mix` is one-hot `(1,0,0,0)` (`decoder.py:320-321`), so the first attention site reads stream 0 only — though all four streams start as identical embedding copies, so possibly a no-op at layer 0.
- The final readout collapses with the **last layer's `ffn_pre`** (`decoder.py:335`), a coefficient set with no other consumer — worth confirming that is intended and not an off-by-one.
- `hc_*_scale` is length 3 (one scalar per coefficient set) shared across all `hc_mult` channels — may be weaker than the paper's parameterization.

### engram (dead today, but)

- `token_mask` is hardcoded `None` at the layer call (`decoder.py:327`) though `live_mask` is computed just above — if engram is ever enabled, padded positions would not be gated off.
- `EngramLayout.from_config` returns `None` early on `engram_layer_ids == []` so the prime search never runs — worth a quick confirm no engram state is constructed accidentally.

### cache / generation

- `DeepseekV41CSACache.update` retains `sliding_window - 1` entries but returns `full` (`cache.py:88-93`) — retention width vs the mask's window looks off by one. Untested since `use_cache=False`.
- `entry_count` is bumped only for `"compressor"` (`cache.py:122`), so the indexer key cache has no independent position anchor; if the two desync, indexer scores misalign silently.

### harness / training

- `gradient_checkpointing=True` **raises** `ValueError: DeepseekV41ForCausalLM does not support gradient checkpointing` (reproduced) — no activation-memory escape hatch; and if enabled, the per-forward `shared` dict would likely double-append under backward recompute.
- `collate` sets `labels = input_ids` as the *same tensor object*, unshifted (`data.py:70`) — correct only if `loss_function` shifts internally and never mutates labels in place. A `.clone()` would be cheap insurance.
- The collator emits no `attention_mask`, so `live_mask` is always `None` and every position counts as live — fine for packed data, would silently break on padded data.
- `TimedCheckpoint` saves at the last step even under `save_strategy="no"` (`trainer.py:19`) — intended per its docstring, but if `state.max_steps` were ever ≤ 0 there it would save every step.
- `pt.py:26` routes any kwarg not in `TRAIN_FIELDS` to `build()` — a misspelled arch kwarg colliding with a `TrainingArguments` field name is silently accepted by the Trainer instead of erroring.
- `keep_fp32` replaces `module.forward` with a closure instance attribute (`precision.py:17`) — suspect it does not survive `deepcopy` / `torch.compile` / `save_pretrained`, and double-wraps if called twice.
- `pt.py` sets `max_position_embeddings = seq_len` from the dataset, so RoPE is sized exactly to the training length; a longer-context eval would fall outside the cached range.

### second pass — model / harness wiring (same tier: worth checking)

- `tie_word_embeddings=True` (`model.py:50`) appears to be a **silent no-op**: measured `lm_head.weight.data_ptr() != embed.weight.data_ptr()`, unchanged after an explicit `m.tie_weights()`, and `_tied_weights_keys` is `None`. That would leave 8.2 M of 28.5 M params (**29 %**) as an untied head, and the config field would be lying.
- `num_items_in_batch` never reaches `self.loss_function` (`decoder.py:405-407`): our forward passes `**kwargs` to the *backbone* but names the loss args explicitly, while upstream Llama forwards `**kwargs` into the loss. `ForCausalLMLoss` accepts it. Suspect the classic mean-vs-token-normalized **gradient-accumulation** bias whenever `gradient_accumulation_steps > 1` (our default is 1, so latent).
- `model.to(torch.bfloat16)` casts every `_keep_in_fp32_modules_strict` param to bf16 (measured: `hc_attn_scale`, `attn_sink`, `gate.bias` all become bf16) — that list only binds through `from_pretrained`. Safe under our autocast harness, but any `.to(dtype)` silently unpins the fp32-critical tensors.
- `output_hidden_states` returns 7 tensors for 6 layers, all `attn_norm` outputs taken *before* `model.norm` (`decoder.py:255-259`) — so `hidden_states[-1]` is **not** the final hidden state, unlike every other HF model. A trap for probing/representation work.
- Recorded `attentions` have per-layer-varying widths (layer 0 `[2,4,128,128]`, compressed layers wider), so `torch.stack(outputs.attentions)` would fail; and the recorded probs exclude the sink column, so rows do not sum to 1.
- `position_ids` is a plain `arange` that ignores `attention_mask` (`decoder.py:293-296`), so left-padded rows get wrong absolute positions — and `compress_lens = (position_ids+1)//ratio` (`indexer.py:141`) inherits the error. Latent under packed training; would bite on any padded batch.
- Never-read config fields **beyond** gap 7's list (grepped all 81 against every owlet1 + harness module): `topk_method`, `num_key_value_heads`, `attention_bias`, `pad_token_id`, all five `dspark_*`, and the six vision fields (`patch_size`, `downsample_ratio`, `max_image_tokens`, `min_pixels`, `max_wh_ratio`, `image_token_id`). Setting any of them does nothing and nothing warns — the same silent-wrong-config class as `n_shared_experts`.
- `o_groups` is documented as "must divide `heads * head_dim`" (`model.py:18`) but is never validated; a bad value just floor-divides inside `DeepseekV41GroupedLinear` (`attention.py:36`). Same for `sliding_window`, which `__post_init__` never checks against `compress_ratios` or `seq_len`.

## Long shots — low confidence, listed for completeness

Weaker, more speculative, or things I could not form a view on. **These may well be nothing** — recorded only so they are not lost, and so nothing had to be cut for space. Dismiss freely.

### numerics / precision

- `DeepseekV41UnweightedRMSNorm`'s own default eps is `1e-6`, but the decoder constructs it with `rms_norm_eps` (`decoder.py:59`), i.e. `1e-20` — the class default suggests someone once intended `1e-6` there.
- `eager_attention_forward` subtracts the row max *and* then softmaxes (`attention.py:66-67`) — a double stabilization; harmless, but it suggests the sink concat was bolted on later.
- `F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)` passes the tensor's own dtype, which is a no-op and defeats autocast's usual fp32 upcast; only safe because the `cat` with `sinks.float()` already promoted.
- `hc_collapse` and the pointwise parts of `hc_expand` do stay fp32 under autocast (only the matmuls downcast) — so the mHC damage, if any, may be partial rather than total.

### quantization

- `_FP4_LUT_CACHE` is keyed by the `torch.device` **object** (`quant.py:16`), so `cuda` and `cuda:0` are distinct keys — duplicate LUTs on multi-GPU, and the module-level dict is never cleared.
- A `head_dim` not divisible by 32 (or `index_head_dim` not divisible by 32, or a post-RoPE width not divisible by 16) would silently drop the corresponding QAT path — the same `n % block_size` escape hatch, reachable purely from `build()` kwargs.
- `_pow2_ceil_scale` does an fp32 `view(torch.int32)` (`quant.py:26`); fine today because callers `.float()` first, but it would misbehave on any non-fp32 input rather than raise.

### attention / CSA2

- `scaling = head_dim**-0.5` uses the **full** head width including the nope channels; some MLA variants scale by the rope+nope split separately. Probably standard, unverified.
- `eager_attention_forward` receives `sliding_window=` and ignores it (`attention.py:224`) — the windowing is entirely in the model-level mask; fine, but a different `attention_interface` might expect the kwarg to be authoritative.
- Layer 0 and 1 are pure sliding-window (`compress_ratio == 0`) and therefore see **only 128 tokens** with no global path at all; with 6 layers that is a third of the model. Probably the intended schedule, noted anyway.
- `DeepseekV41CSACache._layer_type = "shared_compressed_attention"` vs `config.layer_types` entries — if HF's mask utils ever dispatch per layer type, a compressed layer might get a different mask than the sliding one it currently shares.

### indexer / RoPE

- `dim = int(head_dim * partial_rotary_factor)` (`norm_rope.py:97`) truncates; an odd `qk_rope_head_dim / head_dim` ratio would silently shrink the rope width by one channel instead of erroring.
- `index_topk=512` and `candidate_topk_blocks=2048` defaults (`config.py:250-252`) are sized for the 40-layer model; a tiny config that forgets to override gets a k larger than the table. Clamped by `min(...)`, so likely harmless.
- *Correction to my earlier `max_position_embeddings` item:* the rotary recomputes `cos`/`sin` per call from `position_ids` with no table, so there may be **no real length cap** — the field mostly feeds YaRN / dynamic-rope scaling. Treat that earlier suspicion as weak.

### MoE

- `y[idx] += self.experts[i](...)` (`moe.py:97`) is a non-atomic gather/scatter read-modify-write; correct because indices are unique per expert, but fragile if the dispatch ever changes.
- `weights / (weights.sum(-1, keepdim=True) + 1e-20)` (`moe.py:40`) — with `sqrtsoftplus` scores the sum is well away from zero, so the epsilon may be dead code inherited from a softmax gate.
- The expert loop runs in index order and accumulates into `y` — float addition order is deterministic here, but would change under any reordering/fusion, so bitwise A/B comparisons across dispatch rewrites will not hold.

### mHC

- `build()` does not expose `hc_mult`, `hc_sinkhorn_iters`, `compress_ratios`, `kv_source_layer_ids` or `engram_layer_ids`, so none of them can be swept from the `pt` CLI without editing `model.py` — an ergonomics gap, not a bug.
- `mix_hc = (2 + hc_mult) * hc_mult` makes `hc_attn_fn` a `[24, 1024]` matrix at `hc_mult=4` — the mHC projection is ~25 K params per site, 12 sites; small here but quadratic in `hc_mult`.
- 20 Sinkhorn iterations run **every site, every layer, every token** in fp32; probably cheap relative to attention, but it is an unmeasured fixed cost.

### engram

- `engram_pad_id = 2` (`config.py:281`) is a hardcoded literal unrelated to our tokenizer (which happens to also use 2) — would silently mismatch under a different tokenizer.
- `engram_compressed_vocab_size = 99092` and `engram_vocab_size = 16000000` are released-checkpoint constants with no relation to a 32 k-vocab tiny model; nothing scales them down.

### data / packing

- Documents are packed across block boundaries with **no document-level attention masking**, so a block's first tokens attend to the tail of an unrelated story. Standard for packing, and the 128-token window limits the blast radius, but worth a conscious decision.
- `pad_token_id == eos_token_id == 2` for this tokenizer — `generate()` warns it cannot infer an attention mask from a batched input for exactly this reason.
- `val_docs=2000` (`data.py:41`) is dead whenever the dataset ships its own validation split (TinyStories does — 10 135 packed blocks), so the flag silently does nothing there.
- Eval uses `ds["validation"].select(range(256))` (`pt.py:63`) — the *first* 256 blocks, unshuffled; fine for comparability, but it is one contiguous slice of one part of the corpus.
- `packed()`'s `group` drops the remainder **per `map` batch** (`data.py:25`, default 1000 docs), not once globally — a small, systematic token loss rather than a single tail.
- Tokenization is `tok(t + tok.eos_token)`, which the TinyLlama tokenizer renders as `['<s>', …, '</s>']` (checked) — so BOS/EOS are both present per document; noting only because string-concatenating a special token is fragile across tokenizers.

### harness / device

- `Trainer.compute_loss` forces `return_outputs=True` on every step (`trainer.py:28`) even when the outputs are unused, keeping the fp32 logits alive a little longer than necessary.
- `_log_router_load` walks `named_modules()` on every log call (`trainer.py:58`) looking for a buffer no owlet1 module has — harmless, but it means the "router load" logging path is currently pure overhead.
- `pt.py` mutates `os.environ` for `WANDB_PROJECT` / `WANDB_DIR` (`pt.py:23`, `:38`) — process-global, so two runs in one process would share the second's dir.
- `keep_fp32(model)` runs before the Trainer moves the model to GPU (`pt.py:31`); the wrapper closes over `hidden_states.device.type` at call time so it should be fine, but the ordering is implicit.
- Cross-layer state lives in a per-forward `shared` dict whose tensors belong to the *source* layer's device; under `device_map`-style model parallel a consumer on another device would hit a device mismatch. `_no_split_modules` only protects the decoder layer, not this coupling. DDP (one full replica per GPU) is unaffected.
- Nothing asserts the model's params are fp32 at train start, so a future `.to(bf16)` plus `bf16=True` would silently become pure-bf16 training rather than autocast.
- *Checked and clean, recorded so nobody re-checks:* every parameter has a real init branch — no tensor is left at whatever `torch.empty()` returned (all weights std ≈ 0.02, all norms exactly 1.0, `attn_sink`/`hc_*_base`/`gate.bias` exactly 0, all finite). `_init_weights` looks complete.
