# owlet1 — training notes from the DeepSeek-V4.1-Flash tech report

Training-relevant distillation of `../dsv4/DeepSeek_V41_Tech_Report.pdf`. Architecture lives in
[README.md](README.md) + the per-module docs ([moe.md](moe.md), [indexer.md](indexer.md),
[quant.md](quant.md)). Unlabelled claims carry a `§` cite; **[inference]** marks my judgement, not
the paper; "paper does not specify" means exactly that — nothing borrowed from other DeepSeek papers.

## 1. TL;DR

- **Two §4.2.2 mechanisms are missing from our code**: the aux-loss-free bias update (speed
  `0.001`) and the sequence-level balance loss (weight `0.0001`). Our loss is plain CE and
  `bias`/`bias_vl` get no gradient and no update rule → routing is frozen at init. Fix before
  reading anything into MoE runs ([moe.md](moe.md)).
- **[inference]** At 20–100 steps, bias speed `0.001` moves the bias ≤0.1 against O(1) scores — a
  no-op at our horizon. Raise it, or lean on the balance loss for smoke runs.
- **Upstream is not AdamW** (§2.5/§4.2.2): Muon (head-wise for Q/K) for linear weight matrices,
  AdamW only for norm weights + non-matrix params, Sinkhorn-balanced momentum for embeddings/head.
  So the paper's LR `2.6e-4` does **not** transfer as a number to our AdamW-everything harness.
- **Transferable AdamW numbers** (§4.2.2): `β1=0.9`, `β2=0.95`, `ε=1e-20`, `wd=0.1`. Only `wd`
  matches our default today.
- **Schedule is WSD, not cosine** (§4.2.2): linear warmup 2000 steps → constant → cosine 10× decay
  → constant tail.
- **No dense-attention warmup needed** (§1): sparse trained from scratch at 64K. **[inference]** at
  `seq_len=512` the claim is untestable anyway — see §3.
- **No indexer auxiliary loss anywhere in the paper** — the indexer trains only through attention.
- **Stability**: "45T tokens … with no instability" (§4.2.2) is the paper's *entire* stability
  discussion. No loss-spike, divergence or router-collapse guidance exists.

## 2. Paper hyperparameters

### Optimizer (§2.5, §4.2.2)

| Item | Value | Cite |
|---|---|---|
| Linear weight matrices | Muon, Nesterov momentum, decoupled weight decay | §2.5, §4.2.2 |
| Query **and Key** weights | head-wise Muon (split by head before the update) | §2.5 |
| RMSNorm weights, biases, scaling factors | AdamW | §2.5, §4.2.2 |
| Embeddings, prediction head, Engram tables | momentum update + Sinkhorn balancing (Alg. 1) | §2.5 |
| AdamW | `β1=0.9`, `β2=0.95`, `ε=1e-20`, `weight_decay=0.1` | §4.2.2 |
| Muon | `momentum=0.95`, `weight_decay=0.1`, update-RMS rescale `0.18` | §4.2.2 |
| Sinkhorn update | same momentum + LR-correction as Muon, `K=11`, `τ=1e-3`, `ε=1e-20`, `γ=0.18` | §2.5, §4.2.2 |
| Weight decay applied to | norm weights **yes**; biases and scaling factors **no** | §2.5 |
| Engram LR multiplier | `5×` | §4.2.2 |
| Gradient clipping | **paper does not specify** | — |
| Initialization scheme / std | **paper does not specify** | — |

### Schedule / budget (§4.2.2)

| Item | Value |
|---|---|
| Tokens | 45T multimodal |
| Batch size | fixed 100.6M tokens throughout (no ramp) |
| Warmup | linear, first 2000 steps |
| Peak LR | `2.6e-4`, held constant until 28T tokens |
| Decay | cosine `2.6e-4 → 2.6e-5` between 28T and 40T tokens |
| Tail | constant `2.6e-5` from 40T to 45T |
| Seq len | 64K from scratch with sparse attention; extended to **1M at 34T tokens** |
| Context-extension *method* (RoPE scaling / YaRN / etc.) | **paper does not specify** |
| Attention masking | sample-level, "similar to DeepSeek-V4" |
| Total optimizer steps | not stated; `45T/100.6M ≈ 447k` **[inference, arithmetic]** |

### MoE / balance (§4.2.2, §2.1.1)

| Item | Value |
|---|---|
| Bias update speed | `0.001`, separately for image and text tokens |
| Sequence-level balance loss weight | `0.0001` — "to avoid extreme imbalance within single sequences" |
| Correction bias role | steers *selection* only; "the original routing scores are retained for weighting the selected expert outputs" (§2.1.1) |
| Bias update rule | "updated independently according to their respective expert loads" after each step (§2.1.1); exact rule **not restated** — cites Wang et al. 2024a |
| Balance-loss formula | **paper does not specify** |
| Router scoring fn, `gate_temp`, `routed_scaling_factor`, `norm_topk_prob` | **paper does not specify** (zero hits for "softmax"/"softplus"/"sigmoid" in a routing context) |

### Released-model dims (§4.2.1) — reference only, we do not match these

40 layers (20 enc / 20 dec), `d=5120`. First 2 layers pure SWA; 18 enc layers CSA2 `m=2` in 3 groups
of 6 (Full + 5 Reuse); 20 dec layers CSA2 `m=1` in 5 groups of 4 (group 1 Full + 3 Reuse; groups 2–5
Reindex + 3 Reuse). Indexer 32 heads × dim 128, attention top-k 512. Main attention 64 heads × dim
512, q-compression 1280. Hierarchical pool 2048 blocks × 8 = 16,384 candidates. 8 output-projection
groups, intermediate attention output dim 1024, `n_win=128`. SwiGLU **clamped at 10**. MoE in every
block: 1 shared + 384 routed, expert intermediate 2304, 6 routed active. mHC `hc_mult=4`,
`hc_sinkhorn_iters=20`. 552B backbone params, 8B active prefill / 16B decode.
Vocab size, tokenizer, `rms_norm_eps`, dropout: **paper does not specify**.

### Data (§4.1)

7:1 text-only:multimodal token ratio; best-fit packing, padding rate ≤ `1e-4`; ultra-long docs
deterministically pre-split; sample overlap minimized across pre-training and context extension.

## 3. What scales down to our 28.5M smoke config — and what does not

| Paper item | Scales to 28.5M / `seq_len=512`? | What to do |
|---|---|---|
| Bias update speed `0.001` | **No** — **[inference]** it is tuned for ~447k steps and 384 experts. Over 100 steps the bias moves ≤0.1 against O(1) scores. With 8 experts + top-2 the load signal per step is also far coarser. | Implement the update rule, then sweep `0.003 → 0.03` for short runs. Watch per-expert token share, not loss. |
| Balance loss weight `1e-4` | **Partly** — **[inference]** the usual sequence-level form is normalized by expert count and tokens, so the *weight* is roughly scale-free; the *variance* is not. 512 tokens × top-2 over 8 experts gives a very noisy per-sequence load estimate vs 64K × top-6 over 384. | Start at `1e-4` as the paper says. If experts collapse, `1e-3` is the first thing to try (**[inference]**). |
| "Sparse from scratch at 64K, no dense warmup" (§1) | **Vacuous at our scale** — **[inference]**: with `sliding_window=128` and `index_topk=64`, every position < ~192 already sees essentially its whole causal context, and even at position 511 the SWA+top-k union covers most of it. We are not training sparse attention; we are training near-dense attention with a top-k op bolted on. | Do not treat a clean 512-token run as evidence the indexer works. To actually exercise sparsity, raise `seq_len` or drop `sliding_window`/`index_topk` hard (e.g. `--sliding_window=32 --index_topk=16`). |
| Batch 100.6M tokens | **No.** Ours is 8×512 = 4096 tokens, 4 orders of magnitude smaller. | Use `gradient_accumulation_steps` if you want a batch-size sweep to mean anything; otherwise accept the noise. |
| Warmup 2000 steps | **As a fraction, yes**: 2000 × 100.6M ≈ 201B ≈ **0.45%** of the 45T budget. Our default (10/100 = 10%) is ~20× more warmup by fraction. | **[inference]** for short runs keep proportionally long warmup — short runs are dominated by warmup anyway. Do not copy 0.45%. |
| WSD shape (constant → cosine 10× → tail) | **Yes, shape-wise.** | HF has no WSD-with-tail; `warmup_stable_decay` exists via `lr_scheduler_type="warmup_stable_decay"`. Cosine is a defensible substitute at 100 steps (**[inference]**). |
| LR `2.6e-4` | **No** — it is a Muon-RMS-matched LR at 552B params. | Keep an AdamW-appropriate LR for 28.5M. |
| AdamW `β2=0.95`, `ε=1e-20`, `wd=0.1` | **Yes** — these are optimizer-intrinsic, not scale-bound (**[inference]**). | Pass them explicitly; only `wd` matches our default today. |
| SwiGLU clamp at 10 (§4.2.1) | **Yes** — already matched: `swiglu_limit=10.0`. | nothing to do |
| Engram LR `5×`, Engram FP8 tables | Only if Engram is enabled; see [engram.md](engram.md). | — |
| Multimodal (7:1 ratio, image biases, ViT stages) | **Irrelevant** — TinyStories is text-only. `bias_vl` is dead weight in our runs. | — |
| Hierarchical candidate pool | **Post-training only** (§2.3.2): "training-aware and introduced in post-training". | Not a pre-training concern. |

## 4. Precision / QAT

| Tensor | Paper format | Cite |
|---|---|---|
| Indexer Q/K | FP4 QAT, **OCP MXFP4** (block 32, ue8m0) — inherited from DeepSeek-V4 | §2.4.4 |
| Main KV cache | E2M1 + one E4M3 scale per 16 channels (NVFP4 minus the second-level global scale); quantized **after** RoPE; same format for RoPE and non-RoPE parts | §2.4.4 |
| SWA KV cache | **FP8 retained** — "due to its sensitivity to quantization" | §2.4.4 |
| Engram tables + Engram K/V projections | FP8 | §2.4.2, §3.1.3 |
| Compute precision | never stated for training; §1 Fig.2 only weights BF16/FP8/FP4 as 1/0.5/0.25 for a *decode-FLOPs* accounting | §1 |

Range argument for dropping NVFP4's global scale (§2.4.4): largest trained RMSNorm weight ≈ 1, so
the 512-channel KV latent has $\|x\|_2 \lesssim \sqrt{512} \approx 22.6$ (RoPE-preserved), and the
**max magnitude observed during training ≈ 10** — a useful sanity target for our own KV latents.

**Implications for our bf16-autocast harness:**
- Main-KV FP4 QAT is a **post-training** stage upstream (§2.4.4: "To enable FP4 main KV cache
  storage in DeepSeek-V4.1-Flash, we introduce QAT during post-training"), so a pre-training
  smoke run with no KV quantization is faithful to the paper.
- **The paper contradicts itself on this.** §1 (p4) says the opposite: "At the cache-precision
  level, we use FP4 global KV caches **during training** with only marginal performance
  degradation." §2.4.4 places the same QAT in post-training. The report never reconciles the two.
  Only the *indexer* q/k FP4 QAT is unambiguous — inherited from V4 and implicitly always-on.
  If you later care whether main-KV quantization belongs in pre-training, this is unresolved
  upstream, not something our port got wrong.
- `keep_fp32()` already wraps `*Router` forwards in fp32; `_keep_in_fp32_modules_strict` pins
  `hc_attn_*`, `hc_ffn_*`, `attn_sink`, `bias`, `bias_vl`. That covers exactly the tensors the
  released checkpoint stores in F32 — consistent with §2.5 treating scaling factors as a distinct
  (AdamW, no-weight-decay) parameter class.
### Weight-decay hazard (measured, not inferred)

§2.5 splits parameters three ways for weight decay: norm weights **yes**, biases and scaling
factors **no**. Our harness gets two of those three wrong, because HF `Trainer.create_optimizer`
derives its no-decay set structurally — `get_parameter_names(model, ALL_LAYERNORM_LAYERS)` then
drop anything whose name contains `"bias"`. The mHC tensors are raw `nn.Parameter` attributes
named `hc_attn_scale` / `hc_ffn_base` / `attn_sink`: no `bias` substring, not inside an
`nn.LayerNorm`, so they fall through into the decay group.

Measured by running HF's exact selection logic on the 6-layer smoke model:

| param class | n | gets `wd=0.1`? | §2.5 says | |
|---|---|---|---|---|
| `hc_*_scale` | 12 | **ALL** | scaling factors: no | ✗ hazard |
| `hc_*_base` | 12 | **ALL** | biases: no | ✗ hazard |
| `attn_sink` | 6 | **ALL** | scaling factor: no | ✗ hazard |
| `hc_*_fn` | 12 | ALL | weight matrices: yes | ✓ |
| `RMSNorm.weight` | 29 | ALL | norm weights: **yes** | ✓ |
| `gate.bias`, `gate.bias_vl` | 12 | none | biases: no | ✓ |

**30 params are decayed that should not be.** `_init_weights` sets `hc_*_scale = 1`,
`hc_*_base = 0`, `attn_sink = 0`. Decoupled AdamW applies $p \mathrel{-}= \eta\,\lambda\,p$ each
step; at `lr=6e-4, wd=0.1` the per-step factor is $(1 - 6\times10^{-5})$:

| steps | `hc_*_scale` drift from 1.0 |
|---|---|
| 100 (smoke) | ×0.994 — negligible |
| 1 000 | ×0.94 |
| 10 000 | **×0.55** |

`scale` multiplies the mix logits before the sigmoid/softmax in `hc_mixes`, so shrinking it
flattens `pre`/`post` toward their midpoints and drives `comb` toward uniform — a slow, silent
damping of the hyper-connection routing that the loss curve would not obviously reveal. The
zero-init params are harmless at init but are fought once they learn an offset.

Invisible at our current 20-step smoke scale; real over thousands of steps. **This is the one
divergence that lives in the harness rather than the model** — every other gap is in the vendored
code. Fix would be an explicit no-decay param group in `mzoo/train/pt.py` matching these names
instead of the `bias` substring. Not applied. Affects `dsv4` identically.

## 5. Training stages (upstream order)

1. **DeepSeek-ViT alone** — SigLIP contrastive (47B pairs) → autoregressive fine-tune on a 4B MoE
   LLM (236B tokens), LLM then discarded (§4.2.2). Irrelevant to us.
2. **Backbone pre-training** — 45T tokens, 64K seq, vision encoder **frozen** (final norm + VL
   projector trainable), unfrozen at LR-decay onset with "a smaller learning rate" (value **not
   specified**, §2.5). **MTP omitted** (§2.1).
3. **Context extension 64K→1M at 34T tokens** — same run, during the cosine decay (§4.2.2).
4. **DSpark** — dedicated stage after pre-training, "we train only DSpark while keeping the backbone
   frozen" (§2.4.3). All DSpark hyperparameters **unspecified**.
5. **Post-training** — SFT → RL → OPD, "no algorithmic innovation" (§5.1); also where the
   hierarchical candidate pool (§2.3.2), main-KV FP4 QAT (§2.4.4) and train-aware SWA-bounded-replay
   simulation (§3.2.2) enter. DSpark keeps training alongside the backbone "without propagating
   gradients from the DSpark objective into the backbone" (§2.4.3).

**Only stage 2 is in scope for us.** `num_nextn_predict_layers=0` in `model.build()` matches §2.1.

## 6. Stability / numerical notes

The paper is almost silent here. Everything it says:

- §4.2.2: "45T tokens of multimodal data **with no instability**". No spike/restart/rollback discussion.
- §4.2.1: SwiGLU **clamped at threshold 10**.
- §2.1.1: modality-split correction biases "contributes to stable and efficient multimodal training".
- §2.5 / Alg. 1: Sinkhorn rows with $\rho_i \le \tau\bar\rho$ are masked "for numerical stability";
  `τ=1e-3`, `ε=1e-20`.
- §2.4.4: the KV magnitude bound above (max observed ≈10) is the only activation-scale statement.
- §3.1.2: shared indexers use shadow replicas with one logical owner for optimization/checkpointing,
  kept consistent by "parameter synchronization and gradient aggregation". Single-GPU this is
  automatic, but it confirms shared indexer params must receive **summed** gradients from every
  consuming layer — which plain module reuse already gives us.
- **No loss-spike, divergence, router-collapse, z-loss, or gradient-clipping guidance exists.**
- `rms_norm_eps=1e-20` in our config is **not** from the paper — see §8 below.

## 7. Proposed starting recipe (PROPOSAL — not paper fact)

Paper-backed parts are cited; the rest is my judgement for our harness.

```
# smoke (20 steps), unchanged arch defaults
just src/mzoo/ pt --arch=owlet1 --max_steps=20 --per_device_train_batch_size=8 \
  --adam_beta1=0.9 --adam_beta2=0.95 --adam_epsilon=1e-20 --weight_decay=0.1   # §4.2.2

# small run (~2k steps) once the balance mechanisms exist
just src/mzoo/ pt --arch=owlet1 --exp=260912-0001-balance --run=u0.01 \
  --max_steps=2000 --per_device_train_batch_size=8 --gradient_accumulation_steps=8 \
  --learning_rate=6e-4 --lr_scheduler_type=warmup_stable_decay --warmup_steps=100 \
  --adam_beta2=0.95 --adam_epsilon=1e-20 --weight_decay=0.1 --max_grad_norm=1.0
```

- `lr=6e-4` (harness default): keep. Paper's `2.6e-4` is Muon-at-552B and does not transfer.
- `β2=0.95`, `ε=1e-20`, `wd=0.1`: straight from §4.2.2.
- `max_grad_norm=1.0`: HF default; the paper says nothing about clipping.
- Balance knobs (once implemented): start at paper values `bias_speed=0.001`,
  `balance_loss_weight=1e-4` (§4.2.2), then sweep `bias_speed ∈ {0.001, 0.01, 0.03}` — my
  expectation is that only the two larger values do anything at 2k steps.
- Log per-expert token share every `logging_steps`; it is the only way to tell the balance
  machinery is alive, since it barely moves the CE loss.
- To actually stress the indexer, add a second run with `--sliding_window=32 --index_topk=16`.
- Do **not** add an indexer KL/auxiliary loss to "match the paper" — the paper has none.

## 8. Where the paper is silent (things we are guessing at)

Gradient clipping · weight init scheme and std · total step count · vocab size and tokenizer ·
`rms_norm_eps` (the two `1e-20` values in §2.5/§4.2.2 are the **AdamW and Sinkhorn optimizer
epsilons**, not a norm epsilon — our config default is from the HF port, not the paper) ·
router scoring function (`sqrtsoftplus`: zero hits), `gate_temp`, `routed_scaling_factor`,
`norm_topk_prob` · exact sequence-level balance-loss formula · exact correction-bias update rule ·
context-extension method for 64K→1M · the "smaller learning rate" for the unfrozen vision encoder ·
separate base LRs for Muon vs AdamW groups (the RMS-`0.18` rescale implies one shared LR) ·
dropout (never mentioned) · any indexer-specific loss · attention-sink initialization ·
all DSpark and all SFT/RL/OPD hyperparameters · training compute dtype (BF16/FP8) for the backbone.
