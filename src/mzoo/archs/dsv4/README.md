# dsv4

Vendored DeepSeek-V4.1 text backbone, copied from an **unmerged** HuggingFace
PR ([transformers#48721](https://github.com/huggingface/transformers/pull/48721)),
tracking `deepseek-ai/DeepSeek-V4.1-Flash`. `model.py` exposes `build(tok,
seq_len, ...)` for `mzoo.train.pt`. Local env: transformers 5.17.0, torch
2.14.0+cu130, single GB10. A 28.5M-param smoke config trained 20 steps,
loss 10.31 -> 8.00 — the backbone runs, but see the open issues below before
trusting a real run.

## Blocking gotchas (already worked around in `model.py`)

| Issue | Symptom | Workaround |
|---|---|---|
| `use_cache=True` (default) | `TypeError: DeepseekV41CSACache.__init__() missing 1 required positional argument: 'config'` — transformers 5.17 builds dynamic cache layers via `DYNAMIC_LAYER_TYPE_MAPPING[layer_type](**layer_kwargs)` without passing `config`, but the vendored CSA cache layer requires it positionally | `use_cache=False`. Training needs no cache; **generation/inference is currently broken** on this vendored code + transformers 5.17 combo |
| `dspark_noise_token_id` default (128799) | out of range for a 32k vocab, trips config validation | point at `tok.eos_token_id` — inert either way, see DSpark below |
| `num_hidden_layers < 6` | CSA2 layer-sharing path never executes: at 4 layers `compress_ratios` auto-derives to `[0,0,2,1]`, where every compressed layer is its own KV source | use `>= 6` layers, e.g. 6 derives `[0,0,2,2,1,1]` with `kv_source_layer_ids=[2,4]`, giving real consumer layers (3, 5) |
| `o_groups` not dividing `heads * head_dim` | shape error in output projection | pick `o_groups` as a divisor |

## Not implemented upstream (in this vendored copy)

- **DSpark / multi-token prediction is entirely absent from the modeling code.** The config carries a full field set (`num_nextn_predict_layers`, `dspark_block_size`, `dspark_noise_token_id`, `dspark_markov_rank`, `dspark_n_routed_experts`, `dspark_num_experts_per_tok`, `dspark_target_layer_ids`), but grepping the modeling code for `nextn`/`dspark` returns nothing. `num_nextn_predict_layers=0` is therefore trivially safe (set in `build()`). Implementing MTP is on whoever wants it.
- **No MoE load-balancing loss.** `router_aux_loss_coef` / `output_router_logits` exist as config fields, but nothing computes or adds a balancing loss — total loss is plain cross-entropy from `labels`. `router_logits` are captured for inspection only. Our `mzoo.train.trainer.Trainer` MoE logging (`moe/aux_loss`, per-layer expert load) keys off an `aux_loss` output attribute that's never populated, so it never fires here. **Expert load is unmonitored** — worth fixing before a real run.

## No gradient (verified: 20 no-grad params in the 6-layer smoke config)

- **Sparse-attention indexer is frozen.** All 8 indexer params (`wq_b`, `wk`, `k_norm`, `weights_proj`) on the KV-source layers get zero gradient, so the selector stays at **random init** for the whole run. Two causes: top-k selection is hard/non-differentiable, and — the decisive one — `_fake_quant_fp4_block(x, 32)` on the indexer q/k returns a tensor **fully detached from the autograd graph** (`requires_grad=False`), while the fp8 path on window KV is a proper straight-through estimator. The model still trains, but around a random sparse-attention selector — the most significant open issue for a real run. Note the tech report describes **no** auxiliary/KL loss for the indexer (an earlier version of this file wrongly claimed one, by analogy with V3.2-Exp's DSA); §3.1.2 shows the indexer is trained jointly by gradient, so the gap here is the broken STE, not a missing loss.
- **MoE router `gate.bias` / `gate.bias_vl` (12 params) get no gradient — expected**, not a bug: `topk_method="noaux_tc"` updates these correction biases by a separate rule, not SGD. Caveat: nothing in the vendored code appears to implement that separate update rule either. Treat as "expected to be grad-free, but verify the bias-update rule actually exists" before relying on it.

## Harness interaction

- **`eval_loss` is not logged for this arch** (eval line has only runtime metrics), unlike `dense` which logs it fine on the same harness. Cause: the vendored forward signature exposes both `labels` and `shift_labels`, so HF's `find_labels` returns `["labels", "shift_labels"]`; `Trainer.prediction_step` requires *all* label names to be present in the inputs, and our collator only supplies `labels`, so it decides there are no labels and skips loss computation. **Fix (not yet applied):** set `label_names=["labels"]` in `TrainingArguments` in `mzoo/train/pt.py`.

## Constraints (by design, leave alone)

- **Eager attention only** — `_supports_flash_attn` / `_supports_sdpa` / `_supports_flex_attn` are all False. flash-attn caps `head_dim` at 256 while V4.1 uses 512; SDPA has no per-head sink term; the compressed-attention branch concatenates onto the KV axis inside the block, after the model-level mask is built.
- `rms_norm_eps` defaults to 1e-20 — not a typo, but its provenance is the HF port, not the tech report (the paper states no RMSNorm epsilon; its two `1e-20` values are the AdamW and Sinkhorn optimizer epsilons).
- This is **V4.1**, not V2/V3 MLA naming: no `kv_lora_rank`, `qk_nope_head_dim`, `v_head_dim`, `first_k_dense_replace`, `moe_layer_freq`, `n_group`, `topk_group`. Every backbone layer is MoE — no dense/MoE layer schedule. Attention shape comes from `head_dim` + `q_lora_rank` + `qk_rope_head_dim`.
- `__post_init__` auto-derives and cross-validates `compress_ratios`, `kv_source_layer_ids`, `index_source_layer_ids`, `candidate_source_layer_id`, `layer_types` from the layer count, with a web of invariants between them. Leave them `None` — hand-picking means replicating all the invariants yourself.
- Engram (conditional memory) is disabled by default (`engram_layer_ids=[]`) and we leave it that way.

## Open questions before a real run

- Sparse-attention indexer never trains (random init) — the fp4 fake-quant on its q/k detaches the graph; needs a straight-through estimator.
- MoE expert load is unmonitored (no aux loss, harness hook never fires).
- `eval_loss` isn't logged. Fixed in `owlet1` (its forward takes `shift_labels` via `**kwargs`, so `find_labels` returns just `["labels"]`); left as-is here to keep this baseline frozen.
- Generation/inference is broken (`use_cache` incompatibility) — training-only for now.
