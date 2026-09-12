# config.py

Verbatim copy of the vendored `configuration_deepseek_v41.py`. `DeepseekV41TextConfig` is the
text-only config used by `build()`; `DeepseekV41Config` is the composite image-text wrapper
(`text_config` + `vision_config`) that only exists so released checkpoints parse.

## Field tables (`DeepseekV41TextConfig`)

`attribute_map`: `intermediate_size` -> `moe_intermediate_size`, `num_local_experts` ->
`n_routed_experts` (standard HF names read by the shared-expert MLP base class and FP8/TP).
V4.1 has **no** `kv_lora_rank`, `qk_nope_head_dim`, `v_head_dim`, `first_k_dense_replace`,
`moe_layer_freq`, `n_group` or `topk_group` — every layer is MoE and KV is a single latent.

### Core shape

| name | default | meaning |
|---|---|---|
| `vocab_size` | 129280 | token vocabulary |
| `hidden_size` | 5120 | residual width (per hyper-connection stream) |
| `num_hidden_layers` | 40 | backbone layers; DSpark draft layers appended after |
| `max_position_embeddings` | 1048576 | max context |
| `tie_word_embeddings` | `False` | tie LM head to embedding |

### Attention / MLA / RoPE

| name | default | meaning |
|---|---|---|
| `num_attention_heads` | 64 | query heads |
| `num_key_value_heads` | 1 | one KV latent broadcast to all heads |
| `head_dim` | 512 | per-head dim (= KV latent dim) |
| `q_lora_rank` | 1280 | low-rank query projection rank |
| `qk_rope_head_dim` | 64 | trailing rotary channels per head |
| `o_groups` | 8 | head groups in grouped low-rank output projection |
| `o_lora_rank` | 1024 | per-group intermediate dim of output projection |
| `rope_theta` | 10000.0 | RoPE base for sliding-window branch (`main`) |
| `compress_rope_theta` | 160000.0 | RoPE base for compressed branch (`compress`) |
| `rope_scaling` | `None` | YaRN dict applied **only** to `compress` |
| `rope_parameters` | `None` | derived `{main, compress}`; `partial_rotary_factor = qk_rope_head_dim / head_dim` |

### CSA2 sparse attention

| name | default | meaning |
|---|---|---|
| `sliding_window` | 128 | sliding-window ring kept by every layer |
| `compress_ratios` | `None` -> derived | per-layer ratio: 0 window-only, 1 full-res shared KV, r>1 pooled |
| `kv_source_layer_ids` | `None` -> derived | layers owning a compressor; publish shared compressed KV |
| `index_source_layer_ids` | `None` -> derived | layers running the indexer; publish top-k indices |
| `candidate_source_layer_id` | `None` -> last KV source | two-level top-k candidate source; `< 0` disables |
| `candidate_topk_blocks` | 2048 | candidate blocks kept |
| `candidate_block_size` | 8 | compressed positions per block |
| `index_n_heads` / `index_head_dim` | 32 / 128 | indexer heads / head dim |
| `index_topk` | 512 | compressed positions each query attends to |

### MoE

| name | default | meaning |
|---|---|---|
| `moe_intermediate_size` | 2304 | expert FFN width (routed and shared) |
| `n_routed_experts` | 384 | routed experts |
| `n_shared_experts` | 1 | always-on experts |
| `num_experts_per_tok` | 6 | active routed experts |
| `scoring_func` | `"sqrtsoftplus"` | router activation (`sqrtsoftplus` / `softmax` / `sigmoid`) |
| `topk_method` | `"noaux_tc"` | bias steers selection only, weights from unbiased scores |
| `norm_topk_prob` | `True` | renormalise selected weights (`+1e-20` floor) |
| `routed_scaling_factor` | 1.5 | multiplier on routed output |
| `gate_temp` | 1.0 | temperature on router logits |
| `swiglu_limit` | 10.0 | clamp on expert SwiGLU pre-activations |

### mHC, engram, DSpark, misc

| name | default | meaning |
|---|---|---|
| `hc_mult` | 4 | parallel residual streams |
| `hc_sinkhorn_iters` | 20 | Sinkhorn-Knopp iterations on combine matrix |
| `hc_eps` | 1e-6 | Sinkhorn / `pre` gate floor |
| `engram_layer_ids` | `None` -> `[]` | engram layers; empty disables |
| `engram_num_embeddings` | `None` -> `[0]*len(engram_layer_ids)` | table rows per engram layer |
| `engram_vocab_size` / `_max_ngram_size` / `_n_heads` / `_head_dim` / `_pad_id` / `_compressed_vocab_size` | 16000000 / 4 / 8 / 256 / 2 / 99092 | hash-table geometry |
| `num_nextn_predict_layers` | 3 | DSpark draft layers (unimplemented in modeling) |
| `dspark_block_size` / `dspark_noise_token_id` | 5 / 128799 | draft block / filler token |
| `dspark_target_layer_ids` | `None` -> derived | backbone layers feeding the draft model |
| `dspark_markov_rank` / `dspark_n_routed_experts` / `dspark_num_experts_per_tok` | 256 / 128 / 3 | draft-side MoE |
| `rms_norm_eps` | 1e-20 | every RMSNorm (unusually small) |
| `initializer_range` | 0.02 | init std |
| `use_cache` | `True` | return past KV |
| `layer_types` | `None` -> derived | `"shared_compressed_attention"` at KV sources, else `"sliding_attention"` |

## `__post_init__` invariants

When `compress_ratios` is `None` and `num_hidden_layers != 40`, the released pattern
(2 window-only, ratio-2 "encoder" block, ratio-1 "decoder" block) is scaled to $n$:

```
n_slide = min(2, n);  n_enc = (n - n_slide + 1) // 2
compress_ratios = [0]*n_slide + [2]*n_enc + [1]*(n - n_slide - n_enc)   (+ [0]*num_nextn)
kv_source_layer_ids    = first layer of each same-ratio run with ratio > 0
index_source_layer_ids = kv_source_layer_ids           (released 40-layer case: [2,8,14,20,24,28,32,36])
candidate_source_layer_id = kv_source_layer_ids[-1]    (-1 if none)
dspark_target_layer_ids   = last 3 non-KV-source layers
layer_types[i] = "shared_compressed_attention" if i in kv_sources else "sliding_attention"
```

Verified by running the config: **n=6** gives `compress_ratios=[0,0,2,2,1,1]`,
`kv_source_layer_ids=[2,4]`, `index_source_layer_ids=[2,4]`, `candidate_source_layer_id=4`,
`dspark_target_layer_ids=[1,3,5]` — layers 3 and 5 reuse the cache of 2 and 4, so the
sharing path runs. **n=4** gives `[0,0,2,1]` with `kv=[2,3]`: every compressed layer is its
own source and sharing never executes, which is why `build()` defaults to 6 layers.
**n=40** reproduces the released `[0,0]+[2]*18+[1]*20`, `kv=[2,8,14,20]`, `dspark=[37,38,39]`.

`ValueError` is raised when:
- `len(compress_ratios) < num_hidden_layers`, or `> num_hidden_layers + num_nextn_predict_layers`
- any `kv_source_layer_ids` entry is outside `[0, n)` or has `compress_ratios[i] == 0`
- any `index_source_layer_ids` entry has ratio 0, or has no KV source at or before it
- `candidate_source_layer_id >= 0` and not in `index_source_layer_ids`
- any layer with ratio `> 0` has no KV source at or before it
- `len(engram_num_embeddings) != len(engram_layer_ids)`

RoPE: `rope_parameters` becomes `{main: rope_theta, compress: rope_scaling + compress_rope_theta}`
with `attention_factor=1.0` when `compress` is yarn (no mscale). `DeepseekV41Config.__post_init__`
also adopts flat text-field kwargs (a saved text-only `config.json`) into `text_config` rather
than falling back to release-sized defaults.
