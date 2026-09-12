# model.py

`build(tok, seq_len, **kw)` constructs a tiny `DeepseekV41ForCausalLM` (~28.5M params); the
`mzoo.train.pt` harness calls it via `importlib.import_module(f"mzoo.archs.{arch}").build(...)`.

## Signature

| kwarg | default | maps to |
|---|---|---|
| `tok` | required | `vocab_size=len(tok)`, `bos/eos_token_id` |
| `seq_len` | required | `max_position_embeddings` |
| `layers` | 6 | `num_hidden_layers` |
| `hidden` | 256 | `hidden_size` |
| `heads` | 4 | `num_attention_heads` |
| `head_dim` | 64 | `head_dim` |
| `qk_rope_head_dim` | 16 | `qk_rope_head_dim` |
| `q_lora_rank` | 128 | `q_lora_rank` |
| `o_lora_rank` | 128 | `o_lora_rank` |
| `o_groups` | 2 | `o_groups` |
| `expert_ffn` | 256 | `moe_intermediate_size` |
| `routed` | 8 | `n_routed_experts` |
| `top_k` | 2 | `num_experts_per_tok` |
| `shared` | 1 | `n_shared_experts` |
| `index_n_heads` | 4 | `index_n_heads` |
| `index_head_dim` | 32 | `index_head_dim` |
| `index_topk` | 64 | `index_topk` |
| `sliding_window` | 128 | `sliding_window` |
| `candidate_topk_blocks` | 64 | `candidate_topk_blocks` |

Fixed (not kwargs): `tie_word_embeddings=True`, `num_nextn_predict_layers=0`,
`use_cache=False`, `dspark_noise_token_id=tok.eos_token_id`. `compress_ratios`,
`kv/index_source_layer_ids` and `layer_types` are left `None` and auto-derived (see `config.md`).

## Why each workaround

- **`use_cache=False`**: the CSA cache layer is not constructible by transformers 5.17's
  `DynamicCache` (`TypeError: DeepseekV41CSACache.__init__() missing 1 required positional
  argument: 'config'`). Generation/inference is therefore broken; training is unaffected.
- **`num_nextn_predict_layers=0`**: DSpark/MTP draft layers are entirely unimplemented in the modeling code.
- **`dspark_noise_token_id=tok.eos_token_id`**: inert (no DSpark), but the 128799 default is
  out of range for the 32k TinyStories vocab and would fail validation.
- **`layers=6`**: at 4 layers the derived schedule `[0,0,2,1]` makes every compressed layer
  its own KV source, so the shared-cache path never runs; 6 yields `[0,0,2,2,1,1]` with
  `kv_source_layer_ids=[2,4]` and real reuse at layers 3 and 5.
- **`o_groups=2`**: must divide `heads * head_dim` (4 * 64 = 256).

## Invocation

```
just src/mzoo/ pt --arch=owlet1 --proj=owlet1 --exp=smoke --max_steps=20
```
