# 0012 dsv4 model: `shared` dict accumulates compressed KV across kv-source groups (confirmed, fixed)

- status: done (confirmed + fixed 2026-09-14, `src/mzoo/archs/dsv4/model_test.py`)
- depends on: nothing; model-side, `src/mzoo/archs/dsv4/modeling_deepseek_v41.py`

## Observation

In a whole-model forward without a cache (`past_key_values is None`), one mutable `shared` dict
is passed to every layer and never reset at a kv-source group boundary. `DeepseekV41Attention.forward`
does `shared["compress_kv"] = rotated if shared.get("compress_kv") is None else torch.cat([...])`,
so a later kv-source layer (a new group, possibly with a different `compress_ratio`) appends its
latents onto the previous group's compressed KV instead of starting fresh. The same pattern exists
for `shared["index_k"]`. With the default 6-layer smoke schedule (`archs/dsv4/model.py`) layer 4
opens a `ratio=1` group after a `ratio=2` group, so its KV axis would mix both.

## Why it was not chased

CLAUDE.md: no bug chasing without approval. The golden reference test sidestepped it with a
single-group 2-layer config (`golden_ref_test.py::_build_tiny_model`).

## To verify

- Run the 6-layer smoke config once with and once without a `DynamicCache`; compare layer-4
  attention inputs (`kv.shape[2]`) captured via `eager_attention_forward` -- with a cache the
  per-layer `cache_layer.compressed_kv` is used and the axis should be `S + S//1`, without it
  `S + S//2 + S//1`.
- If confirmed: reset `shared["compress_kv"]` / `shared["index_k"]` / `shared["topk_bias"]` /
  `shared["candidates"]` at every kv-source layer (the group's first layer), or key the dict by
  source layer id. Small change, model side; upstream if the vendored file tracks a reference.

## Result (2026-09-14)

Confirmed. 4-layer fp32 model, `compress_ratios=[0,2,2,1]`, `kv_source_layer_ids=[1,3]`,
heads 4 / head_dim 64 / B 2 / S 130 / sliding_window 96, one prefill forward each way.
Layer 3 (the second kv source, ratio 1) got `key.shape[2] = 325 = 130 + 65 + 130` without a
cache (layer 1's 65 ratio-2 entries still in `shared`) vs `260 = 130 + 130` with a cache;
`shared["compress_kv"]` was `[2,1,195,64]` vs `[2,1,130,64]`, `index_k` likewise, and the
logits differed by `max|Δ| = 0.388`. Layers 0-2 matched in both runs.

Fix: in `DeepseekV41Attention.forward`, a kv-source layer with `cache_layer is None` pops
`compress_kv` / `index_k` / `topk_bias` / `candidates` from `shared` before the compressor
runs (cache path untouched). Both runs now agree bit-for-bit (`max|Δ| = 0.0`). This is a
**backport, not a divergence**: the unmerged PR head (62d7ebd, 2026-09-13) already clears
`shared["compress_kv"]` at the same spot and `shared["index_k"]` in the indexer — the rest
of that PR head has moved on a lot (indexer publishes `topk_idx` and attention gathers the
selected entries instead of masking), so only the reset was taken.

## Notes for future sessions

- **There is no dense-FFN knob in this arch** (README: "Every backbone layer is MoE — no
  `first_k_dense_replace` / `moe_layer_freq`"). The smallest FFN that instantiates, and what
  test models should use, is `n_routed_experts=1, num_experts_per_tok=1` (+ the always-built
  shared expert). No grouped/repeated-tensor MoE error was hit with it.
- `DynamicCache(config=...)` still cannot build this arch's cache layers (README gotcha,
  `DeepseekV41CSACache.__init__() missing 1 required positional argument: 'config'`). Build
  the cache by hand: `Cache(layers=[DeepseekV41CSACache(cfg) if t == "shared_compressed_attention"
  else DynamicSlidingWindowLayer(sliding_window=cfg.sliding_window) for t in cfg.layer_types])`
  — see `_build_cache` in `src/mzoo/archs/dsv4/model_test.py`.
