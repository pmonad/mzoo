# 0012 dsv4 model: `shared` dict may accumulate compressed KV across kv-source groups (suspected)

- status: todo, unverified (observation from the golden-reference work, 2026-09-14)
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
