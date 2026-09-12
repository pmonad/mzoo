# attention.py

CSA2 attention block of the V4.1 backbone: low-rank query, a single shared K=V latent per token, a per-head
sink in the softmax, and a grouped low-rank output projection. Layers with a compressed branch additionally
attend over a group-shared compressed KV (published by a KV-source layer, selected by an indexer's top-k).

## Shapes

| symbol | shape / value | meaning |
|---|---|---|
| `B`, `S` | 8, 512 | batch, query length |
| `D` | 256 | `hidden_size` |
| `H`, `d` | 4, 64 | `num_attention_heads`, `head_dim` — dim of each query head **and** of the KV latent |
| `d_r` | 16 | `qk_rope_head_dim`: trailing rotary channels of `d`; leading `d - d_r = 48` are "nope" |
| `r_q`, `r_o`, `G` | 128, 128, 2 | `q_lora_rank`, `o_lora_rank`, `o_groups` |
| `W` | 128 | `sliding_window` (window mask is built at model level) |
| `N_c` | 256 at layers 2/3 | compressed entries appended on the KV axis (`S / compress_ratio` per source) |
| `x` | `[B, S, D]` | normalised residual input |
| `q` | `[B, H, S, d]` | rotated queries |
| `kv` | `[B, 1, T, d]` | rotated latent, `T = S` (window only) or `S + N_c` |
| `attn_sink` | `[H]` | learnable per-head sink logit $\sigma_h$ |
| `wo_a.weight` | `[G r_o, H d / G]` = `[256, 128]` | block-diagonal output stage |

V4.1 naming: there is no `kv_lora_rank` / `qk_nope_head_dim` / `v_head_dim` (V2/V3). `num_key_value_heads=1` is
the single latent; `num_key_value_groups = H` and the `[B,1,T,d]` latent broadcasts against all heads inside the
matmuls (never materialised `H` times).

## `DeepseekV41GroupedLinear`

`nn.Linear` subclass whose weight `[G r_o, H d / G]` is read as `G` independent blocks $W_g \in \mathbb{R}^{Hd/G \times r_o}$.
Input `[..., G, Hd/G]` (contiguous head chunks: group $g$ = heads $gH/G \dots (g+1)H/G - 1$) is multiplied
per group with `bmm`, giving `[..., G, r_o]`; nothing crosses groups here — mixing is `wo_b`'s job.

$$y_g = x_g W_g, \qquad g = 0..G-1$$

Constraint: `o_groups` must divide `H * head_dim` (`in_features_per_group = Hd // G` and the
`reshape(B, S, G, -1)` in the caller). At full scale this replaces a `32768 -> D` dense projection with
`G` blocks of `32768/G -> r_o` plus one `G r_o -> D` mix.

## `eager_attention_forward`

Shared-KV attention with the sink as one extra logit column that is used in the normaliser and then dropped.
`key is value` (same tensor), so the output is a convex combination of rotated latents.

$$s_{h,ij} = \frac{q_{h,i} \cdot k_j}{\sqrt{d}} + M_{ij}, \qquad
p_{h,ij} = \frac{e^{s_{h,ij}}}{\sum_{j'} e^{s_{h,ij'}} + e^{\sigma_h}}, \qquad
o_{h,i} = \sum_j p_{h,ij}\, k_j$$

The sink $\sigma_h$ is unscaled and unmasked; a fully masked row therefore yields a finite (all-zero) output
instead of NaN. Logits are max-subtracted before `softmax`. Returns `[B, S, H, d]` and the sink-free probabilities.

## `DeepseekV41Attention`

Per token: $q = W_{qb}\,\mathrm{RMS}(W_{qa} x)$ reshaped to `H` heads; $kv = \mathrm{RMS}(W_{kv} x) \in \mathbb{R}^{d}$.
Both get partial interleaved RoPE (only the last `d_r = 16` channels rotate, as `(even, odd)` pairs; `cos/sin`
are `[B, S, d_r/2]`), with `main` rope on ratio-0 layers and `compress` rope (`compress_rope_theta`) on any layer
with a compressed branch. The window latent is then **fp8 fake-quantised** (block 32, ue8m0 scales, rope tail
included) before it enters the cache. Output path:

$$\tilde o_i = R(-\theta_i)\, o_i, \qquad
y = W_{ob}\, \big[\, W_{oa,0}\tilde o_i^{(0)} \,\|\, \dots \,\|\, W_{oa,G-1}\tilde o_i^{(G-1)} \big]$$

The `inverse=True` rotation applies $R(-\theta_i)$ (query position) to the rope slice of each head. Since
$o_i = \sum_j p_{ij} R(\theta_j) kv_j$, this leaves $\sum_j p_{ij} R(\theta_j - \theta_i) kv_j$ — relative
position only — while the shared cache keeps storing latents in rotated form.

Branch structure (the `shared` dict is fresh per model forward; `cache_layer` is `None` when `use_cache=False`):

```
q, kv = project + rope; kv = fp8_fake_quant(kv); kv = cache.update(kv)   # every layer
if compress_ratio > 0:                                                   # layers 2..5 in smoke cfg
    if is_kv_source:      latent, pos0 = compressor(x, cache_layer)      # pre-rope, [B, S/ratio, d]
    if is_index_source:   indexer(x, q_residual, latent, pos0, ...)      # publishes shared["topk_bias"]
    block_bias = shared.get("topk_bias")                                 # [B,1,S,N_c]: 0 kept / -inf dropped
    if latent is not None:                                               # KV source only
        rotated = rope(latent, compress rope at pos0 + ratio*k)          # latent positions, own rotary
        rotated = fp4_fake_quant(rotated, block 16, e4m3 scales)         # [B,1,N_c,d]
        cache_layer.update(...)  or  shared["compress_kv"] (+)= rotated
    if is_kv_source and cache_layer: shared["compress_kv"] = running cache
    kv = cat([kv, shared["compress_kv"]], dim=2)                         # T = S + N_c
    mask = cat([mask, block_bias]) if block_bias else pad(mask, -inf)    # extend the [B,1,S,S] window mask
attn = eager(q, kv, kv, mask, sink)   ->  inverse rope  ->  wo_a (grouped)  ->  wo_b
```

Three layer kinds: plain sliding-window (`ratio 0`, layers 0-1: `T = S`, `main` rope); KV source (layers 2, 4:
owns `compressor` + `compress_rotary`, and here also the `indexer`, since the smoke config's index sources equal
its KV sources); consumer (layers 3, 5: no compressor/indexer, reads `compress_kv` and `topk_bias` from `shared`).
The indexer sees the pre-rope latent, so it must run before the rotation. The compress rotary is owned by the
source layer because latent positions `pos0 + ratio*k` are only known after the compressor runs.

## Notes / gotchas

- **Eager only.** `decoder.py` sets `_supports_flash_attn = _supports_sdpa = _supports_flex_attn = False`:
  flash-attn caps `head_dim` at 256 (V4.1 uses 512 at scale); SDPA has no per-head sink term; and the compressed
  branch concatenates onto the KV axis inside the block, after the model-level mask was built.
- `o_groups` must divide `heads * head_dim`.
- The fp8/fp4 fake quant is applied even in unquantised runs, but silently skipped when `head_dim` is not a
  multiple of the block size (32 / 16). With `d = 64` both are active.
- Without a cache (`use_cache=False`, the smoke path) a second KV source **appends** to `shared["compress_kv"]`
  and `shared["index_k"]` instead of replacing them: observed at layer 4, `compress_kv` is `[B,1,768,64]` =
  256 ratio-2 entries from layer 2 + 512 ratio-1 entries of its own, and the indexer's visibility rule
  `entry < (pos+1)/ratio` is then applied with `ratio = 1` across both. The cache path replaces
  (`shared["compress_kv"] = cache_layer.compressed_kv[...]`). Unclear: whether this is intended; not changed here.
- If `attention_mask` is `None` the compressed entries receive no mask at all; the `-inf` pad / bias concat only
  runs when the mask is a tensor.
