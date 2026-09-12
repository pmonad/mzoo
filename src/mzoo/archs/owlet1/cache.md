# cache.py

CSA2 (compressed sparse attention v2) state: a per-layer KV cache that adds a shared
compressed branch on top of the sliding-window ring, plus the compressor that pools
`compress_ratio` tokens into one KV latent.

## Shapes

| symbol | shape | meaning |
|---|---|---|
| $x$ | `[B, S, D]` | hidden states; B=8, S=512, D=256 |
| $W$ | scalar | `sliding_window` = 128 |
| $d$ | scalar | `head_dim` = 64 (the single K=V head) |
| $d_i$ | scalar | `index_head_dim` = 32 |
| $r$ | scalar | `compress_ratios[layer]`; 0 / 2 / 1 |
| $G$ | scalar | groups completed so far, $G = \lfloor S_{\text{seen}} / r \rfloor$; 256 at r=2, 512 at r=1 |
| `keys` = `values` | `[B, 1, <=W-1, d]` | sliding-window ring (K == V, one head) |
| `buffer_kv["compressor"]` | `[B, <r, d]` | partial-group leftovers (fp32) |
| `buffer_gate["compressor"]` | `[B, <r, d]` | matching gate logits; `None` at r=1 |
| `compressed_kv["compressor"]` | `[B, 1, G, d]` | running post-RoPE, fp4-quantised latents (filled by attention.py) |
| `compressed_kv["indexer"]` | `[B, 1, G, d_i]` | running indexer keys (filled by indexer.py) |
| `entry_count["compressor"]` | int | $G$; next group starts at token $G \cdot r$ |

## `compress_ratios` / KV-source sharing

`compress_ratios[i] = 0` means layer $i$ is sliding-window only. A value $r > 0$ means the
layer also attends over KV latents where $r$ consecutive tokens are pooled into one entry
($r=1$: every token is an entry but the projection is done once, not per layer).
Only the **KV source** layers (`kv_source_layer_ids`, the first layer of each same-ratio
run) own a `DeepseekV41Compressor` and a `DeepseekV41CSACache`; every later layer with the
same ratio holds a plain sliding layer and reads the source's entries through the
per-forward `shared` dict. Smoke config: ratios `[0,0,2,2,1,1]`, sources `[2,4]`, so layer 2
publishes 256 ratio-2 latents that layer 3 reuses, and layer 4 publishes 512 ratio-1 latents
that layer 5 reuses.

## `DeepseekV41CSACache`

Subclass of `DynamicSlidingWindowLayer` (`_layer_type = "shared_compressed_attention"`).
`update` is the window ring; `store_compression_weights` / `update_compressor_states` are the
group state. The three batch-reorder hooks also permute the buffers and compressed entries.

```
update(k, v):                                      # K == V; attention mask selects the window
    full  = cat(keys, k)   on the seq axis         # returns everything seen: cached (<=W-1) + new S
    keys  = values = full[-(W-1):]                 # keep the last 127 entries
    return full, full

store_compression_weights(name, kv, gate, r):
    first_pos = entry_count[name] * r              # absolute token index of the first new group
    kv, gate  = cat(buffer, kv), cat(buffer_gate, gate)
    usable    = (len(kv) // r) * r                 # longest group-aligned prefix
    buffer, buffer_gate = kv[usable:], gate[usable:]   # remainder carries to the next call
    return kv[:usable], gate[:usable], first_pos

update_compressor_states(name, new):               # new: [B, 1, g, *]
    compressed_kv[name] = cat(compressed_kv[name], new) on axis 2
    if name == "compressor": entry_count += g
    return compressed_kv[name]
```

Sliding window and compressed entries meet in `attention.py`: the KV axis is
`cat(window_kv, compressed_kv)`, the model-level sliding-causal mask covers the window part,
and the indexer's `topk_bias` (0 / $-\infty$ per query and group) is appended to the mask
for the compressed part. A ratio-$r$ layer therefore sees the last $W$ raw tokens plus up to
`index_topk` selected compressed groups from the whole prefix.

## `DeepseekV41Compressor`

Per KV-source layer. Projects $x$ to a single $d$-dim latent per token and, for $r>1$,
pools each group of $r$ tokens with a learned per-channel softmax gate (fp32), then RMSNorm.
Returns the latents **before** RoPE (the indexer keys are derived from the unrotated form)
and the absolute position of the first returned group.

$$\tilde k_t = W_{kv}\,x_t,\qquad g_t = W_{gate}\,x_t \qquad (\text{fp32})$$

$$\alpha_{t}^{(j)} = \frac{\exp g_t}{\sum_{t' \in \text{group } j} \exp g_{t'}}\ \text{(per channel)},\qquad
c_j = \mathrm{RMSNorm}\Big(\sum_{t \in \text{group } j} \alpha_t^{(j)} \odot \tilde k_t\Big)$$

At $r = 1$ there is no gate: $c_t = \mathrm{RMSNorm}(W_{kv}\,x_t)$ in the model dtype.
With a cache the input first goes through `store_compression_weights`; without one (training)
the trailing $S \bmod r$ tokens are simply dropped. Returns `None` when no group completed.

## Notes / gotchas

- **Inference is broken.** transformers 5.17's `DynamicCache` builds layers as
  `DYNAMIC_LAYER_TYPE_MAPPING[layer_type](**layer_kwargs)` without passing `config`, so
  constructing this cache raises `TypeError: DeepseekV41CSACache.__init__() missing 1
  required positional argument: 'config'`. `model.py` sets `use_cache=False`; training
  (the cache-less path, `cache_layer is None`) is unaffected, generation is not.
- `update` returns the *full* concatenation, not just the window; correctness depends on the
  caller's sliding mask. It also stores only $W-1$ entries so window + new token = $W$.
- Only the `"compressor"` counter is bumped; `"indexer"` entries carry no position anchor.
- RoPE and fp4 fake-quant of the stored latent happen in `attention.py`, not here.
