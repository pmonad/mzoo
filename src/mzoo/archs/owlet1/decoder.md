# decoder.py

The block-and-model level of the owlet1 DeepSeek-V4.1 backbone: the decoder layer with its
multi-stream hyper-connection (mHC) residual, the shared `PreTrainedModel` base, the text
backbone (embedding, mask, layer loop, final norm) and the causal-LM head with loss.

## Shapes

| symbol | shape | meaning |
|---|---|---|
| `B`, `S`, `D`, `hc`, `V` | `8`, `512`, `256`, `4`, `32000` | batch, seq len, `hidden_size`, `hc_mult` (parallel residual streams), vocab |
| `hidden_streams`, `residual` | `[B, S, hc, D]` | the mHC residual: `hc` copies of a `D`-wide stream |
| `flat` | `[B, S, hc·D]` = `[8, 512, 1024]` | streams flattened and RMS-normalised, one statistic per token |
| `hc_*_fn`, `hc_*_base`, `hc_*_scale` | `[(2+hc)·hc, hc·D]` = `[24, 1024]`, `[24]`, `[3]` | mix projection, its bias, one gain per coefficient set (fp32) |
| `pre`, `post`, `pre_mix` | `[B, S, hc]` (fp32) | collapse / expand weights |
| `comb` | `[B, S, hc, hc]` (fp32) | doubly-stochastic residual mixing matrix |
| `collapsed`, `attn_output`, `ffn_output` | `[B, S, D]` | one sublayer input / output |
| `logits` | `[B, S, V]` (fp32) | lm_head output |

## `DeepseekV41DecoderLayer`

One V4.1 block. Instead of `x + f(norm(x))` on a single `[B, S, D]` stream, the residual is `hc`
streams; each sublayer site (attention, then FFN) *collapses* them to one input, runs the sublayer,
and *expands* the output back while re-mixing the streams. Optional engram lookup precedes the block
(`engram_layer_ids=[]` in the smoke config, so `self.engram is None`).

**`hc_mixes`** -- one projection yields all three coefficient sets. With $h \in \mathbb{R}^{hc \cdot D}$
the flattened stream for a token, $\gamma = $ `scale`, $\beta = $ `base`, $\epsilon = $ `hc_eps` $= 10^{-6}$:

$$\hat h = \frac{h}{\sqrt{\tfrac{1}{hc \cdot D}\sum_i h_i^2 + \epsilon_{\text{rms}}}}, \qquad m = W \hat h \in \mathbb{R}^{(2+hc)\,hc}$$

$$\text{pre} = \sigma\!\left(\gamma_0\, m_{[0:hc]} + \beta_{[0:hc]}\right) + \epsilon, \qquad
\text{post} = 2\,\sigma\!\left(\gamma_1\, m_{[hc:2hc]} + \beta_{[hc:2hc]}\right)$$

$$L = \operatorname{reshape}\!\left(\gamma_2\, m_{[2hc:]} + \beta_{[2hc:]},\; hc \times hc\right), \qquad
C^{(0)}_{jk} = \operatorname{softmax}_k(L_{jk}) + \epsilon$$

Sinkhorn projection of $C$ onto the doubly-stochastic manifold, `hc_sinkhorn_iters` $= T = 20$:

$$C \leftarrow \frac{C}{\sum_j C_{jk} + \epsilon} \quad\text{(columns)}, \qquad
\text{then } (T-1)\times:\; C \leftarrow \frac{C}{\sum_k C_{jk} + \epsilon}\ \text{(rows)},\;\;
C \leftarrow \frac{C}{\sum_j C_{jk} + \epsilon}\ \text{(columns)}$$

So the softmax is the first row normalisation; in total there are $T$ row and $T$ column passes, ending on a
column pass, which is why $\sum_j C_{jk} = 1$ holds (to $\epsilon$) at the output.

**`hc_collapse`** -- streams to one sublayer input, computed in fp32 and cast back:

$$\text{collapsed} = \sum_{k=1}^{hc} \text{pre}_k \cdot \text{stream}_k \in \mathbb{R}^{D}$$

**`hc_expand`** -- sublayer output $x \in \mathbb{R}^D$ back into the streams, residual re-mixed through $C$:

$$\text{out}_k = \text{post}_k \cdot x + \sum_{j=1}^{hc} C_{jk}\, \text{residual}_j$$

Because columns of $C$ sum to one, each output stream carries a convex combination of the input streams
plus a gated (`post` $\in (0, 2)$) copy of the sublayer output.

**The cross-site hand-off.** The `pre` a site computes is *not* used at that site. Each site collapses with
the `pre` produced by the previous site, and the layer returns its FFN `pre` for the next layer:

```
forward(hidden_streams, pre_mix, ...):            # pre_mix came from the previous layer's ffn site
    residual = hidden_streams
    attn_pre, attn_post, attn_comb = hc_mixes(hidden_streams, hc_attn_*)
    collapsed = hc_collapse(hidden_streams, pre_mix)          # <- previous site's pre, not attn_pre
    attn_out  = attn(attn_norm(collapsed), shared=shared)
    hidden_streams = hc_expand(attn_out, residual, attn_post, attn_comb)

    residual = hidden_streams
    ffn_pre, ffn_post, ffn_comb = hc_mixes(hidden_streams, hc_ffn_*)
    collapsed = hc_collapse(hidden_streams, attn_pre)         # <- attention site's pre
    ffn_out   = ffn(ffn_norm(collapsed))
    hidden_streams = hc_expand(ffn_out, residual, ffn_post, ffn_comb)
    return hidden_streams, ffn_pre                            # consumed by the next layer's attn site
```

The `post` / `comb` of a site are used at that same site; only `pre` is shifted by one site.

## `DeepseekV41PreTrainedModel`

Shared HF base for both models. `config_class = DeepseekV41TextConfig` (flat) serves the backbone; `ForCausalLM`
overrides it with the composite `DeepseekV41Config` because the released `config.json` is composite and its top-level
`quantization_config` must reach the quantizer (the bare text config silently dropped it). Eager attention only
(head_dim 512, per-head sink, in-block KV concat); `_is_stateful = True`. `_keep_in_fp32_modules_strict` pins exactly
the checkpoint's fp32 tensors: the six `hc_*` parameters, `attn_sink`, router `bias` / `bias_vl`.
`_init_weights` (std $= 0.02$): `hc_*_fn` $\sim \mathcal{N}(0, \sigma^2)$, `hc_*_base = 0`, `hc_*_scale = 1`;
router weight normal, biases zero; `attn_sink = 0`; engram tables normal; RoPE `inv_freq` buffers are rebuilt here
because `from_pretrained` constructs on the meta device.

## `DeepseekV41TextModel`

`embed` -> `hc` copies -> `num_hidden_layers` blocks -> final collapse -> `norm`.

- Mask: `create_sliding_window_causal_mask` (`sliding_window=128`); a per-layer-type dict from `generate()` is
  collapsed to any entry since both layer types share the window. Rotary tables (`"main"`, `"compress"`) computed once.
- Streams are initialised as $\text{stream}_k = e$ for all $k$ (embedding broadcast), and the initial `pre_mix`
  is one-hot $(1, 0, 0, 0)$: the first attention site reads stream 0 only.
- `shared: dict = {}` is created per forward and threaded through all layers. Attention uses it for CSA2
  KV sharing: KV-source layers publish `shared["compress_kv"]`, index-source layers publish
  `shared["topk_bias"]`, and the in-between layers of the group read them.
- After the loop: $\text{last\_hidden} = \operatorname{RMSNorm}\!\left(\sum_k \text{pre\_mix}_k\, \text{stream}_k\right)$
  using the final layer's `ffn_pre`.
- `_can_record_outputs`: `hidden_states` records each layer's `attn_norm` in/out (collapsed block inputs, not the raw
  `[B, S, hc, D]` stream); `router_logits` from the router; `attentions` from attention. Engram plumbing is inert here.

## `DeepseekV41ForCausalLM`

`model` (text backbone) + `lm_head: Linear(D, V, bias=False)`; `tie_word_embeddings=True` in the smoke config ties
`lm_head.weight` to `embed.weight` via `post_init`. Logits are upcast to fp32 and the loss is HF's standard
`loss_function` (plain next-token cross-entropy, `-100` ignored):

$$\mathcal{L} = -\frac{1}{N}\sum_{t:\, y_{t+1} \neq -100} \log p_\theta\!\left(y_{t+1} \mid x_{\le t}\right)$$

`shift_labels`, when present, is already aligned with `logits` and takes precedence over `labels`.
`_reorder_cache` permutes the cache and the engram n-gram history along the beam axis.

## Notes / gotchas

- **owlet1 divergence from `dsv4`** (the only intentional code difference): `shift_labels` is popped from `**kwargs`
  instead of being a named parameter. HF `find_labels` returns every forward arg containing "label", and
  `Trainer.prediction_step` requires *all* of them in the batch; with `shift_labels` in the signature a
  `{input_ids, labels}` collator looked label-free and eval loss was silently skipped (no `eval_loss` logged).
- No auxiliary / load-balancing loss anywhere; the returned `loss` is cross-entropy only.
- `use_cache=False` is forced in `model.py` (CSA cache vs transformers 5.17), so `DynamicCache` creation and the
  `past_key_values` path here are unused in training.
- mHC maths is deliberately fp32: the stream is `.float()`-ed *before* `hc_input_norm`, and `fn`/`scale`/`base`
  are re-`.float()`-ed in the mix, so a bf16 model never rounds the stream before the projection. `hc_collapse` /
  `hc_expand` also compute in fp32 and cast back to the stream dtype.
- `pre` gets `+ hc_eps` so no stream is ever fully switched off; `post` ranges over $(0, 2)$, not $(0, 1)$.
- The mHC parameters are raw `nn.Parameter` attributes (not `nn.Linear`), matching checkpoint keys `layers.N.hc_attn_fn` etc.
