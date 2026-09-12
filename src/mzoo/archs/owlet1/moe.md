# moe.py

The per-layer sparse MoE block: a top-k router (`noaux_tc` selection bias, `sqrtsoftplus` scores), one SwiGLU expert class, and the block that combines top-2-of-8 routed experts with one always-on shared expert. Every backbone layer in V4.1 is MoE; there is no dense-layer schedule (`first_k_dense_replace` / `moe_layer_freq` do not exist here).

## Shapes

| symbol | shape | meaning |
|---|---|---|
| `B, S` | `8, 512` | batch, sequence |
| `D` | `256` | hidden size |
| `N = B*S` | `4096` | tokens, after flattening `[B, S, D] -> [N, D]` |
| `E` | `8` | routed experts (`n_routed_experts`) |
| `K` | `2` | activated experts per token (`num_experts_per_tok`) |
| `F` | `256` | expert intermediate size (`moe_intermediate_size`) |
| `gate.weight` | `[E, D]` | router projection |
| `gate.bias`, `gate.bias_vl` | `[E]` fp32 | selection correction bias (text / image-span tokens) |
| `weights`, `indices` | `[N, K]` | combining weights and chosen expert ids |
| `w1`, `w3` / `w2` | `[F, D]` / `[D, F]` | expert gate, up / down projections |

## `DeepseekV41TopKRouter`

Scores every token against every expert in fp32, picks the top-$K$ by *biased* score, but combines with the *unbiased* score. Nothing is a softmax: `sqrtsoftplus` is a positive, monotone, per-expert transform.

$$\ell = \frac{x\,W_g^\top}{\tau}, \qquad s_e = \sqrt{\operatorname{softplus}(\ell_e)} = \sqrt{\log\!\left(1+e^{\ell_e}\right)}, \qquad \tau = \texttt{gate\_temp} = 1$$

$$\mathcal{I}(x) = \operatorname{top}K_e\,(s_e + b_e), \qquad b = \begin{cases}\texttt{bias\_vl} & \text{image-span token}\\ \texttt{bias} & \text{otherwise}\end{cases}$$

$$w_i = s_i \ (i \in \mathcal{I}), \qquad w_i \leftarrow \frac{w_i}{\sum_{j\in\mathcal{I}} w_j + 10^{-20}} \ \ (\texttt{norm\_topk\_prob}), \qquad w_i \leftarrow \gamma\, w_i,\ \gamma = \texttt{routed\_scaling\_factor} = 1.5$$

So after renormalisation the $K$ weights sum to $\gamma = 1.5$ per token. The bias only moves the `topk` argmax; it never touches $w$.

## `DeepseekV41Expert`

One SwiGLU MLP, computed in fp32 up to the down projection. `swiglu_limit` $c=10$ clamps the *pre-activations*: the up branch on both sides, the gate branch from above only (SiLU already bounds it below). `hidden_act = silu`.

$$g = \min(W_1 x,\ c), \qquad u = \operatorname{clip}(W_3 x,\ -c,\ c), \qquad y = W_2\big(w \cdot \operatorname{SiLU}(g) \odot u\big)$$

The optional routing weight $w$ is applied before $W_2$ (equivalent to scaling the output, since $W_2$ is linear). The shared expert is called with $w = $ `None`.

## `DeepseekV41SparseMoeBlock`

Routed top-$K$ mixture plus one shared expert every token passes through, accumulated in fp32 and cast back:

$$\operatorname{MoE}(x) = \sum_{i \in \mathcal{I}(x)} w_i\, E_i(x) \;+\; E_{\text{sh}}(x)$$

Dispatch is an eager Python loop over the $E$ experts: `bincount` the chosen ids, skip empty experts, `torch.where(indices == i)` gathers that expert's tokens and their slot in `weights`. Spec-grade, not a serving path.

## Notes / gotchas

- **No load-balancing loss.** `router_aux_loss_coef`, `output_router_logits` and `router_jitter_noise` are config fields only; nothing computes an auxiliary loss and `DeepseekV41ForCausalLM` returns `MoeCausalLMOutputWithPast` without `aux_loss`. Total loss is plain cross-entropy. `mzoo.train.trainer.Trainer._track` is guarded by `outputs.aux_loss is not None`, so the `moe/aux_loss` and `moe/{max,min}_load_l*` logs never fire: **expert load is unmonitored** in current runs.
- **`gate.bias` / `gate.bias_vl` are frozen at zero.** They are zero-initialised (`decoder.py::_init_weights`), kept in fp32, and receive no gradient (they only feed the non-differentiable `topk`). That is expected for `noaux_tc` — upstream updates them with a separate sign-of-load rule — but no such update exists in this repo. Net effect: selection reduces to plain top-$K$ on $s$.
- `n_shared_experts` (config, smoke `shared=1`) is never read by `moe.py`; the block always builds exactly one `shared_experts` MLP.
- `image_mask` / `bias_vl` is a VL-training leftover; text runs never pass a mask, so `bias` is always used.
- `decoder.py` records the router's forward output under the `router_logits` key; that output is `(weights, indices)`, not raw logits.
