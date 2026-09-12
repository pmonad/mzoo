# norm_rope.py

RMS normalization (weighted and unweighted) and the two-base interleaved-pair RoPE used by
attention, the compressor and the indexer. RoPE only touches the *trailing*
`qk_rope_head_dim` channels of a head; the leading "nope" channels pass through.

## Shapes

| symbol | shape | meaning |
|---|---|---|
| `B`, `S` | `8`, `512` | batch, sequence length |
| `D` | `256` | hidden size (`attn_norm`, `ffn_norm`, final `norm`) |
| `H`, `d_h` | `4`, `64` | attention heads, head dim (`kv_norm`, cache norm) |
| `d_r` | `16` | `qk_rope_head_dim`: rotated tail of each head |
| `r`, `d_i` | `128`, `32` | `q_lora_rank` (`q_norm`), `index_head_dim` (indexer `k_norm`) |
| `hc` | `4` | `hc_mult`, number of parallel residual streams |
| `cos`, `sin` | `[B, S, d_r/2]` = `[8, 512, 8]` | one angle per channel *pair*, not duplicated |
| `q` / `kv` | `[B, S, H, d_h]` / `[B, S, d_h]` | 4-D / 3-D inputs to `apply_rotary_pos_emb` |

## `DeepseekV41RMSNorm`

T5/LLaMA RMSNorm with a learnable gain. Stats in fp32; the result is cast back to the
input dtype *before* the gain is applied.

$$\mathrm{RMSNorm}(x) = g \odot \frac{x}{\sqrt{\tfrac{1}{n}\sum_{i} x_i^2 + \epsilon}}, \qquad g_0 = \mathbf 1$$

Every instance uses `eps=config.rms_norm_eps` = **1e-20** (the class default `1e-6` is
never used). With fp32 stats this is effectively "no epsilon". Provenance is the HF port,
not the paper (see the gotcha below).

## `DeepseekV41UnweightedRMSNorm`

Same formula with $g \equiv \mathbf 1$: no parameter. Used once per decoder layer on the
mHC (hyper-connection) residual stream flattened across all `hc` copies, so one statistic
normalizes the whole `hc·D` = `1024`-wide vector per token:

$$\hat x = \frac{x}{\sqrt{\tfrac{1}{hc\,D}\sum_{i=1}^{hc\,D} x_i^2 + \epsilon}}, \qquad x \in \mathbb{R}^{B\times S\times hc\,D}$$

The caller upcasts to fp32 first; the output feeds the linear mix projection that yields
the pre/post/comb hyper-connection weights (`decoder.py`).

## `DeepseekV41RotaryEmbedding`

One `inv_freq` buffer per rope layer type from `config.rope_parameters`:

| type | base $\theta$ | scaling | used by |
|---|---|---|---|
| `main` | `rope_theta` = 10000 | none | attention in layers with `compress_ratio == 0` |
| `compress` | `compress_rope_theta` = 160000 | optional YaRN, `attention_factor` forced to 1.0 | compressed-layer attention, compressor latents, indexer q/k |

The rotated width comes from the *attention* head dim whatever tensor it is applied to:
$d_r = \lfloor d_h \cdot \tfrac{d_r}{d_h} \rfloor = 16$. Only $d_r/2 = 8$ unique
frequencies are stored (no `cat([freqs, freqs])`):

$$\omega_i = \theta^{-2i/d_r},\ i < d_r/2, \qquad
\cos_{b,s,i} = \cos(p_{b,s}\,\omega_i), \quad \sin_{b,s,i} = \sin(p_{b,s}\,\omega_i)$$

`forward(x, position_ids, layer_type)` returns `(cos, sin)` as `[B, S, d_r/2]` in
`x.dtype`, computed in fp32 with autocast off. The larger `compress` base exists because
one latent stands for `compress_ratio` tokens, so latent positions
$p = p_0 + \text{ratio}\cdot k$ are further apart. The model precomputes both types at
token positions once per forward; kv-source layers and the indexer own their own instance
because latent positions are only known after the compressor runs.

## `apply_rotary_pos_emb(x, cos, sin, inverse=False)`

Interleaved-pair RoPE on the last `2 * cos.shape[-1]` = 16 channels of `x`, for `[B, S, D]`
or `[B, S, H, D]`. Adjacent channels $(x_{2i}, x_{2i+1})$ form one complex number:

$$\begin{pmatrix} y_{2i} \\ y_{2i+1} \end{pmatrix} =
\begin{pmatrix} \cos\phi_i & -\sin\phi_i \\ \sin\phi_i & \cos\phi_i \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}, \qquad \phi_i = p\,\omega_i$$

`inverse=True` negates `sin`, i.e. rotates by $-\phi_i$ (conjugate), exactly undoing a
forward rotation at the same position. Math is fp32, cast back; the nope prefix is
concatenated back unchanged.

Use of `inverse=True`: in attention K == V and the cache holds the *rotated* vector, so
values enter the weighted sum rotated. After attention,
`apply_rotary_pos_emb(attn_output, cos, sin, inverse=True)` applies $R(-\phi(p_q))$ at the
query position to the output's rope slice before `wo_a`/`wo_b` mix heads, so a single
rotated KV buffer serves as both K and V.

## Notes / gotchas

- `cos`/`sin` are half-width. Pairing is (even, odd), not (first half, second half); a
  `rotate_half`-style helper expecting duplicated `[.., d_r]` would be wrong here.
- The indexer's 32-wide q/k still rotate only their last 16 channels, since $d_r$ comes
  from `config.head_dim`, not the tensor width.
- `rms_norm_eps = 1e-20` comes from the vendored HF port's config default, **not** from the
  tech report — the paper never states an RMSNorm epsilon (its two `1e-20` values are the
  AdamW and Sinkhorn *optimizer* epsilons, §2.5/§4.2.2). Treat it as unexplained-but-deliberate
  upstream rather than paper-backed; see [TRAINING.md](TRAINING.md).
