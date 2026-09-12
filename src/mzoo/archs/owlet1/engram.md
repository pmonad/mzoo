# engram.py

DeepSeek's "Engram" conditional memory: the last $n$ tokens at each position are hashed into a huge sparse embedding table and the fetched rows are written into the hyper-connection residual streams at selected layers, through a per-stream sigmoid gate.

**Disabled by default.** `engram_layer_ids=[]` in the config and in the smoke config, so `EngramLayout.from_config` returns `None`, no `DeepseekV41Engram` is built and no hash state exists. Everything below is dead code in current runs; it is exercised only by the released DeepSeek-V4.1-Flash checkpoint (`engram_layer_ids=[1, 14]`, tables of ~384M rows each).

## Shapes

| symbol | shape | meaning |
|---|---|---|
| `B, S, D` | `8, 512, 256` | batch, sequence, hidden |
| `hc` | `4` | hyper-connection streams (`hc_mult`); residual is `[B, S, hc, D]` |
| `n` | `4` | `engram_max_ngram_size`: hashes 2-, 3-, 4-grams |
| `H` | `8` | `engram_n_heads` (independent hash functions per n-gram size) |
| `C = (n-1)*H` | `24` | hash columns per position per layer |
| `d_e` | `256` | `engram_head_dim`, table row width |
| `L` | `0` (released `2`) | number of engram layers |
| `V_e` | `16 000 000` | `engram_vocab_size`: primes are drawn just above this |
| `V_c` | `99 092` | `engram_compressed_vocab_size` |
| `N_l` | released `384 006 168` | table rows for layer $l$ (`engram_num_embeddings[l]`) $= \sum$ of that layer's 24 primes |
| `hash_ids` | `[B, S, L, C]` | output of `DeepseekV41NgramHashState`; layer slice `[B, S, C]` goes to each engram |
| `wkv` | `[(hc+1)*D, C*d_e]` = `[1280, 6144]` | rows to `hc` keys + one value |

## `DeepseekV41EngramEmbedding`

Row gather $T[h]$, $T \in \mathbb{R}^{N_l \times d_e}$. When the table is stored fp8 (`float8_e4m3fn`) each 32-channel block $j$ is rescaled by its per-row scale $\sigma_{h,j}$ (`scale: [N_l, d_e/32]`); in bf16/fp32 the `scale` parameter is allocated but unused.

$$v_{h,[32j:32j+32]} = \tilde T_{h,[32j:32j+32]} \cdot \sigma_{h,j}$$

## `DeepseekV41Engram`

Fetches the $C$ rows for a position, projects them to one key per hc stream plus one shared value, and adds the value into each stream gated by the stream/key agreement.

$$r = \operatorname{concat}_{c=1..C} T[h_c] \in \mathbb{R}^{C d_e}, \qquad [k_1,\dots,k_{hc},\, v] = W_{kv}\, r, \quad k_c, v \in \mathbb{R}^{D}$$

$$\omega = q \odot \kappa \in \mathbb{R}^{hc \times D} \ (\texttt{q\_weight} \odot \texttt{k\_weight}, \text{only ever used as this product})$$

$$a_c = \frac{\langle h_c \odot \omega_c,\ k_c\rangle}{\sqrt{D}\ \operatorname{rms}(h_c)\ \operatorname{rms}(k_c)}, \qquad \operatorname{rms}(z) = \sqrt{\operatorname{mean}(z^2) + \epsilon}$$

$$g_c = \sigma\!\left(\operatorname{sign}(a_c)\sqrt{\max(|a_c|,\,10^{-6})}\right), \qquad h_c' = h_c + g_c\, v$$

Normalisation is per (token, stream) over $D$, not jointly over streams. `token_mask == False` forces $g_c = 0$ so those positions pass through unchanged. All of this runs in fp32 and is cast back to the stream dtype.

## `EngramLayout`, `_find_next_prime`

Frozen bucket layout. For each layer, n-gram size $m \in \{2..n\}$ and head $j$, a distinct prime $p_{l,m,j}$ is drawn as the next prime strictly above $V_e - 1$ not yet used, in `(l, m, j)` order. Each `(m, j)` pair owns the disjoint row range $[\text{off}_{l,m,j},\ \text{off}_{l,m,j} + p_{l,m,j})$ with offsets being the running sum over the flat `(m, j)` order, restarting at 0 per layer. Hence $N_l = \sum_{m,j} p_{l,m,j} \approx 24 \cdot 16\text{M}$. `_find_next_prime` is a linear scan using `sympy.isprime`.

## `build_compressed_token_map`

Maps every tokenizer id to a smaller id space in which tokens that normalise identically collapse (`" The"`, `"the"`, `"THE"` become one id). Normaliser: NFKC, NFD, strip accents, lowercase, collapse whitespace to one space, strip (a private-use sentinel keeps the lone-space token alive); partial-UTF-8 byte tokens (`�` in the decode) are keyed by their raw token string. Returns `(lookup[V], V_c)`; `V_c` must equal `engram_compressed_vocab_size` or the hash state raises, because the multipliers depend on it.

## `compute_hash_multipliers`

One odd int64 per (layer, look-back), from a per-layer RNG so layers hash differently:

$$a_{l,i} = 2u_{l,i} + 1, \qquad u_{l,i} \sim \mathcal{U}\{0,\ \lfloor (2^{63}-1)/V_c \rfloor / 2\}, \quad i = 0..n-1, \quad \text{seed} = 10007\cdot \text{layer\_id}$$

Odd multipliers are invertible modulo $2^{64}$ (no collisions from the multiply itself); the bound keeps $t \cdot a_{l,i} < 2^{63}$ for $t < V_c$, so the products never overflow int64.

## `DeepseekV41NgramHashState`

Plain Python object (not an `nn.Module`, not part of `past_key_values`) that turns `input_ids` into `[B, S, L, C]` hash ids once per forward. It keeps an absolute-position `history: [B, T]` of compressed ids, grown geometrically and written via `scatter_` at `position_ids`, so prefill, chunked prefill and one-token decode steps all reconstruct their look-back from the same buffer. `DEAD = -1` marks image-span tokens (`token_mask == False`); n-grams never cross one.

```
history[b, position_ids] = token_map[input_ids]      (DEAD where token_mask is False)
blocked = False
for shift in 0 .. n-1:                                # shift 0 = current token
    src      = history[b, pos - shift]
    blocked |= (pos < shift) | (src == DEAD)          # sticky: once blocked, stays blocked
    t[shift] = pad_id if blocked else src            # pad_id = token_map[engram_pad_id]
```

Then multiplicative-XOR rolling hash, reduced by each head's prime and offset into its bucket range:

$$\rho_{l,i} = \bigoplus_{j=0}^{i} a_{l,j}\, t_j, \qquad h_{l,m,j} = \big(\rho_{l,m-1} \bmod p_{l,m,j}\big) + \text{off}_{l,m,j}, \quad m = 2..n$$

The same $\rho_{l,m-1}$ (the $m$-gram hash) is reduced modulo $H$ different primes, giving $H$ near-independent buckets per n-gram; distinct primes are what decorrelate the heads and guarantee the ranges tile the table without overlap. `_reorder_cache` in `decoder.py` permutes `history` along the batch for beam search.

## Notes / gotchas

- Enabling requires calling the model's tokenizer hook (`decoder.py` builds the hash state from a tokenizer) and passing `input_ids`; `inputs_embeds` raises because ids cannot be recovered.
- `engram_num_embeddings` defaults to `[0]*L`; when enabling you must set it to the layout's prime sums. Unclear: nothing in this module asserts `sum(primes) == engram_num_embeddings[l]`, so a short table fails only at lookup time.
- `EngramLayout.from_config` runs `sympy.isprime` over a few hundred integers per layer at model init; trivial cost, but it is why `sympy` is a dependency.
- `history` is mutable state on the model, keyed by absolute position, and survives across `generate` calls. A fresh request starting at position 0 overwrites everything it reads, so this is safe; the hazard is starting a fresh `past_key_values` at a position $> 0$ without rewriting the earlier positions.
