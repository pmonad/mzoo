# Split out of the frozen dsv4 baseline `src/mzoo/archs/dsv4/modeling_deepseek_v41.py`
# (see there for upstream provenance + Apache-2.0 license). Covers: the Engram n-gram memory (embedding, module, layout, hashing state).

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sympy import isprime
from torch import nn

from .config import DeepseekV41TextConfig


class DeepseekV41EngramEmbedding(nn.Module):
    """The n-gram hash table: fp8 rows with per-row / per-32-channel E8M0 scales in the
    checkpoint, dequantized on lookup. When the model is loaded in a float dtype the
    fp8 values are represented exactly (e4m3 ⊂ bf16/fp32), so the table is a plain
    embedding and the scales are unused. ~98 GB per table in the released checkpoint —
    memory-map friendly (pure row gather)."""

    def __init__(self, num_embeddings: int, head_dim: int, block_size: int = 32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.head_dim = head_dim
        self.block_size = block_size
        self.weight = nn.Parameter(torch.empty(num_embeddings, head_dim))
        self.scale = nn.Parameter(torch.empty(num_embeddings, head_dim // block_size))

    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        values = F.embedding(hash_ids, self.weight)
        if self.weight.dtype == torch.float8_e4m3fn:
            scales = F.embedding(hash_ids, self.scale).float()
            values = values.float().unflatten(-1, (-1, self.block_size)) * scales.unsqueeze(-1)
            values = values.flatten(-2)
        return values


class DeepseekV41Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it matches.

    The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per hc stream
    plus a shared value. The gate is a sigmoid of the signed sqrt of a normalized dot
    product between the stream and the key (weights `q_weight * k_weight`, used only as
    a product). Hash ids are computed once per forward by
    :class:`DeepseekV41NgramHashState` and passed in per engram layer."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.layer_hash_index = config.engram_layer_ids.index(layer_idx)
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        n_hash_cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.embed = DeepseekV41EngramEmbedding(
            config.engram_num_embeddings[self.layer_hash_index], config.engram_head_dim
        )
        self.wkv = nn.Linear(
            n_hash_cols * config.engram_head_dim,
            config.hidden_size * (config.hc_mult + 1),
            bias=False,
        )
        self.q_weight = nn.Parameter(torch.empty(config.hc_mult, config.hidden_size))
        self.k_weight = nn.Parameter(torch.empty(config.hc_mult, config.hidden_size))

    def forward(
        self, hidden_streams: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """hidden_streams: [B, S, hc, D]; hash_ids: [B, S, n_hash_cols]; token_mask:
        [B, S], False shuts the gate so those positions pass through untouched."""
        # The dequantized fp8 rows are exact in any dtype; cast to the consumer's.
        kv = self.wkv(self.embed(hash_ids).flatten(-2).to(self.wkv.weight.dtype))
        key, value = kv.split([self.hc_mult * self.hidden_size, self.hidden_size], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        weight = self.q_weight.float() * self.k_weight.float()  # only ever used as a product
        h = hidden_streams.float()
        # Normalized per (token, hc stream) over D — NOT jointly over the streams.
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * weight * key).sum(-1) * rstd * self.hidden_size**-0.5
        # Signed sqrt before the sigmoid, matching the training kernel.
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        out = h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)
        return out.to(hidden_streams.dtype)


def _find_next_prime(start: int, seen: set) -> int:
    candidate = start + 1
    while not isprime(candidate) or candidate in seen:
        candidate += 1
    return candidate


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables: a position is hashed as
    `max_ngram_size - 1` n-grams (2..N-gram), each split over `n_heads` heads; every
    (n-gram size, head) pair owns a disjoint prime-sized bucket range — the primes are
    drawn in order above `engram_vocab_size` and never reused."""

    max_ngram_size: int
    layer_ids: tuple
    num_embeddings: tuple
    primes: tuple
    n_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, config: DeepseekV41TextConfig):
        layer_ids = tuple(config.engram_layer_ids)
        if not layer_ids:
            return None
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(config.engram_max_ngram_size - 1):
                sizes, current = [], config.engram_vocab_size - 1
                for _ in range(config.engram_n_heads):
                    current = _find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=config.engram_max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(config.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=config.engram_n_heads,
            head_dim=config.engram_head_dim,
        )


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike
    collapse together (" The", "the", "THE" hash the same). The compressed vocab size
    must match `engram_compressed_vocab_size` — every hash multiplier derives from it,
    so a mismatch means the whole table rehashes to garbage."""
    from tokenizers import Regex, normalizers

    # A private-use char, so a token that is exactly one space survives Strip() instead
    # of collapsing to the empty string and merging with unrelated tokens.
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # The raw Rust tokenizer, matching what training decodes with.
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # A partial UTF-8 byte token: nothing to normalize, key it by its raw form.
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized or text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(layer_ids: tuple, max_ngram_size: int, compressed_vocab_size: int) -> torch.Tensor:
    """One multiplier per (layer, look-back), from a per-layer RNG so layers hash
    differently. Kept odd and bounded so `id * multiplier` cannot overflow int64."""
    multiplier_bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


class DeepseekV41NgramHashState:
    """Maps each position to the hash ids of the n-grams ending there — once per
    forward, for all engram layers.

    Ids go through the compressed table, then each position is hashed with the
    `max_ngram_size - 1` tokens before it; look-back stops at the start of the sequence
    and at any DEAD token (an image-span token), so an n-gram never spans one. The
    absolute-position history buffer carries all of this across the prefill / chunked
    prefill / decode split (a request resumed mid-sequence reconstructs its look-back
    from the positions it writes, exactly like the reference's position-indexed
    cache)."""

    DEAD = -1

    def __init__(self, config: DeepseekV41TextConfig, tokenizer):
        self.layout = EngramLayout.from_config(config)
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != config.engram_compressed_vocab_size:
            raise ValueError(
                f"The tokenizer-derived compressed vocabulary size ({vocab_size}) does not match "
                f"`engram_compressed_vocab_size` ({config.engram_compressed_vocab_size}); the hash "
                "multipliers would silently rehash the whole engram table."
            )
        self.pad_id = token_map[config.engram_pad_id]
        self.max_ngram_size = self.layout.max_ngram_size
        # Primes as [L, n-gram-1, heads] (the per-step modulus is over all heads of one
        # n-gram size). Bucket offsets are PER LAYER over the flat (n-gram, head) order:
        # each layer's table is addressed from its own start, and the ranges inside it
        # are disjoint because the primes are drawn in order and never reused.
        self.primes = torch.tensor(self.layout.primes)  # [L, n-gram-1, heads]
        offsets = []
        for layer in self.layout.primes:
            flat = [p for per in layer for p in per]
            row, total = [], 0
            for p in flat:
                row.append(total)
                total += p
            offsets.append(row)
        self.offsets = torch.tensor(offsets)  # [L, n_cols]
        self.multipliers = compute_hash_multipliers(self.layout.layer_ids, self.max_ngram_size, vocab_size)
        self.token_map = torch.tensor(token_map)
        self.history: torch.Tensor | None = None

    def _to(self, device: torch.device):
        for name in ("primes", "offsets", "multipliers", "token_map", "history"):
            tensor = getattr(self, name)
            if tensor is not None and tensor.device != device:
                setattr(self, name, tensor.to(device))

    def __call__(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, token_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Returns `[B, S, n_engram_layers, n_hash_cols]` hash ids."""
        batch, seq_len = input_ids.shape
        device = input_ids.device
        self._to(device)

        max_pos = int(position_ids.max().item()) + 1
        if self.history is None or self.history.shape[0] < batch or self.history.shape[1] < max_pos:
            # Geometric growth: a decode step advances max_pos by one, so a fixed-size
            # increment would reallocate and copy O(T) on every step.
            shape = (
                max(batch, 2 * self.history.shape[0] if self.history is not None else 0),
                max(max_pos, 2 * self.history.shape[1] if self.history is not None else 0),
            )
            grown = torch.full(shape, self.DEAD, dtype=torch.long, device=device)
            if self.history is not None:
                grown[: min(self.history.shape[0], shape[0]), : self.history.shape[1]] = self.history[
                    : shape[0], : shape[1]
                ]
            self.history = grown

        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, torch.full_like(compressed, self.DEAD))
        self.history.scatter_(1, position_ids.long(), compressed)

        positions = position_ids.long()
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.max_ngram_size):
            source = self.history.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, torch.full_like(source, self.pad_id), source))
        tokens = torch.stack(tokens, dim=-1)  # [B, S, max_ngram_size]

        # XOR the multiplied ids one look-back at a time: after step i the running value
        # is the hash of the (i+1)-gram, landing in its own prime bucket range.
        products = tokens.unsqueeze(2) * self.multipliers.to(device)  # [B, S, L, max_ngram_size]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets.unsqueeze(0)
