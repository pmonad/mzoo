# Split out of the frozen dsv4 baseline `src/mzoo/archs/dsv4/modeling_deepseek_v41.py`
# (see there for upstream provenance + Apache-2.0 license). Covers: the CSA2 cache layer and the ratio-2 KV compressor.

import torch
from torch import nn

from transformers.cache_utils import DynamicSlidingWindowLayer

from .config import DeepseekV41TextConfig
from .norm_rope import DeepseekV41RMSNorm


class DeepseekV41CSACache(DynamicSlidingWindowLayer):
    r"""Cache layer for a V4.1 **KV-source** layer (CSA2). On top of the shared-KV
    sliding-window ring every layer keeps, it holds the group state of the shared
    compressed branch:

      * `buffer_kv` / `buffer_gate` — source tokens arrived since the last complete
        compress group; once `compress_ratio` tokens accumulate the compressor closes a
        group and drains the buffer. This is what makes chunked prefill seamless:
        partial groups simply carry across forward calls.
      * `compressed_kv["compressor"]` — the running compressed KV entries (one per
        complete group, at `head_dim`), published to the whole group via the per-forward
        shared state; consumer layers hold plain sliding layers and read this one.
      * `compressed_kv["indexer"]` — the running *indexer keys* (one per complete
        group, at `index_head_dim`), derived from the same pooled latents by the
        source's indexer (`wk` + `k_norm`): the whole group scores against one key set.
      * `entry_count["compressor"]` — groups emitted so far, so
        `entry_count * compress_ratio` is the absolute position of the next group's
        first source token.

    The compress ratio is passed per call by the compressor (it is a per-layer config
    value, not a per-layer-type one, so it cannot be resolved at cache-construction
    time).
    """

    _layer_type = "shared_compressed_attention"

    def __init__(self, config: "DeepseekV41TextConfig", **kwargs):
        super().__init__(sliding_window=config.sliding_window)
        self.buffer_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.buffer_gate: dict[str, torch.Tensor | None] = {"compressor": None}
        self.compressed_kv: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
        # Only the compressor counter is read (group positions); indexer keys are
        # appended without needing a position anchor.
        self.entry_count: dict[str, int] = {"compressor": 0}

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        # The base class permutes only the sliding-window keys; the group state
        # (partial-group buffers, shared compressed KV, indexer keys) is per-batch-row
        # and must follow the beams too, or beams silently attend each other's groups.
        super().reorder_cache(beam_idx)
        for name, tensor in self.compressed_kv.items():
            if tensor is not None:
                self.compressed_kv[name] = tensor.index_select(0, beam_idx.to(tensor.device))
        for attr in ("buffer_kv", "buffer_gate"):
            buffer = getattr(self, attr)
            for name, tensor in buffer.items():
                if tensor is not None:
                    buffer[name] = tensor.index_select(0, beam_idx.to(tensor.device))

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        for name, tensor in self.compressed_kv.items():
            if tensor is not None:
                self.compressed_kv[name] = tensor.repeat_interleave(repeats, dim=0)
        for attr in ("buffer_kv", "buffer_gate"):
            buffer = getattr(self, attr)
            for name, tensor in buffer.items():
                if tensor is not None:
                    buffer[name] = tensor.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        for name, tensor in self.compressed_kv.items():
            if tensor is not None:
                self.compressed_kv[name] = tensor[indices, ...]
        for attr in ("buffer_kv", "buffer_gate"):
            buffer = getattr(self, attr)
            for name, tensor in buffer.items():
                if tensor is not None:
                    buffer[name] = tensor[indices, ...]

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        """Sliding-window K=V update: return everything seen so far (the attention
        mask selects the window), keep the last `sliding_window - 1` entries cached."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self.values = self.keys
        self.cumulative_length += key_states.shape[-2]
        full = torch.cat([self.keys, key_states], dim=-2)
        self.keys = full[:, :, -self.sliding_window + 1 :, :]
        self.values = self.keys
        return full, full

    def store_compression_weights(
        self, name: str, kv: torch.Tensor, gate: torch.Tensor | None, compress_ratio: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        r"""Concatenate the newly projected `(kv, gate)` with the buffer, peel off the
        longest group-aligned prefix, keep the remainder buffered, and return
        `(chunk_kv, chunk_gate, first_group_position)` — the absolute position of the
        first group's first source token. `gate` is `None` at ratio 1 (no pooling:
        every token is its own group)."""
        first_group_position = self.entry_count[name] * compress_ratio
        buffered_kv, buffered_gate = self.buffer_kv[name], self.buffer_gate[name]
        if buffered_kv is not None and buffered_kv.shape[1]:
            kv = torch.cat([buffered_kv, kv], dim=1)
            if gate is not None:
                gate = torch.cat([buffered_gate, gate], dim=1)
        usable = (kv.shape[1] // compress_ratio) * compress_ratio
        self.buffer_kv[name] = kv[:, usable:]
        self.buffer_gate[name] = None if gate is None else gate[:, usable:]
        return kv[:, :usable], None if gate is None else gate[:, :usable], first_group_position

    def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        r"""Append freshly emitted entries to `compressed_kv[name]`, bump the group
        count, and return the running tensor."""
        if self.compressed_kv[name] is None:
            self.compressed_kv[name] = compressed
        elif compressed.shape[2] > 0:
            self.compressed_kv[name] = torch.cat([self.compressed_kv[name], compressed], dim=2)
        if name == "compressor":
            self.entry_count[name] += compressed.shape[2]
        return self.compressed_kv[name]


class DeepseekV41Compressor(nn.Module):
    r"""Pools `compress_ratio` consecutive tokens into one KV latent with a learned
    softmax gate (pooling in fp32). Returns the latent **before RoPE** — the indexer
    derives its keys from the unrotated form. At ratio 1 there is no pooling and no
    gate: a plain per-token projection (the CED "decoder" branch — full-resolution KV
    projected once by the source layer instead of per layer)."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.head_dim = config.head_dim
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.head_dim, bias=False) if self.compress_ratio > 1 else None
        self.norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, cache_layer: DeepseekV41CSACache | None
    ) -> tuple[torch.Tensor | None, int]:
        """Returns the pre-RoPE latents of the groups completing in this call (`None`
        while a group is still filling — decode steps between group boundaries), plus
        the absolute position of the first returned group."""
        if self.wgate is None:  # ratio 1: every token is a group, no pooling
            latent = self.norm(self.wkv(hidden_states))
            first_group_position = 0 if cache_layer is None else cache_layer.entry_count["compressor"]
            return latent, first_group_position

        # Pooling math runs in fp32 (the reference stores `wkv` fp32 above ratio 1);
        # upcast the weight explicitly so a half-precision model does not crash.
        kv = nn.functional.linear(hidden_states.float(), self.wkv.weight.float())
        # The gate is computed in fp32 regardless of the storage dtype (the released
        # checkpoint stores `wgate` in BF16).
        gate = nn.functional.linear(hidden_states.float(), self.wgate.weight.float())
        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            chunk_kv, chunk_gate, first_group_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_group_position = cache_layer.store_compression_weights(
                "compressor", kv, gate, self.compress_ratio
            )
        if chunk_kv.shape[1] == 0:
            return None, first_group_position
        n_groups = chunk_kv.shape[1] // self.compress_ratio
        kv = chunk_kv.view(chunk_kv.shape[0], n_groups, self.compress_ratio, -1)
        gate = chunk_gate.view(chunk_gate.shape[0], n_groups, self.compress_ratio, -1)
        latent = (kv * gate.softmax(dim=2, dtype=torch.float32)).sum(dim=2)
        return self.norm(latent.to(hidden_states.dtype)), first_group_position
