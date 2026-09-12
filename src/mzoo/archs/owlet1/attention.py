# Split out of the frozen dsv4 baseline `src/mzoo/archs/dsv4/modeling_deepseek_v41.py`
# (see there for upstream provenance + Apache-2.0 license). Covers: the grouped output projection, eager attention and the CSA2 attention module.

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from .cache import DeepseekV41Compressor
from .config import DeepseekV41TextConfig
from .indexer import DeepseekV41Indexer
from .norm_rope import DeepseekV41RMSNorm, DeepseekV41RotaryEmbedding, apply_rotary_pos_emb
from .quant import _fake_quant_fp4_block, _fake_quant_fp8_block


class DeepseekV41GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear of the attention output projection.

    The stacked attention output is `num_heads * head_dim`-dim (32768 for the released
    model) — a direct projection to `hidden_size` would dominate the per-token cost.
    Instead the heads are split into `o_groups` groups, each projected independently
    to `o_lora_rank`, then mixed to `hidden_size` by `wo_b`. This module owns the
    per-group block (`wo_a`)."""

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs,
):
    """Eager shared-KV attention with the per-head learnable sink of V4.1.

    The sink joins the softmax as one extra logit column and is then dropped — i.e. it
    only grows the denominator, matching the reference kernel (where
    `sum_exp += exp(attn_sink - max)` and the output is normalized by it). Rows whose
    every slot is masked still get a finite result thanks to the sink."""
    # The shared K=V head ([B, 1, T, D]) broadcasts against the query heads in the
    # matmuls — no Hx materialization of the KV tensor.
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    sinks = module.attn_sink.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks.float()], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # the sink only appears in the denominator
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training).to(value.dtype)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous(), attn_weights  # [B, S, H, D]


class DeepseekV41Attention(nn.Module):
    r"""Latent shared-KV attention over two KV sources: a sliding window of raw KV plus,
    when the layer has a compressed branch, the *shared* compressed KV of its group.

    - Q and the output projection are low-rank; the output projection is grouped
      (block-diagonal `wo_a` over `o_groups`, then the mixing `wo_b`).
    - K=V is a single latent (`wkv` + `kv_norm`); the attention output's rope slice is
      inverse-rotated so the shared rotated cache works.
    - Per-head learnable attention sink (`attn_sink`), like gpt-oss.
    - **KV sharing (CSA2)**: only `kv_source_layer_ids` layers own a
      :class:`DeepseekV41Compressor` — the layers in between read the source's
      compressed cache through the per-forward `shared` dict ("Reuse" mode). Index
      sources run the indexer and publish the top-k bias; the layers in between reuse
      it. `shared` is keyed per forward, but the persistent state (group buffers,
      compressed KV, indexer keys) lives on the source's cache layer, so this stays
      correct across prefill / chunked prefill / decode."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.num_heads = config.num_attention_heads
        # Shared-KV latent attention: a single KV head broadcast to all heads.
        self.num_key_value_groups = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_layer_type = "compress" if self.compress_ratio else "main"
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = DeepseekV41RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = DeepseekV41GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.attn_sink = nn.Parameter(torch.empty(self.num_heads))

        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids
        self.compressor = DeepseekV41Compressor(config, layer_idx) if self.is_kv_source else None
        self.indexer = DeepseekV41Indexer(config, layer_idx) if self.is_index_source else None
        # The compress-rope cos/sin here are evaluated at the LATENT positions
        # (first_group_position + ratio*k), which are only known after the compressor
        # runs; the model-level rotary cannot precompute them in `position_embeddings`.
        # Only kv-source layers own one (shared by their group).
        self.compress_rotary = DeepseekV41RotaryEmbedding(config) if self.is_kv_source else None  # trf-ignore: TRF050

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        shared: dict,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # position_ids flows through **kwargs (TRF043): the decoder layer passes the
        # model-level kwargs straight in, and it must stay available to the attention
        # interface below (padding-free paths read it from kwargs).
        position_ids = kwargs["position_ids"]
        batch, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(q_residual).view(batch, seq_len, self.num_heads, self.head_dim)
        q = apply_rotary_pos_emb(q, cos, sin).transpose(1, 2)  # [B, H, S, D]

        kv = self.kv_norm(self.wkv(hidden_states))
        kv = apply_rotary_pos_emb(kv, cos, sin).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        # QAT semantics: the window KV cache stores FP8-quantized values (one ue8m0
        # scale per 32 channels, RoPE tail included) — part of the model, applied
        # even in otherwise-unquantized runs.
        kv = _fake_quant_fp8_block(kv, block_size=32)
        if past_key_values is not None:  # K == V
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        block_bias = None
        if self.compress_ratio:
            cache_layer = past_key_values.layers[self.layer_idx] if past_key_values is not None else None
            latent, first_group_position = (None, 0)
            if self.is_kv_source:
                latent, first_group_position = self.compressor(hidden_states, cache_layer)

            # The indexer consumes the PRE-rope latent; it must run before the latent is
            # rotated into the main compressed cache.
            if self.is_index_source:
                self.indexer(
                    hidden_states, q_residual, latent, first_group_position, position_ids, cache_layer, shared
                )
            block_bias = shared.get("topk_bias")

            if latent is not None:
                positions = first_group_position + self.compress_ratio * torch.arange(
                    latent.shape[1], device=latent.device
                )
                cos_c, sin_c = self.compress_rotary(
                    latent, position_ids=positions.unsqueeze(0).expand(batch, -1), layer_type="compress"
                )
                rotated = apply_rotary_pos_emb(latent, cos_c, sin_c)
                # QAT semantics: the compressed KV cache stores FP4-quantized latents
                # (e2m1 grid, one e4m3 scale per 16 channels).
                rotated = _fake_quant_fp4_block(rotated, block_size=16, e4m3_scales=True)
                rotated = rotated.unsqueeze(1)  # [B, 1, G, hd]
                if cache_layer is not None:
                    cache_layer.update_compressor_states("compressor", rotated)
                else:
                    shared["compress_kv"] = (
                        rotated
                        if shared.get("compress_kv") is None
                        else torch.cat([shared["compress_kv"], rotated], dim=2)
                    )
            if self.is_kv_source and cache_layer is not None:
                # Publish the RUNNING compressed cache — a decode step between group
                # boundaries emits nothing new, but the group still attends over
                # everything emitted so far.
                shared["compress_kv"] = cache_layer.compressed_kv["compressor"]
            compressed_kv = shared.get("compress_kv")
            if compressed_kv is not None:
                kv = torch.cat([kv, compressed_kv], dim=2)

        # The compressed branch concatenated extra entries onto the KV axis after the
        # model-level mask was built: extend the mask with the indexer's per-query
        # block bias instead of zero-padding (which would attend everywhere).
        if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
            if block_bias is not None:
                attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
            else:
                # A compressed branch with no index source in this forward (legal but
                # unusual schedule): mask every compressed entry off instead of
                # attending all of them.
                attention_mask = F.pad(
                    attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=float("-inf")
                )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            q,
            kv,
            kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        # K == V carried RoPE on its rope slice; remove the query's rotation from the
        # output before the grouped projection mixes the heads.
        attn_output = apply_rotary_pos_emb(attn_output, cos, sin, inverse=True)
        grouped = attn_output.reshape(batch, seq_len, self.config.o_groups, -1)
        output = self.wo_b(self.wo_a(grouped).flatten(2))
        return output, attn_weights
