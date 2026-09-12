# Split out of the frozen dsv4 baseline `src/mzoo/archs/dsv4/modeling_deepseek_v41.py`
# (see there for upstream provenance + Apache-2.0 license). Covers: the decoder layer (mHC residual streams), the PreTrainedModel base, the text backbone and the causal-LM head.

import torch
import torch.nn.functional as F
from torch import nn

from transformers import initialization as init  # mzoo: was relative `...` import
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoTokenizer
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import OutputRecorder, capture_outputs

from .attention import DeepseekV41Attention
from .cache import DeepseekV41CSACache  # re-exported via __all__
from .config import DeepseekV41Config, DeepseekV41TextConfig
from .engram import (
    DeepseekV41Engram,
    DeepseekV41EngramEmbedding,  # re-exported via __all__
    DeepseekV41NgramHashState,
    EngramLayout,
)
from .moe import DeepseekV41SparseMoeBlock, DeepseekV41TopKRouter
from .norm_rope import DeepseekV41RMSNorm, DeepseekV41RotaryEmbedding, DeepseekV41UnweightedRMSNorm


class DeepseekV41DecoderLayer(GradientCheckpointingLayer):
    r"""A V4.1 block: the residual stream is `hc_mult` parallel copies (hyper-
    connections), with the engram lookup injected at its layers before the block.

    The single-pass mHC mapping computes all three coefficient sets (pre / post / comb)
    from ONE projection of the flattened stream — but the `pre` a site computes is
    consumed by the *next* site: attention collapses with the previous site's mix and
    the FFN with the attention's. The mHC parameters are raw layer attributes
    (`hc_attn_fn` / `hc_attn_base` / `hc_attn_scale`, fp32), matching the checkpoint."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.attn = DeepseekV41Attention(config, layer_idx)
        self.ffn = DeepseekV41SparseMoeBlock(config, layer_idx)
        self.attn_norm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # CODEPATH: DeepSeek-V4.1-Flash ships n-gram hash tables on layers 1 and 14
        # (engram_layer_ids=[1, 14]); every other checkpoint — and the tiny test
        # configs — sets engram_layer_ids=[] and takes the None side (no engram path).
        self.engram = DeepseekV41Engram(config, layer_idx) if layer_idx in config.engram_layer_ids else None
        self.hc_input_norm = DeepseekV41UnweightedRMSNorm(eps=config.rms_norm_eps)
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def hc_mixes(self, hidden_streams: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
        """One projection of the normalized flattened stream → (pre, post, comb), with
        `comb` Sinkhorn-projected onto the doubly-stochastic manifold. Normalization is
        over the whole flattened hc·D stream (one statistic per token)."""
        hc = self.hc_mult
        # fp32 from the start — the reference upcasts before normalizing, so a
        # bf16/fp16 model must not round the stream before the mix projection.
        flat = self.hc_input_norm(hidden_streams.flatten(start_dim=2).float())
        mixes = F.linear(flat, fn.float())
        pre = torch.sigmoid(mixes[..., :hc] * scale[0].float() + base[:hc].float()) + self.hc_eps
        post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1].float() + base[hc : 2 * hc].float())
        comb_logits = (mixes[..., 2 * hc :] * scale[2].float() + base[2 * hc :].float()).view(
            *mixes.shape[:-1], hc, hc
        )
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre, post, comb

    @staticmethod
    def hc_collapse(hidden_streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        """Collapse the hc copies into one sublayer input, weighted by `pre`."""
        return (pre.unsqueeze(-1) * hidden_streams.float()).sum(dim=2).to(hidden_streams.dtype)

    @staticmethod
    def hc_expand(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        """Place the sublayer output into the streams and mix the residual through
        `comb`: out_k = post_k * x + Σ_j comb[j, k] * residual_j."""
        mixed = torch.einsum("bsjk,bsjd->bskd", comb.float(), residual.float())
        out = post.unsqueeze(-1) * x.float().unsqueeze(-2) + mixed
        return out.to(residual.dtype)

    def forward(
        self,
        hidden_streams: torch.Tensor,
        pre_mix: torch.Tensor,
        hash_ids: torch.Tensor | None,
        token_mask: torch.Tensor | None,
        shared: dict,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_streams: [B, S, hc, hidden]
        if self.engram is not None and hash_ids is not None:
            hidden_streams = self.engram(hidden_streams, hash_ids[:, :, self.engram.layer_hash_index, :], token_mask)

        residual = hidden_streams
        attn_pre, attn_post, attn_comb = self.hc_mixes(
            hidden_streams, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        collapsed = self.hc_collapse(hidden_streams, pre_mix)
        attn_output, _ = self.attn(self.attn_norm(collapsed), shared=shared, **kwargs)
        hidden_streams = self.hc_expand(attn_output, residual, attn_post, attn_comb)

        residual = hidden_streams
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(
            hidden_streams, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        collapsed = self.hc_collapse(hidden_streams, attn_pre)
        ffn_output = self.ffn(self.ffn_norm(collapsed))
        hidden_streams = self.hc_expand(ffn_output, residual, ffn_post, ffn_comb)
        return hidden_streams, ffn_pre


# Deliberate: this base serves BOTH model types. The text backbone chain
# (DeepseekV41TextModel, registered as `deepseek_v41_text`) needs the flat
# DeepseekV41TextConfig; DeepseekV41ForCausalLM overrides with the composite
# DeepseekV41Config because the released checkpoint's config.json is composite and
# its top-level quantization_config must reach the quantizer.
@auto_docstring
class DeepseekV41PreTrainedModel(PreTrainedModel):  # trf-ignore: TRF001
    config_class = DeepseekV41TextConfig

    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV41DecoderLayer"]
    # Eager-only, same reasons as V4: FA caps head_dim at 256 (V4.1 uses 512); SDPA has
    # no per-head sink term; the compressed branch concatenates entries onto the KV axis
    # inside the block, after the model-level mask was built.
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    # The compressor's group-buffer state isn't rewindable across drafts.
    _is_stateful = True
    # DSpark draft layers, the vision tower, the aligner and the image delimiter
    # embeddings ship in the checkpoint but their modules land in follow-up PRs.
    _keys_to_ignore_on_load_unexpected = [
        r"(^|\.)mtp\..*",
        r"^vision\..*",
        r"^aligner\..*",
        r"^image_(start|end|newline)$",
    ]
    # fp32-critical parameters: exactly the tensors the released checkpoint stores
    # in F32 — the raw mHC parameters, the attention sinks and the gate biases
    # (`bias` / `bias_vl`; every other linear is bias-free). The norms and the
    # ratio-2 compressor gate ship BF16 and stay in the model dtype (the reference
    # upcasts them at load; the forward computes in fp32 either way).
    _keep_in_fp32_modules_strict = [
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
        "hc_ffn_fn",
        "hc_ffn_base",
        "hc_ffn_scale",
        "attn_sink",
        "bias",
        "bias_vl",
    ]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, DeepseekV41TopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.bias)
            init.zeros_(module.bias_vl)
        elif isinstance(module, DeepseekV41Attention):
            init.zeros_(module.attn_sink)
        elif isinstance(module, DeepseekV41DecoderLayer):
            init.normal_(module.hc_attn_fn, mean=0.0, std=std)
            init.zeros_(module.hc_attn_base)
            init.ones_(module.hc_attn_scale)
            init.normal_(module.hc_ffn_fn, mean=0.0, std=std)
            init.zeros_(module.hc_ffn_base)
            init.ones_(module.hc_ffn_scale)
        elif isinstance(module, DeepseekV41Engram):
            init.normal_(module.embed.weight, mean=0.0, std=std)
            init.ones_(module.embed.scale)
            init.ones_(module.q_weight)
            init.ones_(module.k_weight)
        elif isinstance(module, DeepseekV41RotaryEmbedding):
            # `from_pretrained` builds on the meta device, so the inv_freq buffers
            # computed in __init__ never materialize — rebuild them here.
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


@auto_docstring
class DeepseekV41TextModel(DeepseekV41PreTrainedModel):
    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepseekV41DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV41RotaryEmbedding(config)
        self.engram_layout = EngramLayout.from_config(config)
        self.engram_hash_state: DeepseekV41NgramHashState | None = None
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def set_input_embeddings(self, value: nn.Module):
        self.embed = value

    def bind_tokenizer(self, tokenizer):
        """Build the engram hash state (tokenizer-derived compressed token map, prime
        bucket layout, per-layer hash multipliers). The hashes must be replicated
        exactly or the pretrained tables are meaningless. Called automatically on the
        first forward when the model was loaded from a hub checkpoint; call it
        explicitly otherwise."""
        self.engram_hash_state = DeepseekV41NgramHashState(self.config, tokenizer)
        return self

    def _ensure_hash_state(self, device: torch.device):
        if self.engram_layout is None or not self.engram_layout.layer_ids:
            return
        if self.engram_hash_state is None:
            if not getattr(self.config, "_name_or_path", ""):
                raise ValueError(
                    "The engram layers need the tokenizer to build their n-gram hash state "
                    "(compressed token map + hash multipliers). Call "
                    "`model.model.bind_tokenizer(tokenizer)` on a `DeepseekV41ForCausalLM` "
                    "(or `model.bind_tokenizer(tokenizer)` on a `DeepseekV41TextModel`), "
                    "or load the model from a hub checkpoint (it binds automatically)."
                )
            self.bind_tokenizer(AutoTokenizer.from_pretrained(self.config._name_or_path))

    _can_record_outputs = {
        "router_logits": OutputRecorder(DeepseekV41TopKRouter),
        # The residual stream is `hc_mult` parallel copies; the recorded
        # `hidden_states` are the collapsed per-block inputs (each layer's
        # `attn_norm` in/out, plus the initial embedding collapse).
        "hidden_states": OutputRecorder(DeepseekV41RMSNorm, layer_name="attn_norm"),
        "attentions": DeepseekV41Attention,
    }

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        elif self.engram_layout:
            # The engram hashes token ids; there is no way to recover them from embeddings.
            raise ValueError("engram layers require `input_ids` (the hash state cannot consume `inputs_embeds`)")
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.shape[0], -1)
        if isinstance(attention_mask, dict):
            # `generate()` may pass a per-layer-type mask dict; both V4.1 layer types
            # attend over the same sliding window, so any of them works.
            causal_mask = next(iter(attention_mask.values()))
        else:
            causal_mask = create_sliding_window_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        position_embeddings = {
            "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }
        self._ensure_hash_state(inputs_embeds.device)
        # Pads (attention_mask == 0) are hashed as DEAD so n-grams never span them.
        # Only a 2D mask carries per-token liveness; generate()'s per-layer-type mask
        # dict has no 2D form to read it from.
        live_mask = (
            attention_mask.bool() if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2 else None
        )
        hash_ids = (
            self.engram_hash_state(input_ids, position_ids, live_mask) if self.engram_hash_state is not None else None
        )

        shared: dict = {}
        # One-hot initial mix: the first site collapses stream 0 only.
        pre_mix = hidden_states.new_zeros(*hidden_states.shape[:-1], dtype=torch.float32)
        pre_mix[..., 0] = 1.0
        for layer in self.layers:
            hidden_states, pre_mix = layer(
                hidden_states,
                pre_mix,
                hash_ids,
                None,
                shared,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
            )
        # Final collapse with the last site's pre mix, then the shared norm.
        hidden_states = DeepseekV41DecoderLayer.hc_collapse(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


@auto_docstring
class DeepseekV41ForCausalLM(DeepseekV41PreTrainedModel, GenerationMixin):
    # The released checkpoint's config.json is composite (top-level `quantization_config`,
    # `text_config`, `vision_config`) with `architectures: [DeepseekV41ForCausalLM]`, so this
    # class must accept the composite config: `get_hf_quantizer` only sees `quantization_config`
    # on the config produced from `config_class` — pointing it at the bare text config silently
    # dropped the FP8 quantization config and fp8 tensors then failed to load. The text config
    # is unwrapped in `__init__` (same pattern as `MllamaForCausalLM`); a text config passed
    # directly is returned unchanged by `get_text_config()`.
    config_class = DeepseekV41Config

    def __init__(self, config):
        super().__init__(config.get_text_config())
        self.model = DeepseekV41TextModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `-100` (see `input_ids` docstring). Tokens with indices set to `-100` are
            ignored (masked), the loss is computed over tokens with labels in `[0, ..., config.vocab_size]`.
        shift_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Already-shifted next-token targets, aligned with `logits` (used for sequence and context
            parallel training, where the shift must happen before sharding). When given, they take
            precedence over `labels` for the loss. Accepted via `**kwargs` rather than as an explicit
            argument -- see the note below.
        """
        # owlet1: `shift_labels` is popped from **kwargs instead of being an explicit
        # parameter. HF's `find_labels` returns every forward arg containing "label", and
        # `Trainer.prediction_step` then requires ALL of them to be present in the batch.
        # With `shift_labels` in the signature a plain {input_ids, labels} collator looks
        # label-free, so eval loss is silently skipped and no eval_loss is ever logged.
        shift_labels = kwargs.pop("shift_labels", None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()
        loss = None
        if labels is not None or shift_labels is not None:
            # `shift_labels` carries already-aligned targets (sequence / context
            # parallel training); `self.loss_function` shifts plain `labels` itself.
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, shift_labels=shift_labels
            )
        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def _reorder_cache(self, past_key_values: "Cache", beam_idx: torch.LongTensor) -> "Cache":
        """Beam-search support: the cache layers' `reorder_cache` permutes the
        sliding-window and group state, and the engram n-gram history (which lives
        on the model, not the cache) must follow the beams too — otherwise a beam
        hashes its next tokens against another beam's predecessor history."""
        past_key_values.reorder_cache(beam_idx)
        hash_state = self.model.engram_hash_state
        if hash_state is not None and hash_state.history is not None:
            hash_state.history = hash_state.history.index_select(0, beam_idx.to(hash_state.history.device))
        return past_key_values


__all__ = [
    "DeepseekV41PreTrainedModel",
    "DeepseekV41TextModel",
    "DeepseekV41ForCausalLM",
    "DeepseekV41CSACache",
    "DeepseekV41EngramEmbedding",
    "EngramLayout",
    "DeepseekV41NgramHashState",
]
