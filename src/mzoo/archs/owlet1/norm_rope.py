# Split out of the frozen dsv4 baseline `src/mzoo/archs/dsv4/modeling_deepseek_v41.py`
# (see there for upstream provenance + Apache-2.0 license). Covers: RMSNorm, unweighted RMSNorm, rotary embeddings and RoPE application.

import torch
from torch import nn

from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import maybe_autocast

from .config import DeepseekV41Config


@use_kernel_forward_from_hub("RMSNorm")
class DeepseekV41RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        DeepseekV41RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class DeepseekV41UnweightedRMSNorm(nn.Module):
    """RMS normalization without a learned weight — used on the flattened
    hyper-connection stream before the mix projection."""

    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class DeepseekV41RotaryEmbedding(nn.Module):
    """Same two-rope scheme as V4, reused verbatim: `main` = plain `rope_theta` for
    pure sliding-window layers, `compress` = `compress_rope_theta` + optional YaRN for
    layers with a compressed branch (one latent stands for `compress_ratio` tokens, so
    its positions are further apart — hence the larger base). The config folds the
    checkpoint's flat `rope_scaling` into `rope_parameters` the same way V4 does."""

    @deprecate_kwarg("device", version="5.18")
    def __init__(self, config: DeepseekV41Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        # Only the nested per-rope-type sub-dicts are real layer types — the top-level
        # `rope_type` key that ``convert_rope_params_to_dict`` may leave on
        # ``config.rope_parameters`` is a flat-shape leftover, not a layer.
        self.layer_types = [k for k, v in config.rope_parameters.items() if isinstance(v, dict)]
        self.rope_type = {}
        for layer_type in self.layer_types:
            rope_params = config.rope_parameters[layer_type]
            self.rope_type[layer_type] = rope_params["rope_type"]
            rope_init_fn = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            inv_freq, attention_scaling = rope_init_fn(config, device, layer_type=layer_type)
            setattr(self, f"{layer_type}_inv_freq", nn.Buffer(inv_freq, persistent=False))
            setattr(self, f"{layer_type}_original_inv_freq", nn.Buffer(inv_freq.clone(), persistent=False))
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)

    @staticmethod
    @deprecate_kwarg("device", version="5.18")
    def compute_default_rope_parameters(
        config: DeepseekV41Config, device=None, layer_type: str | None = None, **kwargs
    ) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            layer_type (`str`, *optional*):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters[layer_type]["rope_theta"]
        # key difference to gemma3: partial rope
        partial_rotary_factor = config.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq.to(device), attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids, layer_type=None):
        # Key difference vs Laguna's forward: no `torch.cat([freqs, freqs], dim=-1)`
        # duplication. V4's interleaved RoPE pairs consecutive channels, so we only need
        # `rope_head_dim // 2` unique θ entries — the `apply_rotary_pos_emb` helper does
        # the `repeat_interleave(2)` next to the rotation math, where the link between
        # the doubled dim and `rotate_half` is local and obvious.
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            cos = freqs.cos() * attention_scaling
            sin = freqs.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False):
    """Interleaved-pair RoPE on the trailing rope slice of `x`.

    DeepSeek-V4.1 pairs *adjacent* channels of the rope slice (even, odd) and rotates
    each pair as a complex number — matching the reference implementation's
    ``torch.view_as_complex(x.unflatten(-1, (-1, 2)))`` convention. `cos` / `sin`
    carry one entry per pair (``rope_dim // 2``). The leading nope channels pass
    through. ``inverse=True`` conjugates the rotation: the attention output carries
    the query's RoPE, and the inverse rotation removes it so the shared rotated cache
    stays in one form. Accepts ``[B, S, D]`` and ``[B, S, H, D]``.
    """
    if inverse:
        sin = -sin
    rope_dim = cos.shape[-1] * 2
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    if x.ndim == 4:
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)  # [B, S, 1, rd/2]
    even, odd = rope[..., 0::2].float(), rope[..., 1::2].float()
    rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)
    return torch.cat([nope, rotated.to(x.dtype)], dim=-1)
