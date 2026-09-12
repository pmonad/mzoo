# Split out of the frozen dsv4 baseline `src/mzoo/archs/dsv4/modeling_deepseek_v41.py`
# (see there for upstream provenance + Apache-2.0 license). Covers: the top-k router, a single expert MLP and the sparse MoE block.

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN

from .config import DeepseekV41TextConfig


class DeepseekV41TopKRouter(nn.Module):
    """MoE gate. The correction bias (`bias`) steers expert *selection* only; the
    routing weights come from the unbiased scores. Image-span tokens switch to a
    separate `bias_vl` (training `noaux_tc_for_vl`)."""

    def __init__(self, config: DeepseekV41TextConfig, n_experts: int, n_activated: int):
        super().__init__()
        self.top_k = n_activated
        self.score_fn = ACT2FN[config.scoring_func]
        self.gate_temp = config.gate_temp
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(n_experts, config.hidden_size))
        self.bias = nn.Parameter(torch.empty(n_experts, dtype=torch.float32))
        self.bias_vl = nn.Parameter(torch.empty(n_experts, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor, image_mask: torch.Tensor | None = None):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
        scores = F.linear(flat, self.weight.float()) / self.gate_temp
        scores = self.score_fn(scores)
        bias = self.bias
        if image_mask is not None and image_mask.any():
            bias = torch.where(image_mask.reshape(-1, 1), self.bias_vl, self.bias)
        indices = (scores + bias).topk(self.top_k, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.top_k > 1:
            # `+1e-20` on the sum — NOT `rms_norm_eps`; matches training.
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.routed_scaling_factor, indices


class DeepseekV41Expert(nn.Module):
    """One SwiGLU expert (`w1` gate / `w3` up / `w2` down). The clamps come from
    training: they keep fp8/fp4 activations in range — up on both sides, gate above."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        inter = config.moe_intermediate_size
        self.w1 = nn.Linear(config.hidden_size, inter, bias=False)
        self.w2 = nn.Linear(inter, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, inter, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        y = self.act_fn(gate) * up
        if weight is not None:
            y = weight * y
        return self.w2(y.to(dtype))


class DeepseekV41SparseMoeBlock(nn.Module):
    """Top-k routed experts plus one shared expert every token goes through. Eager
    dispatch loops over the experts that received tokens — spec-grade, not a serving
    path."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        # DSpark draft layers (M2) have their own expert counts; this block is only
        # ever built for backbone layers.
        self.n_experts = config.n_routed_experts
        n_activated = config.num_experts_per_tok
        self.gate = DeepseekV41TopKRouter(config, self.n_experts, n_activated)
        self.experts = nn.ModuleList([DeepseekV41Expert(config) for _ in range(self.n_experts)])
        self.shared_experts = DeepseekV41Expert(config)

    def forward(self, hidden_states: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        weights, indices = self.gate(hidden_states, image_mask)
        y = torch.zeros_like(flat, dtype=torch.float32)
        # eager dispatch loop: per-expert host reads are inherent to it (spec-grade,
        # not a serving path) — the counts stay a tensor (TRF056).
        counts = torch.bincount(indices.flatten(), minlength=self.n_experts)
        for i in range(self.n_experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y[idx] += self.experts[i](flat[idx], weights[idx, top, None]).float()
        y += self.shared_experts(flat).float()
        return y.to(hidden_states.dtype).view(shape)
