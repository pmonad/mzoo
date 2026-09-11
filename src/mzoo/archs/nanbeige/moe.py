"""DeepSeek-V4 MoE block (from transformers) as the MLP of Nanbeige decoder layers.

First `dense_layers` physical layers keep Nanbeige's dense MLP. HF keeps DeepSeek's aux-loss-free
balancing bias (`e_score_correction_bias`) as a static buffer, so `Router` applies the
DeepSeek-V3 bias update (bias -= bias_update * sign(load - mean load)) on each training forward.
"""

import re

import torch
from torch import nn
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock, DeepseekV4TopKRouter

DEFAULTS = dict(
    dense_layers=2,
    n_routed_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=None,  # None: ~ffn / (1 + top_k) so shared + top_k active matches the dense MLP
    routed_scaling_factor=1.5,
    scoring_func="sqrtsoftplus",
    swiglu_limit=10.0,
    bias_update=1e-3,
    experts_implementation="grouped_mm",
    share_shared_expert=["middle"],  # LoopSplit groups whose MoE layers use one shared expert
)


def is_moe_layer(config, layer_idx):
    moe = getattr(config, "moe", None)
    return bool(moe) and layer_idx >= moe["dense_layers"]


def groups(config):
    """LoopSplit groups of physical layers; without LoopSplit every layer is looped (middle)."""
    n = config.num_hidden_layers
    middle = config.loop_middle_layers if config.enable_double_loop_split else n
    head = (n - middle) // 2
    return {"head": range(head), "middle": range(head, head + middle), "tail": range(head + middle, n)}


def share_shared_experts(layers, config):
    """MoE layers within each group in moe['share_shared_expert'] reuse the group's first shared expert.

    Returns HF `_tied_weights_keys` (module-name regexes) so checkpoints drop the aliases and re-tie on load.
    """
    moe = getattr(config, "moe", None)
    ties = {}
    for group in moe["share_shared_expert"] if moe else []:
        idx = [i for i in groups(config)[group] if is_moe_layer(config, i)]
        for i in idx[1:]:
            layers[i].mlp.shared_experts = layers[idx[0]].mlp.shared_experts
            ties[re.escape(f"layers.{i}.mlp.shared_experts.")] = re.escape(f"layers.{idx[0]}.mlp.shared_experts.")
    return ties


class Router(DeepseekV4TopKRouter):
    """fp32 routing; counts expert load and updates the balancing bias while training."""

    def __init__(self, config, bias_update):
        super().__init__(config)
        self.bias_update = bias_update
        self.register_buffer("load_counts", torch.zeros(self.num_experts), persistent=False)

    def forward(self, hidden_states):
        with torch.autocast(hidden_states.device.type, enabled=False):
            logits, weights, indices = super().forward(hidden_states.float())
        if self.training:
            with torch.no_grad():
                load = torch.bincount(indices.flatten(), minlength=self.num_experts).float()
                self.load_counts += load
                self.e_score_correction_bias -= self.bias_update * torch.sign(load - load.mean())
        return logits, weights, indices


def block(config, layer_idx):
    moe = config.moe
    v4 = DeepseekV4Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        hidden_act=config.hidden_act,
        moe_intermediate_size=moe["moe_intermediate_size"],
        n_routed_experts=moe["n_routed_experts"],
        num_experts_per_tok=moe["num_experts_per_tok"],
        routed_scaling_factor=moe["routed_scaling_factor"],
        scoring_func=moe["scoring_func"],
        swiglu_limit=moe["swiglu_limit"],
        mlp_layer_types=["moe"] * config.num_hidden_layers,
    )
    v4._experts_implementation = moe["experts_implementation"]
    m = DeepseekV4SparseMoeBlock(v4, layer_idx)
    m.gate = Router(v4, moe["bias_update"])
    # DeepSeek's init lives in its PreTrainedModel; Nanbeige's _init_weights only covers Linear/Embedding
    for p in (m.gate.weight, m.experts.gate_up_proj, m.experts.down_proj):
        nn.init.normal_(p, std=config.initializer_range)
    return m
