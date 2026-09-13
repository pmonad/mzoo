"""Tiny DeepSeek-V4.1: sparse (CSA2) attention, mHC residual streams, 1 shared + top-k MoE."""

from .configuration_deepseek_v41 import DeepseekV41TextConfig
from .modeling_deepseek_v41 import DeepseekV41ForCausalLM


def build(
    tok,
    seq_len,
    layers=6,  # >=4 so the auto-derived schedule has both a ratio-2 and a ratio-1 group
    hidden=256,
    heads=4,
    head_dim=64,
    qk_rope_head_dim=16,
    q_lora_rank=128,
    o_lora_rank=128,
    o_groups=2,  # must divide heads * head_dim
    expert_ffn=256,
    routed=8,
    top_k=2,
    shared=1,
    index_n_heads=4,
    index_head_dim=32,
    index_topk=64,
    sliding_window=128,
    candidate_topk_blocks=64,
):
    config = DeepseekV41TextConfig(
        vocab_size=len(tok),
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        head_dim=head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        o_groups=o_groups,
        max_position_embeddings=seq_len,
        sliding_window=sliding_window,
        candidate_topk_blocks=candidate_topk_blocks,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        moe_intermediate_size=expert_ffn,
        n_routed_experts=routed,
        num_experts_per_tok=top_k,
        n_shared_experts=shared,
        num_nextn_predict_layers=0,  # no DSpark draft layers (unimplemented in modeling)
        use_cache=False,  # training only; the CSA cache layer is not constructible by DynamicCache
        dspark_noise_token_id=tok.eos_token_id,  # inert, but must be in-vocab to pass validation
        tie_word_embeddings=True,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        # compress_ratios / kv+index source ids / layer_types auto-derive in __post_init__
    )
    return DeepseekV41ForCausalLM(config)
