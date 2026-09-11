"""Tiny MoE: Llama-style attention, 1 shared + top-k of N routed experts (Qwen2-MoE)."""

from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM


def build(
    tok,
    seq_len,
    layers=6,
    hidden=512,
    heads=8,
    kv_heads=8,
    expert_ffn=688,  # half of dense ffn: shared + top-1 matches dense active compute
    routed=7,
    top_k=1,
    aux=0.01,
    dense_first=0,  # make the first N layers plain dense MLPs instead of MoE
    ffn=1376,  # dense layer width, same default as the dense arch
):
    config = Qwen2MoeConfig(
        vocab_size=len(tok),
        hidden_size=hidden,
        intermediate_size=ffn,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=seq_len,
        qkv_bias=False,
        moe_intermediate_size=expert_ffn,
        shared_expert_intermediate_size=expert_ffn,
        num_experts=routed,
        num_experts_per_tok=top_k,
        mlp_only_layers=list(range(dense_first)),
        norm_topk_prob=False,  # must stay False for top-1, else router gets no gradient
        output_router_logits=True,
        router_aux_loss_coef=aux,
        tie_word_embeddings=True,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return Qwen2MoeForCausalLM(config)
