"""Small dense decoder using the Llama architecture."""

from transformers import LlamaConfig, LlamaForCausalLM


def build(tok, seq_len, layers=6, hidden=384, heads=8, kv_heads=8, ffn=1024):
    config = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=hidden,
        intermediate_size=ffn,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=seq_len,
        tie_word_embeddings=True,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return LlamaForCausalLM(config)
