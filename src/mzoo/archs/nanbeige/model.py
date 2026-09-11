"""Nanbeige4.2 looped transformer at small scale, using the copied modeling code.

Default: LoopSplit (3, 3x3, 3) = 9 unique layers, 15 effective. Nanbeige features (num_loops,
loop_share_kv, mHC, n-gram embeddings) are NanbeigeConfig kwargs passed via **kw.
`moe=True` swaps the MLP of layers >= dense_layers for a DeepSeek-V4 MoE block (see moe.py);
MoE settings are moe.DEFAULTS keys, e.g. --moe=True --n_routed_experts=16.
"""

from . import moe as moe_lib
from .configuration_nanbeige import NanbeigeConfig
from .modeling_nanbeige import NanbeigeForCausalLM


def build(tok, seq_len, layers=9, hidden=384, heads=8, kv_heads=8, ffn=1024, moe=False, **kw):
    moe_cfg = {k: kw.pop(k, v) for k, v in moe_lib.DEFAULTS.items()}
    if moe_cfg["moe_intermediate_size"] is None:  # multiple of 8: grouped_mm needs 16-byte strides in bf16
        moe_cfg["moe_intermediate_size"] = round(ffn / (1 + moe_cfg["num_experts_per_tok"]) / 8) * 8
    kw = dict(
        enable_double_loop_split=True,
        loop_middle_layers=3,
        loop_middle_repeats=3,
        enable_depth_attention=False,  # anchors re-added on each LoopSplit repeat; unreviewed
    ) | kw
    config = NanbeigeConfig(
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
        pad_token_id=None,  # tok pad is eos; a padding_idx would freeze the eos embedding
        use_cache=False,
        moe=moe_cfg if moe else None,
        **kw,
    )
    return NanbeigeForCausalLM(config)
