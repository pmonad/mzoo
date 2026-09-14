"""Regression test for ticket 0012: the per-forward ``shared`` dict must not carry a
kv-source group's compressed KV into the next group.

Without a cache, ``shared`` is the only group state and is handed to every layer, so a
second kv-source layer used to *append* its latents onto the previous group's (see
``DeepseekV41Attention.forward``). The with-cache path never had the bug (each source
owns a ``DeepseekV41CSACache``), so the two runs are the oracle for each other.
"""

import pytest
import torch
from transformers.cache_utils import Cache, DynamicSlidingWindowLayer

from mzoo.archs.dsv4 import modeling_deepseek_v41 as modeling
from mzoo.archs.dsv4.configuration_deepseek_v41 import DeepseekV41TextConfig
from mzoo.archs.dsv4.modeling_deepseek_v41 import DeepseekV41CSACache, DeepseekV41ForCausalLM

DEVICE = "cuda"
S = 130  # non-power-of-two, > sliding_window, and > compress_ratio*index_topk-free


def _build(compress_ratios, kv_sources):
    """Tiny fp32 model. Every backbone layer is MoE in this arch (no dense/MoE
    schedule, no ``first_k_dense_replace`` -- see README), so the "dense FFN" here is
    the smallest MoE that instantiates: 1 routed expert, top-1, + the shared expert."""
    config = DeepseekV41TextConfig(
        vocab_size=64, hidden_size=128, num_hidden_layers=len(compress_ratios),
        num_attention_heads=4, head_dim=64, qk_rope_head_dim=16,
        q_lora_rank=64, o_lora_rank=64, o_groups=2,
        compress_ratios=compress_ratios, kv_source_layer_ids=kv_sources,
        index_source_layer_ids=kv_sources, candidate_source_layer_id=-1,
        sliding_window=96, index_n_heads=4, index_head_dim=32, index_topk=64,
        moe_intermediate_size=64, n_routed_experts=1, num_experts_per_tok=1, n_shared_experts=1,
        num_nextn_predict_layers=0, use_cache=False,
        dspark_noise_token_id=0, tie_word_embeddings=True, bos_token_id=0, eos_token_id=1,
        max_position_embeddings=4096,
    )
    model = DeepseekV41ForCausalLM(config).to(DEVICE, dtype=torch.float32).eval()
    model.config._attn_implementation = "eager"
    return model


def _build_cache(config):
    """``DynamicCache(config=...)`` cannot build this arch's cache layers (README
    gotcha: transformers 5.17 calls the layer class without ``config``), so assemble
    the layer list by hand."""
    return Cache(
        layers=[
            DeepseekV41CSACache(config)
            if t == "shared_compressed_attention"
            else DynamicSlidingWindowLayer(sliding_window=config.sliding_window)
            for t in config.layer_types
        ]
    )


def _run(model, input_ids, cache):
    """One prefill forward, recording per layer the KV length handed to
    ``eager_attention_forward``. Returns ``({layer_idx: kv_len}, logits)``."""
    kv_lens = {}
    original = modeling.eager_attention_forward

    def spy(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        kv_lens[module.layer_idx] = key.shape[2]
        return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

    modeling.eager_attention_forward = spy
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=cache, use_cache=cache is not None)
    finally:
        modeling.eager_attention_forward = original
    return kv_lens, out.logits


@pytest.mark.parametrize(
    "compress_ratios,kv_sources",
    [
        ([0, 2, 2, 1], [1, 3]),  # two groups at DIFFERENT ratios -- the 0012 case
        ([0, 2], [1]),  # feature-off: a single group, where the bug could not fire
    ],
    ids=["two_groups", "single_group"],
)
def test_no_cache_matches_cache(compress_ratios, kv_sources):
    torch.manual_seed(0)
    model = _build(compress_ratios, kv_sources)
    input_ids = torch.randint(0, 64, (2, S), device=DEVICE)

    kv_lens_nocache, logits_nocache = _run(model, input_ids, None)
    kv_lens_cache, logits_cache = _run(model, input_ids, _build_cache(model.config.get_text_config()))

    # The last kv-source layer opens the final group: its KV axis is
    # window (S) + its OWN group only (S // ratio), never the earlier group's entries.
    last = kv_sources[-1]
    assert kv_lens_cache[last] == S + S // compress_ratios[last]
    assert kv_lens_nocache == kv_lens_cache
    torch.testing.assert_close(logits_nocache, logits_cache, atol=0.0, rtol=0.0)
