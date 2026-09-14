"""End-to-end GPU check of ``csa_attn`` against the vendored DSV4.1 model.

``swa_attn`` could replay layer 0 (``compress_ratio == 0``) of the tiny smoke config
directly. ``csa_attn`` is the *dense* compressed layer, and the tiny config of
``golden_ref_test._build_tiny_model`` is **sparse** there: its layer 1 has
``index_topk=64`` against ``compressed_len = S // 2 = 96``, so the indexer drops
visible groups and the eager path is top-k, not dense.

So this test builds the same config with one field changed -- ``index_topk`` larger
than any ``compressed_len`` -- which makes the indexer's step 5 pick *every*
group-causally-visible entry (the invisible ones score ``-inf`` and go to the dummy
slot), so ``shared["topk_bias"]`` is exactly the dense group-causal mask
``g < (t + 1) // ratio``. That is ``csa_attn``'s semantics, and the test asserts the
bias really is dense before trusting it. Setting ``index_source_layer_ids=[]``
instead would **not** work: with no index source the model masks every compressed
entry off (``F.pad(..., value=-inf)`` in ``DeepseekV41Attention.forward``).

The capture helpers are imported from ``../golden_ref_test.py`` rather than copied;
the fp8/fp4 fake-quants happen *before* the capture, so the kernel sees the already
dequantized bf16 tensors -- the v1 contract of ticket 0003.
"""

import torch

from mzoo.archs.dsv4.configuration_deepseek_v41 import DeepseekV41TextConfig
from mzoo.archs.dsv4.modeling_deepseek_v41 import DeepseekV41ForCausalLM
from mzoo.layers.attn.csa_attn.attn import attn
from mzoo.layers.attn.csa_attn.ref import to_kernel_inputs
from mzoo.layers.attn.dense_attn.ref import assert_within_2x_torch
from mzoo.layers.attn.golden_ref import golden
from mzoo.layers.attn.golden_ref_test import _capture_attention_calls, _eager_output_and_lse

WINDOW, RATIO = 128, 2


def _build_dense_compressed_model():
    """``golden_ref_test._build_tiny_model``'s config with ``index_topk`` raised past any
    ``compressed_len``, so layer 1's compressed branch is dense group-causal."""
    config = DeepseekV41TextConfig(
        vocab_size=64, hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, head_dim=64, qk_rope_head_dim=16,
        q_lora_rank=128, o_lora_rank=128, o_groups=2,
        compress_ratios=[0, RATIO], kv_source_layer_ids=[1], index_source_layer_ids=[1],
        candidate_source_layer_id=-1, sliding_window=WINDOW,
        index_n_heads=4, index_head_dim=32, index_topk=4096,  # >= any compressed_len -> dense
        moe_intermediate_size=256, n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
        num_nextn_predict_layers=0, use_cache=False,
        dspark_noise_token_id=0, tie_word_embeddings=True, bos_token_id=0, eos_token_id=1,
        max_position_embeddings=4096,
    )
    model = DeepseekV41ForCausalLM(config).to("cuda", dtype=torch.float32).eval()
    model.config._attn_implementation = "eager"
    return model


def test_attn_matches_model_compressed_layer():
    torch.manual_seed(6)
    model = _build_dense_compressed_model()
    batch, seq = 2, 192  # > window, and a multiple of every T_q = block_M // H at H=4
    input_ids = torch.randint(0, 64, (batch, seq), device="cuda")
    c = _capture_attention_calls(model, input_ids)[1]  # layer 1: compress_ratio == RATIO
    groups = c["key"].shape[2] - seq
    assert groups == seq // RATIO  # window (S raw) + compressed (S // ratio groups)

    # the compressed tail of the mask must be the *dense* group-causal bias, not a top-k one
    main_bias = c["attention_mask"][..., seq:].squeeze(1)  # == shared["topk_bias"]
    t = torch.arange(seq, device="cuda").view(-1, 1)
    g = torch.arange(groups, device="cuda").view(1, -1)
    assert torch.equal(main_bias[0] == 0, g < ((t + 1) // RATIO)), "indexer did not select every visible group"

    o_ref, _ = _eager_output_and_lse(c["query"], c["key"], c["value"], c["attention_mask"],
                                     c["scaling"], c["attn_sink"])
    q, kv, main_kv, sinks = to_kernel_inputs(c, seq)
    assert abs(c["scaling"] - q.shape[-1]**-0.5) < 1e-9

    o = attn(q, kv, main_kv, window=WINDOW, compress_ratio=RATIO, sinks=sinks)
    o_bf16, _ = golden(q, kv, level="compressed", window=WINDOW, main_kv=main_kv,
                       compress_ratio=RATIO, sinks=sinks, dtype=torch.bfloat16)
    assert_within_2x_torch(o, o_ref, o_bf16, "o vs model eager")
