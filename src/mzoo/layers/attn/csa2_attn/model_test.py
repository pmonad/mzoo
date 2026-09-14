"""End-to-end GPU check of ``csa2_attn`` against the vendored DSV4.1 model.

``csa_attn`` had to *disable* the top-k (``index_topk`` past every ``compressed_len``) to
find a dense compressed layer in the tiny smoke config. ``csa2_attn`` is the sparse kernel,
so it replays that config unchanged: ``golden_ref_test._build_tiny_model`` has layer 1 with
``compress_ratio=2`` and ``index_topk=64 < compressed_len = S // 2 = 80``, i.e. the indexer
really does drop visible groups.

The model expresses the selection as the additive bias ``shared["topk_bias"]``, concatenated
verbatim onto the KV axis of ``attention_mask``; ``golden_ref_test._mask_to_indices`` is the
test-only inverse that turns that ``0 / -inf`` bias back into the ``[B, S, topk]`` index list
this kernel consumes. Everything else (fp8/fp4 fake-quant of the caches, the compressor, the
latent RoPE) happens before the capture, so the kernel sees already-dequantized bf16 tensors.
"""

import torch

from mzoo.layers.attn.csa2_attn.attn import attn
from mzoo.layers.attn.csa2_attn.fwd import fwd
from mzoo.layers.attn.csa2_attn.ref import to_kernel_inputs
from mzoo.layers.attn.dense_attn.ref import assert_within_2x_torch
from mzoo.layers.attn.golden_ref import golden
from mzoo.layers.attn.golden_ref_test import (
    _build_tiny_model,
    _capture_attention_calls,
    _eager_output_and_lse,
    _mask_to_indices,
)

WINDOW, RATIO, TOPK = 128, 2, 64


def test_attn_matches_model_sparse_layer():
    torch.manual_seed(6)
    model = _build_tiny_model()
    batch, seq = 2, 160  # S > window, and a multiple of T_q = 64 // heads = 16
    input_ids = torch.randint(0, 64, (batch, seq), device="cuda")
    c = _capture_attention_calls(model, input_ids)[1]  # layer 1: compress_ratio == RATIO
    groups = c["key"].shape[2] - seq
    assert groups == seq // RATIO > TOPK  # real sparsity: more visible groups than index_topk

    main_bias = c["attention_mask"][..., seq:].squeeze(1)  # == shared["topk_bias"]
    indices = _mask_to_indices(main_bias, topk=TOPK)
    assert (main_bias == 0).sum(-1).max() == TOPK, "the indexer did not fill a single top-k list"

    o_ref, lse_ref = _eager_output_and_lse(c["query"], c["key"], c["value"], c["attention_mask"],
                                           c["scaling"], c["attn_sink"])
    q, kv, main_kv, sinks = to_kernel_inputs(c, seq)
    assert abs(c["scaling"] - q.shape[-1]**-0.5) < 1e-9

    o = attn(q, kv, main_kv, indices, window=WINDOW, compress_ratio=RATIO, sinks=sinks)
    o_bf16, _ = golden(q, kv, level="sparse", window=WINDOW, main_kv=main_kv, compress_ratio=RATIO,
                       indices=indices, sinks=sinks, dtype=torch.bfloat16)
    assert_within_2x_torch(o, o_ref, o_bf16, "o vs model eager")

    # lse against the logsumexp of eager's own (fp32) combined logits: the kernel's inputs are
    # bf16 round-trips of them, so this is a bf16-scale check, not the fp32 one fwd_test does.
    _, lse = fwd(q, kv, main_kv, indices, window=WINDOW, compress_ratio=RATIO, sinks=sinks)
    torch.testing.assert_close(lse, lse_ref, atol=5e-2, rtol=5e-2)
