"""End-to-end GPU check of ``swa_attn`` against the vendored DSV4.1 model.

This is the first package in the series whose semantics a *real* model layer has:
a ``compress_ratio == 0`` layer of ``archs/dsv4/modeling_deepseek_v41.py`` is
exactly "sliding window of ``sliding_window`` raw tokens over a shared K==V
latent, plus a per-head sink". So instead of trusting ``golden`` alone, this test
runs the tiny 2-layer smoke config once, captures the exact ``(query, key, value,
attention_mask, scaling, attn_sink)`` tuple layer 0 hands to
``eager_attention_forward``, replays those tensors through ``swa_attn.attn`` in
bf16, and checks the output against eager's own fp32 result with the usual
<= 2x criterion (baseline: ``golden(..., dtype=bfloat16)`` on the same bf16
tensors).

The two helpers are imported **explicitly from ``golden_ref_test``** rather than
copied: they are the same capture, and duplicating a monkeypatch of
``modeling.eager_attention_forward`` in two files is how the two drift apart.
The model's fp8 fake-quant of the window cache happens *before* the capture, so
whatever the kernel sees is already the dequantized bf16 tensor -- which is the
v1 contract of ticket 0002 (in-kernel dequant is ticket 0010).
"""

import torch

from mzoo.layers.attn.dense_attn.ref import assert_within_2x_torch
from mzoo.layers.attn.golden_ref import golden
from mzoo.layers.attn.golden_ref_test import _build_tiny_model, _capture_attention_calls, _eager_output_and_lse
from mzoo.layers.attn.swa_attn.attn import attn

WINDOW = 128  # == the tiny config's sliding_window


def test_attn_matches_model_window_layer():
    torch.manual_seed(6)
    model = _build_tiny_model()
    assert model.config.sliding_window == WINDOW
    batch, seq = 2, 192  # > window, and a multiple of every T_q = block_M // H at H=4
    input_ids = torch.randint(0, 64, (batch, seq), device="cuda")
    c = _capture_attention_calls(model, input_ids)[0]  # layer 0: compress_ratio == 0
    assert c["key"].shape[1] == 1 and c["key"].shape[2] == seq  # shared latent, window only

    o_ref, _ = _eager_output_and_lse(c["query"], c["key"], c["value"], c["attention_mask"],
                                     c["scaling"], c["attn_sink"])
    q = c["query"].transpose(1, 2).to(torch.bfloat16).contiguous()  # [B,S,H,D]
    kv = c["key"].transpose(1, 2).to(torch.bfloat16).contiguous()  # [B,S,1,D]
    assert q.shape[-1] == c["key"].shape[-1] and abs(c["scaling"] - q.shape[-1]**-0.5) < 1e-9

    o = attn(q, kv, window=WINDOW, sinks=c["attn_sink"].float())
    o_bf16, _ = golden(q, kv, level="window", window=WINDOW, sinks=c["attn_sink"].float(),
                       dtype=torch.bfloat16)
    assert_within_2x_torch(o, o_ref, o_bf16, "o vs model eager")
