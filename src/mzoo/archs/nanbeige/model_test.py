"""Smoke tests for the Nanbeige LoopSplit build (tiny sizes, no full-size models)."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from mzoo.archs.nanbeige.model import build

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOK = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
TINY = dict(hidden=64, heads=4, kv_heads=2, ffn=128)


def test_layer_order_default_and_nanbeige_formula():
    order = build(TOK, 32, **TINY).model._get_layer_order()
    assert order == [0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 6, 7, 8]

    order_no_repeats = build(TOK, 32, loop_middle_repeats=None, **TINY).model._get_layer_order()
    assert order_no_repeats == [0, 1, 2] + [3, 4, 5] * 4 + [6, 7, 8]


def test_tied_embeddings_and_no_padding_idx():
    model = build(TOK, 32, **TINY)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    assert model.model.embed_tokens.padding_idx is None


def test_forward_backward_loss_matches_fp32_reference():
    model = build(TOK, 32, layers=3, loop_middle_layers=1, loop_middle_repeats=2, **TINY).to(DEVICE)
    input_ids = torch.randint(0, len(TOK), (2, 32), device=DEVICE)

    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        out = model(input_ids=input_ids, labels=input_ids)
    assert out.logits is None
    assert torch.isfinite(out.loss)
    out.loss.backward()

    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        logits = model(input_ids=input_ids).logits
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), input_ids[:, 1:].reshape(-1))
    assert torch.isclose(out.loss.float(), ref, rtol=1e-3)


def test_plain_loop_path_forward_backward():
    model = build(TOK, 32, enable_double_loop_split=False, num_loops=2, **TINY).to(DEVICE)
    input_ids = torch.randint(0, len(TOK), (2, 32), device=DEVICE)
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        loss = model(input_ids=input_ids, labels=input_ids).loss
    assert torch.isfinite(loss)
    loss.backward()
