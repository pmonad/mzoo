"""Smoke tests for the DeepSeek-V4 MoE swap-in (tiny sizes, no full-size models)."""

import torch
from transformers import AutoTokenizer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock

from mzoo.archs.nanbeige.model import build
from mzoo.archs.nanbeige.modeling_nanbeige import NanbeigeMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOK = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
TINY = dict(hidden=64, heads=4, kv_heads=2, ffn=128, layers=6, loop_middle_layers=2, loop_middle_repeats=2)
MOE = dict(moe=True, n_routed_experts=4, num_experts_per_tok=2)


def mlp_types(model):
    return [type(layer.mlp) for layer in model.model.layers]


def test_dense_then_moe_layers_split_on_dense_layers():
    types = mlp_types(build(TOK, 32, **TINY, **MOE))
    assert types[:2] == [NanbeigeMLP, NanbeigeMLP]
    assert types[2:] == [DeepseekV4SparseMoeBlock] * 4

    types3 = mlp_types(build(TOK, 32, **TINY, **MOE, dense_layers=3))
    assert types3[:3] == [NanbeigeMLP] * 3
    assert types3[3:] == [DeepseekV4SparseMoeBlock] * 3

    types_dense = mlp_types(build(TOK, 32, **TINY, moe=False))
    assert types_dense == [NanbeigeMLP] * 6


def test_expert_and_router_params_initialized():
    moe_layer = build(TOK, 32, **TINY, **MOE).model.layers[2].mlp
    for p in (moe_layer.gate.weight, moe_layer.experts.gate_up_proj, moe_layer.experts.down_proj):
        assert torch.isfinite(p).all()
        assert 0.01 < p.std().item() < 0.04


def test_train_forward_backward_updates_router_bias_and_load_counts():
    model = build(TOK, 32, **TINY, **MOE).to(DEVICE)
    moe_layer = model.model.layers[2].mlp
    input_ids = torch.randint(0, len(TOK), (2, 32), device=DEVICE)

    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        loss = model(input_ids=input_ids, labels=input_ids).loss
    assert torch.isfinite(loss)
    loss.backward()

    assert moe_layer.gate.weight.grad.abs().sum() > 0
    assert moe_layer.experts.gate_up_proj.grad.abs().sum() > 0
    assert moe_layer.experts.down_proj.grad.abs().sum() > 0
    assert moe_layer.gate.load_counts.sum() > 0
    assert not torch.equal(moe_layer.gate.e_score_correction_bias, torch.zeros_like(moe_layer.gate.e_score_correction_bias))


def test_share_shared_expert_within_loopsplit_groups():
    # head=[0,1], middle=[2,3], tail=[4,5]; dense_layers=1 -> layer 0 dense, 1-5 MoE.
    layers = build(TOK, 32, **TINY, **MOE, dense_layers=1).model.layers

    assert layers[2].mlp.shared_experts is layers[3].mlp.shared_experts  # default: share within "middle"
    assert layers[2].mlp.gate is not layers[3].mlp.gate
    assert layers[2].mlp.experts is not layers[3].mlp.experts
    assert layers[4].mlp.shared_experts is not layers[5].mlp.shared_experts  # tail: no sharing by default
    assert layers[1].mlp.shared_experts is not layers[2].mlp.shared_experts  # head vs middle: no sharing

    no_share = build(TOK, 32, **TINY, **MOE, dense_layers=1, share_shared_expert=[]).model.layers
    assert no_share[2].mlp.shared_experts is not no_share[3].mlp.shared_experts

    all_share = build(
        TOK, 32, **TINY, **MOE, dense_layers=1, share_shared_expert=["head", "middle", "tail"]
    ).model.layers
    assert all_share[4].mlp.shared_experts is all_share[5].mlp.shared_experts

    shared_model = build(TOK, 32, **TINY, **MOE, dense_layers=1)
    unshared_model = build(TOK, 32, **TINY, **MOE, dense_layers=1, share_shared_expert=[])
    assert shared_model.num_parameters() < unshared_model.num_parameters()


def test_eval_mode_does_not_update_bias():
    model = build(TOK, 32, **TINY, **MOE).to(DEVICE)
    model.eval()
    moe_layer = model.model.layers[2].mlp
    input_ids = torch.randint(0, len(TOK), (2, 32), device=DEVICE)

    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        model(input_ids=input_ids)
    assert torch.equal(moe_layer.gate.e_score_correction_bias, torch.zeros_like(moe_layer.gate.e_score_correction_bias))
