"""Checkpoint round-trip smoke tests: save_pretrained -> from_pretrained must reproduce the model exactly."""

import pytest
import torch
from transformers import AutoTokenizer

from mzoo.archs.nanbeige.model import build
from mzoo.archs.nanbeige.modeling_nanbeige import NanbeigeForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOK = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
TINY = dict(layers=6, loop_middle_layers=2, loop_middle_repeats=2, hidden=64, heads=4, kv_heads=2, ffn=128)
MOE = dict(moe=True, n_routed_experts=4, dense_layers=1, share_shared_expert=["middle", "tail"])

CASES = {"dense": {}, "moe": MOE}


@pytest.mark.parametrize("case", CASES)
def test_save_and_load_roundtrip(tmp_path, case):
    model = build(TOK, 32, **TINY, **CASES[case]).to(DEVICE).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            if hasattr(layer.mlp, "gate"):  # moe layers only
                layer.mlp.gate.e_score_correction_bias.normal_(0, 0.01)
        input_ids = torch.randint(0, len(TOK), (2, 32), device=DEVICE)
        ref_logits = model(input_ids=input_ids).logits

    model.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").exists()

    loaded = NanbeigeForCausalLM.from_pretrained(tmp_path).to(DEVICE).eval()
    with torch.no_grad():
        out_logits = loaded(input_ids=input_ids).logits
    assert torch.allclose(ref_logits, out_logits, atol=1e-6, rtol=0)

    sd1 = {**dict(model.named_parameters(remove_duplicate=False)), **dict(model.named_buffers(remove_duplicate=False))}
    sd2 = {**dict(loaded.named_parameters(remove_duplicate=False)), **dict(loaded.named_buffers(remove_duplicate=False))}
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), k

    assert loaded.lm_head.weight.data_ptr() == loaded.model.embed_tokens.weight.data_ptr()
    if case == "moe":
        L = loaded.model.layers
        assert L[2].mlp.shared_experts is L[3].mlp.shared_experts  # "middle" group tied
        assert L[4].mlp.shared_experts is L[5].mlp.shared_experts  # "tail" group tied
