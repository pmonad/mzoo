"""GPU tests for the shared golden torch reference (``golden_ref.py``).

Levels are checked bottom-up (dense -> latent -> window -> compressed/sparse), then
the whole thing is checked end-to-end against the vendored DSV4.1 model: a tiny
2-layer config (layer 0 pure sliding-window, layer 1 window + compressed + sparse
top-k) is run once in fp32, ``eager_attention_forward``'s exact inputs are captured
per layer via a monkeypatch, and ``golden`` is checked against that captured
ground truth -- the only real test that the level semantics match the model, not
just each other.

Indexer helpers: only a unit test on ``topk_indices``' -1 dummy-slot behavior is
included (not checked against the model's indexer), because capturing the
indexer's internal q/k/weights would need patching code inside a method body
rather than a swappable module-level function.
"""

import pytest
import torch

from mzoo.archs.dsv4 import modeling_deepseek_v41 as modeling
from mzoo.archs.dsv4.configuration_deepseek_v41 import DeepseekV41TextConfig
from mzoo.archs.dsv4.modeling_deepseek_v41 import DeepseekV41ForCausalLM
from mzoo.layers.attn.dense_attn.ref import sdpa_ref, sdpa_ref_lse
from mzoo.layers.attn.golden_ref import golden, gpt_oss_ref, topk_indices

DEVICE = "cuda"
TOL = dict(atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------------------
# Level-by-level checks
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("use_sinks", [False, True])
def test_dense_matches_sdpa_ref(causal, use_sinks):
    torch.manual_seed(0)
    b, s, h, d = 2, 37, 4, 32
    q, k, v = (torch.randn(b, s, h, d, device=DEVICE) for _ in range(3))
    sinks = torch.randn(h, device=DEVICE) if use_sinks else None
    o, lse = golden(q, k, level="dense", causal=causal, sinks=sinks, v=v)
    torch.testing.assert_close(o, sdpa_ref(q, k, v, causal, sinks), **TOL)
    torch.testing.assert_close(lse, sdpa_ref_lse(q, k, v, causal, sinks), **TOL)


def test_latent_matches_dense_broadcast():
    torch.manual_seed(1)
    b, s, h, d = 2, 29, 4, 16
    q = torch.randn(b, s, h, d, device=DEVICE)
    kv = torch.randn(b, s, 1, d, device=DEVICE)
    o, lse = golden(q, kv, level="latent", causal=True)
    kv_full = kv.expand(-1, -1, h, -1)
    torch.testing.assert_close(o, sdpa_ref(q, kv_full, kv_full, True), **TOL)
    torch.testing.assert_close(lse, sdpa_ref_lse(q, kv_full, kv_full, True), **TOL)


def test_window_ge_seqlen_matches_latent():
    torch.manual_seed(2)
    b, s, h, d = 2, 20, 4, 16
    q = torch.randn(b, s, h, d, device=DEVICE)
    kv = torch.randn(b, s, 1, d, device=DEVICE)
    o_w, lse_w = golden(q, kv, level="window", window=s, causal=True)
    o_l, lse_l = golden(q, kv, level="latent", causal=True)
    torch.testing.assert_close(o_w, o_l, **TOL)
    torch.testing.assert_close(lse_w, lse_l, **TOL)


def test_window_one_is_self_only():
    """Empirically verified off-by-one (see module docstring in golden_ref.py):
    ``window`` counts the query token itself, so ``window=1`` means "attend only to
    self" -- softmax over a single key collapses to weight 1, so the output must
    equal that one V exactly, for every query."""
    torch.manual_seed(3)
    b, s, h, d = 1, 6, 2, 8
    q = torch.randn(b, s, h, d, device=DEVICE)
    kv = torch.randn(b, s, 1, d, device=DEVICE)
    o, _ = golden(q, kv, level="window", window=1, causal=True)
    torch.testing.assert_close(o, kv.expand(-1, -1, h, -1), **TOL)


def test_compressed_matches_manual_duplicate_stream():
    """Simple closed-form check of the compressed-level KV concatenation + softmax,
    independent of ``golden``'s own code: with ``compress_ratio=1`` and a full
    window, the main-branch group-causal condition (``g < t+1``) is identical to the
    window's token-causal condition, so feeding the SAME latent stream as both the
    window and the main KV is exactly "attend twice to an identical causal stream" --
    a closed form we can hand-derive without reusing golden's internals."""
    torch.manual_seed(4)
    b, s, h, d = 2, 16, 4, 8
    q = torch.randn(b, s, h, d, device=DEVICE)
    kv = torch.randn(b, s, 1, d, device=DEVICE)
    o, lse = golden(q, kv, level="compressed", window=s, main_kv=kv, compress_ratio=1)

    qt = q.transpose(1, 2)  # [B,H,S,D]
    kt = kv.transpose(1, 2)  # [B,1,S,D]
    logits = torch.matmul(qt, kt.transpose(-1, -2)) * d**-0.5
    causal = torch.triu(torch.ones(s, s, dtype=torch.bool, device=DEVICE), 1)
    logits = logits.masked_fill(causal, float("-inf"))
    logits_cat = torch.cat([logits, logits], dim=-1)  # main branch is the same stream
    lse_ref = torch.logsumexp(logits_cat, dim=-1)
    probs = torch.softmax(logits_cat, dim=-1)
    o_ref = (torch.matmul(probs[..., :s], kt) + torch.matmul(probs[..., s:], kt)).transpose(1, 2)

    torch.testing.assert_close(o, o_ref, **TOL)
    torch.testing.assert_close(lse, lse_ref, **TOL)


def test_sparse_with_all_visible_indices_matches_compressed():
    """Consistency check (golden vs golden, not an independent reference): if
    ``indices`` lists exactly the group-causally-visible entries, sparse must
    reduce to compressed."""
    torch.manual_seed(5)
    b, s, h, d, ratio = 2, 10, 4, 8, 2
    g = s // ratio
    q = torch.randn(b, s, h, d, device=DEVICE)
    kv = torch.randn(b, s, 1, d, device=DEVICE)
    main_kv = torch.randn(b, g, 1, d, device=DEVICE)
    o_c, lse_c = golden(q, kv, level="compressed", window=s, main_kv=main_kv, compress_ratio=ratio)

    t_idx = torch.arange(s, device=DEVICE).view(1, -1, 1)
    g_idx = torch.arange(g, device=DEVICE).view(1, 1, -1)
    visible = g_idx < ((t_idx + 1) // ratio)
    indices = g_idx.expand(b, s, g).clone()
    indices[~visible.expand(b, s, g)] = -1
    o_s, lse_s = golden(q, kv, level="sparse", window=s, main_kv=main_kv, compress_ratio=ratio, indices=indices)

    torch.testing.assert_close(o_s, o_c, **TOL)
    torch.testing.assert_close(lse_s, lse_c, **TOL)


@pytest.mark.parametrize("level,bad_kwargs", [
    ("dense", {}),                                                   # missing v
    ("dense", dict(v=None, window=1)),                                # extra window
    ("latent", dict(v=torch.zeros(1))),                               # v must be None
    ("window", {}),                                                   # missing window
    ("compressed", dict(window=1)),                                   # missing main_kv/compress_ratio
    ("sparse", dict(window=1, main_kv=torch.zeros(1), compress_ratio=1)),  # missing indices
])
def test_level_kwarg_switch_is_enforced(level, bad_kwargs):
    q = torch.randn(1, 2, 1, 4, device=DEVICE)
    kv = torch.randn(1, 2, 1, 4, device=DEVICE)
    with pytest.raises(ValueError):
        golden(q, kv, level=level, **bad_kwargs)


def test_topk_indices_dummy_slot():
    scores = torch.tensor([[[1.0, 2.0, float("-inf"), float("-inf")]]], device=DEVICE)  # [B=1,S=1,G=4]
    idx = topk_indices(scores, topk=3)
    assert idx.shape == (1, 1, 3)
    assert set(idx[0, 0, :2].tolist()) == {0, 1}  # the two finite scores, order not asserted
    assert idx[0, 0, 2].item() == -1  # third pick had -inf score -> dummy slot

    padded = topk_indices(scores[..., :2], topk=4)  # topk > G: pads past the end too
    assert padded.shape == (1, 1, 4)
    assert set(padded[0, 0, :2].tolist()) == {0, 1}
    assert padded[0, 0, 2:].tolist() == [-1, -1]


# --------------------------------------------------------------------------------------
# gpt_oss_ref: implementation-independent cross-check (real transformers eager path)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(2, 256, 4, 64), (1, 128, 64, 128)])
@pytest.mark.parametrize("use_sinks", [False, True])
@pytest.mark.parametrize("causal", [True, False])
def test_gpt_oss_ref_matches_golden_latent(shape, use_sinks, causal):
    torch.manual_seed(7)
    b, s, h, d = shape
    q = torch.randn(b, s, h, d, device=DEVICE)
    kv = torch.randn(b, s, 1, d, device=DEVICE)
    sinks = torch.randn(h, device=DEVICE) if use_sinks else None
    o_golden, _ = golden(q, kv, level="latent", causal=causal, sinks=sinks)
    o_gpt_oss = gpt_oss_ref(q, kv, causal=causal, sinks=sinks)
    torch.testing.assert_close(o_gpt_oss, o_golden, **TOL)


def test_gpt_oss_ref_window_matches_golden_window():
    torch.manual_seed(8)
    b, s, h, d = 2, 256, 4, 64
    q = torch.randn(b, s, h, d, device=DEVICE)
    kv = torch.randn(b, s, 1, d, device=DEVICE)
    o_golden, _ = golden(q, kv, level="window", window=32, causal=True)
    o_gpt_oss = gpt_oss_ref(q, kv, window=32, causal=True)
    torch.testing.assert_close(o_gpt_oss, o_golden, **TOL)


def test_gpt_oss_ref_bf16_runs():
    """Not a numerical match to golden's bf16 path (gpt-oss softmaxes in the logits
    dtype -- bf16 here -- whereas golden always keeps the softmax in fp32 and only
    casts probs to bf16 for the second matmul), just proof the bf16 path executes and
    is in the right ballpark."""
    torch.manual_seed(9)
    b, s, h, d = 2, 64, 4, 32
    q = torch.randn(b, s, h, d, device=DEVICE)
    kv = torch.randn(b, s, 1, d, device=DEVICE)
    sinks = torch.randn(h, device=DEVICE)
    o_golden, _ = golden(q, kv, level="latent", causal=True, sinks=sinks, dtype=torch.bfloat16)
    o_gpt_oss = gpt_oss_ref(q, kv, causal=True, sinks=sinks, dtype=torch.bfloat16)
    assert o_gpt_oss.dtype == torch.bfloat16
    torch.testing.assert_close(o_gpt_oss, o_golden, atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------------------
# End-to-end vs the vendored model
# --------------------------------------------------------------------------------------


def _build_tiny_model():
    """2 layers: layer 0 pure sliding-window (compress_ratio 0), layer 1 owns a
    compress_ratio=2 branch and is its own kv/index source -- one self-contained
    group, so there is no cross-group interaction through the per-forward ``shared``
    dict to worry about. ``S=160 > sliding_window=128`` exercises window truncation;
    ``compressed_len = S // 2 = 80 > index_topk = 64`` exercises real top-k
    sparsity (some group-causally-visible entries get dropped). Hierarchical
    candidates disabled (``candidate_source_layer_id=-1``) and fp4/fp8 fake-quant of
    kv is a model detail baked into whatever ``eager_attention_forward`` receives --
    golden doesn't need to know about it, per the module docstring.

    Built from ``DeepseekV41TextConfig`` directly (not ``archs.dsv4.model.build``,
    which needs a tokenizer only to set ``vocab_size``) with the same heads=4,
    head_dim=64 etc as that module's smoke config.
    """
    config = DeepseekV41TextConfig(
        vocab_size=64, hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, head_dim=64, qk_rope_head_dim=16,
        q_lora_rank=128, o_lora_rank=128, o_groups=2,
        compress_ratios=[0, 2], kv_source_layer_ids=[1], index_source_layer_ids=[1],
        candidate_source_layer_id=-1, sliding_window=128,
        index_n_heads=4, index_head_dim=32, index_topk=64,
        moe_intermediate_size=256, n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
        num_nextn_predict_layers=0, use_cache=False,
        dspark_noise_token_id=0, tie_word_embeddings=True, bos_token_id=0, eos_token_id=1,
        max_position_embeddings=4096,
    )
    model = DeepseekV41ForCausalLM(config).to(DEVICE, dtype=torch.float32).eval()
    model.config._attn_implementation = "eager"  # not pre-registered -> falls back to our patched default
    return model


def _capture_attention_calls(model, input_ids):
    """Run one prefill forward, recording the exact (query, key, value,
    attention_mask, scaling, attn_sink) each attention layer hands to
    ``eager_attention_forward`` -- the compressed layer's ``attention_mask`` tail is
    exactly ``shared["topk_bias"]`` (the model concatenates it in verbatim, see
    ``DeepseekV41Attention.forward``), so no separate capture of ``shared`` itself
    is needed."""
    captured = {}
    original = modeling.eager_attention_forward

    def spy(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        captured[module.layer_idx] = dict(
            query=query.detach().clone(), key=key.detach().clone(), value=value.detach().clone(),
            attention_mask=None if attention_mask is None else attention_mask.detach().clone(),
            scaling=scaling, attn_sink=module.attn_sink.detach().clone(),
        )
        return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

    modeling.eager_attention_forward = spy
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        modeling.eager_attention_forward = original
    return captured


def _eager_output_and_lse(query, key, value, attention_mask, scaling, attn_sink):
    """Recompute eager_attention_forward's output and the logsumexp of its combined
    (logits + sink) column straight from the captured inputs, per the task's ask to
    verify lse against "logsumexp of eager's combined logits"."""
    o_ref, _ = modeling.eager_attention_forward(
        type("M", (), {"attn_sink": attn_sink, "training": False})(),
        query, key, value, attention_mask, scaling,
    )
    logits = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        logits = logits + attention_mask
    sink_col = attn_sink.float().reshape(1, -1, 1, 1).expand(logits.shape[0], -1, logits.shape[2], 1)
    lse_ref = torch.logsumexp(torch.cat([logits.float(), sink_col], dim=-1), dim=-1)
    return o_ref, lse_ref


def _mask_to_indices(main_bias, topk):
    """Test-only inverse of golden's sparse-level mask construction: turn a
    0/-inf additive bias [B,S,G] (the captured ``topk_bias``) back into golden's
    ``indices`` [B,S,topk] int64 (-1 padding), by definition of what "listed in the
    token's indices" means."""
    b, s, g = main_bias.shape
    allowed = main_bias == 0
    assert allowed.sum(-1).max() <= topk, "more allowed entries than index_topk -- config assumption broke"
    order = torch.argsort((~allowed).to(torch.uint8), dim=-1, stable=True)  # allowed entries first
    idx_full = torch.arange(g, device=main_bias.device).view(1, 1, g).expand(b, s, g)
    picked = torch.gather(idx_full, -1, order)[..., :topk]
    picked_valid = torch.gather(allowed, -1, order)[..., :topk]
    return torch.where(picked_valid, picked, torch.full_like(picked, -1))


def test_golden_matches_model_window_and_sparse_layers():
    torch.manual_seed(6)
    model = _build_tiny_model()
    b, s = 2, 160
    input_ids = torch.randint(0, 64, (b, s), device=DEVICE)
    captured = _capture_attention_calls(model, input_ids)
    assert set(captured) == {0, 1}

    # --- layer 0: pure sliding window ---------------------------------------------
    c0 = captured[0]
    q0 = c0["query"].transpose(1, 2)  # [B,S,H,D]
    kv0 = c0["key"].transpose(1, 2)  # [B,S,1,D]
    o0_ref, lse0_ref = _eager_output_and_lse(c0["query"], c0["key"], c0["value"], c0["attention_mask"],
                                              c0["scaling"], c0["attn_sink"])
    o0, lse0 = golden(q0, kv0, level="window", window=128, sinks=c0["attn_sink"], causal=True)
    torch.testing.assert_close(o0, o0_ref, **TOL)
    torch.testing.assert_close(lse0, lse0_ref, **TOL)

    # --- layer 1: window + compressed + sparse top-k --------------------------------
    c1 = captured[1]
    key1, mask1 = c1["key"], c1["attention_mask"]
    assert key1.shape[2] == s + s // 2  # window (S raw) + compressed (S//2 groups, ratio 2)
    g = key1.shape[2] - s
    assert g > 64  # index_topk=64 -- real sparsity, not "everything visible"
    q1 = c1["query"].transpose(1, 2)
    win_kv1 = key1[:, :, :s].transpose(1, 2)
    main_kv1 = key1[:, :, s:].transpose(1, 2)
    main_bias = mask1[..., s:].squeeze(1)  # == shared["topk_bias"], concatenated verbatim by the model
    indices = _mask_to_indices(main_bias, topk=64)

    o1_ref, lse1_ref = _eager_output_and_lse(c1["query"], c1["key"], c1["value"], c1["attention_mask"],
                                              c1["scaling"], c1["attn_sink"])
    o1, lse1 = golden(q1, win_kv1, level="sparse", window=128, main_kv=main_kv1, compress_ratio=2,
                       indices=indices, sinks=c1["attn_sink"], causal=True)
    torch.testing.assert_close(o1, o1_ref, **TOL)
    torch.testing.assert_close(lse1, lse1_ref, **TOL)
