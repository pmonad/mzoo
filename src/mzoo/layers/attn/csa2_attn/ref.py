"""csa2_attn: the torch side of the gathered compressed source.

There is deliberately **no numeric reference here**: the reference is
``../golden_ref.py::golden(level="sparse", window=W, main_kv=..., compress_ratio=m,
indices=...)`` fed the *same* index list as the kernel,
verified against the vendored model in ``../golden_ref_test.py``, and
``../dense_attn/ref.py::assert_within_2x_torch`` is the acceptance criterion. This
module holds only what is genuinely csa-specific and torch-side:

**Everything the compressed source needs before the kernel stays in torch** (ticket
0003 v1, unchanged here): ``DeepseekV41Compressor`` (the fp32 gated softmax pool, or
the plain per-token projection at ratio 1), the latent-position RoPE at
``first_group_position + ratio * k``, and ``_fake_quant_fp4_block(x, block_size=16,
e4m3_scales=True)``. The kernel consumes the already-dequantized bf16 ``main_kv
[B, G, 1, D]``. Moving the dequant in-kernel is ticket 0010; see
``../csa_attn/README.md`` -> "Main-cache layout" for the byte layout.

**The index list has no torch side here either.** ``indices [B, S, topk]`` is produced
by ``indexer.attn`` (ticket 0006) or, in a model replay, recovered from the model's own
``shared["topk_bias"]`` with ``golden_ref_test._mask_to_indices``; both already speak
the ``-1`` = empty-slot contract the kernel and ``golden`` share.

The one helper below extracts ``(kv, main_kv)`` from a captured model attention call,
because the model hands ``eager_attention_forward`` the two sources already
concatenated on the KV axis (``kv = torch.cat([kv, compressed_kv], dim=2)``).
"""

import torch


def split_captured_kv(key: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a captured ``key [B, 1, S + G, D]`` into ``(kv [B, S, 1, D], main_kv [B, G, 1, D])``.

    ``DeepseekV41Attention.forward`` concatenates the compressed entries onto the KV
    axis *after* the raw window latents, so the first ``seq_len`` entries are the raw
    (fp8 fake-quantized) window cache and the rest are the ``ratio``-pooled,
    latent-rotated, fp4 fake-quantized compressed latents -- in group order, entry
    ``g`` at index ``g``. Both come back in the kernel's BSHD-with-one-KV-head layout.
    """
    assert key.shape[1] == 1, f"expected a shared K == V latent [B, 1, T, D], got {tuple(key.shape)}"
    assert key.shape[2] >= seq_len, f"captured KV axis {key.shape[2]} is shorter than seq_len {seq_len}"
    return key[:, :, :seq_len].transpose(1, 2), key[:, :, seq_len:].transpose(1, 2)


def to_kernel_inputs(capture: dict, seq_len: int, dtype=torch.bfloat16) -> tuple:
    """A captured ``eager_attention_forward`` call -> ``(q, kv, main_kv, sinks)`` for ``attn``."""
    kv, main_kv = split_captured_kv(capture["key"], seq_len)
    q = capture["query"].transpose(1, 2)  # [B, H, S, D] -> [B, S, H, D]
    return (q.to(dtype).contiguous(), kv.to(dtype).contiguous(), main_kv.to(dtype).contiguous(),
            capture["attn_sink"].float())
