# Split out of the frozen dsv4 baseline `src/mzoo/archs/dsv4/modeling_deepseek_v41.py`
# (see there for upstream provenance + Apache-2.0 license). Covers: candidate-block selection and the lightning indexer (sparse attention scoring).

import torch
import torch.nn.functional as F
from torch import nn

from .cache import DeepseekV41CSACache
from .config import DeepseekV41TextConfig
from .norm_rope import DeepseekV41RMSNorm, DeepseekV41RotaryEmbedding, apply_rotary_pos_emb
from .quant import _fake_quant_fp4_block


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k (hierarchical sparse indexer): keep the
    `topk_blocks` highest-scoring blocks of `block_size` compressed positions per query.

    `logits` is `[..., n_positions]` with unreachable positions already at -inf, which
    makes a block score of -inf mean "not reachable yet". The block holding the query's
    newest position is only partly filled, so it is pinned in — it holds the most recent
    tokens but could otherwise be outscored by an older, full block. Returns a bool mask
    shaped like `logits`."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=float("-inf"))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    # `compress_lens` carries a trailing axis: [.., 1] against the [.., blocks] scores.
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device).view(-1) == last, torch.inf)

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    # Fewer reachable blocks than topk_blocks: leftover picks come back -inf — drop them.
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class DeepseekV41Indexer(nn.Module):
    r"""Sparse indexer of a CSA2 **index-source** layer. Scores each query against the
    shared indexer keys (one per compressed group, at `index_head_dim`) and keeps the
    top `index_topk` groups per query; the resulting per-query block bias is published
    to the group's other layers ("Reuse" mode).

    Key sharing: the keys are `k_norm(wk(latent))` of the *compressor latent* — only a
    layer that also owns its compressor (`kv_source_layer_ids`) can produce them
    (`owns_k`); every later index source ("Reindex" mode) rescores with its own weights
    against the keys published by its group's source. Queries come from the attention's
    low-rank residual (`q_norm(wq_a(x))`) through `wq_b`, rotated with the compress
    rope. The candidate source layer additionally publishes the two-level-top-k
    candidate mask that constrains all later index sources."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.owns_k = layer_idx in config.kv_source_layer_ids
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.is_candidate_source = layer_idx == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_idx
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.heads_scaling = self.num_heads**-0.5
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.rotary_emb = DeepseekV41RotaryEmbedding(config)
        if self.owns_k:
            self.wk = nn.Linear(config.head_dim, self.head_dim, bias=False)
            self.k_norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        latent: torch.Tensor | None,
        first_group_position: int,
        position_ids: torch.Tensor,
        cache_layer: DeepseekV41CSACache | None,
        shared: dict,
    ) -> None:
        """Publishes `shared["topk_bias"]` (the per-query block bias over the shared
        compressed KV), and `shared["candidates"]` at the candidate source layer."""
        batch, seq_len, _ = hidden_states.shape
        ratio = self.compress_ratio

        # 1. Publish the index keys of the groups that completed in this call. The keys
        #    are derived from the PRE-rope latent, before the compressor rotates the
        #    same values into the main cache.
        if self.owns_k:
            if latent is not None:
                k = self.k_norm(self.wk(latent))
                positions = first_group_position + ratio * torch.arange(latent.shape[1], device=k.device)
                cos, sin = self.rotary_emb(
                    k, position_ids=positions.unsqueeze(0).expand(batch, -1), layer_type="compress"
                )
                k = apply_rotary_pos_emb(k, cos, sin)
                # QAT semantics: indexer keys are FP4-quantized (ue8m0 scale per 32
                # channels) before they land in the shared key cache.
                k = _fake_quant_fp4_block(k, block_size=32)
                k = k.unsqueeze(1)  # [B, 1, G, idh]
                if cache_layer is not None:
                    cache_layer.update_compressor_states("indexer", k)
                else:
                    shared["index_k"] = (
                        k if shared.get("index_k") is None else torch.cat([shared["index_k"], k], dim=2)
                    )
            # Publish the RUNNING key cache — decode steps between group boundaries emit
            # nothing new but the group still scores against everything emitted so far.
            if cache_layer is not None:
                shared["index_k"] = cache_layer.compressed_kv["indexer"]
        index_k = shared.get("index_k")
        compressed_len = 0 if index_k is None else index_k.shape[2]
        if compressed_len == 0:
            return

        # 2. Score the queries against the shared keys.
        cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type="compress")
        q = self.wq_b(q_residual).view(batch, seq_len, self.num_heads, self.head_dim)
        q = apply_rotary_pos_emb(q, cos_q, sin_q)
        # QAT semantics: the indexer query is FP4-quantized too, so the top-k
        # selection matches the trained quantized scoring.
        q = _fake_quant_fp4_block(q, block_size=32)
        scores = torch.einsum("bshd,btd->bsht", q.float(), index_k[:, 0].float())
        scores = scores.relu_() * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.heads_scaling
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]

        # 3. Visibility: a group becomes visible once the query passed its last token —
        # in ABSOLUTE positions, so a chunk that starts mid-sequence sees exactly the
        # groups it could see in a one-shot prefill. `compress_lens` keeps a trailing
        # axis so every broadcast below is explicit: `masked_fill` expands its result
        # to the broadcast shape, and a sloppy `[T] >= [B, S]` mask would silently
        # grow the score tensor.
        entry_indices = torch.arange(compressed_len, device=index_scores.device).view(1, 1, -1)
        compress_lens = (position_ids.long().unsqueeze(-1) + 1) // ratio  # [B, S, 1]
        index_scores = index_scores.masked_fill(entry_indices >= compress_lens, float("-inf"))

        # 4. Two-level top-k: the candidate source publishes its block mask; every
        #    later index source scores only inside it.
        if self.is_candidate_source:
            shared["candidates"] = select_candidate_blocks(
                index_scores, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates and shared.get("candidates") is not None:
            index_scores = index_scores.masked_fill(~shared["candidates"], float("-inf"))

        # 5. Top-k per query. Early queries can have fewer visible groups than
        #    `index_topk`, so some picks come back with a -inf score; clamp those into
        #    the dummy slot past the end (dropped by the slice) — scattering them at
        #    their raw index would leak future groups into the attention.
        top_k = min(self.index_topk, compressed_len)
        block_bias = index_scores.new_full((batch, 1, seq_len, compressed_len + 1), float("-inf"))
        if top_k > 0:
            topk = index_scores.topk(top_k, dim=-1, sorted=False)
            valid = topk.values > float("-inf")
            safe = torch.where(valid, topk.indices, torch.full_like(topk.indices, compressed_len))
            block_bias.scatter_(-1, safe.unsqueeze(1), 0.0)
        shared["topk_bias"] = block_bias[..., :compressed_len]
