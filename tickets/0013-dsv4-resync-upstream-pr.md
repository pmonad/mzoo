# 0013 dsv4: re-sync the vendored modeling file with HF PR #48721 once it merges

- status: todo (blocked on upstream: PR open, last push 2026-09-13)
- depends on: nothing; consumers are `golden_ref.py`, the `model_test.py` files under
  `layers/attn/{swa,csa,csa2}_attn` and `archs/dsv4`, and `archs/dsv4/model.py`
- location: `src/mzoo/archs/dsv4/modeling_deepseek_v41.py` (+ `configuration_deepseek_v41.py`)

## Why

Vendored at PR commit `f026c83`; PR head `62d7ebd` is 22 commits ahead (checked 2026-09-14).
It is not a drop-in: upstream moved compressed attention from "concat compressed KV onto the KV
axis + additive `topk_bias` mask" to an explicit gather (`shared["topk_idx"]`, `selected_kv` /
`selected_valid` into `eager_attention_forward`), the indexer publishes `topk_idx`, scoring is
chunked with a padding mask, and engram history moved into the cache layer. Our golden reference
and four model-replay tests capture the mask form.

## Known state of our concerns at `62d7ebd`

- cross-group `shared` accumulation (ticket 0012): fixed upstream (`754bc79`); our backport is
  equivalent. Drop the `mzoo:` backport block on re-sync.
- indexer gets no gradient: NOT fixed upstream (the fp4 quantizer's code/LUT ops cut the graph;
  there is no explicit detach to remove). Our STE in `layers/attn/indexer/attn.py` stays.
- `DeepseekV41CSACache.__init__(config, ...)` positional: unchanged upstream; `use_cache=True`
  still breaks on transformers 5.17 here.

## Do

1. Wait for merge (or a stable head). Copy the file, reapply the `mzoo:` import edits, drop the
   0012 backport block, keep the header provenance line updated to the new commit.
2. Re-run `archs/dsv4/model_test.py`, `layers/attn/golden_ref_test.py` and the three
   `model_test.py` replays; adapt the capture helpers to the gather form (the mask tail is no
   longer `topk_bias`; take `topk_idx` directly, which is closer to the kernels' contract).
3. Re-check the README gotchas table (cache ctor, `num_hidden_layers` derivation, dense-FFN
   absence) and the "no gradient" section against the new file.
4. Re-vendor `configuration_deepseek_v41.py` alongside; diff `__post_init__` derivations.

## Notes for future sessions

- PR watch: `gh pr view 48721 --repo huggingface/transformers --json state,headRefOid,updatedAt`.
- Dependent PR #48768 (vision) builds on this one; not a replacement.
