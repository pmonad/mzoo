# __init__.py

Package entry for owlet1, a split-into-small-files fork of the frozen `dsv4` baseline.
Apache-2.0 header (HuggingFace, transformers PR #48721); class names stay `DeepseekV41*`
so diffs against `src/mzoo/archs/dsv4/` remain clean.

## Re-exports (`__all__`)

| name | from |
|---|---|
| `build` | `model` — harness entry point |
| `DeepseekV41Config`, `DeepseekV41TextConfig` | `config` |
| `DeepseekV41ForCausalLM`, `DeepseekV41TextModel` | `decoder` |
| `DeepseekV41CSACache` | `cache` |
| `DeepseekV41EngramEmbedding`, `DeepseekV41NgramHashState`, `EngramLayout` | `engram` |
