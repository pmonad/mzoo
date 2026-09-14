# invoke from repo root: just <recipe>

# GPU smoke suites; more sub-tasks get appended here as they exist
smoke:
    just src/mzoo/layers/attn/ test -q
    just src/mzoo/kernels/ test -q
