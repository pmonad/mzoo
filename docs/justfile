# invoke from repo root: just docs/ <recipe>

[no-cd]
build:
    uv run --with mkdocs --with pymdown-extensions mkdocs build --strict --site-dir site

[no-cd]
serve *args:
    uv run --with mkdocs --with pymdown-extensions mkdocs serve {{args}}

[no-cd]
publish:
    uv run --with mkdocs --with pymdown-extensions mkdocs gh-deploy --force
