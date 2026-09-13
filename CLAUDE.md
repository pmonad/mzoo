- Do not do more than asked, but offer user ideas for next steps
- Do not go chasing bugs without user approval, propose strategies and then proceed. Generally bugs will be on our side and very simple changes on our end are prefered
- avoid writing large chunks of code. also deliberate what libraries can simplify task at hand
- prefer short files. when they grow offer user to convert split it. add a new pkg when it makes sense
- keep tests next to code _test.py


standards
- fire for simple cli and huggingface args large list. 
- reuse existing huggingface ecosystem as much as possible
- prefer gpu for torch tests
- always use `uv run --env-file .env` (machine-specific env like CUDA_HOME lives in gitignored repo-root .env)
- source builds: clone into ~/git/<org>/<proj>, checkout the required tag, install editable
- use justfiles(minimal cmd) and define at source folder. but they will always be invoked from repo root `just path/to/folder <cmd>` so assume that to be cwd and adjust justfile accordingly

goal:
Experiment various model architectures

data layout: /data/pmonad/mzoo/000N-<proj>/yymmdd-000N-<experiment>/000N-<run>/...

