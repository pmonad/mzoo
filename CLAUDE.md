- Do not do more than asked, but offer user ideas for next steps
- Do not go chasing bugs without user approval, propose strategies and then proceed. Generally bugs will be on our side and very simple changes on our end are prefered
- avoid writing large chunks of code. also deliberate what libraries can simplify task at hand
- prefer short files. when they grow offer user to convert split it. add a new pkg when it makes sense
- keep tests next to code _test.py
- tests must include non-power-of-two / non-tile-aligned sizes for every new extent, and a feature-off case that equals the previous version bit-for-bit; power-of-two-only matrices hide padding bugs
- library/compiler bugs are rare; if you believe you hit one, stop and ask the user instead of chasing the rabbit hole
- commit (or tag) right after a package is independently verified, before the next worker edits the tree; a stopped worker leaves a half-edited tree and uncommitted verified state is lost
- while implementing a chapter, park each problem hit in that chapter's `docs/evolution/<part>/<chapter>-impl.md`: one short section per problem (problem, what the measurement showed, how we fixed it), nothing more; the full write-up comes later


standards
- fire for simple cli and huggingface args large list. 
- reuse existing huggingface ecosystem as much as possible
- prefer gpu for torch tests
- always use `uv run --env-file .env` (machine-specific env like CUDA_HOME lives in gitignored repo-root .env)
- source builds: clone into ~/git/<org>/<proj>, checkout the required tag, install editable
- use justfiles(minimal cmd) and define at source folder. but they will always be invoked from repo root `just path/to/folder <cmd>` so assume that to be cwd and adjust justfile accordingly
- benches: compare only against baselines built for the same task (a torch path doing the same math, or the previous package); never against SDPA with masks it was not designed for
- run only the tests of the files you changed (`uv run --env-file .env pytest <package> -q`); `just smoke` (the full GPU suite) only when a shared file (golden_ref, ref helpers, justfiles) or several packages changed
- attention kernels: read src/mzoo/layers/attn/csa2_attn_design.md first (conventions live there); known TileLang/sm121 bugs and workarounds: tickets/0001-tilelang-issues.md; `just smoke` runs all GPU smoke tests

goal:
Experiment various model architectures

data layout: /data/pmonad/mzoo/000N-<proj>/yymmdd-000N-<experiment>/000N-<run>/...

