#!/usr/bin/env bash
# Sync rendered site/ to the gh-pages branch and push.
# Build happens in docker via `just docs/ render` (the publish recipe depends
# on it); git operations run on the host.
set -exuo pipefail
git fetch origin gh-pages
git worktree prune
git branch -f gh-pages origin/gh-pages
wt="$(mktemp -d)"
rmdir "$wt"
trap 'git worktree remove --force "$wt" 2>/dev/null || true' EXIT
git worktree add "$wt" gh-pages
rsync -a --delete --exclude=.git site/ "$wt"/
touch "$wt/.nojekyll"
git -C "$wt" add -A
if git -C "$wt" diff --cached --quiet; then
    echo "nothing new to publish"
else
    git -C "$wt" commit -m "Built site for $(git rev-parse --short HEAD)"
    git push origin gh-pages
fi
