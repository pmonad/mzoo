#!/usr/bin/env bash
# Whole-book single-column pdf: docs/pdf.sh [a4|a5] [out.pdf]
# runs quarto+latex inside docker (image built once from docs/docker/Dockerfile)
set -exuo pipefail
papersize="${1:-a4}"
out="${2:-book.pdf}"
docker image inspect mzoo-quarto >/dev/null 2>&1 || docker build -t mzoo-quarto docs/docker
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp/home -v "$PWD:/data" -w /data mzoo-quarto \
    quarto render docs/book.qmd --to pdf -M papersize:"$papersize" \
        -M title:"mzoo — Evolution" --output "$out"
