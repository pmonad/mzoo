# pandoc + minimal XeLaTeX for chapter PDFs (linux/arm64 has no pandoc/latex image)
# build: docker build -t mzoo-pandoc docs/latex
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        pandoc \
        texlive-xetex \
        texlive-latex-recommended \
        texlive-fonts-recommended \
        lmodern \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /data
