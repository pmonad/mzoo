# quarto + xelatex, built for this machine's arch (quarto-dev images are amd64-only)
# build once: docker build -t mzoo-quarto docs/docker
FROM ubuntu:24.04
ARG QUARTO_VERSION=1.7.33
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates tar \
        texlive-xetex texlive-fonts-recommended texlive-plain-generic lmodern \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sL "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-arm64.tar.gz" \
       | tar -xz -C /opt && ln -s /opt/quarto-${QUARTO_VERSION}/bin/quarto /usr/local/bin/quarto
