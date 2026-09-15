FROM node:22-bookworm-slim

ARG CODEX_VERSION=0.145.0
ARG ELAN_VERSION=v4.1.2
ARG LEAN_TOOLCHAIN=leanprover/lean4:v4.31.0

ENV DEBIAN_FRONTEND=noninteractive
ENV ELAN_HOME=/elan
ENV PATH=/elan/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        ripgrep \
        xz-utils \
        zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl --fail --location --silent --show-error \
        "https://raw.githubusercontent.com/leanprover/elan/${ELAN_VERSION}/elan-init.sh" \
        -o /tmp/elan-init.sh \
    && sh /tmp/elan-init.sh -y --no-modify-path --default-toolchain "${LEAN_TOOLCHAIN}" \
    && rm /tmp/elan-init.sh \
    && lean --version

RUN npm install --global --omit=dev "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force \
    && codex --version

RUN mkdir -p /opt/mathlib-template /mathlib-cache
COPY lean-toolchain lakefile.lean /opt/mathlib-template/

COPY requirements.toml /etc/codex/requirements.toml

RUN mkdir -p /work /run/codex-home \
    && chmod 0755 /work /run/codex-home \
    && rm -rf /root/.cache /tmp/*

WORKDIR /work
