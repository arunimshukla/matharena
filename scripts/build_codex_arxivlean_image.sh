#!/usr/bin/env bash
set -euo pipefail

IMAGE="matharena-codex-arxivlean:lean4.31-codex0.145.0"
CACHE_VOLUME="matharena-lean431-cache"

docker build \
  --file docker/codex-arxivlean/Dockerfile \
  --tag "${IMAGE}" \
  docker/codex-arxivlean

docker volume inspect "${CACHE_VOLUME}" >/dev/null 2>&1 \
  || docker volume create "${CACHE_VOLUME}" >/dev/null

docker run --rm \
  --network bridge \
  --mount "type=volume,src=${CACHE_VOLUME},dst=/mathlib-cache" \
  "${IMAGE}" \
  sh -c '
    set -eu
    cd /mathlib-cache
    if [ ! -f .lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean ]; then
      cp /opt/mathlib-template/lean-toolchain /opt/mathlib-template/lakefile.lean .
      lake update
      lake exe cache get
    fi
    chmod -R a+rX /mathlib-cache
  '

docker run --rm --network none "${IMAGE}" lean --version
docker run --rm --network none "${IMAGE}" codex --version
docker run --rm --network none \
  --mount "type=volume,src=${CACHE_VOLUME},dst=/mathlib-cache,readonly" \
  "${IMAGE}" \
  sh -c '
    printf "%s\n" "import Mathlib" "" "example : True := by trivial" > /tmp/Smoke.lean
    cd /mathlib-cache
    lake env lean /tmp/Smoke.lean
  '
