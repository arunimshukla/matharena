#!/usr/bin/env bash
set -euo pipefail

IMAGE="matharena-harness-arxivlean:lean4.31"
CACHE_VOLUME="matharena-lean431-cache"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

docker build \
  --file docker/harness-arxivlean/Dockerfile \
  --tag "${IMAGE}" \
  docker/harness-arxivlean

docker volume inspect "${CACHE_VOLUME}" >/dev/null 2>&1 \
  || docker volume create "${CACHE_VOLUME}" >/dev/null

LOCAL_LEAN_PROJECT="${REPO_ROOT}/external/comparator_project"
if docker run --rm --network none \
  --mount "type=volume,src=${CACHE_VOLUME},dst=/mathlib-cache,readonly" \
  "${IMAGE}" \
  test -f /mathlib-cache/.lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean
then
  echo "Reusing populated ${CACHE_VOLUME}."
elif [[ -f "${LOCAL_LEAN_PROJECT}/.lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean" ]]
then
  docker run --rm \
    --network none \
    --mount "type=bind,src=${LOCAL_LEAN_PROJECT},dst=/host-cache,readonly" \
    --mount "type=volume,src=${CACHE_VOLUME},dst=/mathlib-cache" \
    "${IMAGE}" \
    sh -c 'cp -a /host-cache/. /mathlib-cache/ && chmod -R a+rX /mathlib-cache'
else
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
fi

docker run --rm --network none "${IMAGE}" lean --version
docker run --rm --network none \
  --mount "type=volume,src=${CACHE_VOLUME},dst=/mathlib-cache,readonly" \
  "${IMAGE}" \
  sh -c '
    printf "%s\n" "import Mathlib" "" "example : True := by trivial" > /tmp/Smoke.lean
    export LEAN_PATH="/mathlib-cache/.lake/packages/Cli/.lake/build/lib/lean:/mathlib-cache/.lake/packages/batteries/.lake/build/lib/lean:/mathlib-cache/.lake/packages/Qq/.lake/build/lib/lean:/mathlib-cache/.lake/packages/aesop/.lake/build/lib/lean:/mathlib-cache/.lake/packages/proofwidgets/.lake/build/lib/lean:/mathlib-cache/.lake/packages/importGraph/.lake/build/lib/lean:/mathlib-cache/.lake/packages/LeanSearchClient/.lake/build/lib/lean:/mathlib-cache/.lake/packages/plausible/.lake/build/lib/lean:/mathlib-cache/.lake/packages/mathlib/.lake/build/lib/lean:/mathlib-cache/.lake/build/lib/lean:/elan/toolchains/leanprover--lean4---v4.31.0/lib/lean"
    lean /tmp/Smoke.lean
  '
