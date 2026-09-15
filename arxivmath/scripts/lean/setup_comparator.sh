#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
COMPARATOR_DIR="${ROOT_DIR}/external/comparator"
EXPORT_DIR="${ROOT_DIR}/external/lean4export"
LANDRUN_ROOT="${ROOT_DIR}/external/landrun"
LANDRUN_SOURCE_DIR="${LANDRUN_ROOT}/src"
LANDRUN_BIN_DIR="${LANDRUN_ROOT}/bin"
LANDRUN_BIN="${LANDRUN_BIN_DIR}/landrun"
PROJECT_DIR="${ROOT_DIR}/external/comparator_project"
LEAN_TOOLCHAIN="leanprover/lean4:v4.31.0"
LANDRUN_REVISION="811cfff51ceaf3d9843708aa6d22e9b84ccac8b4"
COMPARATOR_LANDRUN_FIX="9badaf470d8f724346d33738bd273efacd78df76"

[ -d "${COMPARATOR_DIR}/.git" ] || git clone https://github.com/leanprover/comparator "${COMPARATOR_DIR}"
[ -d "${EXPORT_DIR}/.git" ] || git clone https://github.com/leanprover/lean4export "${EXPORT_DIR}"
[ -d "${LANDRUN_SOURCE_DIR}/.git" ] || git clone https://github.com/Zouuup/landrun.git "${LANDRUN_SOURCE_DIR}"

git -C "${COMPARATOR_DIR}" fetch --tags
git -C "${COMPARATOR_DIR}" checkout v4.31.0
git -C "${COMPARATOR_DIR}" fetch origin "${COMPARATOR_LANDRUN_FIX}"
if ! grep -Fq 'args ++ #["--", spawnArgs.cmd]' "${COMPARATOR_DIR}/Main.lean"; then
  git -C "${COMPARATOR_DIR}" show --format= "${COMPARATOR_LANDRUN_FIX}" -- Main.lean |
    git -C "${COMPARATOR_DIR}" apply -
fi

git -C "${LANDRUN_SOURCE_DIR}" fetch origin "${LANDRUN_REVISION}"
git -C "${LANDRUN_SOURCE_DIR}" checkout --detach "${LANDRUN_REVISION}"
mkdir -p "${LANDRUN_BIN_DIR}"
(cd "${LANDRUN_SOURCE_DIR}" && go build -o "${LANDRUN_BIN}" ./cmd/landrun)

if ! elan toolchain list | grep -Fqx "${LEAN_TOOLCHAIN}"; then
  elan toolchain install "${LEAN_TOOLCHAIN}"
fi

printf '%s\n' "${LEAN_TOOLCHAIN}" > "${EXPORT_DIR}/lean-toolchain"
(cd "${EXPORT_DIR}" && lake update && lake build)
(cd "${COMPARATOR_DIR}" && lake build)

mkdir -p "${PROJECT_DIR}"
printf '%s\n' "${LEAN_TOOLCHAIN}" > "${PROJECT_DIR}/lean-toolchain"
cat > "${PROJECT_DIR}/lakefile.lean" <<'EOF'
import Lake
open Lake DSL

package comparatorcheck where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.31.0"

lean_lib Challenge where
  roots := #[`Challenge]

lean_lib Solution where
  roots := #[`Solution]
EOF

cat > "${PROJECT_DIR}/Challenge.lean" <<'EOF'
import Mathlib

theorem comparator_template_challenge : True := by
  trivial
EOF

cat > "${PROJECT_DIR}/Solution.lean" <<'EOF'
import Mathlib

theorem comparator_template_challenge : True := by
  trivial
EOF

(cd "${PROJECT_DIR}" && lake update && lake build Challenge Solution)

LEAN_BIN_DIR="$(dirname "$(cd "${PROJECT_DIR}" && elan which lake)")"
echo "export PATH=\"${LANDRUN_BIN_DIR}:${EXPORT_DIR}/.lake/build/bin:${LEAN_BIN_DIR}:\$PATH\""
