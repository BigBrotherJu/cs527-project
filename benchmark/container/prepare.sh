#!/usr/bin/env bash
# In a fresh evaluation container: apply a patch minus the protected paths
# (listed in <out_dir>/patch_files.txt and patch_excluded.txt; non-zero exit if
# the rest does not apply). The warmed JIT cache stays: the JIT re-stages
# changed sources and ninja tracks header dependencies, so patched modules are
# rebuilt on first use.
#
#   prepare.sh <patch|none> <out_dir>
set -euo pipefail

PATCH=$1
OUT=$2
mkdir -p "${OUT}"
cd /workspace/flashinfer

# Paths the agent was told not to touch (/task is outside the repo).
PROTECTED=("tests/*" "benchmarks/*" "conftest.py")

if [ "${PATCH}" != "none" ] && [ -s "${PATCH}" ]; then
  git apply --numstat "${PATCH}" > "${OUT}/patch_files.txt"
  : > "${OUT}/patch_excluded.txt"
  EXCLUDES=()
  for p in "${PROTECTED[@]}"; do
    EXCLUDES+=(--exclude="${p}")
    awk -v pat="${p}" 'BEGIN { gsub(/\./, "\\.", pat); gsub(/\*/, ".*", pat); pat = "^" pat "$" }
                       $3 ~ pat { print $3 }' "${OUT}/patch_files.txt" >> "${OUT}/patch_excluded.txt"
  done
  git apply --index "${EXCLUDES[@]}" "${PATCH}"
  echo "applied ${PATCH}; excluded: $(wc -l < "${OUT}/patch_excluded.txt") file(s)"
else
  echo "no patch (baseline)"
fi

