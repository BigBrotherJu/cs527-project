#!/usr/bin/env bash
# Runs inside a fresh evaluation container: apply a candidate patch with the
# protected paths filtered out, then drop the JIT build cache so the patched
# sources are what gets compiled.
#
#   prepare.sh <patch|none> <out_dir>
#
# Writes <out_dir>/patch_files.txt (every path the patch touches, with
# added/deleted line counts) and <out_dir>/patch_excluded.txt (the protected
# paths that were dropped). Exits non-zero if the remaining patch does not apply.
set -euo pipefail

PATCH=$1
OUT=$2
mkdir -p "${OUT}"
cd /workspace/flashinfer

# Paths the agent was told not to touch; edits there are discarded and reported.
# /task is outside the repo, so a repo diff cannot reach it.
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

# Compiled modules and generated sources only; downloaded artifacts stay.
echo "clearing JIT build cache:"
find /root/.cache/flashinfer -maxdepth 4 -type d \( -name cached_ops -o -name generated \) \
  -prune -print -exec rm -rf {} + 2>/dev/null || true
