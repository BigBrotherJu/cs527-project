#!/usr/bin/env bash
# Build cs527/<task_id>:base (docker build, no GPU), then run the task's
# WARMUP_CMD on a GPU and commit the result as cs527/<task_id>:<arch>. Run it on
# a host with a GPU of each architecture you need.
#
#   scripts/build_task.sh <task_id> [gpu_index]
set -euo pipefail

TASK_ID=${1:?usage: build_task.sh <task_id> [gpu_index]}
GPU=${2:-0}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"
load_task "${TASK_ID}" "${GPU}"
if [[ ! "${REF_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "REF_COMMIT must be a full 40-char hash (GitHub refuses short ones), got: ${REF_COMMIT}" >&2
  exit 1
fi

echo "== build cs527/${TASK_ID}:base"
${DOCKER} build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg REF_COMMIT="${REF_COMMIT}" \
  --build-context task="${TASK_DIR}/task" \
  -t "cs527/${TASK_ID}:base" \
  "${ROOT}/docker"

# The JIT cache lands in /root/.cache/flashinfer, so agent and evaluation
# containers must run as root with HOME=/root. The amend folds anything the
# warm-up left in the tree into the initial commit, out of the agent's diff.
# FLASHINFER_JIT_VERBOSE is exported inside bash: a -e variable would be baked
# into the committed image.
echo "== warm up on GPU ${GPU} -> ${IMAGE}"
WARM="warm-${TASK_ID}-$$"
${DOCKER} run --name "${WARM}" --gpus "\"device=${GPU}\"" --ipc=host \
  -w /workspace/flashinfer "cs527/${TASK_ID}:base" \
  bash -c "set -e; export FLASHINFER_JIT_VERBOSE=1; ${WARMUP_CMD}; git add -A; git commit -q --amend --no-edit; git status --short"
${DOCKER} commit --change 'CMD ["sleep", "infinity"]' "${WARM}" "${IMAGE}" >/dev/null
${DOCKER} rm "${WARM}" >/dev/null
${DOCKER} images "cs527/${TASK_ID}"
