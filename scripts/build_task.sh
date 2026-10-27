#!/usr/bin/env bash
# Build the image for one task: docker build, then warm the JIT cache on a GPU
# and commit the result.
#
#   scripts/build_task.sh <task_id> [gpu_index]
#
# Reads tasks/<task_id>/task.env (BASE_IMAGE, REF_COMMIT, WARMUP_CMD) and uses
# tasks/<task_id>/task/ as the agent-visible /task directory.
# Produces cs527/<task_id>:base (unwarmed) and cs527/<task_id>:latest (warmed).
set -euo pipefail

TASK_ID=${1:?usage: build_task.sh <task_id> [gpu_index]}
GPU=${2:-0}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TASK_DIR="${ROOT}/tasks/${TASK_ID}"
# shellcheck source=/dev/null
source "${TASK_DIR}/task.env"
: "${BASE_IMAGE:?}" "${REF_COMMIT:?}" "${WARMUP_CMD:?}"
if [[ ! "${REF_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "REF_COMMIT must be a full 40-char hash, got: ${REF_COMMIT}" >&2
  exit 1
fi

IMAGE="cs527/${TASK_ID}"
DOCKER=${DOCKER:-docker}   # DOCKER="sudo docker" when the user is not in the docker group

echo "== build ${IMAGE}:base"
${DOCKER} build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg REF_COMMIT="${REF_COMMIT}" \
  --build-context task="${TASK_DIR}/task" \
  -t "${IMAGE}:base" \
  "${ROOT}/docker"

# docker build has no GPU, so the JIT warm-up runs in a container that is then
# committed. The cache lands in /root/.cache/flashinfer, so agent and evaluation
# containers must run as root with HOME=/root to hit it. The amend folds anything
# the warm-up left in the tree into the initial commit, so it never shows up in
# the agent's diff.
echo "== warm JIT cache on GPU ${GPU}"
WARM="warm-${TASK_ID}-$$"
${DOCKER} run --name "${WARM}" --gpus "\"device=${GPU}\"" --ipc=host \
  -w /workspace/flashinfer "${IMAGE}:base" \
  bash -c "set -e; ${WARMUP_CMD}; git add -A; git commit -q --amend --no-edit; git status --short"
${DOCKER} commit --change 'CMD ["sleep", "infinity"]' "${WARM}" "${IMAGE}:latest" >/dev/null
${DOCKER} rm "${WARM}" >/dev/null

echo "== done: ${IMAGE}:latest"
${DOCKER} images "${IMAGE}"
