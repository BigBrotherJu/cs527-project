#!/usr/bin/env bash
# Correctness evaluation of one candidate patch in a fresh, offline container.
#
#   scripts/eval_correctness.sh <task_id> <patch|none> <out_dir> [gpu_index]
#
# Applies the patch (protected paths filtered, see in_container/prepare.sh),
# clears the JIT build cache, then runs the visible tests, the hidden tests
# (tasks/<id>/eval/test_hidden_*.py) and bench_task.py --check on the visible
# and hidden shapes. Results land in <out_dir>: JUnit XML, check JSON, logs, and
# status.txt with one "<step> <exit code>" line per step.
set -uo pipefail

TASK_ID=${1:?usage: eval_correctness.sh <task_id> <patch|none> <out_dir> [gpu_index]}
PATCH=${2:?}
OUT=${3:?}
GPU=${4:-0}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TASK_DIR="${ROOT}/tasks/${TASK_ID}"
# shellcheck source=/dev/null
source "${TASK_DIR}/task.env"
DOCKER=${DOCKER:-docker}

mkdir -p "${OUT}"
OUT=$(cd "${OUT}" && pwd)
PATCH_MOUNT=()
PATCH_ARG=none
if [ "${PATCH}" != "none" ]; then
  PATCH_MOUNT=(-v "$(cd "$(dirname "${PATCH}")" && pwd)/$(basename "${PATCH}"):/eval/patch.diff:ro")
  PATCH_ARG=/eval/patch.diff
fi

${DOCKER} run --rm --network none --gpus "\"device=${GPU}\"" --ipc=host \
  -v "${ROOT}/scripts/in_container:/eval/bin:ro" \
  -v "${TASK_DIR}/eval:/eval/task:ro" \
  -v "${TASK_DIR}/task:/task:ro" \
  "${PATCH_MOUNT[@]}" \
  -v "${OUT}:/out" \
  -w /workspace/flashinfer \
  "cs527/${TASK_ID}:latest" \
  bash -c "
    step() { local name=\$1; shift; \"\$@\" > /out/\${name}.log 2>&1; local rc=\$?; echo \"\${name} \${rc}\" >> /out/status.txt; echo \"\${name}: exit \${rc}\"; }
    : > /out/status.txt
    bash /eval/bin/prepare.sh ${PATCH_ARG} /out || { echo 'prepare 1' >> /out/status.txt; exit 1; }
    echo 'prepare 0' >> /out/status.txt
    cp /eval/task/test_hidden_*.py tests/moe/
    step visible_tests  pytest -q ${VISIBLE_TESTS} --junitxml=/out/visible_tests.xml
    step hidden_tests   pytest -q tests/moe/test_hidden_*.py --junitxml=/out/hidden_tests.xml
    step check_visible  python /task/bench_task.py --check --repeat-ms 20 --json /out/check_visible.json
    step check_hidden   python /task/bench_task.py --check --repeat-ms 20 --shapes /eval/task/hidden_shapes.json --json /out/check_hidden.json
  "
echo "== ${OUT}/status.txt"
cat "${OUT}/status.txt"
