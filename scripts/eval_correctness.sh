#!/usr/bin/env bash
# Correctness of one patch in a fresh, offline container: visible tests, hidden
# tests (eval/test_hidden_*.py) and bench_task.py --check on the visible and
# hidden shapes. <out_dir> gets JUnit XML, check JSON, logs, status.txt ("<step>
# <exit code>" per step) and test_counts.json.
#
#   [BASE_OUT=<dir>] scripts/eval_correctness.sh <task_id> <patch|none> <out_dir> [gpu_index]
#
# BASE_OUT, a baseline result from the same architecture, also lists every test
# whose outcome differs from it (see test_counts.py).
set -uo pipefail

TASK_ID=${1:?usage: eval_correctness.sh <task_id> <patch|none> <out_dir> [gpu_index]}
PATCH=${2:?}
OUT=${3:?}
GPU=${4:-0}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"
load_task "${TASK_ID}" "${GPU}" || exit 1
mkdir -p "${OUT}"
OUT=$(cd "${OUT}" && pwd)
eval_mounts "${OUT}" "${PATCH}"

${DOCKER} run --rm --network none --gpus "\"device=${GPU}\"" --ipc=host \
  "${MOUNTS[@]}" -w /workspace/flashinfer "${IMAGE}" \
  bash -c "
    step() { local name=\$1; shift; \"\$@\" > /out/\${name}.log 2>&1; local rc=\$?; echo \"\${name} \${rc}\" >> /out/status.txt; echo \"\${name}: exit \${rc}\"; }
    : > /out/status.txt
    bash /eval/bin/prepare.sh ${PATCH_ARG} /out || { echo 'prepare 1' >> /out/status.txt; exit 1; }
    echo 'prepare 0' >> /out/status.txt
    cp /eval/task/test_hidden_*.py ${HIDDEN_TEST_DIR}/
    step visible_tests  pytest -q ${VISIBLE_TESTS} --junitxml=/out/visible_tests.xml
    step hidden_tests   pytest -q ${HIDDEN_TEST_DIR}/test_hidden_*.py --junitxml=/out/hidden_tests.xml
    step check_visible  python /task/bench_task.py --check --repeat-ms 20 --json /out/check_visible.json
    step check_hidden   python /task/bench_task.py --check --repeat-ms 20 --shapes /eval/task/hidden_shapes.json --json /out/check_hidden.json
  "
echo "== ${OUT}/status.txt"
cat "${OUT}/status.txt"
echo "== test outcomes"
python3 "${ROOT}/scripts/test_counts.py" "${OUT}" ${BASE_OUT:+--base "${BASE_OUT}"}
