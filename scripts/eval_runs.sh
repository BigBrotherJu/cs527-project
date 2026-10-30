#!/usr/bin/env bash
# Evaluate every agent run of a task: correctness of each run's scored.patch
# against the baseline, then one interleaved perf eval of the baseline, the
# reference and every correct run, then a summary table.
#
#   scripts/eval_runs.sh <task_id> <runs_dir> <eval_dir> [gpu_index] [rounds]
#   e.g. scripts/eval_runs.sh c3f7334d4 ~/runs/c3f7334d4 ~/evals/c3f7334d4 0 5
#
# <runs_dir> holds run_agent.py outputs (<arm>-<k>/meta.json, scored.patch).
# <eval_dir> gets base/ and <arm>-<k>/ (eval_correctness.sh), perf/
# (eval_perf.sh), and summary.md / runs.csv (summarize_runs.py). Runs already
# evaluated are kept, so the script can be restarted; perf is always rerun.
# A run is correct when every step in its status.txt passed and every test has
# the same outcome as in the baseline.
set -uo pipefail

TASK_ID=${1:?usage: eval_runs.sh <task_id> <runs_dir> <eval_dir> [gpu_index] [rounds]}
RUNS_DIR=${2:?}
EVAL_DIR=${3:?}
GPU=${4:-0}
ROUNDS=${5:-5}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "${EVAL_DIR}"

correct() {  # eval dir of one patch
  [ -f "$1/status.txt" ] && awk '$2 != 0 { bad = 1 } END { exit bad }' "$1/status.txt" &&
    python3 -c 'import json, sys; sys.exit(any(json.load(open(sys.argv[1])).get("differs_from_base", {}).values()))' \
      "$1/test_counts.json"
}

[ -f "${EVAL_DIR}/base/status.txt" ] ||
  "${ROOT}/scripts/eval_correctness.sh" "${TASK_ID}" none "${EVAL_DIR}/base" "${GPU}"

ARMS=(base=none "ref=${ROOT}/tasks/${TASK_ID}/eval/ref.patch")
for meta in "${RUNS_DIR}"/*/meta.json; do
  run=$(dirname "${meta}")
  name=$(basename "${run}")
  if [ ! -s "${run}/scored.patch" ]; then
    echo "== ${name}: empty scored patch (tests never passed, or passed only on unchanged code), skipping"
    continue
  fi
  if [ ! -f "${EVAL_DIR}/${name}/status.txt" ]; then
    echo "== ${name}: correctness"
    BASE_OUT="${EVAL_DIR}/base" "${ROOT}/scripts/eval_correctness.sh" "${TASK_ID}" "${run}/scored.patch" \
      "${EVAL_DIR}/${name}" "${GPU}"
  fi
  if correct "${EVAL_DIR}/${name}"; then
    ARMS+=("${name}=${run}/scored.patch")
  else
    echo "== ${name}: incorrect, left out of perf"
  fi
done

echo "== perf: ${#ARMS[@]} arms"
rm -rf "${EVAL_DIR}/perf"
"${ROOT}/scripts/eval_perf.sh" "${TASK_ID}" "${EVAL_DIR}/perf" "${GPU}" "${ROUNDS}" "${ARMS[@]}"

python3 "${ROOT}/scripts/summarize_runs.py" "${RUNS_DIR}" "${EVAL_DIR}" | tee "${EVAL_DIR}/summary.md"
