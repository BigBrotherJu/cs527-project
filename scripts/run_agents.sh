#!/usr/bin/env bash
# Run <runs_per_arm> agent attempts per arm on one GPU, PARALLEL at a time,
# started in alternating arm order (correctness-1, timing-1, ..., correctness-2,
# ...) so no arm runs all of its attempts first. Agents share the GPU through
# the tools' per-GPU lock, so their GPU work never overlaps. Runs land in
# <out_root>/<task_id>/<arm>-<k> (log: <arm>-<k>.log beside it); a run that
# already has meta.json is skipped, so the script can be restarted.
#
#   scripts/run_agents.sh <task_id> <gpu_index> <runs_per_arm> [out_root] [-- run_agent.py options]
#   e.g. PARALLEL=3 scripts/run_agents.sh c3f7334d4 0 3 ~/runs -- --model openai/gpt-6.1-sol
#
#   ARMS="..."      arms to run, in this order within each round
#                   (default: all five, see run_agent.py)
#   PARALLEL=N      agents at a time (default 1). Each builds with
#                   MAX_JOBS = cores / N (one bgmv build averages ~4 cores)
#   PYTHON          interpreter with mini-swe-agent installed (default: python3)
set -uo pipefail

TASK_ID=${1:?usage: run_agents.sh <task_id> <gpu_index> <runs_per_arm> [out_root] [-- options]}
GPU=${2:?}
RUNS=${3:?}
shift 3
OUT_ROOT=${HOME}/runs
if [ $# -gt 0 ] && [ "$1" != "--" ]; then OUT_ROOT=$1; shift; fi
[ "${1:-}" = "--" ] && shift
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARMS=${ARMS:-correctness timing ncu-select ncu-full ncu-summary}
PARALLEL=${PARALLEL:-1}
MAX_JOBS=$(( $(nproc) / PARALLEL > 2 ? $(nproc) / PARALLEL : 2 ))
mkdir -p "${OUT_ROOT}/${TASK_ID}"

run_one() {  # arm k out [run_agent.py options]
  "${PYTHON:-python3}" "${ROOT}/scripts/run_agent.py" "${TASK_ID}" "$1" "$3" --gpu "${GPU}" \
    --max-jobs "${MAX_JOBS}" "${@:4}" > "${OUT_ROOT}/${TASK_ID}/$1-$2.log" 2>&1
  echo "== $1-$2: done (exit $?)"
}

for ((k = 1; k <= RUNS; k++)); do
  for arm in ${ARMS}; do
    out="${OUT_ROOT}/${TASK_ID}/${arm}-${k}"
    if [ -f "${out}/meta.json" ]; then echo "== ${arm}-${k}: done earlier, skipping"; continue; fi
    if [ -e "${out}" ]; then echo "== ${arm}-${k}: ${out} exists without meta.json (crashed?), skipping" >&2; continue; fi
    while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]; do wait -n; done
    echo "== ${arm}-${k}: start -> ${out}"
    run_one "${arm}" "${k}" "${out}" "$@" &
    sleep "${STAGGER:-10}"  # stagger container starts (volume copy-up)
  done
done
wait
