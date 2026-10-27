#!/usr/bin/env bash
# Interleaved timing of several arms (baseline, agent patch, reference patch) on
# one GPU.
#
#   scripts/eval_perf.sh <task_id> <out_dir> <gpu_index> <rounds> <name>=<patch|none> ...
#   e.g. scripts/eval_perf.sh c3f7334d4 results/perf 0 5 base=none ref=tasks/c3f7334d4/eval/ref.patch
#
# Each arm gets its own offline container with its patch applied and the JIT
# module rebuilt once, so the timed rounds include no compilation. Rounds then
# run bench_task.py on the visible and hidden shapes, one arm at a time, rotating
# the arm order every round so no arm always runs first. Results:
# <out_dir>/<name>/r<k>_{visible,hidden}.json, summarized by summarize_perf.py.
#
# Lock the GPU clocks on the host first if you can (nvidia-smi -lgc), and keep
# other work off this GPU while it runs.
set -euo pipefail

TASK_ID=${1:?usage: eval_perf.sh <task_id> <out_dir> <gpu_index> <rounds> <name>=<patch|none> ...}
OUT=${2:?}
GPU=${3:?}
ROUNDS=${4:?}
shift 4
[ $# -ge 2 ] || { echo "need at least two arms" >&2; exit 1; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TASK_DIR="${ROOT}/tasks/${TASK_ID}"
DOCKER=${DOCKER:-docker}
mkdir -p "${OUT}"
OUT=$(cd "${OUT}" && pwd)

NAMES=()
CONTAINERS=()
cleanup() { for c in "${CONTAINERS[@]}"; do ${DOCKER} rm -f "${c}" >/dev/null 2>&1 || true; done; }
trap cleanup EXIT

for arm in "$@"; do
  name=${arm%%=*}
  patch=${arm#*=}
  c="perf-${TASK_ID}-${name}-$$"
  # /task comes from the host, read-only: evaluation always uses the current,
  # untouched tools rather than the copy baked into the image.
  mounts=(-v "${ROOT}/scripts/in_container:/eval/bin:ro" -v "${TASK_DIR}/eval:/eval/task:ro"
          -v "${TASK_DIR}/task:/task:ro" -v "${OUT}:/out")
  patch_arg=none
  if [ "${patch}" != "none" ]; then
    mounts+=(-v "$(cd "$(dirname "${patch}")" && pwd)/$(basename "${patch}"):/eval/patch.diff:ro")
    patch_arg=/eval/patch.diff
  fi
  mkdir -p "${OUT}/${name}"
  ${DOCKER} run -d --name "${c}" --network none --gpus "\"device=${GPU}\"" --ipc=host \
    "${mounts[@]}" -w /workspace/flashinfer "cs527/${TASK_ID}:latest" sleep infinity >/dev/null
  CONTAINERS+=("${c}")
  NAMES+=("${name}")
  echo "== ${name}: prepare + build"
  ${DOCKER} exec "${c}" bash /eval/bin/prepare.sh "${patch_arg}" "/out/${name}"
  # One untimed pass compiles the module and warms the GPU for this arm.
  ${DOCKER} exec "${c}" python /task/bench_task.py --repeat-ms 20 > "${OUT}/${name}/warmup.log" 2>&1
done

n=${#NAMES[@]}
for ((r = 1; r <= ROUNDS; r++)); do
  for ((i = 0; i < n; i++)); do
    k=$(((i + r - 1) % n))
    name=${NAMES[$k]}
    c=${CONTAINERS[$k]}
    echo "== round ${r}/${ROUNDS}: ${name}"
    ${DOCKER} exec "${c}" python /task/bench_task.py --json "/out/${name}/r${r}_visible.json" > /dev/null
    ${DOCKER} exec "${c}" python /task/bench_task.py --shapes /eval/task/hidden_shapes.json \
      --json "/out/${name}/r${r}_hidden.json" > /dev/null
  done
done

python3 "${ROOT}/scripts/summarize_perf.py" "${OUT}" "${NAMES[@]}" | tee "${OUT}/summary.txt"
