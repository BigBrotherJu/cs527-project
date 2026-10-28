#!/usr/bin/env bash
# Interleaved timing of two or more arms (baseline, reference, agent patches) on
# one GPU. Each arm gets its own offline container, built and warmed once
# untimed; each round then runs bench_task.py on the visible and hidden shapes
# for every arm, rotating the arm order. Results: <out_dir>/<name>/r<k>_{visible,
# hidden}.json and summary.txt (summarize_perf.py).
#
#   scripts/eval_perf.sh <task_id> <out_dir> <gpu_index> <rounds> <name>=<patch|none> ...
#
# The SM clock is locked (-lgc) for the rounds and reset on exit, by default at
# the GPU's default applications clock. The memory clock cannot be fixed at run
# time on H100 (-lmc is unsupported and exits 0 anyway; -lmcd needs a driver
# reload; -ac changes only the reported applications clock, measured: HBM stays
# at 2619 MHz under load either way), so it is checked against the default
# applications memory clock instead. clocks_log.csv samples clocks, power,
# temperature and throttle reasons every 500 ms during the rounds; the summary
# flags power or thermal throttling and SM or memory clocks below their targets.
#
#   GPU_CLOCK=<MHz>       SM clock to lock to instead
#   GPU_MEM_CLOCK=<MHz>   expected memory clock instead
#   LOCK_CLOCKS=0     do not lock (e.g. no root); clocks.txt says so
#   NVSMI             lock/reset command (default: "sudo nvidia-smi")
set -euo pipefail

TASK_ID=${1:?usage: eval_perf.sh <task_id> <out_dir> <gpu_index> <rounds> <name>=<patch|none> ...}
OUT=${2:?}
GPU=${3:?}
ROUNDS=${4:?}
shift 4
[ $# -ge 2 ] || { echo "need at least two arms" >&2; exit 1; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"
load_task "${TASK_ID}" "${GPU}"
NVSMI=${NVSMI:-sudo nvidia-smi}
LOCK_CLOCKS=${LOCK_CLOCKS:-1}
mkdir -p "${OUT}"
OUT=$(cd "${OUT}" && pwd)

NAMES=()
CONTAINERS=()
LOCKED=0
SAMPLER=
cleanup() {
  if [ -n "${SAMPLER}" ]; then kill "${SAMPLER}" 2>/dev/null || true; fi
  for c in "${CONTAINERS[@]}"; do ${DOCKER} rm -f "${c}" >/dev/null 2>&1 || true; done
  if [ "${LOCKED}" = 1 ]; then ${NVSMI} -i "${GPU}" -rgc >/dev/null || echo "WARNING: failed to reset GPU ${GPU} clocks" >&2; fi
}
trap cleanup EXIT

for arm in "$@"; do
  name=${arm%%=*}
  patch=${arm#*=}
  c="perf-${TASK_ID}-${name}-$$"
  eval_mounts "${OUT}" "${patch}"
  mkdir -p "${OUT}/${name}"
  ${DOCKER} run -d --name "${c}" --network none --gpus "\"device=${GPU}\"" --ipc=host \
    "${MOUNTS[@]}" -w /workspace/flashinfer "${IMAGE}" sleep infinity >/dev/null
  CONTAINERS+=("${c}")
  NAMES+=("${name}")
  echo "== ${name}: prepare + build"
  ${DOCKER} exec "${c}" bash /eval/bin/prepare.sh "${PATCH_ARG}" "/out/${name}"
  # One untimed pass compiles the module and warms the GPU for this arm.
  ${DOCKER} exec "${c}" python /task/bench_task.py --repeat-ms 20 > "${OUT}/${name}/warmup.log" 2>&1
done

# Lock only around the timed rounds: the per-arm builds above are CPU-bound.
if [ "${LOCK_CLOCKS}" != 0 ]; then
  query() { nvidia-smi -i "${GPU}" --query-gpu="$1" --format=csv,noheader,nounits; }
  CLOCK=${GPU_CLOCK:-$(query clocks.default_applications.graphics)}
  MEM_CLOCK=${GPU_MEM_CLOCK:-$(query clocks.default_applications.memory)}
  [[ "${CLOCK}${MEM_CLOCK}" =~ ^[0-9]+$ ]] ||
    { echo "no default applications clocks (${CLOCK}, ${MEM_CLOCK}); set GPU_CLOCK and GPU_MEM_CLOCK" >&2; exit 1; }
  LOCKED=1
  ${NVSMI} -i "${GPU}" -lgc "${CLOCK},${CLOCK}" >/dev/null
  echo "locked GPU ${GPU} SM clock at ${CLOCK} MHz; expecting memory at ${MEM_CLOCK} MHz"
  printf 'sm_lock_mhz=%s\nmem_lock_mhz=%s\n' "${CLOCK}" "${MEM_CLOCK}" > "${OUT}/clocks.txt"
else
  echo "GPU ${GPU} clocks NOT locked (LOCK_CLOCKS=0)"
  printf 'sm_lock_mhz=none\nmem_lock_mhz=none\n' > "${OUT}/clocks.txt"
fi

# Newer drivers renamed clocks_throttle_reasons to clocks_event_reasons.
REASONS=clocks_event_reasons.active
nvidia-smi -i "${GPU}" --query-gpu="${REASONS}" --format=csv,noheader >/dev/null 2>&1 ||
  REASONS=clocks_throttle_reasons.active
nvidia-smi -i "${GPU}" -lms 500 --format=csv,noheader,nounits \
  --query-gpu=timestamp,clocks.sm,clocks.mem,clocks.max.memory,power.draw,temperature.gpu,"${REASONS}" \
  > "${OUT}/clocks_log.csv" &
SAMPLER=$!

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

kill "${SAMPLER}" 2>/dev/null || true
SAMPLER=

python3 "${ROOT}/scripts/summarize_perf.py" "${OUT}" "${NAMES[@]}" | tee "${OUT}/summary.txt"
