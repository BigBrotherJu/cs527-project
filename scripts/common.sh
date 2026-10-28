# Sourced by the host scripts after they set ROOT.
#
# Images are per GPU architecture, cs527/<task_id>:<arch> (sm90 on H100), since
# the warmed JIT cache and downloaded cubins serve only the architecture they
# were built on. A task runs only on its SUPPORTED_SM: elsewhere its tests skip
# themselves, so a patch could not be judged.

# load_task <task_id> <gpu_index>: sets TASK_DIR, the task.env variables, DOCKER
# and IMAGE (the warmed image for that GPU); fails on an unsupported GPU.
load_task() {
  TASK_DIR="${ROOT}/tasks/$1"
  # shellcheck source=/dev/null
  source "${TASK_DIR}/task.env"
  DOCKER=${DOCKER:-docker}   # DOCKER="sudo docker" when not in the docker group
  local cc arch
  cc=$(nvidia-smi -i "$2" --query-gpu=compute_cap --format=csv,noheader) || return 1
  arch=${cc//./}
  if [[ " ${SUPPORTED_SM} " != *" ${arch} "* ]]; then
    echo "task $1 is tested only on SM ${SUPPORTED_SM}; GPU $2 is SM${arch}" >&2
    return 1
  fi
  IMAGE="cs527/$1:sm${arch}"
}

# eval_mounts <out_dir> <patch|none>: sets MOUNTS (scripts, eval material, /task
# read-only from the host, the patch, /out) and PATCH_ARG for prepare.sh.
# ${ROOT}/scripts/in_container -> /eval/bin
# ${TASK_DIR}/eval             -> /eval/task
# ${TASK_DIR}/task             -> /task
# <out_dir>                    -> /out
# <patch>                      -> /eval/patch.diff (if provided)
eval_mounts() {
  MOUNTS=(-v "${ROOT}/scripts/in_container:/eval/bin:ro" -v "${TASK_DIR}/eval:/eval/task:ro"
          -v "${TASK_DIR}/task:/task:ro" -v "$1:/out")
  PATCH_ARG=none
  if [ "$2" != "none" ]; then
    MOUNTS+=(-v "$(cd "$(dirname "$2")" && pwd)/$(basename "$2"):/eval/patch.diff:ro")
    PATCH_ARG=/eval/patch.diff
  fi
}

# require_profiling_admin_only: fail unless the host driver restricts GPU
# performance counters to admins (RmProfilingAdminOnly: 1). Only then does
# leaving out --cap-add SYS_ADMIN keep ncu from a timing-only agent; with 0, ncu
# works in every container.
# How to set:
# echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=1' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
# sudo update-initramfs -u
# sudo reboot
require_profiling_admin_only() {
  local v
  v=$(awk '/RmProfilingAdminOnly/ {print $2}' /proc/driver/nvidia/params)
  if [ "${v}" != 1 ]; then
    echo "RmProfilingAdminOnly is '${v}', not 1: ncu would work without SYS_ADMIN" >&2
    return 1
  fi
}
