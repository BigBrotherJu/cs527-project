"""Task configuration (benchmark/tasks/<id>/task.toml) and host helpers shared by the scripts.

Images are per GPU architecture, cs527/<task_id>:sm<arch> (sm90 on H100), since
the warmed JIT cache and downloaded cubins serve only the architecture they
were built on. A task runs only on its supported_sm: elsewhere its tests skip
themselves, so a patch could not be judged.
"""

import contextlib
import fcntl
import os
import shlex
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # the project; scripts run from here with python -m
BENCHMARK = ROOT / "benchmark"
# The docker binary, as mini-swe-agent reads it; needs docker-group membership.
DOCKER = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")


@dataclass
class Task:
    id: str
    dir: Path
    config: dict   # task.toml
    gpu: str
    arch: str      # e.g. "90"
    image: str     # the warmed image for this GPU

    @property
    def visible_tests(self):
        """The visible tests as one shell word list."""
        return shlex.join(self.config["visible_tests"])

    @property
    def cuda_arch(self):
        """FLASHINFER_CUDA_ARCH_LIST for this architecture, e.g. 9.0a."""
        return f"{self.arch[:-1]}.{self.arch[-1]}a"


def load_task(task_id, gpu):
    """The task's config and image for GPU <gpu>; exits on an unsupported GPU."""
    task_dir = BENCHMARK / "tasks" / task_id
    config = tomllib.loads((task_dir / "task.toml").read_text())
    cc = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=compute_cap", "--format=csv,noheader"],
                        capture_output=True, text=True, check=True).stdout.strip()
    arch = cc.replace(".", "")
    if int(arch) not in config["supported_sm"]:
        raise SystemExit(f"task {task_id} is tested only on SM {config['supported_sm']}; GPU {gpu} is SM{arch}")
    return Task(task_id, task_dir, config, str(gpu), arch, f"cs527/{task_id}:sm{arch}")


def eval_mounts(task, out, patch):
    """Mounts of an evaluation container and the patch argument for prepare.sh:

        benchmark/container  -> /eval/bin
        <task>/eval          -> /eval/task
        <task>/task          -> /task
        <out>                -> /out
        <patch>              -> /eval/patch.diff (unless "none")
    """
    mounts = ["-v", f"{BENCHMARK}/container:/eval/bin:ro", "-v", f"{task.dir}/eval:/eval/task:ro",
              "-v", f"{task.dir}/task:/task:ro", "-v", f"{Path(out).resolve()}:/out"]
    if patch == "none":
        return mounts, "none"
    return [*mounts, "-v", f"{Path(patch).resolve()}:/eval/patch.diff:ro"], "/eval/patch.diff"


def require_profiling_admin_only():
    """Exit unless the host driver restricts GPU performance counters to admins
    (RmProfilingAdminOnly: 1). Only then does leaving out --cap-add SYS_ADMIN
    keep ncu from a timing-only agent; with 0, ncu works in every container.

    How to set:
        echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=1' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
        sudo update-initramfs -u
        sudo reboot
    """
    value = None
    for line in Path("/proc/driver/nvidia/params").read_text().splitlines():
        if line.startswith("RmProfilingAdminOnly:"):
            value = line.split(":", 1)[1].strip()
    if value != "1":
        raise SystemExit(f"RmProfilingAdminOnly is {value!r}, not 1: ncu would work without SYS_ADMIN")


def docker(*args, **kwargs):
    """Run docker; raises on failure unless check=False is given."""
    kwargs.setdefault("check", True)
    return subprocess.run([DOCKER, *map(str, args)], **kwargs)


@contextlib.contextmanager
def gpu_lock(gpu):
    """Hold the host-wide lock of one GPU: agents' tools and timing take it, so
    their GPU work never overlaps. Yields the seconds spent waiting."""
    with open(f"/tmp/cs527-gpu{gpu}.lock", "w") as f:
        start = time.time()
        fcntl.flock(f, fcntl.LOCK_EX)
        yield time.time() - start
