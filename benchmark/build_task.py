"""Build cs527/<task_id>:base (docker build, no GPU), then warm it up on a GPU
and commit the result as cs527/<task_id>:sm<arch>. Run it on a host with a GPU
of each architecture you need.

    python3 -m benchmark.build_task <task_id> [gpu_index]

The warm-up runs the visible tests and bench_task.py once, so the JIT modules
and downloaded cubins they need are in the image (agents and evaluation run
offline).
"""

import argparse
import os
import re

from benchmark.task import BENCHMARK, docker, load_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_id")
    ap.add_argument("gpu", nargs="?", default="0")
    a = ap.parse_args()
    task = load_task(a.task_id, a.gpu)
    if not re.fullmatch(r"[0-9a-f]{40}", task.config["ref_commit"]):
        raise SystemExit(f"ref_commit must be a full 40-char hash (GitHub refuses short ones), "
                         f"got: {task.config['ref_commit']}")

    base = f"cs527/{task.id}:base"
    print(f"== build {base}", flush=True)
    docker("build", "--build-arg", f"BASE_IMAGE={task.config['base_image']}",
           "--build-arg", f"REF_COMMIT={task.config['ref_commit']}",
           "--build-context", f"task={task.dir}/task", "-t", base, BENCHMARK / "docker")

    # The JIT cache lands in /root/.cache/flashinfer, so agent and evaluation
    # containers must run as root with HOME=/root. The amend folds anything the
    # warm-up left in the tree into the initial commit, out of the agent's diff.
    # FLASHINFER_JIT_VERBOSE is exported inside bash: a -e variable would be
    # baked into the committed image. FLASHINFER_JIT_DEBUG=0 is required with
    # it: unset, the JIT treats VERBOSE=1 as a debug build (-O0, ~100x slower),
    # whose cache entries a normal run would not reuse.
    print(f"== warm up on GPU {task.gpu} -> {task.image}", flush=True)
    warmup = (f"set -e; export FLASHINFER_JIT_VERBOSE=1 FLASHINFER_JIT_DEBUG=0; "
              f"pytest -q {task.visible_tests}; python /task/bench_task.py; "
              f"git add -A; git commit -q --amend --no-edit; git status --short")
    warm = f"warm-{task.id}-{os.getpid()}"
    try:
        docker("run", "--name", warm, "--gpus", f'"device={task.gpu}"', "--ipc=host",
               "-w", "/workspace/flashinfer", base, "bash", "-c", warmup)
        docker("commit", "--change", 'CMD ["sleep", "infinity"]', warm, task.image, capture_output=True)
    finally:
        docker("rm", "-f", warm, capture_output=True, check=False)
    docker("images", f"cs527/{task.id}")


if __name__ == "__main__":
    main()
