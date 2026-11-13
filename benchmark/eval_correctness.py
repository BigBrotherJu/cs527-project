"""Correctness of one patch in a fresh, offline container: visible tests, hidden
tests (eval/test_hidden_*.py) and bench_task.py --check on the visible and
hidden shapes. <out_dir> gets JUnit XML, check JSON, a log per step, status.txt
("<step> <exit code>" per step) and test_counts.json.

    python3 -m benchmark.eval_correctness <task_id> <patch|none> <out_dir> [gpu_index] [--base DIR]

--base, a baseline result from the same architecture, also lists every test
whose outcome differs from it (see test_counts.py).
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from benchmark.task import ROOT, docker, eval_mounts, load_task


def run(task, patch, out, base=None):
    """Evaluate <patch> into <out>; returns test_counts.py's exit code (1 if any
    test differs from <base>)."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mounts, patch_arg = eval_mounts(task, out, patch)
    status = out / "status.txt"
    status.write_text("")
    hidden_dir = shlex.quote(task.config["hidden_test_dir"])

    def step(name, cmd):
        with open(out / f"{name}.log", "w") as log:
            rc = docker("exec", container, "bash", "-c", cmd,
                        stdout=log, stderr=subprocess.STDOUT, check=False).returncode
        with open(status, "a") as f:
            f.write(f"{name} {rc}\n")
        print(f"{name}: exit {rc}", flush=True)
        return rc

    container = f"correct-{task.id}-{out.name}-{os.getpid()}"
    docker("run", "-d", "--name", container, "--network", "none", "--gpus", f'"device={task.gpu}"',
           "--ipc=host", *mounts, "-w", "/workspace/flashinfer", task.image, "sleep", "infinity",
           stdout=subprocess.DEVNULL)
    try:
        if step("prepare", f"bash /eval/bin/prepare.sh {patch_arg} /out") == 0:
            # A failed copy shows up as a failing hidden_tests step.
            docker("exec", container, "bash", "-c", f"cp /eval/task/test_hidden_*.py {hidden_dir}/", check=False)
            step("visible_tests", f"pytest -q {task.visible_tests} --junitxml=/out/visible_tests.xml")
            step("hidden_tests", f"pytest -q {hidden_dir}/test_hidden_*.py --junitxml=/out/hidden_tests.xml")
            step("check_visible", "python /task/bench_task.py --check --repeat-ms 20 --json /out/check_visible.json")
            step("check_hidden", "python /task/bench_task.py --check --repeat-ms 20 "
                                 "--shapes /eval/task/hidden_shapes.json --json /out/check_hidden.json")
    finally:
        docker("rm", "-f", container, capture_output=True, check=False)

    print(f"== {status}\n{status.read_text()}== test outcomes", flush=True)
    counts = [sys.executable, "-m", "benchmark.test_counts", out, *(["--base", base] if base else [])]
    return subprocess.run(counts, cwd=ROOT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_id")
    ap.add_argument("patch", help="patch file, or none for the baseline")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("gpu", nargs="?", default="0")
    ap.add_argument("--base", type=Path, help="baseline result dir to compare every test against")
    a = ap.parse_args()
    sys.exit(run(load_task(a.task_id, a.gpu), a.patch, a.out_dir, a.base))


if __name__ == "__main__":
    main()
