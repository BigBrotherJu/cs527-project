"""Interleaved timing of two or more arms (baseline, reference, agent patches) on
one GPU. Each arm gets its own offline container, built and warmed once
untimed; each round then runs bench_task.py on the visible and hidden shapes
for every arm, rotating the arm order. Results: <out_dir>/<name>/r<k>_{visible,
hidden}.json and summary.txt (summarize_perf.py).

    python3 scripts/eval_perf.py <task_id> <out_dir> <gpu_index> <rounds> <name>=<patch|none> ...

The SM clock is locked (-lgc) for the rounds and reset afterwards, by default
at the GPU's default applications clock. The memory clock cannot be fixed at
run time on H100 (-lmc is unsupported and exits 0 anyway; -lmcd needs a driver
reload; -ac changes only the reported applications clock, measured: HBM stays
at 2619 MHz under load either way), so it is checked against the default
applications memory clock instead. clocks_log.csv samples clocks, power,
temperature and throttle reasons every 500 ms during the rounds; the summary
flags power or thermal throttling and SM or memory clocks below their targets.
"""

import argparse
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

from task import ROOT, docker, eval_mounts, gpu_lock, load_task


def query(gpu, field):
    return subprocess.run(["nvidia-smi", "-i", gpu, f"--query-gpu={field}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()


def run(task, out, rounds, arms, *, lock_clocks=True, sm_clock=None, mem_clock=None, nvsmi="sudo nvidia-smi"):
    """Time <arms>, a list of (name, patch|"none"), into <out>."""
    if len(arms) < 2:
        raise SystemExit("need at least two arms")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    nvsmi = shlex.split(nvsmi)
    # A kill (SIGTERM) runs the cleanup below too, so the clocks get reset.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    names = [name for name, _ in arms]
    containers, sampler, locked = [], None, False
    try:
        for name, patch in arms:
            c = f"perf-{task.id}-{name}-{os.getpid()}"
            mounts, patch_arg = eval_mounts(task, out, patch)
            (out / name).mkdir(exist_ok=True)
            docker("run", "-d", "--name", c, "--network", "none", "--gpus", f'"device={task.gpu}"',
                   "--ipc=host", *mounts, "-w", "/workspace/flashinfer", task.image, "sleep", "infinity",
                   stdout=subprocess.DEVNULL)
            containers.append(c)
            print(f"== {name}: prepare + build", flush=True)
            docker("exec", c, "bash", "/eval/bin/prepare.sh", patch_arg, f"/out/{name}")
            # One untimed pass compiles the module and warms the GPU for this arm.
            with open(out / name / "warmup.log", "w") as log:
                docker("exec", c, "python", "/task/bench_task.py", "--repeat-ms", "20",
                       stdout=log, stderr=subprocess.STDOUT)

        # Hold the GPU for the timed rounds: agents' tools (gpu_tools.py) take
        # the same lock, so their GPU work waits instead of overlapping the timing.
        with gpu_lock(task.gpu):
            # Lock only around the timed rounds: the per-arm builds above are CPU-bound.
            if lock_clocks:
                sm_clock = sm_clock or query(task.gpu, "clocks.default_applications.graphics")
                mem_clock = mem_clock or query(task.gpu, "clocks.default_applications.memory")
                if not (str(sm_clock).isdigit() and str(mem_clock).isdigit()):
                    raise SystemExit(f"no default applications clocks ({sm_clock}, {mem_clock}); "
                                     "pass --sm-clock and --mem-clock")
                locked = True
                subprocess.run([*nvsmi, "-i", task.gpu, "-lgc", f"{sm_clock},{sm_clock}"],
                               stdout=subprocess.DEVNULL, check=True)
                print(f"locked GPU {task.gpu} SM clock at {sm_clock} MHz; "
                      f"expecting memory at {mem_clock} MHz", flush=True)
                (out / "clocks.txt").write_text(f"sm_lock_mhz={sm_clock}\nmem_lock_mhz={mem_clock}\n")
            else:
                print(f"GPU {task.gpu} clocks NOT locked (--no-lock-clocks)", flush=True)
                (out / "clocks.txt").write_text("sm_lock_mhz=none\nmem_lock_mhz=none\n")

            # Newer drivers renamed clocks_throttle_reasons to clocks_event_reasons.
            reasons = "clocks_event_reasons.active"
            if subprocess.run(["nvidia-smi", "-i", task.gpu, f"--query-gpu={reasons}", "--format=csv,noheader"],
                              capture_output=True).returncode:
                reasons = "clocks_throttle_reasons.active"
            with open(out / "clocks_log.csv", "w") as log:
                sampler = subprocess.Popen(
                    ["nvidia-smi", "-i", task.gpu, "-lms", "500", "--format=csv,noheader,nounits",
                     "--query-gpu=timestamp,clocks.sm,clocks.mem,clocks.max.memory,power.draw,"
                     f"temperature.gpu,{reasons}"], stdout=log)

            for r in range(1, rounds + 1):
                for i in range(len(arms)):
                    k = (i + r - 1) % len(arms)
                    name, c = names[k], containers[k]
                    print(f"== round {r}/{rounds}: {name}", flush=True)
                    docker("exec", c, "python", "/task/bench_task.py", "--json", f"/out/{name}/r{r}_visible.json",
                           stdout=subprocess.DEVNULL)
                    docker("exec", c, "python", "/task/bench_task.py", "--shapes", "/eval/task/hidden_shapes.json",
                           "--json", f"/out/{name}/r{r}_hidden.json", stdout=subprocess.DEVNULL)
    finally:
        if sampler:
            sampler.terminate()
            sampler.wait()
        for c in containers:
            docker("rm", "-f", c, capture_output=True, check=False)
        if locked and subprocess.run([*nvsmi, "-i", task.gpu, "-rgc"], stdout=subprocess.DEVNULL).returncode:
            print(f"WARNING: failed to reset GPU {task.gpu} clocks", file=sys.stderr)

    summary = subprocess.run([sys.executable, ROOT / "scripts" / "summarize_perf.py", out, *names],
                             capture_output=True, text=True)
    print(summary.stdout, end="", flush=True)
    (out / "summary.txt").write_text(summary.stdout)
    if summary.returncode:
        raise SystemExit(summary.stderr)


def add_clock_args(ap):
    ap.add_argument("--sm-clock", help="SM clock (MHz) to lock to (default: the default applications clock)")
    ap.add_argument("--mem-clock", help="expected memory clock (MHz) (default: the default applications clock)")
    ap.add_argument("--lock-clocks", action=argparse.BooleanOptionalAction, default=True,
                    help="lock the SM clock for the rounds (--no-lock-clocks e.g. without root)")
    ap.add_argument("--nvsmi", default="sudo nvidia-smi", help="command that locks and resets clocks")


def clock_kwargs(a):
    return {"lock_clocks": a.lock_clocks, "sm_clock": a.sm_clock, "mem_clock": a.mem_clock, "nvsmi": a.nvsmi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_id")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("gpu")
    ap.add_argument("rounds", type=int)
    ap.add_argument("arms", nargs="+", metavar="name=patch", help="patch file, or none for the baseline")
    add_clock_args(ap)
    a = ap.parse_args()
    if any("=" not in arm for arm in a.arms):
        ap.error("arms are name=patch (or name=none)")
    arms = [tuple(arm.split("=", 1)) for arm in a.arms]
    run(load_task(a.task_id, a.gpu), a.out_dir, a.rounds, arms, **clock_kwargs(a))


if __name__ == "__main__":
    main()
