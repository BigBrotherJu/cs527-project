"""Run <runs_per_arm> agent attempts per arm on one GPU, --parallel at a time,
started in alternating arm order (correctness-1, timing-1, ..., correctness-2,
...) so no arm runs all of its attempts first. Agents share the GPU through the
tools' per-GPU lock, so their GPU work never overlaps. Runs land in
<out_root>/<task_id>/<arm>-<k> (log: <arm>-<k>.log beside it); a run that
already has meta.json is skipped, so the script can be restarted.

    python3 -m agent.run_agents <task_id> <gpu_index> <runs_per_arm> [out_root] [options] [-- run_agent.py options]
    e.g. ~/mswea/bin/python -m agent.run_agents c3f7334d4 0 3 ~/runs --parallel 3 -- --model openai/gpt-6.1-sol

Each agent builds with MAX_JOBS = cores / --parallel (one bgmv build averages
~4 cores).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmark.task import ROOT

ARMS = ["correctness", "timing", "ncu-select", "ncu-full", "ncu-summary"]  # gpu_tools.ARMS


def main():
    argv = sys.argv[1:]
    passthrough = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(usage=__doc__.split("\n\n")[1])
    ap.add_argument("task_id")
    ap.add_argument("gpu")
    ap.add_argument("runs", type=int)
    ap.add_argument("out_root", nargs="?", type=Path, default=Path.home() / "runs")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS, help="arms to run, in this order within each round")
    ap.add_argument("--parallel", type=int, default=1, help="agents at a time")
    ap.add_argument("--stagger", type=float, default=10, help="seconds between starts (volume copy-up)")
    ap.add_argument("--python", default=sys.executable, help="interpreter with mini-swe-agent installed")
    a = ap.parse_args(argv)
    out_dir = a.out_root.resolve() / a.task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    max_jobs = max(2, os.cpu_count() // a.parallel)

    running = {}  # name -> process

    def reap(until_below):
        while True:
            for name, p in list(running.items()):
                if p.poll() is not None:
                    print(f"== {name}: done (exit {p.returncode})", flush=True)
                    del running[name]
            if len(running) < until_below:
                return
            time.sleep(2)

    for k in range(1, a.runs + 1):
        for arm in a.arms:
            name = f"{arm}-{k}"
            out = out_dir / name
            if (out / "meta.json").exists():
                print(f"== {name}: done earlier, skipping", flush=True)
                continue
            if out.exists():
                print(f"== {name}: {out} exists without meta.json (crashed?), skipping", file=sys.stderr)
                continue
            reap(a.parallel)
            print(f"== {name}: start -> {out}", flush=True)
            with open(out_dir / f"{name}.log", "w") as log:
                running[name] = subprocess.Popen(
                    [a.python, "-m", "agent.run_agent", a.task_id, arm, out, "--gpu", a.gpu,
                     "--max-jobs", str(max_jobs), *passthrough], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            time.sleep(a.stagger)
    reap(1)


if __name__ == "__main__":
    main()
