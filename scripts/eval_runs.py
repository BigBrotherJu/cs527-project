"""Evaluate every agent run of a task: correctness of each run's scored.patch
against the baseline, then one interleaved perf eval of the baseline, the
reference and every correct run, then a summary table.

    python3 scripts/eval_runs.py <task_id> <runs_dir> <eval_dir> [gpu_index] [rounds] [clock options, see eval_perf.py]
    e.g. python3 scripts/eval_runs.py c3f7334d4 ~/runs/c3f7334d4 ~/evals/c3f7334d4 0 5

<runs_dir> holds run_agent.py outputs (<arm>-<k>/meta.json, scored.patch).
<eval_dir> gets base/ and <arm>-<k>/ (eval_correctness.py), perf/
(eval_perf.py), and summary.md / runs.csv (summarize_runs.py). Runs already
evaluated are kept, so the script can be restarted; perf is always rerun.
A run is correct when every step in its status.txt passed and every test has
the same outcome as in the baseline.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import eval_correctness
import eval_perf
from task import ROOT, load_task


def is_correct(result):
    """Whether an eval_correctness.py result dir shows a correct patch."""
    status, counts = result / "status.txt", result / "test_counts.json"
    if not status.exists() or not counts.exists():
        return False
    if any(line.split()[1] != "0" for line in status.read_text().splitlines()):
        return False
    return not any(json.loads(counts.read_text()).get("differs_from_base", {}).values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_id")
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("gpu", nargs="?", default="0")
    ap.add_argument("rounds", nargs="?", type=int, default=5)
    eval_perf.add_clock_args(ap)
    a = ap.parse_args()
    task = load_task(a.task_id, a.gpu)
    eval_dir = a.eval_dir.resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)

    base = eval_dir / "base"
    if not (base / "status.txt").exists():
        eval_correctness.run(task, "none", base)

    arms = [("base", "none"), ("ref", str(task.dir / "eval" / "ref.patch"))]
    for meta in sorted(a.runs_dir.glob("*/meta.json")):
        run = meta.parent
        patch = run / "scored.patch"
        if not patch.exists() or patch.stat().st_size == 0:
            print(f"== {run.name}: empty scored patch (tests never passed, or passed only on unchanged code), "
                  "skipping", flush=True)
            continue
        result = eval_dir / run.name
        if not (result / "status.txt").exists():
            print(f"== {run.name}: correctness", flush=True)
            eval_correctness.run(task, patch, result, base)
        if is_correct(result):
            arms.append((run.name, str(patch)))
        else:
            print(f"== {run.name}: incorrect, left out of perf", flush=True)

    print(f"== perf: {len(arms)} arms", flush=True)
    shutil.rmtree(eval_dir / "perf", ignore_errors=True)
    eval_perf.run(task, eval_dir / "perf", a.rounds, arms, **eval_perf.clock_kwargs(a))

    summary = subprocess.run([sys.executable, ROOT / "scripts" / "summarize_runs.py", a.runs_dir, eval_dir],
                             capture_output=True, text=True)
    print(summary.stdout, end="")
    (eval_dir / "summary.md").write_text(summary.stdout)
    if summary.returncode:
        raise SystemExit(summary.stderr)


if __name__ == "__main__":
    main()
