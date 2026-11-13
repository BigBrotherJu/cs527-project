"""Nsight Compute report of one launch of the task's kernel on one shape.

Runs in the GPU worker for an agent's `profile` command:

    python /agent/bin/profile_task.py <select|full> <kernel_regex> [--shape NAME | --shapes FILE]
                                     [--section S ...] [--metrics LIST] [--set SET] [--save]

`select` passes the agent's --section/--metrics/--set to ncu (default: ncu's
basic set); `full` ignores them and collects `--set full`. The shape is one
entry of /task/shapes.json by name (gpu_tools.py passes the task's
PROFILE_SHAPE when the agent names none) or the first entry of the agent's own
shapes file. With --save the report is also written to
/scratch/profile-<n>.txt, where the agent can page through it.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "full"])
    ap.add_argument("kernel")
    ap.add_argument("--shape")
    ap.add_argument("--shapes", default="/task/shapes.json")
    ap.add_argument("--section", action="append", default=[])
    ap.add_argument("--metrics")
    ap.add_argument("--set", dest="metric_set")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    # bench_task.py runs every shape in its file, so give it a file of one.
    spec = json.loads(Path(a.shapes).read_text())
    shapes = [s for s in spec["shapes"] if a.shape in (None, s["name"])]
    if not shapes:
        sys.exit(f"no shape named {a.shape!r} in {a.shapes}; names: {[s['name'] for s in spec['shapes']]}")
    one = Path("/tmp/profile_shape.json")
    one.write_text(json.dumps({"defaults": spec.get("defaults", {}), "shapes": shapes[:1]}))

    what = ["--set", "full"]
    if a.mode == "select":
        what = ([f"--section={s}" for s in a.section] + (["--metrics", a.metrics] if a.metrics else [])
                + (["--set", a.metric_set] if a.metric_set else []))
    # The first matching launch; ncu flushes caches before it by default, like
    # the benchmark's cold-L2 timing.
    cmd = ["ncu", "-k", f"regex:{a.kernel}", "--launch-count", "1", *what,
           "python", "/task/bench_task.py", "--shapes", str(one), "--repeat-ms", "1"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    report = f"shape: {shapes[0]['name']}\n$ {' '.join(cmd)}\n\n{out.stdout}{out.stderr}"
    if a.save:
        n = len(list(Path("/scratch").glob("profile-*.txt"))) + 1
        path = Path(f"/scratch/profile-{n}.txt")
        path.write_text(report)
        report = f"(full report saved to {path})\n{report}"
    print(report)
    sys.exit(out.returncode)


if __name__ == "__main__":
    main()
