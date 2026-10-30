"""Summarize evaluated agent runs (eval_runs.sh): one row per run, then per arm.

    python3 scripts/summarize_runs.py <runs_dir> <eval_dir>

Per run: exit status, steps, tokens, wall time, which patch was scored,
correctness, and speedup over the baseline (geomean over shapes of the median
over rounds; min over hidden shapes too). Per arm: runs, correct runs, median
speedups of the correct runs, and fast_1, the fraction of runs that are correct
and faster than the baseline on the visible shapes. A run whose scored patch is
empty (tests passed only on unchanged code) is "no change": 1.00x, not counted
as correct. Prints Markdown; also writes <eval_dir>/runs.csv.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

from summarize_perf import load


def speedups(perf, arm, shape_set, devices):
    """Per-shape baseline/arm time ratios, or None if the arm was not timed."""
    base, mine = load(perf, "base", shape_set, devices), load(perf, arm, shape_set, devices)
    if not mine:
        return None
    return [statistics.median(base[s]) / statistics.median(mine[s]) for s in base if s in mine]


def correctness(eval_dir):
    status = eval_dir / "status.txt"
    if not status.exists():
        return "not evaluated"
    steps = dict(line.split() for line in status.read_text().splitlines() if line.strip())
    failed = [k for k, v in steps.items() if v != "0"]
    counts = json.loads((eval_dir / "test_counts.json").read_text()) if (eval_dir / "test_counts.json").exists() else {}
    differs = sum(len(v) for v in counts.get("differs_from_base", {}).values())
    if failed or differs:
        return "incorrect: " + ", ".join(failed + ([f"{differs} tests differ from base"] if differs else []))
    return "correct"


def fmt(x, spec=".2f"):
    return "-" if x is None else format(x, spec)


def main():
    runs_dir, eval_dir = Path(sys.argv[1]), Path(sys.argv[2])
    perf = eval_dir / "perf"
    devices = set()
    rows = []
    for name in ["ref"] + sorted(p.parent.name for p in runs_dir.glob("*/meta.json")):
        meta = json.loads((runs_dir / name / "meta.json").read_text()) if name != "ref" else {"arm": "(reference)"}
        vis, hid = (speedups(perf, name, s, devices) for s in ("visible", "hidden"))
        unchanged = name != "ref" and meta.get("scored_from") and \
            not (runs_dir / name / "scored.patch").read_text().strip()
        if unchanged:
            vis = hid = [1.0]
        tokens = meta.get("tokens", {})
        rows.append({
            "run": name, "arm": meta["arm"], "exit": meta.get("exit_status"), "steps": meta.get("steps"),
            "prompt_tokens": tokens.get("prompt_tokens"), "completion_tokens": tokens.get("completion_tokens"),
            "wall_min": round(meta["wall_seconds"] / 60, 1) if "wall_seconds" in meta else None,
            "scored": meta.get("scored_from", "ref.patch") if name != "ref" else "ref.patch",
            "correctness": "correct" if name == "ref" else "no change" if unchanged else
                           (correctness(eval_dir / name) if meta.get("scored_from") else "no passing tests"),
            "visible_x": statistics.geometric_mean(vis) if vis else None,
            "hidden_x": statistics.geometric_mean(hid) if hid else None,
            "hidden_min_x": min(hid) if hid else None,
        })
    if len(devices) > 1:
        sys.exit(f"perf results come from more than one device: {sorted(map(str, devices))}")

    with open(eval_dir / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"device: {devices.pop() if devices else 'unknown'}\n")
    print("| run | arm | exit | steps | tokens in/out | wall min | scored | correctness | visible x | hidden x | hidden min x |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        tok = f"{fmt(r['prompt_tokens'], ',')}/{fmt(r['completion_tokens'], ',')}"
        print(f"| {r['run']} | {r['arm']} | {r['exit'] or '-'} | {fmt(r['steps'], 'd')} | {tok} | "
              f"{fmt(r['wall_min'], '.1f')} | {r['scored'] or '-'} | {r['correctness']} | "
              f"{fmt(r['visible_x'])} | {fmt(r['hidden_x'])} | {fmt(r['hidden_min_x'])} |")

    print("\n| arm | runs | correct | median visible x | median hidden x | fast_1 |")
    print("|---|---|---|---|---|---|")
    for arm in dict.fromkeys(r["arm"] for r in rows if r["run"] != "ref"):
        mine = [r for r in rows if r["arm"] == arm]
        ok = [r for r in mine if r["correctness"] == "correct" and r["visible_x"] is not None]
        med = lambda k: statistics.median(r[k] for r in ok) if ok else None
        fast = sum(r["visible_x"] > 1 for r in ok) / len(mine)
        print(f"| {arm} | {len(mine)} | {len(ok)} | {fmt(med('visible_x'))} | {fmt(med('hidden_x'))} | {fast:.2f} |")


if __name__ == "__main__":
    main()
