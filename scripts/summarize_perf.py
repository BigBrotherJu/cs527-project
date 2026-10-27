"""Summarize eval_perf.sh output: per shape, the median over rounds of each arm's
per-round median time, the spread across rounds, and the speedup of every arm
over the first one (the baseline).

    python3 scripts/summarize_perf.py <out_dir> <baseline_arm> <arm> ...
"""

import json
import statistics
import sys
from pathlib import Path


def load(out, arm, shape_set):
    per_shape = {}
    for f in sorted((out / arm).glob(f"r*_{shape_set}.json")):
        for r in json.loads(f.read_text())["results"]:
            per_shape.setdefault(r["name"], []).append(r["median_us"])
    return per_shape


def main():
    out = Path(sys.argv[1])
    arms = sys.argv[2:]
    base = arms[0]
    for shape_set in ("visible", "hidden"):
        data = {a: load(out, a, shape_set) for a in arms}
        if not data[base]:
            continue
        print(f"\n[{shape_set}]  median over rounds, us (spread = (max-min)/median over rounds)")
        head = f"{'shape':<24}" + "".join(f"{a:>12}{'spread':>8}" for a in arms)
        head += "".join(f"{a + '/x':>10}" for a in arms[1:])
        print(head)
        geo = {a: [] for a in arms[1:]}
        for shape, base_rounds in data[base].items():
            med = {}
            line = f"{shape:<24}"
            for a in arms:
                rounds = data[a].get(shape, [])
                if not rounds:
                    line += f"{'-':>12}{'':>8}"
                    continue
                med[a] = statistics.median(rounds)
                spread = (max(rounds) - min(rounds)) / med[a]
                line += f"{med[a]:>12.2f}{spread:>7.1%} "
            for a in arms[1:]:
                if a in med:
                    sp = med[base] / med[a]
                    geo[a].append(sp)
                    line += f"{sp:>9.2f}x"
                else:
                    line += f"{'-':>10}"
            print(line)
        for a, sps in geo.items():
            if sps:
                g = statistics.geometric_mean(sps)
                print(f"speedup of {a} over {base}: geomean {g:.2f}x, min {min(sps):.2f}x, max {max(sps):.2f}x")


if __name__ == "__main__":
    main()
