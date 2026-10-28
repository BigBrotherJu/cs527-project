"""Summarize eval_perf.sh output: per shape, the median over rounds of each arm's
per-round median time, the spread across rounds, and the speedup of every arm
over the first one (the baseline).

    python3 scripts/summarize_perf.py <out_dir> <baseline_arm> <arm> ...

Timings are only comparable on one device, so it refuses an out_dir whose
result files come from more than one. It also reports the clocks sampled during
the rounds (clocks_log.csv) and warns about anything that makes them suspect.
"""

import json
import statistics
import sys
from pathlib import Path


# Clock event reason bits (NVML) that mean the GPU ran below its clock target.
THROTTLE_BITS = {0x4: "SW power cap", 0x8: "HW slowdown", 0x20: "SW thermal",
                 0x40: "HW thermal", 0x80: "HW power brake"}
GPU_IDLE = 0x1


def clock_report(out):
    """Summarize clocks_log.csv (timestamp, SM, mem, max mem, power, temp, reasons)."""
    log, cfg = out / "clocks_log.csv", out / "clocks.txt"
    if not log.exists():
        return
    lock = dict(l.split("=", 1) for l in cfg.read_text().split()).get("sm_lock_mhz", "none")
    rows = []
    for line in log.read_text().splitlines():
        f = [x.strip() for x in line.split(",")]
        try:
            rows.append((float(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5]), int(f[6], 16)))
        except (IndexError, ValueError):
            continue  # [N/A] fields or a partial last line
    if not rows:
        print("clocks: no samples")
        return
    busy = [r for r in rows if not r[5] & GPU_IDLE] or rows
    sm = sorted(r[0] for r in busy)
    print(f"clocks: SM lock {lock} MHz; {len(rows)} samples; busy SM min {sm[0]:.0f} median "
          f"{statistics.median(sm):.0f} MHz; mem {min(r[1] for r in rows):.0f}-{max(r[1] for r in rows):.0f} "
          f"of max {rows[0][2]:.0f} MHz; power max {max(r[3] for r in rows):.0f} W; "
          f"temp max {max(r[4] for r in rows):.0f} C")
    reasons = {name for r in rows for bit, name in THROTTLE_BITS.items() if r[5] & bit}
    if reasons:
        print(f"WARNING: throttled during the rounds: {', '.join(sorted(reasons))}")
    if lock != "none" and sm[0] < 0.99 * float(lock):
        print(f"WARNING: busy SM clock fell to {sm[0]:.0f} MHz, below the {lock} MHz lock")
    if any(r[1] < r[2] for r in rows):
        print("WARNING: memory clock was below its max during the rounds")


def load(out, arm, shape_set, devices):
    per_shape = {}
    for f in sorted((out / arm).glob(f"r*_{shape_set}.json")):
        data = json.loads(f.read_text())
        devices.add(data.get("device"))
        for r in data["results"]:
            per_shape.setdefault(r["name"], []).append(r["median_us"])
    return per_shape


def main():
    out = Path(sys.argv[1])
    arms = sys.argv[2:]
    base = arms[0]
    devices = set()
    data_by_set = {
        s: {a: load(out, a, s, devices) for a in arms} for s in ("visible", "hidden")
    }
    if len(devices) > 1:
        sys.exit(f"results come from more than one device: {sorted(map(str, devices))}")
    print(f"device: {devices.pop() if devices else 'unknown'}")
    clock_report(out)
    for shape_set, data in data_by_set.items():
        if not data[base]:
            continue
        print(f"\n[{shape_set}]  median over rounds, us (spread = (max-min)/median over rounds)")
        head = f"{'shape':<26}" + "".join(f"{a:>12}{'spread':>8}" for a in arms)
        head += "".join(f"{a + '/x':>10}" for a in arms[1:])
        print(head)
        geo = {a: [] for a in arms[1:]}
        for shape, base_rounds in data[base].items():
            med = {}
            line = f"{shape:<26}"
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
