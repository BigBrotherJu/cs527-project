"""Count test outcomes in an eval_correctness.sh result dir and, optionally,
compare them per test against a baseline result dir from the same architecture.

    python3 scripts/test_counts.py <out_dir> [--base <base_out_dir>]

Reads <out_dir>/{visible,hidden}_tests.xml (JUnit) and writes
<out_dir>/test_counts.json. Tests skip themselves on GPUs they do not support,
so a candidate "passes" only together with the right skip set: with --base,
every test whose outcome differs from the baseline is listed (passed there and
skipped here is how a pass-by-skip shows up) and the exit code is 1 if any did.
The comparison refuses result dirs from different devices.
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SUITES = ("visible", "hidden")


def outcomes(xml_path):
    """{test id: passed|failed|error|skipped} for one JUnit file."""
    result = {}
    if not xml_path.exists():
        return result
    for case in ET.parse(xml_path).iter("testcase"):
        test_id = f"{case.get('classname')}::{case.get('name')}"
        if case.find("failure") is not None:
            result[test_id] = "failed"
        elif case.find("error") is not None:
            result[test_id] = "error"
        elif case.find("skipped") is not None:
            result[test_id] = "skipped"
        else:
            result[test_id] = "passed"
    return result


def device(out):
    f = out / "check_visible.json"
    return json.loads(f.read_text()).get("device") if f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--base", type=Path)
    args = ap.parse_args()

    report = {"device": device(args.out), "suites": {}}
    per_suite = {}
    for suite in SUITES:
        per_suite[suite] = outcomes(args.out / f"{suite}_tests.xml")
        counts = {k: 0 for k in ("passed", "failed", "error", "skipped")}
        for o in per_suite[suite].values():
            counts[o] += 1
        report["suites"][suite] = counts
        print(f"{suite:<8} " + "  ".join(f"{k} {v}" for k, v in counts.items()))

    status = 0
    if args.base is not None:
        base_device = device(args.base)
        if base_device != report["device"]:
            print(f"cannot compare: device {report['device']!r} vs base {base_device!r}",
                  file=sys.stderr)
            (args.out / "test_counts.json").write_text(json.dumps(report, indent=1) + "\n")
            return 2
        report["base"] = str(args.base)
        report["differs_from_base"] = {}
        for suite in SUITES:
            base = outcomes(args.base / f"{suite}_tests.xml")
            cur = per_suite[suite]
            diff = {
                t: {"base": base.get(t, "absent"), "this": cur.get(t, "absent")}
                for t in sorted(base.keys() | cur.keys())
                if base.get(t) != cur.get(t)
            }
            report["differs_from_base"][suite] = diff
            for t, d in diff.items():
                print(f"  {suite}: {t}: {d['base']} in base, {d['this']} here")
            if diff:
                status = 1
        if status == 0:
            print("same outcome as base for every test")

    (args.out / "test_counts.json").write_text(json.dumps(report, indent=1) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
