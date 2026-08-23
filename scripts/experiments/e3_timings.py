#!/usr/bin/env python3
"""Run the interactive-latency benchmark and turn its output into Table VII.

The measurements live in tests/test_renderer/test_interactive_bench.cpp, all
DISABLED_ so that no build gate depends on how busy the GPU was. This script
runs them, parses the "  [BENCH] ..." lines, records the machine they were
taken on, and writes the table the paper's Section VIII-G quotes.

Usage:
    e3_timings.py                 # every case
    e3_timings.py --filter T1     # geometry only
"""

import argparse
import json
import pathlib
import platform
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
BINARY = REPO / "build" / "tests" / "Release" / "libquantiloom_tests.exe"
EVIDENCE = pathlib.Path(r"H:\quantiloom-paper\evidence\e3")

# One space, not two, before `median`: Report() pads the label to 34 characters
# and several labels are longer than that, so the columns close up and a
# two-space rule silently drops exactly the zero-step control rows that the
# stride sweep exists to be compared against.
BENCH = re.compile(
    r"\[BENCH\]\s+(?P<name>.+?)\s+median\s+(?P<median>[\d.]+) ms\s+"
    r"p95\s+(?P<p95>[\d.]+) ms\s+min\s+(?P<min>[\d.]+)\s+max\s+(?P<max>[\d.]+)\s+"
    r"n=(?P<n>\d+)(?:\s{2,}(?P<note>.+))?$")

# Scene open reports once rather than as a distribution, and no target names it.
SETUP = re.compile(r"\[BENCH\]\s+(?P<name>.+?)\s+once\s+(?P<ms>[\d.]+) ms\s+"
                   r"\((?P<note>.+)\)$")

# The replayed-step counts carry no timing, so they do not match either row
# pattern; they are the independent variable panel (b) of Fig. 9 plots against.
REPLAYED = re.compile(r"\[BENCH\]\s+(?P<name>.+?)\s+replayed\s+(?P<steps>[\d.]+) steps")

# The targets Section VIII-G commits to before any of this is measured.
TARGETS_MS = {
    "T1 refit -> first frame": 500.0,
    "T1 rebuild -> first frame": 500.0,
    "T2 backward seek": 2000.0,
}


def target_for(name):
    for prefix, limit in TARGETS_MS.items():
        if name.startswith(prefix):
            return limit
    return None


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filter", default="InteractiveBench*",
                        help="gtest filter; a bare 'T1'/'T2' is expanded")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--from-stdout", type=pathlib.Path,
                        help="re-parse a saved bench_stdout.txt instead of running; "
                             "the suite takes ~24 min, so a parser fix must not "
                             "cost a re-measurement")
    args = parser.parse_args()

    if args.from_stdout:
        stdout = args.from_stdout.read_text(encoding="utf-8", errors="replace")
        gtest_filter = f"(re-parsed from {args.from_stdout.name})"
    else:
        if not BINARY.is_file():
            raise SystemExit(f"{BINARY} not built")

        gtest_filter = args.filter
        if gtest_filter in ("T1", "T2"):
            gtest_filter = {"T1": "InteractiveBench.DISABLED_Geometry*",
                            "T2": "InteractiveBench.DISABLED_Thermal*"}[gtest_filter]

        print(f"running {gtest_filter} ...", flush=True)
        result = subprocess.run(
            [str(BINARY), "--gtest_also_run_disabled_tests",
             f"--gtest_filter={gtest_filter}"],
            cwd=REPO, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=args.timeout)
        print(result.stdout[-3000:], flush=True)
        stdout = result.stdout

    setup, replayed = [], {}
    for line in stdout.splitlines():
        match = SETUP.search(line.rstrip())
        if match:
            setup.append({"name": match.group("name").strip(),
                          "ms": float(match.group("ms")),
                          "note": match.group("note")})
        match = REPLAYED.search(line.rstrip())
        if match:
            replayed[match.group("name").strip()] = float(match.group("steps"))

    rows = []
    for line in stdout.splitlines():
        match = BENCH.search(line.rstrip())
        if not match:
            continue
        row = {k: match.group(k) for k in
               ("name", "median", "p95", "min", "max", "n", "note")}
        row["name"] = row["name"].strip()
        for key in ("median", "p95", "min", "max"):
            row[key] = float(row[key])
        row["n"] = int(row["n"])
        row["target_ms"] = target_for(row["name"])
        # Reported as a factor rather than a tick, so a comfortable pass and a
        # narrow one do not read the same.
        row["margin"] = (row["target_ms"] / row["p95"]) if row["target_ms"] else None
        rows.append(row)

    if not rows:
        raise SystemExit("no [BENCH] lines parsed; the cases may have skipped "
                         "(no ray-tracing GPU) or the scene may be missing")

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": {"gpu": gpu_name(), "platform": platform.platform(),
                 "processor": platform.processor()},
        "gtest_filter": gtest_filter,
        "rows": rows,
        "setup": setup,
        "replayed_steps": replayed,
    }
    (EVIDENCE / "timings.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.from_stdout:
        (EVIDENCE / "bench_stdout.txt").write_text(stdout, encoding="utf-8")

    lines = ["| Measurement | Median (ms) | p95 (ms) | n | Target (ms) | Margin |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        target = f"{row['target_ms']:.0f}" if row["target_ms"] else "\u2014"
        margin = f"{row['margin']:.0f}\u00d7" if row["margin"] else "\u2014"
        name = row["name"] + (f" ({row['note']})" if row["note"] else "")
        lines.append(f"| {name} | {row['median']:.2f} | {row['p95']:.2f} | "
                     f"{row['n']} | {target} | {margin} |")
    table = "\n".join(lines)
    (EVIDENCE / "table7_interactive.md").write_text(table + "\n", encoding="utf-8")

    print("\n" + table)
    missed = [r for r in rows if r["target_ms"] and r["p95"] > r["target_ms"]]
    print(f"\n{len(rows)} measurements, {len(missed)} over target")
    for row in missed:
        print(f"  MISSED: {row['name']} p95 {row['p95']:.1f} ms > {row['target_ms']:.0f} ms")


if __name__ == "__main__":
    main()
