#!/usr/bin/env python3
"""Table III: the two RGB-to-spectrum convention measurements.

Both parts come out of `colour_lab`, which links quantiloom_core and so uses
the renderer's own fitter, observer and illuminant rather than a reimplementation
that could drift from them.

    (a) LUT resolution sweep      what the coefficient table's resolution costs
                                  in CIELab, per resolution
    (b) illuminant round trip     what an authored emissive triple comes back as

A note on part (b) that belongs beside the numbers rather than in a commit
message. The table's third row -- the retired "flat spectrum + downstream white
balance" convention -- CANNOT be reproduced by this script, because the code
that implemented it was removed in 50a6cc6 (2026-08-20) and the flat-illuminant
path no longer exists. What that commit recorded, measured end to end on the
Cornell box before the removal, was 1 : 0.946 : 0.821, 3.57 % off.

Reconstructing the same convention host-side -- sigmoid chromaticity against a
flat illuminant, then the two retired correction constants -- gives 0.61 %, not
3.57 %. The reconstruction reproduces the constants exactly (E integrates to
linear sRGB 1.2703 : 1 : 0.9580, so the corrections are 0.7872 and 1.0439
against the recorded 0.7872 and 1.0437), so the disagreement is not in the
convention but in what was measured through: the recorded figure is a render of
the whole Cornell box through the 32-sample estimator, this is a host-side
integral of the emitter alone at 1 nm. Both are reported, each labelled with
what it is. Neither is quietly dropped.

Usage:
    e0_conventions.py                 # both parts
    e0_conventions.py --skip-sweep    # (b) only; the sweep refits three tables
"""

import argparse
import json
import pathlib
import subprocess
import sys

# The table this prints carries superscripts and a delta. On a Chinese-locale
# Windows shell stdout is GBK, which encodes neither, and the script would die
# on its own output rather than on anything it measured.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path(__file__).resolve().parents[2]
TOOL = REPO / "build" / "src" / "tools" / "Release" / "colour_lab.exe"
EVIDENCE = pathlib.Path(r"H:\quantiloom-paper\evidence\e0")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _winpaths import require_windows_paths  # noqa: E402
require_windows_paths(EVIDENCE)

# What 50a6cc6 recorded for the convention it retired, kept here so the row can
# be carried in the table with its provenance attached rather than as a number
# of unknown origin.
RETIRED = {
    "convention": "flat spectrum + downstream white balance",
    "ratio": [1.0, 0.946, 0.821],
    "deviation_pct": 3.57,
    "source": "commit 50a6cc6 (2026-08-20), measured end to end on the Cornell "
              "box before the code was removed",
    "reproducible_here": False,
}


def run(*args):
    """colour_lab writes JSON to stdout and progress to stderr."""
    if not TOOL.is_file():
        raise SystemExit(f"{TOOL} missing; cmake --build build --config Release "
                         f"--target colour_lab")
    # encoding explicitly, not text=True: that decodes with the locale codec,
    # and this repository's tools have hit that before.
    proc = subprocess.run([str(TOOL), *args], capture_output=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit(f"colour_lab {' '.join(args)} failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def sweep_markdown(data):
    lines = ["| Resolution | Coefficient memory | Mean ΔE | p99 ΔE | Worst ΔE |",
             "|---:|---:|---:|---:|---:|"]
    for row in data["rows"]:
        chosen = " (chosen)" if row["resolution"] == 64 else ""
        lines.append(f"| {row['resolution']}³{chosen} | {row['coefficient_mb']:.1f} MB "
                     f"| {row['mean_dE']:.3f} | {row['p99_dE']:.3f} | {row['worst_dE']:.2f} |")
    return "\n".join(lines)


def illuminant_markdown(data):
    lines = ["| Convention | Round-trip ratio (R:G:B) | Deviation |",
             "|---|---|---:|"]
    for row in data["rows"]:
        r = row["ratio"]
        dev = "—" if row["deviation_pct"] is None else f"{row['deviation_pct']:.2f} %"
        lines.append(f"| {row['convention']} | 1 : {r[1]:.3f} : {r[2]:.3f} | {dev} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-sweep", action="store_true",
                        help="(b) only -- the sweep refits three tables")
    parser.add_argument("--samples", type=int, default=None,
                        help="override the tool's own default sample count")
    args = parser.parse_args()

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    result = {}

    if not args.skip_sweep:
        extra = ["--samples", str(args.samples)] if args.samples else []
        sweep = run("--lut-sweep", *extra)
        result["lut_sweep"] = sweep
        print(sweep_markdown(sweep))
        print()

    illuminant = run("--illuminant")
    # Carry the retired row alongside the two that reproduce, so the table can
    # be assembled from this file without anyone having to find the commit.
    illuminant["retired_convention_as_recorded"] = RETIRED
    result["illuminant"] = illuminant
    print(illuminant_markdown(illuminant))
    print(f"\nas recorded in {RETIRED['source'].split(',')[0]}: "
          f"1 : {RETIRED['ratio'][1]:.3f} : {RETIRED['ratio'][2]:.3f}, "
          f"{RETIRED['deviation_pct']:.2f} % off")

    (EVIDENCE / "table_iii.json").write_text(json.dumps(result, indent=2),
                                             encoding="utf-8")
    markdown = []
    if "lut_sweep" in result:
        markdown += ["(a) LUT resolution sweep", "", sweep_markdown(result["lut_sweep"]), ""]
    markdown += ["(b) Illuminant convention comparison", "",
                 illuminant_markdown(result["illuminant"]), ""]
    (EVIDENCE / "table_iii.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"\nwrote {EVIDENCE / 'table_iii.json'}")


if __name__ == "__main__":
    main()
