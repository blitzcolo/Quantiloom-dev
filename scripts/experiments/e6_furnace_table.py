#!/usr/bin/env python3
"""E6 -- the furnace table, one row per cavity.

The furnace gate answers a yes/no question: does every isothermal cavity
render its own blackbody to within 0.2 per cent.  A table has to answer a
different one -- by how much, for each of them -- and the gate prints
"0.0000%" for five of the eight, which is a statement about the format.

This renders the same eight configurations, records the ratio at full width,
and reports the properties each cavity actually had.  "Actually" is the point:
the emissivity a cavity is solved and shaded with comes from the material as
the renderer resolved it, which is not always what the asset file says -- the
rho1 cavity's glTF names no emissivity curve at all -- so the value is read
out of the render log rather than out of the scene.

Usage:
    e6_furnace_table.py --out-dir evidence/e6
    e6_furnace_table.py --out-dir evidence/e6 --skip-render   # reuse EXRs
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from runlog import Run, REPO, DEFAULT_CLI  # noqa: E402

CHECKER = REPO / "scripts" / "render-tests" / "check_furnace.py"

# The gate's own list, in the gate's own order. Kept here rather than globbed so
# that a cavity added to the directory does not silently join the paper's table.
CASES = [
    ("lwir", "e1"), ("lwir", "e05"), ("lwir", "rho1"),
    ("lwir", "spectral"), ("lwir", "specular"),
    ("mwir", "e1"), ("mwir", "e05"), ("mwir", "rho1"),
]

# check_furnace.py's defaults, which the gate does not override. Recorded here
# because a table quoting a tolerance has to quote the one that was applied.
TOLERANCE = 0.002
CAVITY_TEMPERATURE_K = 300.0

CONFIG_KEYS = {
    "mode": re.compile(r'^\s*mode\s*=\s*"([^"]+)"', re.M),
    "spp": re.compile(r"^\s*spp\s*=\s*(\d+)", re.M),
    "resolution": re.compile(r"^\s*resolution\s*=\s*\[\s*(\d+)\s*,\s*(\d+)", re.M),
    "temperature_k": re.compile(r"^\s*default_temperature_k\s*=\s*([\d.]+)", re.M),
    "output": re.compile(r'^\s*output\s*=\s*"([^"]+)"', re.M),
    "spectral_ref": re.compile(r'^\s*spectral_material_ref\s*=\s*"([^"]+)"', re.M),
}

LOG_IR = re.compile(r"Material '([^']+)': IR override e=([\d.]+) t=([\d.]+) r=([\d.]+)")
LOG_CURVE = re.compile(r"Loaded emissivity curve: (\S+) \(")
LOG_GLTF = re.compile(r"Loading glTF model: (\S+)")
LOG_SPECTRAL = re.compile(
    r"Processing Quantiloom material: '([^']+)' -> type='([^']+)', name='([^']+)'")
CHECK_FIELDS = {
    "reference": re.compile(r"Reference:.*=\s*([\d.eE+-]+)"),
    "roi_mean": re.compile(r"ROI mean:\s*([\d.eE+-]+)"),
    "rel_error": re.compile(r"Rel error:\s*([\d.]+)%"),
    "ratio": re.compile(r"Ratio:\s*([\d.]+)"),
}


def read_config(path):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    out = {}
    for key, pattern in CONFIG_KEYS.items():
        match = pattern.search(text)
        if not match:
            continue
        out[key] = ("x".join(match.groups()) if key == "resolution"
                    else match.group(1))
    return out


def wall_emissivity(log_text):
    """What the wall actually radiates with, in the form the render used it.

    Three forms, and the order between them matters. A cavity bound to a
    measured spectrum is shaded with epsilon(lambda) = 1 - rho(lambda) on the
    GPU, and its scalar override reads e=0.000 -- so reporting that scalar
    would put "0.00" in the table for the one pair of cavities whose whole
    point is a real spectrum. The spectral binding therefore wins, and the
    scalar is only quoted when nothing spectral was bound.
    """
    spectral = LOG_SPECTRAL.search(log_text)
    if spectral:
        return {"kind": "spectral", "material": spectral.group(1),
                "library": spectral.group(2), "reference": spectral.group(3)}

    curve = LOG_CURVE.search(log_text)
    if curve:
        model = LOG_GLTF.search(log_text)
        base = pathlib.Path(model.group(1)).parent if model else REPO
        path = base / curve.group(1)
        values = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = [p.strip() for p in line.replace(",", " ").split()]
                if len(parts) >= 2:
                    values.append(float(parts[1]))
        if values:
            constant = max(values) - min(values) < 1e-9
            return {"kind": "curve", "file": curve.group(1),
                    "epsilon": values[0] if constant else None,
                    "min": min(values), "max": max(values), "constant": constant}

    override = LOG_IR.search(log_text)
    if override:
        return {"kind": "scalar", "material": override.group(1),
                "epsilon": float(override.group(2)),
                "tau": float(override.group(3)), "rho": float(override.group(4))}
    return {}


def describe_emissivity(entry, config_settings):
    if entry.get("kind") == "spectral":
        return f"measured, {entry['reference']}"
    if entry.get("kind") in ("curve", "scalar") and entry.get("epsilon") is not None:
        return f"{entry['epsilon']:.2f}"
    if entry.get("kind") == "curve":
        return f"{entry['min']:.2f}-{entry['max']:.2f}"
    return config_settings.get("spectral_ref", "n/a")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--cli", type=pathlib.Path, default=DEFAULT_CLI)
    parser.add_argument("--skip-render", action="store_true",
                        help="reuse the EXRs already in the repository root")
    args = parser.parse_args()

    rows = []
    with Run("e6_furnace", args.out_dir, cli=args.cli) as run:
        run.note(tolerance=TOLERANCE, cavity_temperature_k=CAVITY_TEMPERATURE_K,
                 checker=str(CHECKER.relative_to(REPO)))

        for band, case in CASES:
            config = REPO / "assets" / "configs" / f"furnace_{band}_{case}.toml"
            settings = read_config(config)
            output = REPO / settings.get("output", f"furnace_{band}_{case}_output.exr")

            if args.skip_render:
                log_text = ""
            else:
                record = run.render(config)
                log_text = pathlib.Path(record["log"]).read_text(encoding="utf-8")

            emissivity = wall_emissivity(log_text)

            # PYTHONIOENCODING because the checker prints "B̄" and "m²", and a
            # Windows console's code page cannot encode either -- the script
            # dies on its own first print line, which reads exactly like a
            # missing EXR. It runs unmodified under the WSL gate, where the
            # locale is UTF-8, so this belongs at the call rather than in it.
            environment = dict(os.environ, PYTHONIOENCODING="utf-8")
            check = subprocess.run(
                [sys.executable, str(CHECKER), str(output), band.upper()],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=str(REPO), env=environment)
            parsed = {}
            for key, pattern in CHECK_FIELDS.items():
                match = pattern.search(check.stdout)
                if match:
                    parsed[key] = float(match.group(1))
            parsed["passed"] = check.returncode == 0
            if not parsed:
                print(f"{band}_{case}: checker produced nothing\n{check.stdout}"
                      f"{check.stderr}", file=sys.stderr)

            rows.append({
                "case": f"{band}_{case}",
                "band": band.upper(),
                "config": str(config.relative_to(REPO)),
                **settings,
                "emissivity": emissivity,
                **parsed,
            })
            ratio = parsed.get("ratio")
            print(f"{band}_{case:<10} ratio {ratio if ratio is None else f'{ratio:.8f}'}"
                  f"  spp {settings.get('spp', '?'):>4}"
                  f"  {'PASS' if parsed.get('passed') else 'FAIL'}")

        deviations = [abs(r["ratio"] - 1.0) for r in rows if "ratio" in r]
        run.note(rows=rows,
                 max_abs_ratio_minus_one=max(deviations) if deviations else None)

    worst = max(deviations) if deviations else float("nan")
    (args.out_dir / "table_iv.json").write_text(
        json.dumps({"tolerance": TOLERANCE, "rows": rows,
                    "max_abs_ratio_minus_one": worst}, indent=2), encoding="utf-8")

    lines = [
        "| Cavity | Band | Wall emissivity | Spectral mode | spp | Ratio to Planck |",
        "|---|---|---|---|---:|---:|",
    ]
    for r in rows:
        eps_text = describe_emissivity(r["emissivity"], r)
        ratio = r.get("ratio")
        # Eight decimals, because six rounds five of the eight cavities to
        # 1.000000 and puts the table back where the gate's "0.0000%" left it.
        ratio_text = "n/a" if ratio is None else f"{ratio:.8f}"
        lines.append(f"| {r['case']} | {r['band']} | {eps_text} | "
                     f"{r.get('mode','?')} | {r.get('spp','?')} | {ratio_text} |")
    lines.append("")
    lines.append(f"Largest deviation |ratio - 1| = {worst:.2e}; "
                 f"acceptance tolerance {TOLERANCE:.1%} "
                 f"(check_furnace.py default, applied to every case). "
                 f"All cavities at {CAVITY_TEMPERATURE_K:.0f} K.")
    table = "\n".join(lines)
    (args.out_dir / "table_iv.md").write_text(table, encoding="utf-8")
    print()
    # The files above are UTF-8; this print has to survive whatever
    # code page the console is on.
    print(table.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
