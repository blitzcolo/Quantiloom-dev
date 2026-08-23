#!/usr/bin/env python3
"""The cross-band gallery strip of Section VIII-F.

One asset, one camera, one set of measured spectra, rendered in five bands.
Nothing changes between the panels except which part of the spectrum is being
asked about -- that is the claim the band-aware convention of Section IV makes,
and a strip is the only way to show it.

Each panel names its own display transform in the caption, because an LWIR
render's radiance sits two orders of magnitude below the display range and
saying "here is the infrared" without saying how it was mapped is not a
measurement. The infrared panels use linear AGC, never CLAHE: CLAHE inverts
half of all brightness-ordered pixel pairs, so a temperature cannot be read off
one, and a figure whose point is the reflection-to-emission transition must not
be shown through an operator that reorders brightness.

Usage:
    e7_gallery.py --strip                 # the five-band figure
    e7_gallery.py --diurnal               # the thermal-shadow sequence
    e7_gallery.py --strip --spp 64        # a quick look
"""

import argparse
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
CLI = REPO / "build" / "src" / "app" / "Release" / "Quantiloom.exe"
CONFIG = REPO / "assets" / "configs" / "gallery" / "kv2_desert_lwir.toml"
WORK = pathlib.Path(r"H:\quantiloom-paper\evidence\e7")

# `mode` and `band` move together. A scene that sets one without the other
# binds each material's visible curve and then samples it in the infrared,
# which is the failure the convention exists to prevent.
BANDS = [
    ("VIS", "vis_fused", "VIS", "sRGB, no tone mapping"),
    ("NIR", "nir_fused", "NIR", "linear AGC"),
    ("SWIR", "swir_fused", "SWIR", "linear AGC"),
    ("MWIR", "mwir_fused", "MWIR", "linear AGC"),
    ("LWIR", "lwir_fused", "LWIR", "linear AGC"),
]

# 15:00, when the shadow is fully developed, plus four more across the day.
# Sun directions come from the same forcing file the solver reads, so the
# shading and the temperature field agree about where the shadow is.
DIURNAL_HOURS = [78.0, 82.0, 86.0, 90.0, 94.0]


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)


def render(overrides, output, work):
    tokens = " ".join(f"{k}={toml_value(v)}" for k, v in
                      {**overrides, "renderer.output": output.as_posix()}.items())
    manifest = work / "_batch.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(f"{CONFIG.as_posix()} | {tokens}\n", encoding="utf-8")
    result = subprocess.run([str(CLI), "batch", str(manifest)], cwd=str(REPO),
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=14400)
    if result.returncode != 0:
        print(result.stdout[-3000:] + result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"render failed: {output.name}")
    return result.stdout


def strip(args):
    work = WORK / "strip"
    work.mkdir(parents=True, exist_ok=True)
    records = []
    for name, mode, band, display in BANDS:
        output = work / f"kv2_{name.lower()}.exr"
        print(f"  rendering {name} ...", flush=True)
        log = render({"spectral.mode": mode, "spectral.band": band,
                      "renderer.spp": args.spp}, output, work)
        # Whether every material actually bound a measured curve in this band
        # is the thing most worth recording: a silent fallback to a base colour
        # is exactly the defect Section VIII-D caught, and it is invisible in
        # the pixels.
        bound = [line.strip() for line in log.splitlines()
                 if "spectral" in line.lower() and
                 ("curve" in line.lower() or "material" in line.lower())]
        records.append({"band": name, "mode": mode, "output": str(output),
                        "display": display, "spp": args.spp,
                        "spectral_log": bound[:12]})
        print(f"    -> {output.name}", flush=True)

    (WORK / "strip.json").write_text(json.dumps(
        {"config": str(CONFIG), "seed": "0x547C", "bands": records}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {WORK / 'strip.json'}")


def diurnal(args):
    work = WORK / "diurnal"
    work.mkdir(parents=True, exist_ok=True)
    records = []
    for hour in DIURNAL_HOURS:
        output = work / f"kv2_lwir_t{hour:g}.exr"
        print(f"  rendering t = {hour} h ...", flush=True)
        render({"thermal.enabled": True, "thermal.time_h": hour,
                "thermal.forcing_file":
                    (REPO / "assets" / "configs" / "desert" /
                     "desert_day.csv").as_posix(),
                "renderer.spp": args.spp}, output, work)
        records.append({"time_h": hour, "output": str(output), "spp": args.spp})
        print(f"    -> {output.name}", flush=True)

    (WORK / "diurnal.json").write_text(json.dumps(
        {"config": str(CONFIG), "seed": "0x547C", "frames": records}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {WORK / 'diurnal.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strip", action="store_true")
    parser.add_argument("--diurnal", action="store_true")
    parser.add_argument("--spp", type=int, default=512)
    args = parser.parse_args()

    if not CLI.is_file():
        raise SystemExit(f"{CLI} not built")
    if not (REPO / "assets" / "models" / "desert" / "desert_kv2.gltf").is_file():
        raise SystemExit("desert_kv2.gltf missing; rebuild it with the command "
                         "in the header of the gallery config")
    if not (args.strip or args.diurnal):
        parser.error("choose --strip and/or --diurnal")

    WORK.mkdir(parents=True, exist_ok=True)
    if args.strip:
        strip(args)
    if args.diurnal:
        diurnal(args)


if __name__ == "__main__":
    main()
