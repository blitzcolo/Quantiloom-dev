#!/usr/bin/env python3
"""Render-level checks for the camera ISP display chain.

Three captures of one LWIR thermal scene exercise the CPU ISP products:

  grey vs ironbow  -- the palette changes only the display branch; the
                      apparent-temperature product must come out bit-identical.
  clahe tone       -- per-tile equalisation is a viewport/GPU path, so the CPU
                      chain degrades it to global equalisation and records the
                      fallback in the display EXR metadata (the degraded image
                      must equal an authored equalize run).

It also pins the display product contract: encoded sRGB kind metadata and a
3-channel [0,1] image.

Run from either WSL or Windows after building the CLI and shaders. The script
uses repo-relative assets and writes temporary configs/outputs under
assets/configs; no tracked fixture is changed. Exit 0 pass, 1 wrong render,
2 no CLI, 3 no usable GPU. All output is ASCII for Windows CJK consoles.
"""

import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import OpenEXR

ROOT = pathlib.Path(__file__).resolve().parents[2]
CLI = pathlib.Path(os.environ.get("CLI", str(ROOT / "build/src/app/Release/Quantiloom.exe")))
BASE = ROOT / "assets/configs/camera_physics_open.toml"

GPU_ABSENT = re.compile(r"No Vulkan-compatible GPUs|Failed to create Vulkan instance|No suitable")
SUCCESS = "Saved spectral image"

SENSOR = """
[sensor.optics]
focal_length_mm = 50.0
f_number = 2.8
pixel_pitch_um = 20.0
sensor_width_px = 64
sensor_height_px = 64
[sensor.thermal]
time_constant_s = 0.008
responsivity_dn_w = 1e13
[[sensor.channels]]
name = "Mono"
[sensor.channels.absorptance]
kind = "thermal_absorptance"
wavelength_nm = [8000.0, 14000.0]
value = [1.0, 1.0]
[sensor.products]
band_measurement = false
raw_dn = false
corrected_device_signal = false
apparent_temperature = true
display = true
"""


class RenderFailure(Exception):
    pass


class NoGpu(Exception):
    pass


def ascii_safe(value):
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def relative(path):
    return path.relative_to(ROOT).as_posix()


def case_config(name, work, tone, palette):
    doc = BASE.read_text(encoding="utf-8")
    changes = [
        ("renderer", "resolution", "[64, 64]"),
        ("renderer", "spp", "8"),
        ("renderer", "output", json.dumps(relative(work / f"{name}.exr"))),
        ("spectral", "wavelength_nm", "10000.0"),
        ("spectral", "band", json.dumps("LWIR")),
        ("camera", "projection", '"perspective"'),
        ("camera", "fov_y", "0.3667"),
        ("sensor", "enabled", "true"),
        ("sensor", "version", "1"),
        ("sensor", "detector", json.dumps("thermal")),
        ("sensor", "device_id", json.dumps("checker_thermal")),
        ("sensor", "cfa", json.dumps("mono")),
        ("lighting", "sun_radiance", "[5.0, 5.0, 5.0]"),
        ("lighting", "solar_lut", json.dumps("assets/luts/flat_sun_sky.csv")),
        ("lighting", "solar_lut_columns", "[4, 3]"),
        ("lighting", "solar_lut_diffuse_is_global", "true"),
    ]
    lines = doc.splitlines()
    for section, key, literal in changes:
        header = f"[{section}]"
        start = lines.index(header)
        end = next((i for i in range(start + 1, len(lines))
                    if lines[i].startswith("[")), len(lines))
        key_re = re.compile(rf"^{re.escape(key)}\s*=")
        for i in range(start + 1, end):
            if key_re.match(lines[i]):
                lines[i] = f"{key} = {literal}"
                break
        else:
            lines.insert(end, f"{key} = {literal}")
    doc = "\n".join(lines) + "\n"
    doc += SENSOR
    doc += f'[isp]\ninfrared_tone = "{tone}"\ninfrared_palette = "{palette}"\n'
    return doc


def run(name, doc, work):
    cfg = work / f"{name}.toml"
    cfg.write_text(doc, encoding="utf-8")
    proc = subprocess.run([str(CLI), relative(cfg)], cwd=ROOT,
                          capture_output=True, encoding="utf-8", errors="replace")
    log = (proc.stdout or "") + (proc.stderr or "")
    if GPU_ABSENT.search(log):
        raise NoGpu("camera isp: no usable GPU, nothing measured")
    output = work / f"{name}.exr"
    if log.count(SUCCESS) != 1 or proc.returncode != 0 or not output.is_file():
        raise RenderFailure(f"{name}: render failed: " +
                            " | ".join(log.splitlines()[-8:]))


def header_value(path, key):
    with OpenEXR.File(str(path)) as exr:
        value = exr.header().get(key)
    if isinstance(value, bytes):
        value = value.decode("ascii", "replace")
    return value


def read_rgb(path, what):
    if not path.is_file():
        raise RenderFailure(f"missing {what} {path}")
    with OpenEXR.File(str(path)) as exr:
        channels = exr.channels()
        if "RGB" in channels:
            pixels = channels["RGB"].pixels
            if pixels.ndim == 2:  # packed luminance fallback
                pixels = np.stack([pixels] * 3, axis=-1)
        else:
            names = sorted(channels)
            if len(names) < 3:
                raise RenderFailure(f"{what} has {len(names)} channels, want 3")
            pixels = np.stack([channels[name].pixels for name in names[:3]], axis=-1)
        pixels = pixels.astype(np.float64)
    if not np.all(np.isfinite(pixels)):
        raise RenderFailure(f"{what} EXR contains nonfinite pixels")
    return pixels


def display_pixels(path):
    pixels = read_rgb(path, "display product")
    if pixels.shape[-1] != 3:
        raise RenderFailure("display EXR does not have three components")
    if pixels.min() < 0.0 or pixels.max() > 1.0:
        raise RenderFailure("display EXR is not encoded into [0, 1]")
    return pixels


def product_array(path):
    # Products other than the display are not necessarily RGB: the
    # apparent-temperature map is a single channel in kelvin.
    if not path.is_file():
        raise RenderFailure(f"missing product {path}")
    with OpenEXR.File(str(path)) as exr:
        channels = exr.channels()
        pixels = np.stack([channels[name].pixels.astype(np.float64)
                           for name in sorted(channels)], axis=-1)
    if not np.all(np.isfinite(pixels)):
        raise RenderFailure(f"product {path} contains nonfinite pixels")
    return pixels


def main():
    if not CLI.is_file():
        print(f"camera isp: CLI not found at {CLI}")
        return 2
    work = ROOT / "assets/configs/camera_isp_work"
    work.mkdir(parents=True, exist_ok=True)

    try:
        variants = {
            "grey": ("linear", "grey"),
            "ironbow": ("linear", "ironbow"),
            "clahe": ("clahe", "grey"),
            "equalize": ("equalize", "grey"),
        }
        for name, (tone, palette) in variants.items():
            run(name, case_config(name, work, tone, palette), work)

        grey_display = work / "grey_display.exr"
        if header_value(grey_display, "camera_signal_kind") != "display_srgb":
            raise RenderFailure(
                "display product kind metadata is " +
                ascii_safe(header_value(grey_display, "camera_signal_kind")) +
                ", expected display_srgb")

        # The palette only recolors the display branch.
        grey = display_pixels(grey_display)
        ironbow = display_pixels(work / "ironbow_display.exr")
        if np.array_equal(grey, ironbow):
            raise RenderFailure("ironbow display equals the grey display")
        grey_tapp = product_array(work / "grey_tapp.exr")
        ironbow_tapp = product_array(work / "ironbow_tapp.exr")
        if not np.array_equal(grey_tapp, ironbow_tapp):
            raise RenderFailure("palette changed the apparent-temperature product")
        print("palette: display recolored, apparent temperature bit-identical")

        # CLAHE on the CPU chain degrades to equalize and says so.
        clahe_display = work / "clahe_display.exr"
        marker = header_value(clahe_display, "camera_ir_clahe_fallback")
        if marker != "equalize":
            raise RenderFailure("CLAHE run lacks the equalize fallback marker, got " +
                                ascii_safe(marker))
        if not np.array_equal(display_pixels(clahe_display),
                              display_pixels(work / "equalize_display.exr")):
            raise RenderFailure("CLAHE fallback does not equal the authored equalize run")
        print("clahe: degraded to equalize with metadata marker, output matches")
    except NoGpu as no_gpu:
        print(no_gpu)
        return 3
    except RenderFailure as failure:
        print(failure)
        return 1
    print("camera isp: all display-chain checks passed")
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
