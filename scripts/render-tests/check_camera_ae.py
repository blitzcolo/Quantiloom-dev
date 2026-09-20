#!/usr/bin/env python3
"""Render-level check of the M4-4 AE/AWB closed loop on the CPU reference chain.

Two sequence renders of one static, deliberately dim scene exercise the
feedback the controller writes into CaptureState:

  auto on   -- the per-frame exposure window (camera_exposure_end_s -
               camera_exposure_start_s in the display EXR metadata) must
               change across frames, stay inside [isp.auto] min/max, and
               settle (the last step is a small fraction of the value, the
               IIR having converged). The settled display must be brighter
               than the fixed-exposure control: AE opened up on a dim scene.
  auto off  -- the control render must show a constant exposure window across
               frames; the loop wrote nothing.

Both runs must number their acquisitions 0, 1, 2, ... in order.

The loop lives entirely in libQuantiloom (postprocess/), so the check runs
on the CPU reference chain; no GPU camera pass is involved. A Vulkan device
is still needed for the path tracing itself.

Run from either WSL or Windows after building the CLI. Exit 0 pass,
1 wrong render, 2 no CLI, 3 no usable GPU. All output is ASCII for Windows
CJK consoles.
"""

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np
import OpenEXR

ROOT = pathlib.Path(__file__).resolve().parents[2]
CLI = pathlib.Path(os.environ.get("CLI", str(ROOT / "build/src/app/Release/Quantiloom.exe")))
BASE = ROOT / "assets/configs/camera_physics_open.toml"
WORK = ROOT / "build" / "camera_ae_check"

GPU_ABSENT = re.compile(r"No Vulkan-compatible GPUs|Failed to create Vulkan instance|No suitable")
SUCCESS = "Saved spectral image"

FRAMES = 8

SENSOR = """
[sensor.optics]
focal_length_mm = 50.0
f_number = 2.8
pixel_pitch_um = 5.0
sensor_width_px = 32
sensor_height_px = 32
[sensor.exposure]
# Deliberately short: the scene lands well under the AE target at this
# exposure, and every pixel stays clear of the ADC ceiling so the loop's
# unsaturated statistics carry a signal.
time_s = 0.00001
[sensor.photon]
full_well_e = 20000.0
dark_current_e_s = 0.0
read_noise_e = 0.0
[[sensor.channels]]
name = "Mono"
[sensor.channels.qe]
kind = "absolute_qe"
wavelength_nm = [500.0, 600.0]
value = [0.5, 0.5]
[sensor.quality]
noise_free = true
[sensor.products]
raw_dn = false
band_measurement = false
corrected_device_signal = false
apparent_temperature = false
display = true
[timeline]
ticks_per_second = 10.0
end_s = 10.0
"""

AUTO = """
[isp]
auto_exposure = true
[isp.auto]
target_luminance = 0.18
smoothing = 0.5
min_exposure_s = 0.0001
max_exposure_s = 0.2
max_gain = 16.0
"""


class RenderFailure(Exception):
    pass


class NoGpu(Exception):
    pass


def relative(path):
    return path.relative_to(ROOT).as_posix()


def case_config(name, work):
    doc = BASE.read_text(encoding="utf-8")
    changes = [
        ("renderer", "resolution", "[32, 32]"),
        ("renderer", "spp", "8"),
        ("renderer", "output", json.dumps(relative(work / f"{name}_spectral.exr"))),
        ("spectral", "band", json.dumps("VIS")),
        ("camera", "projection", '"perspective"'),
        ("camera", "fov_y", "0.3667"),
        ("lighting", "sun_radiance", "[0.05, 0.05, 0.05]"),
        ("lighting", "solar_lut", json.dumps("assets/luts/flat_sun_sky.csv")),
        ("lighting", "solar_lut_columns", "[4, 3]"),
        ("lighting", "solar_lut_diffuse_is_global", "true"),
        ("sensor", "enabled", "true"),
        ("sensor", "version", "1"),
        ("sensor", "detector", json.dumps("photon")),
        ("sensor", "device_id", json.dumps("checker_ae")),
        ("sensor", "cfa", json.dumps("mono")),
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
    if name == "auto":
        doc += AUTO
    return doc


def run_sequence(name, doc, work):
    cfg = work / f"{name}.toml"
    cfg.write_text(doc, encoding="utf-8")
    frame_dir = work / name
    frame_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(CLI), "sequence", relative(cfg),
         "--from-tick", "0", "--to-tick", str(FRAMES - 1), "--every", "1",
         "--output", relative(frame_dir / "frame_{tick:05}.exr")],
        cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace")
    log = (proc.stdout or "") + (proc.stderr or "")
    if GPU_ABSENT.search(log):
        raise NoGpu("camera ae: no usable GPU, nothing measured")
    if proc.returncode != 0:
        raise RenderFailure(f"{name}: sequence failed: " +
                            " | ".join(log.splitlines()[-8:]))
    return sorted(frame_dir.glob("*.exr"))


def frame_records(paths):
    def attr(header, name):
        value = header.get(name)
        if value is None:
            return None
        # The OpenEXR.File header mapping yields plain strings on some
        # bindings and attribute objects on others.
        return str(getattr(value, "value", value))

    records = []
    for path in paths:
        with OpenEXR.File(str(path)) as exr:
            header = exr.header()
            kind = attr(header, "camera_signal_kind")
            if kind is None or kind != "display_srgb":
                continue
            acquisition = int(attr(header, "camera_acquisition_index"))
            start = float(attr(header, "camera_exposure_start_s"))
            end = float(attr(header, "camera_exposure_end_s"))
            channels = exr.channels()
            names = sorted(channels)
            pixels = np.stack([channels[n].pixels.astype(np.float64)
                               for n in names[:3]], axis=-1)
            if pixels.shape[-1] == 1:
                pixels = np.repeat(pixels, 3, axis=-1)
            luma = float(np.mean(0.2126 * pixels[..., 0] +
                                 0.7152 * pixels[..., 1] +
                                 0.0722 * pixels[..., 2]))
            records.append({"acquisition": acquisition,
                            "exposure": end - start, "luma": luma})
    records.sort(key=lambda r: r["acquisition"])
    return records


def fail(message):
    print(f"FAIL: {message}")
    return 1


def main():
    if not CLI.exists():
        print(f"no CLI at {CLI}")
        return 2
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        auto = frame_records(run_sequence("auto", case_config("auto", WORK), WORK))
        control = frame_records(
            run_sequence("control", case_config("control", WORK), WORK))
    except NoGpu as exc:
        print(exc)
        return 3
    except RenderFailure as exc:
        print(f"FAIL: {exc}")
        return 1

    if len(auto) != FRAMES or len(control) != FRAMES:
        return fail(f"expected {FRAMES} display frames per case, got "
                    f"{len(auto)} auto and {len(control)} control")
    for label, records in (("auto", auto), ("control", control)):
        indices = [r["acquisition"] for r in records]
        if indices != list(range(FRAMES)):
            return fail(f"{label}: acquisition indices {indices}, want 0..{FRAMES - 1}")

    # Control: the loop is off, so the exposure window is the authored one on
    # every frame.
    control_exposures = [r["exposure"] for r in control]
    if max(control_exposures) - min(control_exposures) > 1e-9:
        return fail(f"control exposure window drifted: {control_exposures}")

    # Auto: the loop moved, respected its rails, and settled. Frame 0 is the
    # authored exposure, deliberately below the loop's floor so AE opens up;
    # the rails bind from the first feedback application onward.
    auto_exposures = [r["exposure"] for r in auto]
    if max(auto_exposures) - min(auto_exposures) < 1e-12:
        return fail(f"auto exposure never moved: {auto_exposures}")
    min_s, max_s = 0.0001, 0.2
    if not all(min_s - 1e-12 <= e <= max_s + 1e-12 for e in auto_exposures[1:]):
        return fail(f"auto exposure left [{min_s}, {max_s}]: {auto_exposures}")
    settle = abs(auto_exposures[-1] - auto_exposures[-2]) / max(auto_exposures[-1], 1e-12)
    if settle > 0.02:
        return fail(f"auto exposure did not settle (last step {settle:.4f}): {auto_exposures}")
    if auto_exposures[-1] <= control_exposures[0] * 1.5:
        return fail(f"auto exposure {auto_exposures[-1]:.6f} did not open up from "
                    f"the control {control_exposures[0]:.6f} on a dim scene")

    # The settled display must be brighter than the fixed-exposure control.
    if not auto[-1]["luma"] > control[-1]["luma"] * 1.05:
        return fail(f"auto display luma {auto[-1]['luma']:.4f} did not exceed "
                    f"control luma {control[-1]['luma']:.4f}")

    print(f"pass: auto exposure {auto_exposures[0]:.6f} -> "
          f"{auto_exposures[-1]:.6f} s (settled step {settle:.5f}), "
          f"display luma {control[-1]['luma']:.4f} -> {auto[-1]['luma']:.4f}; "
          f"control constant at {control_exposures[0]:.6f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
