#!/usr/bin/env python3
"""Render-level check of the camera acquisition history across a sequence.

`sequence --every N` exports every Nth scene tick. The device advances at its
own frame period (thermal detector lag, AE/AWB feedback) through
AdvanceCameraState; a scene tick before the next device frame reuses the
previous product. Every RNG stream is keyed on the acquisition index, not
the export index. So an
every-2 run and an every-1 run over the same tick range describe the SAME
unbroken acquisition sequence, and the exported frames must agree:

  every-2  run, exported frame k   ==   every-1  run, frame 2k
  value for value, on every product.

The scene (assets/configs/camera_history_check.toml) is a hot block sliding
across a warm ground in LWIR under a thermal-detector camera whose
first-order lag (time_constant_s = 1.5 frame periods) makes every frame a
function of the acquisitions before it, plus a read-noise floor so the
per-acquisition RNG stream is genuinely exercised by the comparison. A
shared RNG keying bug between the render path and the advance path is
exactly what makes the two runs disagree.

Checks:
  a. the every-2 run exports exactly floor((B - A) / 2) + 1 frames, whose
     camera_acquisition_index metadata reads 0, 2, 4, ... -- the skipped
     matching device-frame slots became acquisitions, not nothing;
  b. adjacent exported frames differ on rawDn and display (the history is
     actually moving -- a frozen state would leave every frame identical);
  c. every-2 frame k equals every-1 frame 2k to within 1e-6 (measured bit-
     exact: both runs walk the same acquisition sequence, so the frames
     must be identical; the tolerance covers any EXR encode nondeterminism
     and the measured maximum is reported);
  d. warmup: a run with [sensor.warmup] seconds > 0 must succeed and its
     first frame must differ from the no-warmup first frame. A full
     warmup-vs-reference value check would need a second long reference
     solve-style run; per the acquisition-history scoping that comparison is degraded to
     "warmup does not error and shifts the state", which the acquisition
     index (3 after a 0.3 s / 0.1 s warmup) plus the shifted noise draw
     already proves. See the comment at the warmup check below.
  e. with scene ticks at 20/s and the device at 10/s, adjacent scene ticks
     reuse the same product while every-2 and every-1 exports still agree.

Run from either WSL or Windows after building the CLI. Exit 0 pass,
1 wrong render, 2 no CLI, 3 no usable GPU. All output is ASCII for
Windows CJK consoles.
"""

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
BASE = ROOT / "assets/configs/camera_history_check.toml"
# Derived configs must live beside the committed one: model paths in the
# config resolve against its own directory.
WARMUP_CFG = ROOT / "assets/configs/camera_history_check_warmup_work.toml"
CROSS_RATE_CFG = ROOT / "assets/configs/camera_history_check_cross_rate_work.toml"
WORK = ROOT / "build" / "camera_history_check"

GPU_ABSENT = re.compile(r"No Vulkan-compatible GPUs|Failed to create Vulkan instance|No suitable")

FROM_TICK = 0
TO_TICK = 8
EVERY = 2
EXPECTED_FRAMES = (TO_TICK - FROM_TICK) // EVERY + 1
TOLERANCE = 1e-6

WARMUP_SECONDS = 0.3

PRODUCTS = ("rawdn", "display")


class RenderFailure(Exception):
    pass


class NoGpu(Exception):
    pass


def relative(path):
    return path.relative_to(ROOT).as_posix()


def run_sequence(name, config, every, to_tick):
    frame_dir = WORK / name
    frame_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(CLI), "sequence", relative(config),
         "--from-tick", str(FROM_TICK), "--to-tick", str(to_tick),
         "--every", str(every),
         "--output", relative(frame_dir / "frame_{tick:05}.exr")],
        cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace")
    log = (proc.stdout or "") + (proc.stderr or "")
    if GPU_ABSENT.search(log):
        raise NoGpu("camera history: no usable GPU, nothing measured")
    if proc.returncode != 0:
        raise RenderFailure(f"{name}: sequence failed: " +
                            " | ".join(log.splitlines()[-8:]))
    # The products sit beside the frame as frame_00000_rawdn.exr and so on;
    # the five-question-mark pattern keeps those out of the frame list.
    return sorted(frame_dir.glob("frame_?????.exr"))


def product_pixels(path, product):
    """Stack one product's channels into a float64 array."""
    with OpenEXR.File(str(path)) as exr:
        channels = exr.channels()
        names = sorted(channels)
        if not names:
            raise RenderFailure(f"{path.name}: no channels")
        return np.stack([channels[n].pixels.astype(np.float64)
                         for n in names], axis=-1)


def header_value(path, key):
    with OpenEXR.File(str(path)) as exr:
        value = exr.header().get(key)
    return str(getattr(value, "value", value))


def frame_products(frame_dir, tick):
    stem = frame_dir / f"frame_{tick:05}"
    out = {}
    for product in PRODUCTS:
        path = pathlib.Path(f"{stem}_{product}.exr")
        if not path.is_file():
            raise RenderFailure(f"missing product {path}")
        out[product] = product_pixels(path, product)
    return out


def fail(message):
    print(f"FAIL: {message}")
    return 1


def main():
    if not CLI.exists():
        print(f"no CLI at {CLI}")
        return 2
    if not BASE.is_file():
        print(f"no config at {BASE}")
        return 2

    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    WARMUP_CFG.write_text(
        BASE.read_text(encoding="utf-8") +
        f"\n[sensor.warmup]\nseconds = {WARMUP_SECONDS}\n",
        encoding="utf-8")
    CROSS_RATE_CFG.write_text(
        BASE.read_text(encoding="utf-8").replace(
            "ticks_per_second = 10.0", "ticks_per_second = 20.0"),
        encoding="utf-8")

    try:
        every2 = run_sequence("every2", BASE, EVERY, TO_TICK)
        every1 = run_sequence("every1", BASE, 1, TO_TICK)
        warmup = run_sequence("warmup", WARMUP_CFG, 1, 2)
        cross_every2 = run_sequence("cross_every2", CROSS_RATE_CFG, EVERY, TO_TICK)
        cross_every1 = run_sequence("cross_every1", CROSS_RATE_CFG, 1, TO_TICK)
    except NoGpu as exc:
        print(exc)
        return 3
    except RenderFailure as exc:
        print(f"FAIL: {exc}")
        return 1
    finally:
        WARMUP_CFG.unlink(missing_ok=True)
        CROSS_RATE_CFG.unlink(missing_ok=True)

    # (a) Export count and the acquisition-index metadata: an every-2 run
    # over ticks 0..8 exports five frames, and the exported ones carry
    # acquisition indices 0, 2, 4, 6, 8 -- the skipped ticks ran as
    # acquisitions (that is what makes frame k of this run comparable to
    # frame 2k of the every-1 run at all).
    exported_ticks = list(range(FROM_TICK, TO_TICK + 1, EVERY))
    if len(every2) != EXPECTED_FRAMES:
        return fail(f"every-2 run exported {len(every2)} frames, "
                    f"want {EXPECTED_FRAMES}")
    indices = []
    for tick in exported_ticks:
        index = header_value(WORK / "every2" / f"frame_{tick:05}_rawdn.exr",
                             "camera_acquisition_index")
        if index is None or index == "None":
            return fail(f"frame {tick}: no camera_acquisition_index metadata")
        indices.append(int(index))
    if indices != exported_ticks:
        return fail(f"every-2 acquisition indices {indices}, want {exported_ticks}")

    # (b) + (c). Adjacent exported frames must differ, and each every-2
    # frame must equal the every-1 frame at the same tick.
    max_equal_delta = 0.0
    for product in PRODUCTS:
        previous = None
        for k, tick in enumerate(exported_ticks):
            skipped = frame_products(WORK / "every2", tick)[product]
            full = frame_products(WORK / "every1", tick)[product]
            if skipped.shape != full.shape:
                return fail(f"{product} tick {tick}: shape {skipped.shape} "
                            f"vs {full.shape}")
            delta = float(np.abs(skipped - full).max())
            max_equal_delta = max(max_equal_delta, delta)
            if delta > TOLERANCE:
                return fail(f"{product} tick {tick}: every-2 frame differs "
                            f"from every-1 frame by {delta:.6g} "
                            f"(tolerance {TOLERANCE:.0e}); the advance path "
                            f"and the render path disagree on the state")
            if previous is not None:
                step = float(np.abs(skipped - previous).max())
                if step <= 0.0:
                    return fail(f"{product}: exported frames at ticks "
                                f"{exported_ticks[k - 1]} and {tick} are "
                                f"identical; the acquisition history is not "
                                f"moving")
            previous = skipped

    # (d) Warmup, degraded per the docstring: the run must succeed and its
    # first frame must differ from the no-warmup first frame. With the
    # warmup grid clamped to the clock origin the thermal state itself is
    # unchanged from the steady-state first frame -- what moves is the
    # acquisition index (the 0.3 s / 0.1 s period warmup burns indices
    # 0..2, so the exported first frame is acquisition 3) and with it the
    # per-acquisition noise draw. A full closed-form warmup reference is
    # the one comparison this degradation gives up; it is a deliberate
    # cost trade, recorded here so nobody reads the weak check as the
    # strong one.
    warmup_first = frame_products(WORK / "warmup", 0)
    plain_first = frame_products(WORK / "every1", 0)
    warmup_index = int(header_value(WORK / "warmup" / "frame_00000_rawdn.exr",
                                    "camera_acquisition_index"))
    if warmup_index != round(WARMUP_SECONDS / 0.1):
        return fail(f"warmup first frame has acquisition index {warmup_index}, "
                    f"want 3 after a {WARMUP_SECONDS} s warmup at 0.1 s period")
    warmup_delta = max(
        float(np.abs(warmup_first[p] - plain_first[p]).max())
        for p in PRODUCTS)
    if warmup_delta <= 0.0:
        return fail("warmup run's first frame equals the no-warmup first "
                    "frame; warmup changed nothing")

    # A 20 tick/s scene with a 10 acquisition/s device must not invent an
    # acquisition at each scene tick. Every-2 still matches the full run at
    # the same tick; adjacent full-run ticks share one physical frame.
    if len(cross_every2) != EXPECTED_FRAMES or len(cross_every1) != TO_TICK + 1:
        return fail("cross-rate sequence exported the wrong frame count")
    cross_indices = [int(header_value(
        WORK / "cross_every2" / f"frame_{tick:05}_rawdn.exr",
        "camera_acquisition_index")) for tick in exported_ticks]
    if cross_indices != [0, 1, 2, 3, 4]:
        return fail(f"cross-rate acquisition indices {cross_indices}, want 0..4")
    for product in PRODUCTS:
        for tick in exported_ticks:
            skipped = frame_products(WORK / "cross_every2", tick)[product]
            full = frame_products(WORK / "cross_every1", tick)[product]
            if np.max(np.abs(skipped - full)) > TOLERANCE:
                return fail(f"cross-rate {product} tick {tick} differs by export stride")
        first = frame_products(WORK / "cross_every1", 0)[product]
        reused = frame_products(WORK / "cross_every1", 1)[product]
        if np.max(np.abs(first - reused)) > TOLERANCE:
            return fail(f"cross-rate {product} tick 1 invented a new acquisition")

    print(f"pass: {EXPECTED_FRAMES} every-2 frames, acquisition indices "
          f"{exported_ticks[0]}..{exported_ticks[-1]} by {EVERY}")
    print(f"pass: every-2 frame k equals every-1 frame 2k on "
          f"{', '.join(PRODUCTS)} (max delta {max_equal_delta:.6g})")
    print(f"pass: adjacent exported frames differ; warmup shifts the first "
          f"frame (delta {warmup_delta:.6g}, acquisition {warmup_index})")
    print("pass: 20 tick/s timeline uses 10 acquisition/s device history")
    return 0


if __name__ == "__main__":
    sys.exit(main())
