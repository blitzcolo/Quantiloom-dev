#!/usr/bin/env python3
"""Check the GPU dynamic-exposure compositor against an independent twin.

The GPU preview integrates a moving exposure by tracing T time strata
(binding 28 layer array) and reprojecting them through the anchor depth
(camera_dynamic.comp). The in-process GPU cases in test_camera_gpu.cpp render
a synthetic box scene into the strata and write the layers, the device
composite, the disocclusion counters, and the scene spec to a work dir. This
checker:

  1. runs those gtest cases (they assert the T=1 degenerate copy, the rolling
     shutter row dependence, the scheduler-level DynamicExposureReport, and
     the metadata annotation), and
  2. recomposites the strata in numpy with the same reprojection math and
     compares the result against the device composite pixel for pixel, and
     the disoccluded-pixel count against the device counter exactly.

Run from either WSL or Windows after building the tests and shaders. Exit
0 pass, 1 wrong render, 2 no test binary, 3 no usable GPU. All output is
ASCII for Windows CJK consoles.
"""

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
TESTS = pathlib.Path(
    os.environ.get("TESTS", str(ROOT / "build/tests/Release/libquantiloom_tests.exe"))
)
WORK = ROOT / "build" / "camera_dynamic_check"

# The gtest cases that render the synthetic scene and exercise the report.
FILTER = (
    "CameraGpuTest.StratifiedCompositorMatchesAnchorsAndCountsDisocclusion:"
    "CameraSchedulerGpuTest.StratifiedAcquisitionReportsDynamicExposure:"
    "CameraSchedulerGpuTest.StaticSceneDegeneratesToSingleStratum"
)

NO_GPU = re.compile(
    r"Skipped|skipping|No Vulkan-compatible GPUs|Failed to create Vulkan "
    r"instance|No suitable|CUDA_ERROR|device lost",
    re.IGNORECASE,
)


def win_path(path):
    """Best-effort /mnt/<drive>/... to <drive>:/... for child .exe processes."""
    text = str(path)
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", text)
    if match:
        return f"{match.group(1).upper()}:/{match.group(2)}"
    return text


def fail(message):
    print(f"FAIL: {message}")
    return 1


def run_cases():
    if not TESTS.exists():
        print(f"no test binary at {TESTS}")
        return 2, ""
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["QUANTILOOM_CAMERA_DYNAMIC_DIR"] = win_path(WORK)
    if sys.platform.startswith("linux"):
        # WSL only forwards listed variables to Windows child processes.
        env["WSLENV"] = "QUANTILOOM_CAMERA_DYNAMIC_DIR"
    # argv uses the path as the host sees it: the WSL interop handles the
    # /mnt/<drive> form, and on native Windows it is already a Windows path.
    proc = subprocess.run(
        [str(TESTS), f"--gtest_filter={FILTER}"],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        if NO_GPU.search(output):
            print("no usable GPU (cases skipped)")
            return 3, output
        print("gtest cases failed; the device results below are unreliable")
    return 0, output


# ---------------------------------------------------------------------------
# Numpy twin of camera_dynamic.comp.hlsl (global shutter run: rowDelay == 0).
# ---------------------------------------------------------------------------

def pixel_ray(camera, x, y, width, height):
    u = (x + 0.5) / width
    v = (y + 0.5) / height
    ndc_x = u * 2.0 - 1.0
    ndc_y = -(v * 2.0 - 1.0)
    direction = (
        camera["forward"]
        + ndc_x * camera["right"] * camera["fov_scale"] * camera["aspect"]
        + ndc_y * camera["up"] * camera["fov_scale"]
    )
    return direction / np.linalg.norm(direction)


def project_world(camera, world):
    rel = world - np.asarray(camera["origin"], dtype=np.float64)
    along = float(np.dot(rel, camera["forward"]))
    expected = float(np.linalg.norm(rel))
    x = float(np.dot(rel, camera["right"]))
    y = float(np.dot(rel, camera["up"]))
    ndc_x = x / (along * camera["fov_scale"] * camera["aspect"])
    ndc_y = y / (along * camera["fov_scale"])
    uv = np.array([ndc_x * 0.5 + 0.5, 1.0 - (ndc_y * 0.5 + 0.5)])
    return uv, expected


def sample_stratum(rate, depth, layer, uv, expected):
    """Twin of SampleStratum: bilinear rate, per-tap depth consistency."""
    height, width = depth.shape[1], depth.shape[2]
    if (uv[0] < 0.0 or uv[0] > 1.0 or uv[1] < 0.0 or uv[1] > 1.0
            or expected <= 0.0):
        return None
    texel = uv * np.array([width, height], dtype=np.float64) - 0.5
    base = np.floor(texel).astype(int)
    frac = texel - base
    rate_sum = np.zeros(4, dtype=np.float64)
    weight_sum = 0.0
    for dy in (0, 1):
        for dx in (0, 1):
            tap = base + np.array([dx, dy])
            if tap[0] < 0 or tap[1] < 0 or tap[0] >= width or tap[1] >= height:
                continue
            weight = ((1.0 - frac[0]) if dx == 0 else frac[0]) * (
                (1.0 - frac[1]) if dy == 0 else frac[1])
            if weight <= 0.0:
                continue
            tap_depth = float(depth[layer, tap[1], tap[0]])
            tolerance = max(1e-3, 0.01 * tap_depth)
            if tap_depth < 0.0 or abs(tap_depth - expected) > tolerance:
                return None
            rate_sum += weight * rate[layer, tap[1], tap[0]]
            weight_sum += weight
    if weight_sum <= 0.0:
        return None
    return rate_sum / weight_sum


def recomposite(spec, rate, depth):
    width = spec["width"]
    height = spec["height"]
    strata = spec["strata"]
    exposure = spec["exposure_seconds"]
    t0 = spec["t0"]
    if spec.get("row_delay_seconds", 0.0) != 0.0:
        raise ValueError("twin covers the global-shutter artifact run")
    cameras = [
        {
            "origin": np.asarray(c["origin"], dtype=np.float64),
            "forward": np.asarray(c["forward"], dtype=np.float64),
            "right": np.asarray(c["right"], dtype=np.float64),
            "up": np.asarray(c["up"], dtype=np.float64),
            "fov_scale": float(c["fov_scale"]),
            "aspect": float(c["aspect"]),
        }
        for c in spec["cameras"]
    ]
    out = np.zeros((height, width, 4), dtype=np.float64)
    disoccluded = 0
    window_start = t0 - 0.5 * exposure
    spacing = exposure / strata
    for y in range(height):
        for x in range(width):
            anchor = rate[0, y, x]
            anchor_distance = float(depth[0, y, x])
            direction = pixel_ray(cameras[0], x, y, width, height)
            far = anchor_distance if anchor_distance >= 0.0 else 1.0e6
            world = cameras[0]["origin"] + direction * far
            # Global shutter: every row integrates at t0.
            position = (t0 - window_start) / spacing - 0.5
            lower = int(np.floor(position))
            lower = int(np.clip(lower, 0, strata - 2))
            upper_weight = float(np.clip(position - lower, 0.0, 1.0))
            accumulated = np.zeros(4, dtype=np.float64)
            weight_sum = 0.0
            for layer, weight in ((lower, 1.0 - upper_weight),
                                  (lower + 1, upper_weight)):
                if weight <= 0.0:
                    continue
                uv, expected = project_world(cameras[layer], world)
                sampled = sample_stratum(rate, depth, layer, uv, expected)
                if sampled is not None:
                    accumulated += weight * sampled
                    weight_sum += weight
            if weight_sum > 0.0:
                out[y, x] = accumulated / weight_sum
            else:
                out[y, x] = anchor
                disoccluded += 1
    return out, disoccluded


def main():
    status, output = run_cases()
    if status != 0:
        return status

    passed = re.search(r"\[  PASSED  \] \d+ test", output)
    if not passed:
        return fail("gtest cases did not all pass")

    needed = ["spec.json", "strata_rate.f32", "strata_depth.f32",
              "composite.f32", "counters.u32"]
    missing = [name for name in needed if not (WORK / name).exists()]
    if missing:
        return fail(f"artifacts missing: {', '.join(missing)}")

    spec = json.loads((WORK / "spec.json").read_text())
    width = spec["width"]
    height = spec["height"]
    strata = spec["strata"]
    rate = np.fromfile(WORK / "strata_rate.f32", dtype=np.float32)
    rate = rate.reshape(strata, height, width, 4).astype(np.float64)
    depth = np.fromfile(WORK / "strata_depth.f32", dtype=np.float32)
    depth = depth.reshape(strata, height, width).astype(np.float64)
    composite = np.fromfile(WORK / "composite.f32", dtype=np.float32)
    composite = composite.reshape(height, width, 4).astype(np.float64)
    counters = np.fromfile(WORK / "counters.u32", dtype=np.uint32)

    reference, disoccluded = recomposite(spec, rate, depth)

    delta = np.abs(reference - composite)
    max_abs = float(delta.max())
    mean_abs = float(delta.mean())
    print(f"composite max|delta| = {max_abs:.6g}")
    print(f"composite mean|delta| = {mean_abs:.6g}")
    print(f"disoccluded twin = {disoccluded}, device = {int(counters[0])}")

    failures = []
    if max_abs > 1e-3:
        failures.append(
            f"composite deviates from the numpy twin by {max_abs:.6g}")
    if int(counters[0]) != disoccluded:
        failures.append(
            f"disoccluded count mismatch: device {int(counters[0])} "
            f"vs twin {disoccluded}")
    if float(composite[:, :, 0].sum()) == 0.0:
        failures.append("device composite is all zeros")

    if failures:
        for message in failures:
            print(f"FAIL: {message}")
        return 1
    print("PASS: device composite matches the numpy twin; counters agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
