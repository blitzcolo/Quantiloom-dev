#!/usr/bin/env python3
"""Validate a Vulkan-timestamp camera-chain benchmark, without CPU-time proxies.

The benchmark host writes JSONL: one `type=meta` line containing schema 2,
GPU and driver, 1920x1080 extent, warmup count and cold-start wall time;
then one `type=frame` line per acquisition. Every frame has Vulkan-query
`baseline_trace_ms`, `camera_trace_ms`, `full_camera_ms`, and five post stages:
`psf_ms`, `dynamic_ms`, `detector_ms`, `isp_ms`, `hsv_ms`. Stage intervals are
disjoint. The checker derives spectral-response increment as camera trace
minus matched baseline trace, and added camera time as full camera interval
minus matched baseline trace. The baseline uses one wavelength at the device
response midpoint; its difference from the camera's sampled-wavelength RT
includes spectral path-cost differences and is labelled that way. CPU cold-start time is
recorded separately as `cold_start_wall_ms` and never compared with a GPU gate.

At least 300 acquisitions must follow warmup; exactly the first 300 are gated.
The nearest-rank P95 of the added camera GPU time must be <=5 ms. The checker
prints all stage P95 values, trace P95, GPU/driver, and cold start. This is a
portable stdlib-only measurement reader; collecting Vulkan timestamps belongs
to the renderer host, not to this Python process.

Usage: python3 scripts/render-tests/check_camera_gpu_perf.py metrics.jsonl
Exit 0 pass, 1 invalid/slow measurement, 2 missing metrics. ASCII stdout only.
"""

import json
import math
import pathlib
import sys

POST_STAGES = ("psf_ms", "dynamic_ms", "detector_ms", "isp_ms", "hsv_ms")
MIN_FRAMES = 300
MAX_P95_MS = 5.0


def ascii_safe(value):
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def finite_nonnegative(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0):
        raise ValueError(f"{label} must be a finite nonnegative number")
    return float(value)


def p95(values):
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def read_jsonl(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records or not isinstance(records[0], dict) or records[0].get("type") != "meta":
        raise ValueError("first timing record must be meta")
    if any(not isinstance(frame, dict) or frame.get("type") != "frame"
           for frame in records[1:]):
        raise ValueError("all remaining timing records must be frames")
    document = dict(records[0])
    document["samples"] = records[1:]
    return document


def validate(document):
    if document.get("schema") != 2:
        raise ValueError("unknown GPU timing schema")
    if document.get("timing_source") != "vulkan_timestamp":
        raise ValueError("camera timing must come from Vulkan timestamp queries")
    if document.get("resolution") != [1920, 1080]:
        raise ValueError("camera benchmark must use a 1920x1080 physical array")
    gpu = document.get("gpu_name")
    driver = document.get("driver_version")
    if not isinstance(gpu, str) or not gpu or not isinstance(driver, str) or not driver:
        raise ValueError("GPU name and driver version are required")
    if document.get("baseline_spectral_mode") != "single_mid_response":
        raise ValueError("baseline must declare single_mid_response spectral method")
    warmup = document.get("warmup_frames")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
        raise ValueError("at least one warmup acquisition is required")
    samples = document.get("samples")
    if not isinstance(samples, list) or len(samples) < warmup + MIN_FRAMES:
        raise ValueError("need 300 measured acquisitions after warmup")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("acquisition_index") != index:
            raise ValueError("benchmark acquisitions must be contiguous from zero")
    if "cold_start_wall_ms" not in document:
        raise ValueError("cold start must be reported separately")
    cold_start = finite_nonnegative(document["cold_start_wall_ms"], "cold start")
    measured = samples[warmup:warmup + MIN_FRAMES]
    fields = ("baseline_trace_ms", "camera_trace_ms", "full_camera_ms") + POST_STAGES
    durations = {field: [] for field in fields}
    durations["response_delta_ms"] = []
    durations["added_camera_ms"] = []
    durations["post_total_ms"] = []
    for index, sample in enumerate(measured):
        if sample.get("timing_source", "vulkan_timestamp") != "vulkan_timestamp":
            raise ValueError(f"sample {index} is not a Vulkan timestamp measurement")
        for field in fields:
            durations[field].append(finite_nonnegative(
                sample.get(field), f"sample {index} {field}"))
        baseline = durations["baseline_trace_ms"][-1]
        camera_trace = durations["camera_trace_ms"][-1]
        full = durations["full_camera_ms"][-1]
        if full + 0.01 < camera_trace:
            raise ValueError(f"sample {index} full GPU interval is shorter than camera trace")
        post = max(0.0, full - camera_trace)
        stage_sum = sum(durations[field][-1] for field in POST_STAGES)
        if stage_sum > post + max(0.01, post * 0.01):
            raise ValueError(f"sample {index} stage intervals overlap or exceed post total")
        durations["response_delta_ms"].append(camera_trace - baseline)
        durations["post_total_ms"].append(post)
        # A negative response delta can be measurement jitter, not negative
        # work. Do not give such jitter credit against post-processing cost.
        durations["added_camera_ms"].append(max(0.0, camera_trace - baseline) + post)
    return gpu, driver, warmup, cold_start, {key: p95(value) for key, value in durations.items()}


def main(argv):
    if len(argv) != 2:
        print("usage: check_camera_gpu_perf.py metrics.jsonl", file=sys.stderr)
        return 1
    path = pathlib.Path(argv[1])
    if not path.is_file():
        print(ascii_safe(f"no GPU timing document at {path}"), file=sys.stderr)
        return 2
    try:
        gpu, driver, warmup, cold_start, percentiles = validate(read_jsonl(path))
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(ascii_safe(f"FAIL: {error}"), file=sys.stderr)
        return 1
    print(f"GPU: {ascii_safe(gpu)}; driver: {ascii_safe(driver)}")
    print(f"Extent: 1920x1080; warmup: {warmup}; measured: {MIN_FRAMES}")
    print(f"Cold start CPU wall time: {cold_start:.3f} ms (reported separately)")
    print("Baseline spectral method: single_mid_response; response delta includes path-cost changes")
    print(f"Matched baseline RT GPU P95: {percentiles['baseline_trace_ms']:.3f} ms")
    print(f"Camera RT GPU P95: {percentiles['camera_trace_ms']:.3f} ms")
    print(f"Full camera GPU P95: {percentiles['full_camera_ms']:.3f} ms")
    print(f"RT response/path increment GPU P95: {percentiles['response_delta_ms']:.3f} ms")
    print(f"Camera post total GPU P95: {percentiles['post_total_ms']:.3f} ms")
    for field in POST_STAGES:
        print(f"{field} GPU P95: {percentiles[field]:.3f} ms")
    total = percentiles["added_camera_ms"]
    print(f"Added camera chain GPU P95: {total:.3f} ms (limit {MAX_P95_MS:.3f})")
    if total > MAX_P95_MS:
        print("FAIL: added camera GPU chain exceeds P95 limit", file=sys.stderr)
        return 1
    print("camera GPU performance check PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
