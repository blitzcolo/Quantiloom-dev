#!/usr/bin/env python3
"""Shared pieces for the cross-renderer validation against Mitsuba 3.

The point of this module is that every convention Quantiloom and Mitsuba could
disagree about is written down once, in code, with the reason attached.  A
cross-renderer comparison is mostly an exercise in not comparing two different
questions, and the ways to ask two different questions here are numerous and
quiet:

  * `fov_axis` defaults to `x` in Mitsuba and `y` in Quantiloom's configs;
  * Mitsuba's default reconstruction filter is a Gaussian, which is a different
    image from a box even at convergence;
  * `specfilm` defaults to float16 storage, which is 3 decimal digits;
  * a float or RGB emitter value in a spectral variant is silently wrapped in
    D65, so an "equal-energy" emitter is nothing of the sort;
  * `directional`'s `direction` is where the light *goes*, the opposite of the
    surface-to-sun vector Quantiloom's configs carry.

Each is handled below rather than in the individual experiment scripts.

Radiometric conventions, both verified numerically against closed forms:

  * `specfilm` stores, per channel, the Monte Carlo estimate of the *integral*
    int srf(lambda) L(lambda) dlambda -- not an average.  So a boxcar of height
    1/(band width) yields the band average, which is what Quantiloom's fused IR
    modes write; and a narrow boxcar of height 1/(window width) yields
    L(lambda_c) itself, which is what its single-wavelength mode writes.
  * Mitsuba is not limited to 360-830 nm.  That range bounds the CIE tables,
    the RGB upsampling and the *default* wavelength sampling; with a film
    spectral response the sensor importance-samples the response's own support.
    SWIR and LWIR comparisons therefore need no custom build.
"""

import hashlib
import json
import pathlib
import subprocess
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

def read_spectral_csv(path):
    """(wavelengths_nm, values) from a two-column CSV, `#` comments allowed."""
    wavelengths, values = [], []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 2:
            continue
        try:
            wavelengths.append(float(parts[0]))
            values.append(float(parts[1]))
        except ValueError:
            continue  # a header row
    return np.asarray(wavelengths), np.asarray(values)


def irregular(wavelengths, values):
    """A Mitsuba `irregular` spectrum dict from arbitrary sample positions.

    Mitsuba interpolates these piecewise-linearly, which is bit-identical to
    numpy's interpolation over the same nodes -- so at a wavelength that is a
    node, both renderers evaluate the same number from the same file, and
    there is no spectral discretisation difference left to explain a residual.
    """
    return {
        "type": "irregular",
        "wavelengths": ", ".join(f"{w:.6g}" for w in wavelengths),
        "values": ", ".join(f"{v:.9g}" for v in values),
    }


def flat(value, lo=200.0, hi=15000.0):
    """A constant spectrum over a range wide enough for any band used here."""
    return {"type": "regular", "wavelength_min": lo, "wavelength_max": hi,
            "values": f"{value:.9g}, {value:.9g}"}


def spectrum_from_csv(path):
    return irregular(*read_spectral_csv(path))


# ---------------------------------------------------------------------------
# Film spectral response
# ---------------------------------------------------------------------------

def band_srf(lo_nm, hi_nm):
    """A boxcar of height 1/(hi-lo): the channel becomes the band AVERAGE.

    Which is the unit Quantiloom's fused IR modes write -- band integral over
    band width, W/sr/m^2/nm -- so the two files can be compared directly rather
    than through a conversion that would have to be trusted.
    """
    height = 1.0 / (hi_nm - lo_nm)
    return {"type": "regular", "wavelength_min": lo_nm, "wavelength_max": hi_nm,
            "values": f"{height:.9g}, {height:.9g}"}


def single_srf(centre_nm, half_width_nm=1.0):
    """A boxcar so narrow the channel is L(centre) itself.

    The window has to be narrow enough that the radiance does not curve across
    it and wide enough that the sampler does not starve; +/-1 nm satisfies both
    for every spectrum used here, and +/-0.2 nm gives the same answer, which is
    the check that the window is not itself a parameter of the result.
    """
    height = 1.0 / (2.0 * half_width_nm)
    return {"type": "regular",
            "wavelength_min": centre_nm - half_width_nm,
            "wavelength_max": centre_nm + half_width_nm,
            "values": f"{height:.9g}, {height:.9g}"}


def specfilm(width, height, srf, channel="V"):
    """A single-channel spectral film.

    `box` rather than the default Gaussian: a reconstruction filter is part of
    the image, and Quantiloom writes box-filtered pixels. `float32` rather than
    the default float16, because a ratio quoted to eight significant figures
    cannot come out of a three-digit buffer.
    """
    return {
        "type": "specfilm",
        "width": int(width),
        "height": int(height),
        "component_format": "float32",
        "rfilter": {"type": "box"},
        channel: srf,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(scene_dict, spp, seed):
    """Render a scene dict and return the first channel as float64 (H, W)."""
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("scalar_spectral")
    scene = mi.load_dict(scene_dict)
    image = mi.render(scene, spp=int(spp), seed=int(seed))
    array = np.asarray(image, dtype=np.float64)
    return array[:, :, 0] if array.ndim == 3 else array


def half_split_noise(scene_dict, spp, seeds):
    """The Monte Carlo floor, from two independent renders.

    RMS(A - B)/sqrt(2) estimates the per-pixel standard deviation of one render
    without needing a converged reference. Reported beside any cross-renderer
    difference, because a difference smaller than this floor is not a
    difference between renderers.
    """
    a = render(scene_dict, spp, seeds[0])
    b = render(scene_dict, spp, seeds[1])
    return float(np.sqrt(np.mean((a - b) ** 2) / 2.0)), a, b


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def sha256(path):
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo):
    try:
        result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                capture_output=True, timeout=20)
        if result.returncode == 0:
            return result.stdout.decode("utf-8", "replace").strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def manifest(label, extra=None):
    import mitsuba as mi
    return {
        "label": label,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mitsuba_version": mi.__version__,
        "mitsuba_variant": mi.variant(),
        "quantiloom_commit": git_commit(REPO),
        **(extra or {}),
    }


def write_manifest(path, data):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Quantiloom side
# ---------------------------------------------------------------------------

DEFAULT_QL = REPO / "build" / "src" / "app" / "Release" / "Quantiloom.exe"


def run_quantiloom(config, cli=DEFAULT_QL, cwd=REPO, timeout=7200):
    result = subprocess.run([str(cli), str(config)], cwd=str(cwd),
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"{config} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def read_exr(path, channel=0):
    """First channel of an EXR as float64 (H, W)."""
    import OpenEXR
    f = OpenEXR.File(str(path))
    pixels = list(f.channels().values())[0].pixels
    array = np.asarray(pixels, dtype=np.float64)
    return array[:, :, channel] if array.ndim == 3 else array
