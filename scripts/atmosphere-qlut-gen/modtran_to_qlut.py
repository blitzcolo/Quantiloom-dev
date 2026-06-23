"""
Convert QLTrans output (sun.txt, sky.txt, trans.txt) to .qlut format.

LUT is parameterized by wavelength × solar zenith angle (2D texture).
Each discrete atmospheric configuration (model × weather × ihaze) produces one .qlut file.

Usage:
    python modtran_to_qlut.py --sun sun.txt --sky sky.txt --trans trans.txt \
        --wave 3 --model 2 --weather 0 --ihaze 4 \
        --wl-start 8000 --wl-stop 12000 --wl-step 20 \
        --sun-zen-start 0 --sun-zen-stop 90 --sun-zen-step 1 \
        --out atmosphere.qlut
"""
import argparse
import datetime
import numpy as np

WAVE_RANGES = {1: (380, 760, 50), 2: (3000, 5000, 50), 3: (8000, 12000, 50)}
WAVE_NAMES  = {1: "VIS", 2: "MWIR", 3: "LWIR"}
MODEL_NAMES = {1: "Tropical", 2: "Midlatitude_Summer", 3: "Midlatitude_Winter",
               4: "Subarctic_Summer", 5: "Subarctic_Winter", 6: "US_Standard_1976"}
IHAZE_NAMES = {0: "No_Aerosol", 1: "Rural_23km", 2: "Rural_5km",
               3: "Navy_Maritime", 4: "Maritime_23km", 5: "Urban_5km",
               6: "Tropospheric_50km"}
HEADER_SIZE = 2048


def load_angular_data(path, n_wave):
    """
    Load angle-indexed data (sky or trans).
    File format: angle_index\\n then n_wave lines of 'wl\\tval'
    Returns (angle_indices, data[n_angles, n_wave]) as float32.
    """
    angles, rows = [], []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines):
        angles.append(int(lines[i])); i += 1
        row = []
        for _ in range(n_wave):
            row.append(float(lines[i].split()[1])); i += 1
        rows.append(row)
    return np.array(angles, dtype=np.int32), np.array(rows, dtype=np.float32)


def make_header(wave, model, weather, ihaze,
                wl_start, wl_stop, wl_step, wl_count,
                sun_zen_start, sun_zen_stop, sun_zen_step, n_sun,
                trans_count, path_rad_count):
    sun_values = ", ".join(f"{sun_zen_start + i * sun_zen_step:.1f}" for i in range(n_sun))
    path_rad_offset = HEADER_SIZE + trans_count * 4
    toml = f"""[metadata]
name = "{WAVE_NAMES[wave]} {MODEL_NAMES.get(model, str(model))} ihaze={ihaze} weather={weather}"
source = "QLTrans/MOD4v1r1"
created = "{datetime.date.today().isoformat()}"
atmospheric_model = "{MODEL_NAMES.get(model, str(model))}"
ihaze = {ihaze}
weather = {weather}
format_version = 2

[grid.wavelength]
start = {wl_start:.1f}
stop = {wl_stop:.1f}
step = {wl_step:.1f}
count = {wl_count}

[grid.altitude]
start = 0.0
stop = 0.0
step = 1.0
count = 1

[grid.zenith]
semantic = "solar_zenith"
start = {sun_zen_start:.1f}
stop = {sun_zen_stop:.1f}
step = {sun_zen_step:.1f}
count = {n_sun}
values = [{sun_values}]

[data]
header_size = {HEADER_SIZE}
transmittance_offset = {HEADER_SIZE}
transmittance_count = {trans_count}
path_radiance_offset = {path_rad_offset}
path_radiance_count = {path_rad_count}
dtype = "float32"
byte_order = "little"
"""
    encoded = toml.encode("utf-8")
    assert len(encoded) < HEADER_SIZE, f"Header too large: {len(encoded)} bytes"
    return encoded + b"\x00" * (HEADER_SIZE - len(encoded))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sun",     required=True)
    ap.add_argument("--sky",     required=True)
    ap.add_argument("--trans",   required=True)
    ap.add_argument("--wave",    type=int, required=True, help="1=VIS 2=MWIR 3=LWIR")
    ap.add_argument("--model",   type=int, default=2)
    ap.add_argument("--weather", type=int, default=0)
    ap.add_argument("--ihaze",   type=int, default=4)
    ap.add_argument("--wl-start", type=float, default=0)
    ap.add_argument("--wl-stop",  type=float, default=0)
    ap.add_argument("--wl-step",  type=float, default=0)
    ap.add_argument("--sun-zen-start", type=float, default=0)
    ap.add_argument("--sun-zen-stop",  type=float, default=90)
    ap.add_argument("--sun-zen-step",  type=float, default=1)
    ap.add_argument("--out",     required=True)
    args = ap.parse_args()

    # Wavelength params
    if args.wl_start > 0 and args.wl_stop > 0 and args.wl_step > 0:
        wl_start, wl_stop, wl_step = args.wl_start, args.wl_stop, args.wl_step
    else:
        wl_start, wl_stop, wl_step = WAVE_RANGES[args.wave]
    wl_count = int(round((wl_stop - wl_start) / wl_step)) + 1

    # Solar zenith params
    n_sun = int(round((args.sun_zen_stop - args.sun_zen_start) / args.sun_zen_step)) + 1

    # Load data — sun.txt now also has angular entries
    _, sky_mat = load_angular_data(args.sky, wl_count)
    _, trans_mat = load_angular_data(args.trans, wl_count)

    assert sky_mat.shape[0] == n_sun, \
        f"Sky data has {sky_mat.shape[0]} angles, expected {n_sun}"
    assert trans_mat.shape[0] == n_sun, \
        f"Trans data has {trans_mat.shape[0]} angles, expected {n_sun}"

    # Layout: [wavelength][altitude=1][solar_zenith]
    trans_3d = trans_mat.T[:, np.newaxis, :]   # [n_wave, 1, n_sun]
    sky_3d   = sky_mat.T[:, np.newaxis, :]     # [n_wave, 1, n_sun]

    trans_bytes = trans_3d.astype("<f4").tobytes()
    sky_bytes   = sky_3d.astype("<f4").tobytes()

    header = make_header(args.wave, args.model, args.weather, args.ihaze,
                         wl_start, wl_stop, wl_step, wl_count,
                         args.sun_zen_start, args.sun_zen_stop,
                         args.sun_zen_step, n_sun,
                         len(trans_3d.flatten()), len(sky_3d.flatten()))

    with open(args.out, "wb") as f:
        f.write(header)
        f.write(trans_bytes)
        f.write(sky_bytes)

    print(f"Written: {args.out}  ({wl_count} wavelengths x {n_sun} solar zenith angles)")


if __name__ == "__main__":
    main()
