#!/usr/bin/env python3
"""E2 -- the surface energy balance against a measured station.

The only experiment in the set that compares the solver to the world rather
than to a closed form or to another implementation. Everything else here
establishes that the code computes what it says it computes; this asks whether
what it says is what a real patch of ground does.

**The station is the scene.** SURFRAD's Desert Rock site is flat, open, bare
desert soil -- which is to say, one surface element under an unoccluded sky,
with no geometry around it to exchange with. That is exactly the configuration
`thermal_column_tool` integrates, so this experiment does not render anything.
Putting the station inside a rendered scene would add view factors the station
does not have and then report the difference as physics.

**Two forcing modes, and the difference between them is a result.**

  (a) measured   the sky temperature is inverted from the station's own
                 measured downwelling longwave, T_sky = (LW_down/sigma)^(1/4).
                 The solver absorbs eps*sigma*s*T_sky^4 with s = 1, which is
                 eps*LW_down exactly -- so this injects the measurement with no
                 model in between, and any error is the solver's.
  (b) modelled   the sky temperature comes from the Berdahl-Fromberg clear-sky
                 correlation driven by the station's air temperature and
                 humidity. Everything else is identical.

(b) minus (a) is the correlation's own contribution, which is otherwise
impossible to separate from the solver's.

**The reference is inverted, not measured.** No station measures surface
temperature directly; it measures upwelling longwave. Inverting

    T_s = [ (LW_up - (1 - eps) LW_down) / (eps sigma) ]^(1/4)

requires a broadband emissivity that nobody measured either, so eps is a stated
assumption and the sensitivity to it is reported rather than buried: every
result is given at eps = 0.96 and repeated at 0.94 and 0.98.

**Anti-hindsight.** Thermal properties are taken from the literature range for
dry soil and fixed BEFORE the comparison. If any is tuned, it is tuned on ONE
window and the others are held out; the split is stated in the output. The
attribution candidates for a disagreement -- latent heat, soil moisture,
lateral conduction -- are the ones the manuscript pre-registered, and nothing
is added to that list afterwards.

Usage:
    e2_surfrad.py --fetch --year 2023 --days 150 250
    e2_surfrad.py --select-windows
    e2_surfrad.py --run --out-dir evidence/e2_surfrad
"""

import argparse
import json
import math
import pathlib
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import solar  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
CACHE = pathlib.Path(r"H:\quantiloom-paper\evidence\e2_surfrad\raw")
TOOL = REPO / "build" / "src" / "tools" / "Release" / "thermal_column_tool.exe"

STATION = {
    "name": "Desert Rock, NV",
    "code": "dra",
    "directory": "Desert_Rock_NV",
    "latitude": 36.624,
    "longitude": -116.019,
    "elevation_m": 1007.0,
    "utc_offset_h": -8.0,   # Pacific standard time; SURFRAD timestamps are UTC
}
BASE_URL = "https://gml.noaa.gov/aftp/data/radiation/surfrad"

STEFAN_BOLTZMANN = 5.670374419e-8

# SURFRAD's one-minute record: eight leading fields, then value/QC pairs. The
# mapping below was checked against the file's own arithmetic -- netir must
# equal dw_ir minus uw_ir, and it does, to the printed digit.
LEADING = 8
PAIRS = ["dw_solar", "uw_solar", "direct_n", "diffuse", "dw_ir", "dw_casetemp",
         "dw_dometemp", "uw_ir", "uw_casetemp", "uw_dometemp", "uvb", "par",
         "netsolar", "netir", "totalnet", "temp", "rh", "windspd", "winddir",
         "pressure"]

# Bare desert soil. Literature values, fixed before any comparison; see the
# module docstring on why they are not tuned afterwards.
SOIL = {
    "thermal_conductivity_w_mk": 0.50,   # dry soil, 0.3-1.0
    "density_kg_m3": 1600.0,
    "specific_heat_j_kgk": 875.0,        # rho*c = 1.4 MJ/m3K, within 1.2-1.6
    "thickness_m": 0.50,                 # >= 3x the diurnal skin depth (0.099 m)
    # Selected on the CALIBRATION window (days 151-155) and held fixed for every
    # other window, which is the split this experiment declared before running.
    # The a-priori value was 10, the middle of a textbook "still air to light
    # breeze" range, and it was wrong by 5 K of daytime bias. Two things make 20
    # a correction rather than a fit: the calibration sweep is a clean minimum
    # (5.84 K day RMSE at h=10, 1.18 K at h=20, 4.00 K at h=30), and the
    # station's own wind record puts a standard flat-plate correlation --
    # McAdams, h = 5.7 + 3.8u, at a daytime mean of 4.36 m/s -- at 22.3, which
    # lands within 10% of the minimum without having been told about it.
    "convection_h_w_m2k": 20.0,
    "ir_emissivity": 0.96,
}
NODE_COUNT = 24
TIMESTEP_S = 60.0
SPINUP_DAYS = 2


def fetch(year, day_range):
    CACHE.mkdir(parents=True, exist_ok=True)
    got = 0
    for day in range(day_range[0], day_range[1] + 1):
        name = f"{STATION['code']}{year % 100:02d}{day:03d}.dat"
        target = CACHE / name
        if target.is_file() and target.stat().st_size > 0:
            continue
        url = f"{BASE_URL}/{STATION['directory']}/{year}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                target.write_bytes(response.read())
            got += 1
        except Exception as error:  # noqa: BLE001 - a missing day is normal
            print(f"  {name}: {error}", file=sys.stderr)
    print(f"fetched {got} new file(s) into {CACHE}")


def read_day(path):
    """One SURFRAD day as a list of dicts, QC-screened.

    A QC flag other than 0 means the value failed the network's own checks;
    such a record is dropped whole rather than patched, because the balance
    needs a consistent instant rather than a plausible one.
    """
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < LEADING + 2 * len(PAIRS):
            continue
        try:
            year, jday = int(parts[0]), int(parts[1])
            hour, minute = int(parts[4]), int(parts[5])
        except ValueError:
            continue  # header lines

        record = {"year": year, "jday": jday,
                  "utc_hour": hour + minute / 60.0,
                  "zenith_deg": float(parts[7])}
        ok = True
        for index, key in enumerate(PAIRS):
            value = float(parts[LEADING + 2 * index])
            flag = int(float(parts[LEADING + 2 * index + 1]))
            if flag != 0 and key in ("dw_solar", "direct_n", "diffuse", "dw_ir",
                                     "uw_ir", "temp", "rh", "uw_solar",
                                     "windspd"):
                ok = False
                break
            record[key] = value
        if ok:
            rows.append(record)
    return rows


def clear_sky_score(rows):
    """How closely a day's global irradiance follows a cloudless envelope.

    Haurwitz: GHI_clear = 1098 * cos(z) * exp(-0.059 / cos(z)). Crude by modern
    standards and entirely adequate as a screen -- a cloudy day departs from it
    by tens of percent, and what is being selected for is the absence of cloud
    rather than the value of the envelope.
    """
    residuals = []
    for row in rows:
        cos_z = math.cos(math.radians(row["zenith_deg"]))
        if cos_z <= 0.10:
            continue
        clear = 1098.0 * cos_z * math.exp(-0.059 / cos_z)
        if clear <= 20.0:
            continue
        residuals.append(abs(row["dw_solar"] - clear) / clear)
    if len(residuals) < 60:
        return None
    residuals.sort()
    return residuals[len(residuals) // 2]


def surface_temperature(row, emissivity):
    """The station's own surface temperature, inverted from upwelling longwave."""
    numerator = row["uw_ir"] - (1.0 - emissivity) * row["dw_ir"]
    if numerator <= 0.0:
        return None
    return (numerator / (emissivity * STEFAN_BOLTZMANN)) ** 0.25


def write_forcing(rows, path, mode, sample_minutes=10):
    """The solver's eight-column CSV, at a coarser cadence than one minute.

    Ten minutes rather than one because the sun-visibility table costs a column
    per row and the surface's own time constant is hours; the forcing is
    interpolated between rows, so this resolves the diurnal cycle to well
    inside the measurement's own noise.
    """
    written = []
    for row in rows:
        minutes = round(row["utc_hour"] * 60.0)
        if minutes % sample_minutes != 0:
            continue

        air_k = row["temp"] + 273.15
        humidity = max(1.0, min(100.0, row["rh"]))
        if mode == "measured":
            sky_k = (max(1.0, row["dw_ir"]) / STEFAN_BOLTZMANN) ** 0.25
        else:
            sky_k = solar.clear_sky_temperature_k(air_k, humidity)

        local_hour = row["utc_hour"] + STATION["utc_offset_h"]
        elevation, azimuth = solar.solar_position(
            row["year"], 1, row["jday"], local_hour,
            STATION["latitude"], STATION["longitude"], STATION["utc_offset_h"])

        written.append({
            "time_h": row["absolute_hour"],
            "air_k": air_k,
            "dni": max(0.0, row["direct_n"]),
            "azimuth": azimuth % 360.0,
            "elevation": max(0.0, elevation),
            "sky_k": sky_k,
            "diffuse": max(0.0, row["diffuse"]),
            "rh": humidity,
            "surfrad_elevation": 90.0 - row["zenith_deg"],
        })

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# SURFRAD {STATION['name']}, forcing mode '{mode}'.\n")
        f.write("# time_h is hours from the start of the spin-up window.\n")
        if mode == "measured":
            f.write("# sky_k inverted from the station's measured LW down:\n"
                    "#   T_sky = (LW_down / sigma)^(1/4). The solver absorbs\n"
                    "#   eps*sigma*s*T_sky^4 with s = 1, i.e. eps*LW_down exactly.\n")
        else:
            f.write("# sky_k from Berdahl-Fromberg driven by the station's own\n"
                    "#   air temperature and relative humidity. Everything else\n"
                    "#   is identical to the measured mode.\n")
        f.write("# time_h, air_k, dni, sun_azimuth_deg, sun_elevation_deg, "
                "sky_k, diffuse_w_m2, relative_humidity\n")
        for r in written:
            f.write(f"{r['time_h']:.5f}, {r['air_k']:.3f}, {r['dni']:.2f}, "
                    f"{r['azimuth']:.3f}, {r['elevation']:.3f}, {r['sky_k']:.3f}, "
                    f"{r['diffuse']:.2f}, {r['rh']:.1f}\n")

    # The solar model against the station's own measurement of the same angle.
    # Not decorative: if these disagree the forcing's sun is in the wrong place
    # and every direct-beam term is wrong with it.
    errors = [abs(r["elevation"] - r["surfrad_elevation"])
              for r in written if r["surfrad_elevation"] > 5.0]
    return written, (max(errors) if errors else None)


def albedo_of(rows):
    """Short-wave absorptivity from the station's own up/down solar pair."""
    pairs = [(r["uw_solar"], r["dw_solar"]) for r in rows if r["dw_solar"] > 200.0]
    if not pairs:
        return None
    return sum(u / d for u, d in pairs) / len(pairs)


def write_spec(path, forcing_path, hours, absorptivity, emissivity, nodes=NODE_COUNT,
               interior="adiabatic"):
    path.write_text(f"""# Auto-generated by scripts/experiments/e2_surfrad.py
[column]
thermal_conductivity_w_mk = {SOIL['thermal_conductivity_w_mk']}
density_kg_m3             = {SOIL['density_kg_m3']}
specific_heat_j_kgk       = {SOIL['specific_heat_j_kgk']}
thickness_m               = {SOIL['thickness_m']}
convection_h_w_m2k        = {SOIL['convection_h_w_m2k']}
shortwave_absorptivity    = {absorptivity:.4f}
ir_emissivity             = {emissivity:.4f}
interior_bc               = "{interior}"
interior_temperature_k    = 290.0

[solve]
start_time_h  = 0.0
end_time_h    = {hours:.4f}
timestep_s    = {TIMESTEP_S}
layers        = {nodes}
initial       = "steady"
output_step_h = {10.0 / 60.0:.6f}
checkpoint_stride_h = 24.0

[geometry]
normal       = [0.0, 1.0, 0.0]
sky_fraction = 1.0

[forcing]
file = "{forcing_path.as_posix()}"
""", encoding="utf-8")


def run_column(spec, out_csv):
    result = subprocess.run([str(TOOL), str(spec), "--out", str(out_csv)],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(f"{spec.name} failed:\n{result.stdout}\n{result.stderr}")
    rows = []
    for line in out_csv.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 2:
            rows.append((float(parts[0]), float(parts[1])))
    return rows


def statistics(predicted, observed, is_day, evaluate_from_h):
    """Day and night error, and the phase lag, over the evaluation window only.

    Day and night separately because they are different physics: daytime error
    is dominated by the short-wave terms and the surface's response to them,
    night by the long-wave balance and the heat the slab stored during the day.
    A single figure averages a bias in one against a bias in the other, and the
    literature this is being compared against reports them apart for the same
    reason.
    """
    import numpy as np

    times = np.array([t for t, _ in observed])
    obs = np.array([v for _, v in observed])
    day_mask = np.array(is_day, dtype=bool)
    pred = np.interp(times, [t for t, _ in predicted], [v for _, v in predicted])

    keep = times >= evaluate_from_h
    times, obs, pred, day_mask = times[keep], obs[keep], pred[keep], day_mask[keep]
    if times.size == 0:
        return None

    def summarise(mask):
        if not np.any(mask):
            return None
        residual = pred[mask] - obs[mask]
        return {
            "n": int(mask.sum()),
            "rmse_k": float(np.sqrt(np.mean(residual ** 2))),
            "mae_k": float(np.mean(np.abs(residual))),
            "bias_k": float(np.mean(residual)),
            "observed_range_k": [float(obs[mask].min()), float(obs[mask].max())],
        }

    # Phase: the time of each day's maximum, predicted against observed. A
    # cross-correlation over the whole record would be dominated by the diurnal
    # fundamental and would report the lag of a sinusoid rather than of a peak,
    # which is the quantity a thermal image is sensitive to.
    lags = []
    for start in np.arange(math.floor(times.min() / 24.0) * 24.0,
                           times.max(), 24.0):
        window = (times >= start) & (times < start + 24.0)
        if window.sum() < 60:
            continue
        lags.append(float(times[window][np.argmax(pred[window])]
                          - times[window][np.argmax(obs[window])]))

    return {
        "day": summarise(day_mask),
        "night": summarise(~day_mask),
        "all": summarise(np.ones_like(day_mask)),
        "peak_lag_h": {
            "per_day": lags,
            "median": float(np.median(lags)) if lags else None,
        },
        "series": {"time_h": times.tolist(), "observed_k": obs.tolist(),
                   "predicted_k": pred.tolist(), "is_day": day_mask.tolist()},
    }


def assemble_window(days, year, emissivity):
    """One contiguous run of days as a single time series, hours from its start."""
    rows, observed, is_day = [], [], []
    first = days[0]
    for day in days:
        path = CACHE / f"{STATION['code']}{year % 100:02d}{day:03d}.dat"
        if not path.is_file():
            continue
        for row in read_day(path):
            row["absolute_hour"] = (row["jday"] - first) * 24.0 + row["utc_hour"]
            temperature = surface_temperature(row, emissivity)
            if temperature is None:
                continue
            rows.append(row)
            observed.append((row["absolute_hour"], temperature))
            is_day.append(row["zenith_deg"] < 90.0)
    return rows, observed, is_day


def run(args):
    import numpy as np  # noqa: F401  (statistics needs it; fail early if absent)

    windows = json.loads((CACHE.parent / "windows.json").read_text(encoding="utf-8"))
    selected = windows["windows"][: args.max_windows]
    if not selected:
        raise SystemExit("no windows; run --select-windows first")

    out_dir = pathlib.Path(args.out_dir)
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    results = {"station": STATION, "soil": SOIL, "node_count": NODE_COUNT,
               "timestep_s": TIMESTEP_S, "spinup_days": SPINUP_DAYS,
               "emissivity_nominal": args.emissivity,
               "emissivity_sensitivity": args.emissivity_sensitivity,
               "calibration_window": selected[0][0],
               "windows": []}

    for days in selected:
        label = f"{args.year}_{days[0]:03d}-{days[-1]:03d}"
        print(f"\nwindow {label}  ({len(days)} days, "
              f"{SPINUP_DAYS} of spin-up then {len(days) - SPINUP_DAYS} evaluated)")

        entry = {"label": label, "days": days,
                 "evaluate_from_h": SPINUP_DAYS * 24.0, "modes": {}}

        for emissivity in ([args.emissivity] + args.emissivity_sensitivity):
            rows, observed, is_day = assemble_window(days, args.year, emissivity)
            if len(observed) < 1000:
                print(f"  eps={emissivity}: only {len(observed)} usable records, skipped")
                continue
            absorptivity = 1.0 - (albedo_of(rows) or 0.22)

            for mode in ("measured", "modelled"):
                forcing = work / f"forcing_{label}_{mode}_e{emissivity:.2f}.csv"
                written, solar_error = write_forcing(rows, forcing, mode)
                spec = work / f"spec_{label}_{mode}_e{emissivity:.2f}.toml"
                write_spec(spec, forcing, written[-1]["time_h"],
                           absorptivity, emissivity)
                predicted = run_column(spec, work / f"traj_{label}_{mode}_"
                                                    f"e{emissivity:.2f}.csv")
                stats = statistics(predicted, observed, is_day,
                                   entry["evaluate_from_h"])
                key = f"{mode}_e{emissivity:.2f}"
                entry["modes"][key] = {
                    "mode": mode, "emissivity": emissivity,
                    "shortwave_absorptivity": absorptivity,
                    "solar_elevation_max_error_deg": solar_error,
                    "forcing_rows": len(written),
                    **{k: v for k, v in stats.items() if k != "series"},
                }
                if emissivity == args.emissivity:
                    entry["modes"][key]["series"] = stats["series"]
                day, night = stats["day"], stats["night"]
                print(f"  {key:<20} day RMSE {day['rmse_k']:5.2f} K  "
                      f"night RMSE {night['rmse_k']:5.2f} K  "
                      f"day bias {day['bias_k']:+5.2f}  "
                      f"peak lag {stats['peak_lag_h']['median']:+.2f} h  "
                      f"(sun model err {solar_error:.3f} deg)")

        results["windows"].append(entry)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e2_surfrad.json").write_text(json.dumps(results, indent=2),
                                             encoding="utf-8")
    print(f"\nwrote {out_dir / 'e2_surfrad.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--days", type=int, nargs=2, default=[150, 250])
    parser.add_argument("--select-windows", action="store_true")
    parser.add_argument("--clear-threshold", type=float, default=0.08)
    parser.add_argument("--window-days", type=int, default=3)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--out-dir", default=r"H:\quantiloom-paper\evidence\e2_surfrad")
    parser.add_argument("--max-windows", type=int, default=4)
    parser.add_argument("--emissivity", type=float, default=0.96,
                        help="broadband soil emissivity for the inversion; stated "
                             "rather than measured, which is why it is swept")
    parser.add_argument("--emissivity-sensitivity", type=float, nargs="*",
                        default=[0.94, 0.98])
    args = parser.parse_args()

    if args.fetch:
        fetch(args.year, args.days)

    if args.select_windows:
        scores = {}
        for day in range(args.days[0], args.days[1] + 1):
            path = CACHE / f"{STATION['code']}{args.year % 100:02d}{day:03d}.dat"
            if not path.is_file():
                continue
            rows = read_day(path)
            score = clear_sky_score(rows)
            if score is not None:
                scores[day] = score
        clear = sorted(d for d, s in scores.items() if s <= args.clear_threshold)
        streaks, current = [], []
        for day in clear:
            if current and day == current[-1] + 1:
                current.append(day)
            else:
                if len(current) >= args.window_days + SPINUP_DAYS:
                    streaks.append(current)
                current = [day]
        if len(current) >= args.window_days + SPINUP_DAYS:
            streaks.append(current)
        print(f"days scored: {len(scores)}; clear (median residual <= "
              f"{args.clear_threshold:.0%}): {len(clear)}")
        for streak in streaks:
            print(f"  window {streak[0]}-{streak[-1]} ({len(streak)} days), "
                  f"median residual {max(scores[d] for d in streak):.3f}")
        (CACHE.parent / "windows.json").write_text(
            json.dumps({"scores": scores, "windows": streaks}, indent=2),
            encoding="utf-8")
        print(f"wrote {CACHE.parent / 'windows.json'}")

    if args.run:
        run(args)


if __name__ == "__main__":
    main()
