#!/usr/bin/env python3
"""Write the eight-column forcing CSV for the synthetic desert day.

Synthetic on purpose.  The desert scene exists for the mesh-resolution study,
which measures a discretisation error against a reference integrated from the
*same* forcing -- so the forcing has to be a realistic diurnal cycle, and does
not have to be a measured one.  The measured case is the station validation,
which is a different experiment against a different scene for a different claim.

Solar geometry is nonetheless real: NOAA positions for a named place and date,
so the sun moves the way a sun moves and the shadow sweeps rather than jumps.

Row spacing matters more here than it looks.  The scene path builds one GPU
sun-visibility column per row, so a one-minute file costs 1440 shadow
dispatches over 80,000 elements; the solver interpolates between the two
nearest columns, so the cost of a coarse file is a shadow that moves in steps.
Fifteen minutes is the compromise this scene is built around: 0.25 h of solar
motion is about 3.75 degrees of azimuth, which at the sphere's shadow length is
well under one triangle at every tessellation in the sweep.

Usage:
    make_desert_forcing.py --out assets/configs/desert/desert_day.csv
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import solar  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--latitude", type=float, default=36.626,
                        help="default is Desert Rock, NV -- the station the "
                             "measured-forcing experiment uses, so the two "
                             "experiments at least share a sun")
    parser.add_argument("--longitude", type=float, default=-116.018)
    parser.add_argument("--utc-offset", type=float, default=-8.0)
    parser.add_argument("--date", default="2023-07-15", help="YYYY-MM-DD")
    parser.add_argument("--step-h", type=float, default=0.25,
                        help="row spacing on the LAST day, the one an experiment "
                             "evaluates in. The scene path builds one GPU shadow "
                             "column per row and interpolates between the two "
                             "nearest, so this is what decides whether the shadow "
                             "sits where the geometry puts it.")
    parser.add_argument("--spinup-step-h", type=float, default=1.0,
                        help="row spacing on every earlier day. Coarse on purpose: "
                             "spin-up only has to deliver the right thermal state "
                             "at the start of the last day, and the sun-visibility "
                             "table costs one column per row across every element "
                             "-- at 322k elements that is 1.3 MB a row, so a "
                             "uniformly fine multi-day file is gigabytes of table "
                             "to describe hours nothing is measured in.")
    parser.add_argument("--days", type=int, default=4,
                        help="how many copies of the same day to write, back to "
                             "back. More than one is spin-up, and it is not "
                             "optional: a 0.15 m adiabatic sand slab has a "
                             "diffusion time near 27 h, so a run started from an "
                             "imperfect initial condition still carries several "
                             "kelvin of it fifteen hours later -- measured at 4.2 K "
                             "for an 80 K spread in starting temperature. Two days "
                             "of real forcing ahead of the evaluated hour put that "
                             "below the discretisation the study is trying to see. "
                             "Evaluate at (days-1)*24 + hour.")
    parser.add_argument("--t-min-k", type=float, default=295.0)
    parser.add_argument("--t-max-k", type=float, default=313.0)
    parser.add_argument("--relative-humidity", type=float, default=20.0,
                        help="per cent; 20 is a dry summer desert day")
    parser.add_argument("--transmittance", type=float, default=0.75)
    args = parser.parse_args()

    year, month, day = (int(part) for part in args.date.split("-"))

    # Every day is the same day. Repeating one rather than advancing the date
    # is what lets the spin-up converge to a periodic state, so that "the
    # initial condition has been forgotten" can be checked by comparing
    # consecutive days instead of asserted. Measured on this forcing, an 80 K
    # spread in starting temperature falls by a factor of about six per day:
    # 4.2 K after one, 0.11 K after three, 0.017 K after four.
    def sample(hour_of_day, absolute_hour):
        elevation, azimuth = solar.solar_position(
            year, month, day, hour_of_day, args.latitude, args.longitude,
            args.utc_offset)
        dni, dhi = solar.clear_sky_irradiance(elevation, args.transmittance)
        air_k = solar.diurnal_air_temperature_k(hour_of_day, args.t_min_k, args.t_max_k)
        sky_k = solar.clear_sky_temperature_k(air_k, args.relative_humidity)
        return (absolute_hour, air_k, dni, azimuth, max(0.0, elevation), sky_k, dhi,
                args.relative_humidity)

    rows = []
    for d in range(args.days):
        step = args.step_h if d == args.days - 1 else args.spinup_step_h
        hour = 0.0
        while hour < 24.0 - 1e-9:
            rows.append(sample(hour, hour + 24.0 * d))
            hour += step
    # Close the series so its final row is not held flat across an hour an
    # experiment might evaluate at.
    rows.append(sample(0.0, 24.0 * args.days))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"# Synthetic clear-sky desert day -- NOT measured.\n")
        f.write(f"# Solar geometry: NOAA positions for {args.latitude:.3f} N, "
                f"{args.longitude:.3f} E on {args.date}, UTC{args.utc_offset:+.0f}.\n")
        f.write(f"# Irradiance: Laue clear-sky, transmittance {args.transmittance}, "
                f"diffuse 10% of extraterrestrial horizontal.\n")
        f.write(f"# Sky temperature: Berdahl-Fromberg at RH {args.relative_humidity}%.\n")
        f.write(f"# {args.days} identical days: {args.spinup_step_h} h rows through the "
                f"spin-up, {args.step_h} h on the last. Evaluate at "
                f"{24 * (args.days - 1)} + hour.\n")
        f.write("# time_h, air_k, dni, sun_azimuth_deg, sun_elevation_deg, "
                "sky_k, diffuse_w_m2, relative_humidity\n")
        for row in rows:
            f.write("{:.4f}, {:.3f}, {:.2f}, {:.3f}, {:.3f}, {:.3f}, {:.2f}, {:.1f}\n"
                    .format(*row))

    peak = max(rows, key=lambda r: r[2])
    print(f"{args.out}: {len(rows)} rows at {args.step_h} h")
    print(f"  peak DNI {peak[2]:.0f} W/m2 at {peak[0]:.2f} h, "
          f"elevation {peak[4]:.1f} deg, azimuth {peak[3]:.1f} deg")
    print(f"  air {min(r[1] for r in rows):.1f}-{max(r[1] for r in rows):.1f} K, "
          f"sky {min(r[5] for r in rows):.1f}-{max(r[5] for r in rows):.1f} K")


if __name__ == "__main__":
    main()
