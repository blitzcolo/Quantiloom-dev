#!/usr/bin/env python3
"""Figure 10 and Table VIII from the station-forcing results.

Two panels over one clear-sky window: the trajectory the solver produced
against the surface temperature inverted from the station's own upwelling
longwave, and the residual beneath it. Night is shaded, because the two halves
of the day are different physics and the paper reports them separately -- the
figure should make that split visible rather than leave the reader to infer it
from a table.

Usage:
    e2_figure.py --results evidence/e2_surfrad/e2_surfrad.json --out figures/
"""

import argparse
import json
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def shade_night(axis, times, is_day):
    """Shade the intervals where the sun is below the horizon."""
    start = None
    for index, day in enumerate(is_day):
        if not day and start is None:
            start = times[index]
        elif day and start is not None:
            axis.axvspan(start, times[index], color="0.92", zorder=0, lw=0)
            start = None
    if start is not None:
        axis.axvspan(start, times[-1], color="0.92", zorder=0, lw=0)


def figure(window, out_path, station):
    measured = window["modes"]["measured_windh_e0.96"]
    series = measured["series"]
    times = np.asarray(series["time_h"])
    observed = np.asarray(series["observed_k"])
    predicted = np.asarray(series["predicted_k"])
    is_day = series["is_day"]

    figure_, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.2, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    shade_night(top, times, is_day)
    shade_night(bottom, times, is_day)

    top.plot(times, observed, color="0.25", lw=1.1,
             label="inverted from station LW$\\uparrow$  ($\\varepsilon$ = 0.96)")
    top.plot(times, predicted, color="#c1121f", lw=1.1, ls="--",
             label="surface energy balance")
    top.set_ylabel("surface temperature (K)")
    top.legend(loc="upper right", frameon=False, fontsize=8)

    residual = predicted - observed
    bottom.axhline(0.0, color="0.6", lw=0.7)
    bottom.plot(times, residual, color="#c1121f", lw=0.9)
    bottom.set_ylabel("residual (K)")
    bottom.set_xlabel(f"hours from {window['days'][0]:03d}/"
                      f"{window['label'].split('_')[0]} 00:00 UTC")

    day, night = measured["day"], measured["night"]
    top.set_title(
        f"{station['name']} \u2014 days {window['days'][0]}\u2013{window['days'][-1]}, "
        f"{int(window['evaluate_from_h'] // 24)} d spin-up discarded\n"
        f"wind-driven convection, no fitted parameter — "
        f"day RMSE {day['rmse_k']:.2f} K, night RMSE {night['rmse_k']:.2f} K",
        fontsize=9)

    limit = max(3.0, float(np.abs(residual).max()) * 1.1)
    bottom.set_ylim(-limit, limit)
    for axis in (top, bottom):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure_.savefig(out_path, dpi=300, bbox_inches="tight")
    figure_.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure_)
    return out_path


def table(results):
    rows = []
    lags = []
    for window in results["windows"]:
        for key in ("measured_e0.96", "modelled_e0.96",
                    "measured_windh_e0.96"):
            entry = window["modes"].get(key)
            if not entry:
                continue
            rows.append({
                "window": window["label"],
                "days_evaluated": len(window["days"]) - results["spinup_days"],
                "forcing": entry["mode"],
                "convection": ("wind-driven" if entry.get("wind_driven_h")
                               else "constant"),
                "day_rmse_k": entry["day"]["rmse_k"],
                "day_mae_k": entry["day"]["mae_k"],
                "day_bias_k": entry["day"]["bias_k"],
                "night_rmse_k": entry["night"]["rmse_k"],
                "night_mae_k": entry["night"]["mae_k"],
                "night_bias_k": entry["night"]["bias_k"],
                "peak_lag_h": entry["peak_lag_h"]["median"],
                "calibration": window["days"][0] == results["calibration_window"],
            })
            if entry["mode"] == "measured" and not entry.get("wind_driven_h"):
                lags += entry["peak_lag_h"]["per_day"]

    lines = ["| Window (DOY) | Days | Sky | Convection | Day RMSE | Day MAE | "
             "Day bias | Night RMSE | Night MAE | Night bias | Peak lag |",
             "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        mark = " \u2020" if r["calibration"] else ""
        lines.append(
            f"| {r['window'].split('_')[1]}{mark} | {r['days_evaluated']} | "
            f"{r['forcing']} | {r['convection']} | "
            f"{r['day_rmse_k']:.2f} K | {r['day_mae_k']:.2f} K | "
            f"{r['day_bias_k']:+.2f} K | {r['night_rmse_k']:.2f} K | "
            f"{r['night_mae_k']:.2f} K | {r['night_bias_k']:+.2f} K | "
            f"{r['peak_lag_h']:+.2f} h |")
    return rows, "\n".join(lines), lags


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=pathlib.Path,
                        default=pathlib.Path(r"H:\quantiloom-paper\evidence"
                                             r"\e2_surfrad\e2_surfrad.json"))
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path(r"H:\quantiloom-paper\figures"))
    parser.add_argument("--window-index", type=int, default=1,
                        help="which window to plot; the default is the first "
                             "HELD-OUT one, not the calibration window")
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    path = figure(results["windows"][args.window_index],
                  args.out / "fig10_surfrad_trajectory.png", results["station"])
    rows, markdown, lags = table(results)

    (args.out / "table8_surfrad.md").write_text(
        markdown + "\n\n\u2020 calibration window: the convection coefficient was "
        "selected here and held fixed for every other window.\n"
        f"\nPooled peak lag over {len(lags)} evaluated days: median "
        f"{statistics.median(lags):+.2f} h, interquartile range "
        f"[{sorted(lags)[len(lags) // 4]:+.2f}, "
        f"{sorted(lags)[3 * len(lags) // 4]:+.2f}] h.\n", encoding="utf-8")
    (args.out / "table8_surfrad.json").write_text(json.dumps(rows, indent=2),
                                                  encoding="utf-8")
    print(f"wrote {path}")
    print(f"wrote {args.out / 'table8_surfrad.md'}")
    print()
    # The files are UTF-8; this print has to survive the console's code page.
    print(markdown.encode('ascii', 'replace').decode())


if __name__ == "__main__":
    main()
