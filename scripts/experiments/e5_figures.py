#!/usr/bin/env python3
"""Figure 11: the three sensor-chain curves.

(a) the thermography round-trip, (b) NETD by two independent routes, and
(c) injected fixed-pattern noise against what survives correction. One figure
because they are one claim in three parts: that the chain from a prescribed
temperature to a reported one is closed, calibrated, and separable.

Usage:
    e5_figures.py --out figures/fig11_sensor_curves.png
"""

import argparse
import collections
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EVIDENCE = pathlib.Path(r"H:\quantiloom-paper\evidence\e5")

# These are Windows paths; under WSL they are directory NAMES, not paths.
# See scripts/experiments/_winpaths.py for what that silently does.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _winpaths import require_windows_paths  # noqa: E402
require_windows_paths(EVIDENCE)


def panel_roundtrip(axis):
    data = json.loads((EVIDENCE / "roundtrip.json").read_text(encoding="utf-8"))
    series = collections.defaultdict(list)
    for row in data["rows"]:
        series[(row["band"], row["cavity"])].append((row["T_set_K"], row["error_K"]))

    # The residual is at the precision of the storage, not of the physics: a
    # float32 mantissa is 2^-24, so a temperature near 300 K cannot be written
    # to better than ~0.02 mK whatever the inversion does. Drawing the band
    # says that; joining the points with lines would dress quantisation up as
    # a trend, which is what the first version of this panel did.
    # The true float32 spacing, which is a step function of the binade and not
    # T * 2^-24: it doubles at 256 K, so the band is not a smooth wedge. Using
    # the smooth approximation put one point outside a band it is actually
    # inside.
    grid = sorted({row["T_set_K"] for row in data["rows"]})
    quantum = [1000.0 * float(np.spacing(np.float32(t))) for t in grid]
    axis.fill_between(grid, [-q for q in quantum], quantum, color="0.88",
                      step="mid", zorder=0,
                      label="one float32 ulp at $T$")
    axis.set_ylim(-2.4 * max(quantum), 2.4 * max(quantum))

    for (band, cavity), points in sorted(series.items()):
        points.sort()
        axis.plot([p[0] for p in points], [1000.0 * p[1] for p in points],
                  marker="o", ls="none", ms=4, mfc="none", mew=0.9,
                  label=f"{band} {cavity}")

    axis.axhline(0.0, color="0.6", lw=0.7)
    axis.set_xlabel("prescribed cavity temperature (K)")
    axis.set_ylabel("$T_{\\rm recovered} - T_{\\rm set}$ (mK)")
    axis.set_title("(a) thermography round-trip, sensor off", fontsize=9)
    axis.legend(fontsize=6, ncol=2, frameon=False, loc="upper left")
    within = sum(1 for row in data["rows"]
                 if abs(row["error_K"]) <= float(np.spacing(np.float32(row["T_set_K"])))
                 + 1e-12)
    axis.text(0.98, 0.04,
              f"{within} of {len(data['rows'])} runs land on the nearest float32",
              transform=axis.transAxes, ha="right", fontsize=7.5)


def panel_netd(axis):
    data = json.loads((EVIDENCE / "netd.json").read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if r["netd_empirical_mK"]]
    temperature = [r["T_K"] for r in rows]

    axis.plot(temperature, [r["netd_analytic_mK"] for r in rows],
              "o-", color="0.30", ms=4, lw=1.1,
              label="analytic, $\\sigma_L / (dL/dT)$")
    axis.plot(temperature, [r["netd_empirical_mK"] for r in rows],
              "s--", color="#c1121f", ms=4, lw=1.1,
              label=f"empirical, {data['frames']} frames")

    # The end points use a one-sided difference for dDN/dT and are marked as
    # such rather than quietly included in the agreement claim.
    for index in (0, len(rows) - 1):
        axis.plot([rows[index]["T_K"]], [rows[index]["netd_empirical_mK"]],
                  "s", color="#c1121f", ms=8, mfc="none", mew=1.2)

    axis.set_xlabel("scene temperature (K)")
    axis.set_ylabel("NETD (mK)")
    axis.set_title("(b) NETD, analytic against measured", fontsize=9)
    axis.legend(fontsize=7.5, frameon=False)
    interior = rows[1:-1]
    ratios = [r["netd_empirical_mK"] / r["netd_analytic_mK"] for r in interior]
    axis.text(0.98, 0.86,
              f"interior points agree to {100 * max(abs(r - 1) for r in ratios):.1f} %\n"
              f"(open markers: one-sided $dDN/dT$)",
              transform=axis.transAxes, ha="right", va="top", fontsize=7.5)


def panel_fpn(axis):
    data = json.loads((EVIDENCE / "fpn.json").read_text(encoding="utf-8"))
    bright = [r for r in data["rows"] if r["field"] == "bright"]
    dark = {(r["prnu_sigma"], r["nuc_efficiency"]): r["residual_fpn_dn"]
            for r in data["rows"] if r["field"] == "dark"}

    # What the design predicts: the multiplicative part survives at
    # (1 - efficiency) of what went in, the additive part is whatever the dark
    # field shows, and the two add in quadrature because they are independent.
    predicted, measured, labels = [], [], []
    for row in bright:
        additive = dark[(row["prnu_sigma"], row["nuc_efficiency"])]
        predicted.append(math_hypot(row["predicted_residual_dn"], additive))
        measured.append(row["residual_fpn_dn"])
        labels.append((row["prnu_sigma"], row["nuc_efficiency"]))

    for efficiency, marker in ((0.95, "o"), (0.97, "s"), (0.99, "^")):
        keep = [i for i, l in enumerate(labels) if l[1] == efficiency]
        axis.plot([predicted[i] for i in keep], [measured[i] for i in keep],
                  marker, ms=6, label=f"NUC {efficiency:.0%}")

    limit = max(max(predicted), max(measured)) * 1.1
    axis.plot([0, limit], [0, limit], color="0.6", lw=0.8, ls=":")
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("predicted residual (DN)\n"
                    "$\\sqrt{[(1-\\eta)\\,\\sigma_p S]^2 + \\sigma_{\\rm dark}^2}$",
                    fontsize=8)
    axis.set_ylabel("measured residual (DN)")
    axis.set_title("(c) fixed-pattern noise surviving correction", fontsize=9)
    axis.legend(fontsize=7.5, frameon=False, loc="upper left")

    darks = sorted({round(v, 2) for v in dark.values()})
    axis.text(0.98, 0.06,
              f"dark-field residual {min(darks):.2f}\u2013{max(darks):.2f} DN,\n"
              f"independent of PRNU",
              transform=axis.transAxes, ha="right", fontsize=7.5)


def math_hypot(a, b):
    return (a * a + b * b) ** 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path(r"H:\quantiloom-paper\figures"
                                             r"\fig11_sensor_curves.png"))
    args = parser.parse_args()

    figure, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))
    panel_roundtrip(axes[0])
    panel_netd(axes[1])
    panel_fpn(axes[2])
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=300, bbox_inches="tight")
    figure.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.out}")

    # The numbers the caption and the text quote, so neither is retyped.
    fpn = json.loads((EVIDENCE / "fpn.json").read_text(encoding="utf-8"))
    dark = {(r["prnu_sigma"], r["nuc_efficiency"]): r["residual_fpn_dn"]
            for r in fpn["rows"] if r["field"] == "dark"}
    worst = 0.0
    for row in (r for r in fpn["rows"] if r["field"] == "bright"):
        prediction = math_hypot(row["predicted_residual_dn"],
                                dark[(row["prnu_sigma"], row["nuc_efficiency"])])
        worst = max(worst, abs(row["residual_fpn_dn"] - prediction) / prediction)
    print(f"  FPN: worst departure from the quadrature prediction {worst:.1%}")


if __name__ == "__main__":
    main()
