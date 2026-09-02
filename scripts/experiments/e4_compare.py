#!/usr/bin/env python3
"""E4, part two: the error profiles and the coarsening curve.

Reads the per-mesh element dumps and the pointwise reference, evaluates both
temperature fields along the transect, and produces Fig. 7 (error against
position across the shadow, corrected and uncorrected, one panel per mesh) plus
the scaling curve that turns "refining the mesh would cost two to three orders
of magnitude" from an estimate into a measurement.

The two fields being compared:

  uncorrected   T(x) = T_element, one constant per triangle. What the solver
                produces on its own, and what a renderer without the tangent
                would show.
  corrected     T(x) = T_element + (v(x) - v_element) * dT/dv, with v(x) the
                single binary ray the shader traces and v_element the value the
                solver used. Both are taken from the dump rather than
                recomputed, so this is the arithmetic the shader performs, not
                an idealisation of it.

The |dT/dv| >= 0.1 K gate the shader applies is applied here too: below it the
shader traces no ray and leaves the element temperature alone.

Usage:
    e4_compare.py --out-dir evidence/e4_dtdv
"""

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e4_dtdv_sweep as e4  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _winpaths import require_windows_paths  # noqa: E402

TANGENT_GATE_K = 0.1     # closesthit.rchit skips the correction below this
EDGE_BAND_M = 2.0        # the +/- band the paper reports an RMSE over


def read_lag_columns(path):
    """The `# lag column k: time_h=... sun=x,y,z` lines of a dump."""
    columns = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("# lag column "):
            continue
        fields = {}
        for token in line.split(":", 1)[1].split():
            key, _, value = token.partition("=")
            fields[key] = value
        columns.append({"time_h": float(fields["time_h"]),
                        "sun": [float(v) for v in fields["sun"].split(",")]})
    return columns


def evaluate(divisions, reference, work, suffix=""):
    """Every field at every transect point, for one mesh.

    The multi-lag arm is only evaluated when its dump is there: it is a third
    solve, and a sweep run without --memory-lags does not have one.
    """
    _, corrected = e4.read_dump(work / f"elements_{divisions}_corr{suffix}.csv")
    _, raw = e4.read_dump(work / f"elements_{divisions}_raw{suffix}.csv")

    temperature = corrected["T_K"].astype(float)
    tangent = np.array([float(v) if v.strip() else 0.0
                        for v in corrected["dTdv_K"]])
    element_visibility = corrected["v_element"].astype(float)
    sky = corrected["sky_fraction"].astype(float)
    raw_temperature = raw["T_K"].astype(float)

    # The arm that traces the shadow at the hour it was cast. Its correction is
    # the same one term per carried column plus the remainder against the
    # present sun -- exactly what closesthit.rchit sums, evaluated here so the
    # comparison is against the pointwise reference rather than against a
    # render.
    lag_path = work / f"elements_{divisions}_lag{suffix}.csv"
    lag = None
    if lag_path.is_file():
        _, lag_data = e4.read_dump(lag_path)
        lag_columns = read_lag_columns(lag_path)
        lag = {
            "T_K": lag_data["T_K"].astype(float),
            "total": np.array([float(v) if v.strip() else 0.0
                               for v in lag_data["dTdv_K"]]),
            "v_element": lag_data["v_element"].astype(float),
            "columns": lag_columns,
            "tangent": [lag_data[f"dTdv_lag{k}_K"].astype(float)
                        for k in range(len(lag_columns))],
            "visibility": [lag_data[f"v_lag{k}"].astype(float)
                           for k in range(len(lag_columns))],
        }
        history_times = np.array(reference["history_times_h"])

    rows = []
    for point in reference["points"]:
        index = e4.element_index(point["x"], point["z"], divisions)
        gate = abs(tangent[index]) >= TANGENT_GATE_K
        correction = ((point["visibility_binary"] - element_visibility[index])
                      * tangent[index]) if gate else 0.0
        row = {
            "offset_m": point["offset_m"],
            "element": int(index),
            "T_ref_K": point["T_ref_K"],
            "T_uncorrected_K": float(raw_temperature[index]),
            "T_corrected_K": float(temperature[index] + correction),
            "v_point_binary": point["visibility_binary"],
            "v_point_disc": point["visibility_disc"],
            "v_element": float(element_visibility[index]),
            "dTdv_K": float(tangent[index]),
            "sky_fraction": float(sky[index]),
        }

        if lag is not None:
            history = np.array(point["visibility_history"])
            attributed = 0.0
            total_correction = 0.0
            for k, column in enumerate(lag["columns"]):
                slope = float(lag["tangent"][k][index])
                attributed += slope
                if abs(slope) < TANGENT_GATE_K:
                    continue
                # The point's own visibility at that hour, which is what the
                # shader would have traced toward that column's sun.
                v_point = float(np.interp(column["time_h"], history_times, history))
                total_correction += (v_point - float(lag["visibility"][k][index])) * slope
            remainder = float(lag["total"][index]) - attributed
            if abs(remainder) >= TANGENT_GATE_K:
                total_correction += ((point["visibility_binary"] -
                                      float(lag["v_element"][index])) * remainder)
            row["T_memory_K"] = float(lag["T_K"][index] + total_correction)

        rows.append(row)
    return rows


def statistics(rows):
    offsets = np.array([r["offset_m"] for r in rows])
    reference = np.array([r["T_ref_K"] for r in rows])
    uncorrected = np.array([r["T_uncorrected_K"] for r in rows])
    corrected = np.array([r["T_corrected_K"] for r in rows])

    band = np.abs(offsets) <= EDGE_BAND_M

    def summarise(values):
        error = values - reference
        return {
            "max_abs_K": float(np.max(np.abs(error))),
            "rmse_band_K": float(np.sqrt(np.mean(error[band] ** 2))),
            "rmse_all_K": float(np.sqrt(np.mean(error ** 2))),
            "bias_K": float(np.mean(error)),
        }

    out = {
        "uncorrected": summarise(uncorrected),
        "corrected": summarise(corrected),
        "contrast_K": float(reference.max() - reference.min()),
        "sky_fraction_min": float(min(r["sky_fraction"] for r in rows)),
    }
    if all("T_memory_K" in r for r in rows):
        out["memory"] = summarise(np.array([r["T_memory_K"] for r in rows]))
    return out


def figure(results, reference, out_path):
    meshes = sorted(results, key=lambda d: -d)
    figure_, axes = plt.subplots(len(meshes), 1, figsize=(7.0, 2.05 * len(meshes)),
                                 sharex=True)
    if len(meshes) == 1:
        axes = [axes]

    for axis, divisions in zip(axes, meshes):
        rows = results[divisions]["rows"]
        stats = results[divisions]["stats"]
        offsets = np.array([r["offset_m"] for r in rows])
        reference_t = np.array([r["T_ref_K"] for r in rows])

        axis.axhline(0.0, color="0.65", lw=0.7)
        axis.plot(offsets,
                  np.array([r["T_uncorrected_K"] for r in rows]) - reference_t,
                  color="0.35", lw=1.1, label="per-triangle (no correction)")
        axis.plot(offsets,
                  np.array([r["T_corrected_K"] for r in rows]) - reference_t,
                  color="#c1121f", lw=1.1, label="with $dT/dv$ correction")
        if "memory" in stats:
            axis.plot(offsets,
                      np.array([r["T_memory_K"] for r in rows]) - reference_t,
                      color="#0b6e4f", lw=1.1,
                      label="with the shadow's history")
        edge = results[divisions]["edge_m"]
        axis.set_ylabel("error (K)")
        caption = (f"{divisions}$^2$  ({edge:.2f} m triangle)   "
                   f"max |err| {stats['uncorrected']['max_abs_K']:.2f} K "
                   f"$\\rightarrow$ {stats['corrected']['max_abs_K']:.2f} K")
        if "memory" in stats:
            caption += f" $\\rightarrow$ {stats['memory']['max_abs_K']:.2f} K"
        axis.text(0.015, 0.86, caption,
                  transform=axis.transAxes, fontsize=8.5, va="top")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    axes[-1].set_xlabel("distance across the shadow edge (m)")
    axes[0].set_title(
        f"Error against a pointwise reference, transect across the shadow "
        f"({reference['evaluate_h']:.0f} h; {statistics_note(results)})",
        fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure_.savefig(out_path, dpi=300, bbox_inches="tight")
    figure_.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure_)


def statistics_note(results):
    any_stats = next(iter(results.values()))["stats"]
    return f"sun/shade contrast {any_stats['contrast_K']:.1f} K"


def scaling_figure(results, out_path):
    meshes = sorted(results)
    edges = np.array([results[d]["edge_m"] for d in meshes])
    uncorrected = np.array([results[d]["stats"]["uncorrected"]["rmse_band_K"]
                            for d in meshes])
    corrected = np.array([results[d]["stats"]["corrected"]["rmse_band_K"]
                          for d in meshes])

    figure_, axis = plt.subplots(figsize=(5.4, 4.0))
    axis.loglog(edges, uncorrected, "o-", color="0.35",
                label="per-triangle (no correction)")
    axis.loglog(edges, corrected, "s-", color="#c1121f",
                label="with $dT/dv$ correction")

    # Where refinement alone would have to reach to match the correction.
    slope, intercept = np.polyfit(np.log(edges), np.log(uncorrected), 1)
    target = corrected.mean()
    break_even = float(np.exp((np.log(target) - intercept) / slope))
    axis.axhline(target, color="#c1121f", ls=":", lw=0.9)
    axis.axvline(break_even, color="0.35", ls=":", lw=0.9)
    axis.plot([break_even], [target], "k*", ms=11, zorder=5)

    axis.set_xlabel("triangle edge length (m)")
    axis.set_ylabel(f"RMSE within $\\pm${EDGE_BAND_M:.0f} m of the edge (K)")
    axis.legend(frameon=False, fontsize=8.5, loc="upper left")
    axis.set_title("Refining the mesh versus carrying the tangent\n"
                   "(static shadow; the star is where refinement alone "
                   "would match the correction)", fontsize=9)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    figure_.savefig(out_path, dpi=300, bbox_inches="tight")
    figure_.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure_)
    return {"slope": float(slope), "break_even_edge_m": break_even,
            "target_rmse_K": float(target)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=pathlib.Path, default=e4.WORK)
    parser.add_argument("--divisions", type=int, nargs="+", default=e4.DIVISIONS)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--no-scaling", action="store_true",
                        help="single-mesh run: skip the coarsening curve")
    parser.add_argument("--figures", type=pathlib.Path,
                        default=pathlib.Path(r"H:\quantiloom-paper\figures"))
    args = parser.parse_args()
    # These defaults are Windows paths; under WSL they are directory NAMES.
    # See scripts/experiments/_winpaths.py for what that silently does.
    require_windows_paths(args.figures)

    reference = json.loads(
        (e4.WORK / f"reference{args.suffix}.json").read_text(encoding="utf-8"))
    scene_extent = 120.0

    results = {}
    for divisions in args.divisions:
        rows = evaluate(divisions, reference, e4.WORK, args.suffix)
        results[divisions] = {
            "rows": rows,
            "stats": statistics(rows),
            "edge_m": scene_extent / (divisions - 1),
        }
        stats = results[divisions]["stats"]
        print(f"{divisions:>4}^2  edge {results[divisions]['edge_m']:.2f} m   "
              f"max|err| {stats['uncorrected']['max_abs_K']:6.2f} -> "
              f"{stats['corrected']['max_abs_K']:5.2f} K   "
              f"band RMSE {stats['uncorrected']['rmse_band_K']:6.3f} -> "
              f"{stats['corrected']['rmse_band_K']:5.3f} K")

    figure(results, reference,
           args.figures / f"fig7_shadow_edge_error{args.suffix}.png")
    if args.no_scaling:
        scaling = None
    else:
        scaling = scaling_figure(
            results, args.figures / "fig7b_coarsening_scaling.png")
    if scaling:
        print(f"\nuncorrected RMSE scales as edge^{scaling['slope']:.2f}")
        print(f"refinement would need a {scaling['break_even_edge_m']:.4f} m "
              f"triangle to match the corrected {scaling['target_rmse_K']:.3f} K")
        coarsest = max(args.divisions)
        print(f"  that is {(scene_extent / scaling['break_even_edge_m'] + 1):,.0f} "
              f"divisions per side, "
              f"{(scene_extent / scaling['break_even_edge_m']) ** 2 / (coarsest - 1) ** 2:,.1f}x "
              f"the element count of {coarsest}^2")

    payload = {"evaluate_h": reference["evaluate_h"],
               "edge_band_m": EDGE_BAND_M,
               "tangent_gate_K": TANGENT_GATE_K,
               "penumbra_half_width_m": reference["penumbra_half_width_m"],
               "scaling": scaling,
               "meshes": {str(d): {"edge_m": results[d]["edge_m"],
                                   "stats": results[d]["stats"],
                                   "rows": results[d]["rows"]}
                          for d in args.divisions}}
    (args.out_dir / f"e4_summary{args.suffix}.json").write_text(json.dumps(payload, indent=2),
                                                  encoding="utf-8")
    print(f"\nwrote {args.out_dir / 'e4_summary.json'}")


if __name__ == "__main__":
    main()
