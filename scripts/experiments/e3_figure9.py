#!/usr/bin/env python3
"""Figure 9: the interactivity measurements, and the mechanism they refute.

Three panels, matching the three paragraphs of Section VIII-G:

  (a) A geometry edit, decomposed. One bar per component on a log axis, because
      0.07 ms of TLAS update and a 500 ms target do not share a linear scale.
      The decomposition is the evidence: a fast refit and a refit that quietly
      did nothing look identical as a single number, so the GPU's own trace
      timestamp is plotted beside the wall clock as an independent witness that
      the frame really traced.

  (b) Backward-seek latency against checkpoint stride. This is the negative
      result. Section V describes the stride as a memory-versus-latency dial;
      the replayed step counts on the right-hand axis show it bounding the
      re-solve exactly as described, while the latency on the left does not
      move. The zero-step control -- a seek to the time already displayed --
      is drawn as a horizontal line, and it costs what a 33-step seek costs.

  (c) Backward-seek latency against element count, which is what the cost
      actually tracks. The 2 s target and the reference scene are marked
      because the margin there is 13 ms, and a panel that hid it would be
      making the target look comfortable when it is not.

Usage:
    e3_figure9.py
"""

import argparse
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EVIDENCE = pathlib.Path(r"H:\quantiloom-paper\evidence\e3")
FIGURES = pathlib.Path(r"H:\quantiloom-paper\figures")

REPLAYED = re.compile(r"\[BENCH\]\s+T2 backward seek, stride (?P<stride>[\d.]+) h\s+"
                      r"replayed (?P<steps>[\d.]+) steps")
ELEMENTS = re.compile(r"(?P<elements>\d+) elements")

ACCENT = "#2f4b7c"
WARN = "#b03030"
MUTED = "#8a8a8a"


def load():
    path = EVIDENCE / "timings.json"
    if not path.is_file():
        raise SystemExit(f"{path} missing; run e3_timings.py first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {r["name"]: r for r in payload["rows"]}

    stdout_path = EVIDENCE / "bench_stdout.txt"
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") \
        if stdout_path.is_file() else ""
    # The replayed-step counts print on their own line and carry no timing, so
    # the table parser skips them; they are the whole point of panel (b).
    replayed = {float(m.group("stride")): float(m.group("steps"))
                for m in REPLAYED.finditer(stdout)}
    return payload, rows, replayed


def elements_of(row):
    match = ELEMENTS.search(row.get("note") or "")
    return int(match.group("elements")) if match else None


def panel_a(axis, rows):
    wanted = [("T1 refit alone", "TLAS update"),
              ("T1 GPU trace dispatch", "trace dispatch\n(GPU timestamp)"),
              ("T1 first frame alone", "first frame"),
              ("T1 refit -> first frame", "refit \u2192 frame\n(during a drag)"),
              ("T1 rebuild -> first frame", "rebuild \u2192 frame\n(on release)")]
    present = [(rows[name], label) for name, label in wanted if name in rows]
    if not present:
        axis.text(0.5, 0.5, "no T1 rows", ha="center", va="center", fontsize=8)
        return

    y = np.arange(len(present))
    medians = [r["median"] for r, _ in present]
    p95s = [r["p95"] for r, _ in present]
    colours = [MUTED if "\u2192" not in label else ACCENT for _, label in present]

    axis.barh(y, medians, height=0.62, color=colours, zorder=3)
    axis.errorbar(medians, y, xerr=[[0] * len(present),
                                    [p - m for m, p in zip(medians, p95s)]],
                  fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2.5, zorder=4)
    for index, (value, p95) in enumerate(zip(medians, p95s)):
        axis.text(p95 * 1.35, index, f"{value:.2f} ms", va="center", ha="left",
                  fontsize=6.4, color="#333333", zorder=5)

    axis.axvline(500.0, color=WARN, lw=1.2, ls="--", zorder=2)
    axis.text(430.0, -0.42, "0.5 s target", color=WARN, fontsize=6.8,
              va="center", ha="right")

    axis.set_yticks(y)
    axis.set_yticklabels([label for _, label in present], fontsize=6.6)
    axis.set_xscale("log")
    axis.set_xlim(0.02, 3000.0)
    axis.set_xlabel("milliseconds (log)", fontsize=7.5)
    axis.set_title("(a)  a geometry edit, decomposed\nbars are medians, whiskers p95",
                   fontsize=8.0)
    axis.tick_params(axis="x", labelsize=7)
    axis.grid(axis="x", color="0.9", lw=0.6, zorder=0)
    axis.set_axisbelow(True)


def panel_b(axis, rows, replayed):
    seeks = {}
    for name, row in rows.items():
        match = re.match(r"T2 backward seek, stride ([\d.]+) h$", name)
        if match:
            seeks[float(match.group(1))] = row
    if not seeks:
        axis.text(0.5, 0.5, "no stride sweep", ha="center", va="center", fontsize=8)
        return

    strides = sorted(seeks)
    medians = [seeks[s]["median"] for s in strides]
    p95s = [seeks[s]["p95"] for s in strides]

    axis.errorbar(strides, medians,
                  yerr=[[0] * len(strides), [p - m for m, p in zip(medians, p95s)]],
                  marker="o", ms=5, lw=1.5, color=ACCENT, capsize=3,
                  label="backward seek", zorder=4)

    zero = next((r for n, r in rows.items() if n.startswith("T2 re-seek to the same time")),
                None)
    if zero:
        axis.axhline(zero["median"], color=WARN, lw=1.3, ls=(0, (4, 2)), zorder=3,
                     label=f"zero steps: {zero['median']:.0f} ms")

    axis.set_xscale("log", base=2)
    axis.set_xticks(strides)
    axis.set_xticklabels([f"{s:g}" for s in strides])
    axis.set_xlabel("checkpoint stride (h)", fontsize=7.5)
    axis.set_ylabel("seek latency (ms)", fontsize=7.5)
    # From zero, deliberately. Auto-scaling a set of points that span 501-505 ms
    # fills the panel with a 0.8 % wobble and reads as variation, which is the
    # opposite of what the measurement says; the spread belongs in the caption.
    axis.set_ylim(0.0, max(p95s) * 1.18)
    axis.tick_params(labelsize=7)
    axis.grid(color="0.92", lw=0.6, zorder=0)
    axis.set_axisbelow(True)

    if replayed:
        twin = axis.twinx()
        xs = [s for s in strides if s in replayed]
        twin.plot(xs, [replayed[s] for s in xs], marker="s", ms=4, lw=1.2,
                  color="#5a8f3d", ls=":", zorder=4)
        for s in xs:
            twin.annotate(f"{replayed[s]:.0f}", (s, replayed[s]), fontsize=6.2,
                          color="#3f6b2b", xytext=(0, 5), textcoords="offset points",
                          ha="center")
        twin.set_ylabel("solver steps replayed", fontsize=7.5, color="#3f6b2b")
        twin.tick_params(axis="y", labelsize=7, colors="#3f6b2b")
        twin.set_ylim(0, max(replayed.values()) * 1.45)

    axis.legend(fontsize=5.9, loc="center left", bbox_to_anchor=(0.02, 0.62),
                frameon=False)
    axis.set_title("(b)  the stride bounds the re-solve\nand not the clock", fontsize=8.0)


def panel_c(axis, rows):
    points = []
    for name, row in rows.items():
        if name.startswith("T2 seek @"):
            count = elements_of(row)
            if count:
                points.append((count, row))
    if not points:
        axis.text(0.5, 0.5, "no element sweep", ha="center", va="center", fontsize=8)
        return
    points.sort()
    counts = [c for c, _ in points]
    medians = [r["median"] for _, r in points]

    # Linear on both axes: the claim is that latency is proportional to element
    # count, and proportionality is a straight line through the origin here and
    # a parallel offset on log-log, where it is far harder to read.
    per_element = medians[-1] / counts[-1]
    grid = np.array([0.0, max(counts) * 1.18])
    axis.plot(grid, per_element * grid, lw=1.0, ls=":", color=MUTED, zorder=3,
              label=f"{per_element * 1000:.0f} \u00b5s per element")
    axis.plot(counts, medians, marker="o", ms=5, lw=1.5, color=ACCENT, zorder=4,
              label="measured")

    axis.axhline(2000.0, color=WARN, lw=1.2, ls="--", zorder=2)
    axis.text(grid[1] * 0.02, 2010.0, "2 s target", color=WARN, fontsize=6.8,
              va="bottom", ha="left")

    reference = points[-1]
    axis.annotate(f"reference scene\n{reference[0]:,} elements\n"
                  f"{reference[1]['median']:.0f} ms, "
                  f"{2000 - reference[1]['median']:.0f} ms of margin",
                  xy=(reference[0], reference[1]["median"]),
                  xytext=(-10, -34), textcoords="offset points", fontsize=6.3,
                  ha="right", va="top", color="#333333", linespacing=1.35,
                  arrowprops=dict(arrowstyle="-", lw=0.7, color="#666666"))

    axis.set_xlim(0.0, grid[1])
    axis.set_ylim(0.0, 2400.0)
    axis.set_xticks([0, 25000, 50000, 75000])
    axis.set_xticklabels(["0", "25k", "50k", "75k"])
    axis.set_xlabel("thermal elements", fontsize=7.5)
    axis.set_ylabel("seek latency (ms)", fontsize=7.5)
    axis.tick_params(labelsize=7)
    axis.grid(color="0.92", lw=0.6, zorder=0)
    axis.set_axisbelow(True)
    axis.legend(fontsize=6.3, loc="upper left", frameon=False,
                bbox_to_anchor=(0.0, 0.86))
    axis.set_title("(c)  what a scrub actually costs\nis per-element work", fontsize=8.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path,
                        default=FIGURES / "fig9_interactive_timings.png")
    args = parser.parse_args()

    payload, rows, replayed = load()
    print(f"{len(rows)} benchmark rows, {len(replayed)} stride step counts")
    print(f"host: {payload['host']['gpu']}")

    figure, axes = plt.subplots(1, 3, figsize=(7.16, 2.75))
    panel_a(axes[0], rows)
    panel_b(axes[1], rows, replayed)
    panel_c(axes[2], rows)
    figure.tight_layout(w_pad=2.2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=300, bbox_inches="tight")
    figure.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
