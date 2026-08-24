#!/usr/bin/env python3
"""The three schematic figures: the pipeline, the sensor chain, the build gates.

These carry no measurements, so they are drawn rather than plotted, and the one
rule they follow is that every box names something that exists in the source
tree. A schematic that shows a stage the system does not have is worse than no
schematic, because a reader cannot tell which boxes were drawn from the code and
which from the abstract.

Text is measured rather than guessed. Every label is laid out at a reference
size, its true rendered width read back from the renderer, and the font scaled
so the longest line fits the box it sits in -- eyeballing point sizes against
box widths produces overflow that only shows up after the figure is placed in
the manuscript.

Greyscale-safe by construction: the palette separates stages by fill lightness
as well as by hue, so the figures survive a monochrome print of the proceedings.

Usage:
    schematics.py            # all three
    schematics.py --only 6
"""

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIGURES = pathlib.Path(r"H:\quantiloom-paper\figures")

# These are Windows paths; under WSL they are directory NAMES, not paths.
# See scripts/experiments/_winpaths.py for what that silently does.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _winpaths import require_windows_paths  # noqa: E402
require_windows_paths(FIGURES)

# Lightness rises with each stage so the order survives a greyscale print.
INK = "#1c1c1c"
STAGE = ["#2f4b7c", "#665191", "#a05195", "#c2427a"]
LIGHT = ["#dbe3f0", "#e4dfef", "#f0dcec", "#f7dde5"]
NEUTRAL = "#ededed"
EDGE = "#8f8f8f"

_REFERENCE_PT = 20.0   # large enough that the linear scaling below is accurate


def measure_pt(figure, text, weight="normal"):
    """Width of the widest line, in points, at _REFERENCE_PT."""
    renderer = figure.canvas.get_renderer()
    widest = 0.0
    for line in text.split("\n"):
        artist = figure.text(0.0, 0.0, line, fontsize=_REFERENCE_PT, fontweight=weight)
        widest = max(widest, artist.get_window_extent(renderer).width)
        artist.remove()
    return widest / figure.dpi * 72.0


def fit_pt(axis, text, available_axes_width, cap, weight="normal", pad=0.90):
    """The largest size <= cap at which every line fits the box.

    The width has to come from the axes' own extent, not the figure's: an axes
    with default margins spans 77.5 % of the figure, and scaling against the
    figure instead silently licenses every label to overflow by a third.
    """
    figure = axis.figure
    axes_width_in = axis.get_window_extent(figure.canvas.get_renderer()).width / figure.dpi
    available = available_axes_width * axes_width_in * 72.0 * pad
    reference = measure_pt(figure, text, weight)
    if reference <= 0.0:
        return cap
    return min(cap, _REFERENCE_PT * available / reference)


def box(axis, x, y, w, h, title, body="", face=NEUTRAL, edge=EDGE, title_cap=8.0,
        body_cap=6.5, title_colour=INK, lw=1.1):
    figure = axis.figure
    axis.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.014",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))
    if body:
        axis.text(x + w / 2, y + h * 0.775, title, ha="center", va="center",
                  fontsize=fit_pt(axis, title, w, title_cap, "bold"),
                  fontweight="bold", color=title_colour, zorder=3)
        axis.text(x + w / 2, y + h * 0.335, body, ha="center", va="center",
                  fontsize=fit_pt(axis, body, w, body_cap),
                  color="#333333", zorder=3, linespacing=1.5)
    else:
        axis.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                  fontsize=fit_pt(axis, title, w, title_cap, "bold"),
                  fontweight="bold", color=title_colour, zorder=3)


def arrow(axis, start, end, colour=INK, lw=1.2, rad=0.0, ls="-"):
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10, linewidth=lw,
        color=colour, linestyle=ls, zorder=4,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0.5, shrinkB=0.5))


def blank_axes(width, height):
    figure, axis = plt.subplots(figsize=(width, height))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.canvas.draw()          # so get_renderer() is available to measure_pt
    return figure, axis


def save(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.png"
    figure.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Fig. 1 -- the system pipeline
# ---------------------------------------------------------------------------
def figure1():
    """Four stages over one scene graph, and the loop that makes it one system.

    The feedback arrow is the point of the figure. Every other schematic of a
    rendering pipeline is a left-to-right chain; here the solver's temperature
    field is an input to the transport stage that produced the radiance it was
    solved from, which is why the thermal timeline sits inside the renderer
    rather than beside it as a preprocess.

    The scene graph is drawn as a bus rather than as three long arcs: it feeds
    every stage, and three curves crossing the figure to say so obscures the
    feedback path, which is the part a reader cannot infer.
    """
    figure, axis = blank_axes(7.16, 3.15)

    top, height = 0.395, 0.335
    mid = top + height / 2

    source_w = 0.125
    box(axis, 0.0, top + 0.045, source_w, height - 0.09, "glTF 2.0\nscene graph", "",
        face="#e6e6e6", edge="#6f6f6f", title_cap=7.8, lw=1.2)

    stages = [
        ("Spectral transport", "Cook\u2013Torrance/GGX\nphysical Fresnel\n"
                               "hero-wavelength sampling"),
        ("Energy balance", "1-D conduction\nsix-flux coupling\nCSR view factors\n"
                           "$dT/dv$ tangent"),
        ("Sensor chain", "PSF \u00b7 QE \u00b7 noise\nPRNU / DSNU \u00b7 NUC\n"
                         "12/14/16-bit ADC"),
        ("Output products", "per-band inversion\nNETD \u00b7 fusion\nENVI cube"),
    ]
    gap = 0.062
    width = (1.0 - source_w - 4 * gap) / 4
    xs = [source_w + gap + i * (width + gap) for i in range(4)]

    for index, (x, (title, body)) in enumerate(zip(xs, stages)):
        box(axis, x, top, width, height, title, body, face=LIGHT[index],
            edge=STAGE[index], title_colour=STAGE[index], title_cap=8.2,
            body_cap=6.4, lw=1.2)

    # The scene graph is a bus: one spine, one drop per stage.
    bus_y = top + height + 0.088
    axis.plot([source_w / 2, xs[-1] + width / 2], [bus_y, bus_y], color="#8f8f8f",
              lw=0.9, ls=(0, (3.0, 2.2)), zorder=1)
    axis.plot([source_w / 2, source_w / 2], [top + height - 0.045, bus_y],
              color="#8f8f8f", lw=0.9, ls=(0, (3.0, 2.2)), zorder=1)
    for x in xs:
        arrow(axis, (x + width / 2, bus_y), (x + width / 2, top + height),
              colour="#8f8f8f", lw=0.9, ls=(0, (3.0, 2.2)))
    axis.text(xs[-1] + width / 2, bus_y + 0.045,
              "one scene graph and one spectral representation reach every stage",
              ha="right", va="bottom", fontsize=6.6, color="#666666", style="italic")

    for x, label in zip(xs[:3], ("radiance\nfield", "temperature\nfield", "raw DN")):
        arrow(axis, (x + width, mid), (x + width + gap, mid), lw=1.3)
        axis.text(x + width + gap / 2, mid + 0.022, label, ha="center", va="bottom",
                  fontsize=fit_pt(axis, label, gap, 5.7, pad=0.82),
                  color="#333333", linespacing=1.35)

    # The loop: solved temperatures re-enter transport as self-emission.
    # Routed orthogonally, like the bus above: down, back, up into transport.
    loop_y = top - 0.075
    a, b = xs[0] + width / 2, xs[1] + width / 2
    axis.plot([b, b, a], [top, loop_y, loop_y], color=STAGE[1], lw=1.35,
              solid_joinstyle="miter", zorder=4)
    arrow(axis, (a, loop_y), (a, top), colour=STAGE[1], lw=1.35)
    axis.text((xs[0] + xs[1] + width) / 2, top - 0.235,
              "surface temperature re-enters transport as self-emission\n"
              "\u2014 a checkpointed timeline (\u00a7V), scrubbable, not a preprocess",
              ha="center", va="center", fontsize=6.5, color=STAGE[1], linespacing=1.45)
    save(figure, "fig1_pipeline")


# ---------------------------------------------------------------------------
# Fig. 4 -- the sensor effect chain, in both implementations
# ---------------------------------------------------------------------------
def figure4():
    """Two implementations of one model, and the coefficient that ties them.

    Drawn as two lanes rather than one chain because the contracts differ: the
    offline lane owes bit-exactness, the interactive lane owes a frame budget,
    and the responsivity asserted equal between them is the only thing that
    makes them the same sensor. Each lane takes the radiance field from its own
    left edge, so no arrow crosses the figure to say what a label can.
    """
    figure, axis = blank_axes(7.16, 3.5)

    axis.text(0.5, 0.985, "radiance field  $L(\\lambda)$  from the renderer  "
                          "(\u00a7IV) and the solved temperature field (\u00a7V)",
              ha="center", va="top", fontsize=7.8, fontweight="bold", color=INK)

    lane_x, right = 0.118, 1.0
    span = right - lane_x

    def lane(y, height, entries, colour, light, title_cap, body_cap, label, note):
        gap = 0.0125
        width = (span - (len(entries) - 1) * gap) / len(entries)
        for index, entry in enumerate(entries):
            title, body = entry if isinstance(entry, tuple) else (entry, "")
            x = lane_x + index * (width + gap)
            box(axis, x, y, width, height, title, body, face=light, edge=colour,
                title_cap=title_cap, body_cap=body_cap, title_colour=colour)
            if index:
                arrow(axis, (x - gap, y + height / 2), (x, y + height / 2), lw=1.0)
        # Both lanes stack their label over their note from the lane's top edge,
        # so the short lane cannot collide the two the way centring them does.
        axis.text(0.0, y + height, label, ha="left", va="top",
                  fontsize=fit_pt(axis, label, lane_x, 7.5, "bold"),
                  fontweight="bold", color=colour, linespacing=1.3)
        axis.text(0.0, y + height - 0.082, note, ha="left", va="top",
                  fontsize=fit_pt(axis, note, lane_x, 6.0),
                  color="#5a5a5a", linespacing=1.3)
        arrow(axis, (lane_x - 0.014, y + height / 2), (lane_x, y + height / 2), lw=1.1)

    lane(0.605, 0.305,
         [("Optics", "focal length\nF-number, pitch\n$\\cos^4$ vignetting"),
          ("PSF", "Gaussian $\\sigma$ matched\nto Airy FWHM\nfactor 0.437"),
          ("Detector", "QE, well capacity\nPoisson photon,\nread-out, dark"),
          ("FPN", "PRNU multiplicative\nDSNU additive\nseparate parameters"),
          ("NUC", "efficiency 95\u201399 %\na residual survives\nby design"),
          ("ADC", "12 / 14 / 16-bit\nconfigurable gain")],
         STAGE[0], LIGHT[0], 7.6, 5.6,
         "offline\nCPU", "bit-exact,\ndeterministic")

    lane(0.145, 0.205,
         ["PSF blur\nhorizontal", "PSF blur\nvertical", "radiance\n\u2192 electrons",
          "Poisson +\nread-out noise", "fixed-pattern\nnoise", "quantise\n\u2192 radiance"],
         STAGE[2], LIGHT[2], 6.9, 5.6,
         "interactive\nGPU", "six compute\npasses")

    axis.add_patch(FancyBboxPatch(
        (lane_x, 0.415), span, 0.115, boxstyle="round,pad=0,rounding_size=0.012",
        facecolor="#fff7e2", edgecolor="#c19f42", linewidth=1.0, zorder=1))
    note = ("responsivity $\\mathcal{R}$ (electrons per unit radiance) asserted identical "
            "in both lanes by unit test \u2014 and the same $\\mathcal{R}$ that Eq. (9) "
            "divides to report NETD")
    axis.text(lane_x + span / 2, 0.4725, note, ha="center", va="center",
              fontsize=fit_pt(axis, note, span, 6.7), color="#6b5310")

    footer = ("all stochastic stages draw from one deterministic seed (default 0x548C); "
              "seed 0 enables per-run variation")
    axis.text(0.5, 0.045, footer, ha="center", va="center",
              fontsize=fit_pt(axis, footer, 1.0, 6.5), color="#5a5a5a", style="italic")
    save(figure, "fig4_sensor_chain")


# ---------------------------------------------------------------------------
# Fig. 6 -- the four build gates
# ---------------------------------------------------------------------------
def figure6():
    """Four gates in series, and what each one can and cannot see.

    The failure edge is drawn as prominently as the pass edge because that is
    the property being claimed: the SDK is installed only if all four pass, so a
    red gate is not a warning in a log, it is the end of the build.
    """
    figure, axis = blank_axes(7.16, 2.65)

    gates = [
        ("(i) Unit suite", "1,260 tests in 7.6 s\nlinks the core library,\nnot the shaders"),
        ("(ii) ABI gate", "export list diffed\nagainst a golden\nbaseline of 244 symbols"),
        ("(iii) Furnace gate", "8 isothermal cavities\n(5 LWIR, 3 MWIR) against\n"
                               "band-integrated Planck"),
        ("(iv) Illumination gate", "occlusion, open-sky\nequivalence, Cornell-box\nindirect"),
    ]
    install_w = 0.108
    gap = 0.040
    width = (1.0 - install_w - 4 * gap) / 4
    top, height = 0.435, 0.345

    for index, (title, body) in enumerate(gates):
        x = index * (width + gap)
        box(axis, x, top, width, height, title, body, face=LIGHT[index],
            edge=STAGE[index], title_cap=7.8, body_cap=6.2,
            title_colour=STAGE[index], lw=1.2)
        if index:
            arrow(axis, (x - gap, top + height / 2), (x, top + height / 2), lw=1.3)
            axis.text(x - gap / 2, top + height / 2 + 0.018, "pass", ha="center",
                      va="bottom", fontsize=5.6, color="#4a4a4a")
        arrow(axis, (x + width / 2, top), (x + width / 2, 0.275), colour="#a82c2c",
              lw=1.1, ls=(0, (2.8, 1.9)))

    last = 3 * (width + gap)
    arrow(axis, (last + width, top + height / 2), (1.0 - install_w, top + height / 2),
          lw=1.3)
    axis.text(1.0 - install_w - gap / 2, top + height / 2 + 0.018, "pass", ha="center",
              va="bottom", fontsize=5.6, color="#4a4a4a")
    box(axis, 1.0 - install_w, top + 0.055, install_w, height - 0.11,
        "install\nSDK", "", face="#e3f0e3", edge="#3f7a3f", title_cap=7.6,
        title_colour="#2f5f2f", lw=1.2)

    box(axis, 0.0, 0.135, 1.0, 0.108, "any gate red \u2192 build stops, SDK is not installed",
        "", face="#fdeceb", edge="#a82c2c", title_cap=7.6, title_colour="#8f2424")

    axis.text(0.5, 0.925, "the SDK that Quantiloom-Qt links is installed only on the far "
                          "side of all four",
              ha="center", va="center", fontsize=7.3, color="#5a5a5a", style="italic")
    axis.text(0.5, 0.045, "the first two gates see the library; the last two are the only "
                          "checks that observe a shader at all",
              ha="center", va="center", fontsize=6.5, color="#5a5a5a")
    save(figure, "fig6_build_gates")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", type=int, choices=(1, 4, 6))
    args = parser.parse_args()

    for number, draw in ((1, figure1), (4, figure4), (6, figure6)):
        if args.only in (None, number):
            draw()


if __name__ == "__main__":
    main()
