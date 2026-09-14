# Quantiloom: A Spectral Path Tracer with Coupled Surface Energy Balance

> Targeting Quantiloom v0.3.2 and Quantiloom-Qt v0.3.2.
> This document describes the system as implemented, the rationale behind
> every key design choice, and the verification strategy that guards them.

---

## 1. Overview

Quantiloom is an end-to-end spectral infrared imaging simulation system.
Its pipeline runs:

> Spectral path tracing (physical Fresnel, measured spectral reflectance,
> Jakob–Hanika upsampling, a sampled four-wavelength visible estimator with
> dispersion, rank-one fluorescence) → surface energy balance temperature
> field solver (1-D conduction with optional lateral conduction, six-flux
> coupling, view-factor long-wave/short-wave exchange, trajectory tangents
> with respect to sun visibility and material parameters, piecewise-static
> geometry epochs for a scene that moves) → sensor effect chain
> (PSF / QE / noise / FPN / NUC / ADC) → thermography inversion (NETD) /
> multi-band fusion / hyperspectral cube output (ENVI, spectral EXR, TIFF).

The core is a C++20 / HLSL shared library on Vulkan ray tracing, consumed
by three front ends: a CLI (single scene, batch, timeline sequence, MCP
server), a Qt 6 desktop application with interactive progressive rendering,
and an MCP service for agent-driven workflows.  The complete system—SDK plus
GUI, excluding tests—is approximately 118,000 lines of code.

---

## 2. Codebase Scale

### 2.1 Top-Level Metrics

| Module | Lines | Files |
|---|---:|---:|
| Core C++ source (`src/`, excluding shaders) | 60,999 | — |
| Public headers (`include/quantiloom/`) | 8,369 | 32 |
| Shaders (HLSL: `.rgen` / `.rchit` / `.rmiss` / `.rahit` / `.comp.hlsl` / `.comp` / `.rayq.hlsl` / `.hlsli`) | 13,618 | 27 |
| **Core SDK subtotal** | **~83,000** | — |
| Unit tests (GoogleTest; 1,473 cases / 131 suites, ~12 s) | 34,543 | 84 |
| Quantiloom-Qt GUI (separate repository) | 35,035 | 94 |
| **System total (core SDK + GUI, excluding tests)** | **~118,000** | — |
| **Including tests** | **~152,600** | — |

### 2.2 Core Library Modules (`src/libQuantiloom/`)

| Module | Files | Lines | Responsibility |
|---|---:|---:|---|
| `renderer/` | 51 | 25,540 | Vulkan abstraction, acceleration structures, textures, offline/interactive dual hosts, GPU thermal stepper, thermal preview and epoch builder, timeline state, spectral unmixing |
| `io/` | 14 | 9,044 | glTF / USD / EXR / ENVI / spectral EXR / TIFF / spectral library I/O |
| `thermal/` | 18 | 5,430 | Surface energy balance solver, timeline, epochs, short-wave gains, Crank–Nicolson stepper, solve cache |
| `hs_core/` | 13 | 4,803 | Hyperspectral configuration, adaptive wavelength grid |
| `core/` | 20 | 4,365 | Fundamental types, configuration, logging, SHA-256, CIE / D65 / RGB-to-spectrum tables |
| `scene/` | 14 | 3,037 | Camera, material, mesh, scene editing, motion grammars, multi-model merge |
| `atmos/` | 12 | 1,625 | Atmospheric LUT, ResMLP neural surrogate |
| `postprocess/` | 3 | 1,169 | Sensor effect chain, multi-band fusion, thermography inversion |
| `mcp/` | 11 | 1,126 | MCP service module |

Other modules: `src/app/` (CLI, 2,274 lines), `src/libSpectraForge/`
(IR material generation, 820 lines), `src/tools/` (1,766 lines): `fusion_tool`,
the `QLTrans` MODTRAN wrapper, and three measurement harnesses—
`thermal_column_tool` (one 1-D column through the energy balance with no scene
around it), `sensor_lab` (N frames through *one* `GenericSensor`, the only
arrangement that separates temporal noise from the fixed pattern), and
`colour_lab` (the RGB-to-spectrum conventions: what the coefficient table's
resolution costs in CIELab, and what an authored emissive triple comes back
as).  All four link `quantiloom_core` rather than the DLL, because what they
drive is internal and is not going to be exported to give a measurement script
a way in.

### 2.3 Principal Shaders

| File | Lines |
|---|---:|
| `closesthit.rchit` | 5,400 |
| `common.hlsli` | 1,470 |
| `pbr.hlsli` | 1,131 |
| `SpectralConversion.hlsli` | 657 |
| `clahe.comp.hlsl` | 533 |
| `miss.rmiss` | 469 |
| `volumetric.hlsli` | 403 |
| `spectral_reconstruct.comp` | 365 |
| `thermal_step.comp.hlsl` | 339 |
| `sampling.hlsli` | 329 |
| `blackbody.hlsli` | 270 |
| `thermal_exchange.rayq.hlsl` | 249 |

### 2.4 Test Distribution

`test_renderer/` 26 files; `test_core/` 24; `test_io/` 7; `test_hs_core/` 7;
`test_scene/` 7; `test_postprocess/` 6; `test_mcp/` 2; `test_app/` 2;
plus `test_main.cpp` and the shared `support/VulkanTestDevice`.
The baseline includes 8 SKIPPED tests by design: the 8 BC7 texture
compression cases (disabled after measurement showed a net quality loss).
The 2 EXR multipart cases that used to skip are live since the spectral EXR
writer landed (§9.4).  A further 5 cases are `DISABLED_` by name—the
interactive latency benchmark of `test_interactive_bench.cpp`.  Those are
measurements rather than assertions, and a build gate that went red because a
driver got slower would be noise.

### 2.5 DLL Exports

The shared library exports 264 symbols (`docs/abi/exports.golden`), and
`libSpectraForge` a further 25 (`spectraforge-exports.golden`).  Every new
export must pass through a reviewed three-step process: move the header into
`include/quantiloom/`, annotate with `QL_API`, and update the golden baseline.
A build-time ABI gate rejects any change that skips a step.

Not every SDK/Studio contract is an export.  `ThermalSolveParams`,
`ConfigApplyReport` and `ThermalSolveStatus` are plain structs with no `QL_API`
symbol of their own, so growing one changes the ABI layout without changing the
export list.  The SDK and Studio ship together and `SdkGuard` (§11.1) is the
backstop that refuses a stale pairing.

---

## 3. System Architecture

### 3.1 The Two-Target Rule

The core is compiled once into `quantiloom_core` (an OBJECT library) and
consumed in exactly two ways.  **A target links one or the other, never
both**—two copies of the library's global state (the spdlog logger, static
caches) in one process is a correctness bug:

| Link target | Consumers | Visibility |
|---|---|---|
| `quantiloom_core` | `libquantiloom_tests`, `fusion_tool` | Everything; `QL_API` irrelevant |
| `libQuantiloom` (DLL) | CLI, `libSpectraForge`, Quantiloom-Qt | Only `QL_API`, only `include/quantiloom/` |

Tests link the object library so that internal code is testable without being
exported—otherwise every new unit test would widen the ABI.

### 3.2 Build Pipeline and Gates

Development happens in WSL 2; the product is a Windows binary.  The full
build (`./build_wsl.sh`) runs four gates before installing the SDK:

1. **Unit test suite** — 1,473 tests (1,465 pass, 8 skip) in ~12 s.
2. **ABI gate** — `scripts/check_exports.sh` diffs the current export list
   against `docs/abi/exports.golden`.
3. **Furnace rendering gate** — `run_furnace_suite.sh` renders the 8
   isothermal cavity configurations of `furnace_cases.txt` (5 LWIR, 3 MWIR)
   and verifies each against its own Planck integral.
4. **Illumination gate** — `run_illumination_suite.sh` runs nine arms:
   occlusion in NIR, SWIR and MWIR; open sky in SWIR against a closed form;
   `open:vis`, the same ground in the visible, where the deterministic
   estimator must render with a spread of exactly zero and the sampled one
   must land on its mean; colour bleed (Cornell box); MIS weight
   consistency, once per emission representation; `hero`, the two visible
   estimators against each other (§4.3); `fluor`, a transfer between two
   wavelength bands (§5.5); and view independence (§6.6).

Nothing a gate prints may be non-ASCII.  A Windows console on a CJK locale
encodes stdout as GBK, so a checker printing a combining macron or a
superscript two raised `UnicodeEncodeError`, exited 1, and failed the furnace
gate before it had measured anything—on a machine where every cavity was
correct.  Under WSL the locale is UTF-8 and the same script is fine, which is
how two checkers carried it unnoticed: the gate that fails is the one nobody
had run.

Only after all four pass does the script install the SDK into the sibling
`Quantiloom-SDK` tree, which the Qt front end links against.

**The Windows path must stay equivalent.**  `./build_windows.ps1` followed by
`./install_windows.ps1` installs the same SDK to the same prefix, and
Quantiloom-Qt cannot tell which path produced what it links—so a gate added to
one belongs in the other.  The gates are *ported*, not wrapped: a build agent
has MSVC, CMake and PowerShell and need not have bash or WSL, so
`check_exports.ps1` and the two `run_*_suite.ps1` are PowerShell
reimplementations of their `.sh` twins.  Running the shell scripts through Git
Bash was tried first and rejected—it trades a WSL dependency for a Git Bash one
and calls it portable.

What is shared rather than duplicated, because two copies of a decision drift:
the cavity list (`furnace_cases.txt`, read by both runners) and every checker in
`scripts/render-tests/*.py`, which is where all the thresholds live.  Those stay
Python deliberately—they compute band-integrated Planck radiance and RMSE over
EXR images, which is the measurement rather than shell glue, and Python is a
portable dependency in a way bash is not.  When it is absent the render gates
are skipped *loudly* and the stamp says so.

Two scripts cannot enforce an order between themselves the way one script's
`set -e` does, so `build_windows.ps1` records what passed in
`build/.gates-passed.json` and `install_windows.ps1` refuses without it, or when
the DLL is newer than the stamp.  `-Force` overrides and says that it did.

### 3.3 CLI Modes

| Mode | Invocation | Description |
|---|---|---|
| Single scene | `Quantiloom.exe <config.toml>` | Render one TOML scene |
| Batch | `batch <list.txt>` | Render a manifest of scenes in one GPU session, a renderer per line |
| Sequence | `sequence <config.toml> --from-tick A --to-tick B --every N --output "out/frame_{tick:05}.exr"` | Render a `[timeline]` tick by tick on one renderer (§10.5) |
| Serve | `serve [--port]` | Local MCP server for agent-driven rendering |

**Batch manifest per-job overrides.**  Each line in a manifest may carry
per-frame overrides after a `|` delimiter:

```
plate.toml | material_overrides.CheckerGround.ir_temperature_k=280.0 renderer.output="seq_280K.exr"
plate.toml | material_overrides.CheckerGround.ir_temperature_k=300.0 renderer.output="seq_300K.exr"
```

Overrides use `material_overrides` tables rather than inline `[[materials]]`
arrays because `Config::MergedWith` replaces arrays wholesale—an inline
`[[materials]]` entry would delete every other material in the scene.  Tables
merge by key, which is the semantics per-frame overrides require.  A material
that a multi-model merge renamed (§10.1) is addressed with a quoted key,
`material_overrides."block/Material".ir_temperature_k=320.0`; the tokeniser
keeps the quotes and hands the token to the TOML parser, so it needs no
special case, and a test pins that.

A manifest line carrying `timeline.time_s=` renders one instant of a moving
scene per line, and is the slow path: a renderer per frame, with the solve
cache (§8.10) making the thermal half cheap on a re-run.  `sequence` is the
fast one.

### 3.4 Cross-Validation by Construction

The same `RenderCore` is consumed by CLI, GUI, and unit tests.  Running two
host paths side by side has exposed bugs including: vertically mirrored
environment maps, mip chains that were never sampled, IR emissivity that
ignored the rendering wavelength, a sensor-chain input-unit mismatch of
~7×10⁴ (LWIR rendered near-black), and a specular path where reflected
radiance exceeded incident radiance by 11,000×.

A *third* renderer (§12.3) has since added three more, none of which any
internal gate could have found: a declared reflectance curve that was loaded
and never reached the shader, 64 spectral samples spread over the wrong
wavelength range, and a missing MIS weight on the bound-emission path that put
every indirectly lit surface at 2×.

Two rendering invariants are enforced by the furnace and illumination gates:

- **An isothermal cavity renders its own blackbody.**  Eight furnace
  configurations return 1.0000.
- **Reflected radiance does not exceed incident radiance.**

These are not automatically true—making them gates is precisely because they
are easy to break, and unit tests (which link the core, not the shaders)
cannot catch physics bugs that live in the four shader stages.

---

## 4. Spectral Path Tracing

### 4.1 Vulkan Ray Tracing Pipeline

The renderer builds on the Vulkan `VK_KHR_ray_tracing_pipeline` extension.
BLAS instances are built per mesh; a single TLAS covers the scene.  The
interactive host supports TLAS refit (via
`VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR`) during gizmo drags, with
a full rebuild on mouse release.

Nine rendering modes are available, expressed by the `SpectralMode` enum:
`RGB`, `VIS_Fused`, `VIS_Hero`, `NIR_Fused`, `SWIR_Fused`, `MWIR_Fused`,
`LWIR_Fused`, `Single`, and `Multispectral`.  `VIS_Hero` is appended at 8
rather than inserted beside `VIS_Fused`: the value travels as a specialization
constant, and renumbering would render one mode's scene with another's shader.
A unit test reads the mode defines out of `common.hlsli` and checks them
against the enum.

The visible band is the only one with two estimators, and the nine decisions
that are about the *band* rather than about sampling—which lighting inputs
arrive as RGB, which materials get their reflectance upsampled from a colour,
which emitters need a MIS weight, what a participating medium may assume, what
an output pixel means and whether it gets a PNG beside it—ask `IsVisMode`
rather than naming one mode.

### 4.2 Alpha Mode and the Any-Hit Stage

The `alphaMode` property (`OPAQUE` / `MASK` / `BLEND`) is threaded from
the glTF loader through acceleration-structure flags to the any-hit shader.
Declaring a geometry as opaque lets the driver skip the any-hit stage, but
for alpha-tested or blended geometry that would turn a mesh grille into a
solid wall—visually wrong in the render, and worse in the thermal solver,
where view factors would treat it as a continuous surface and skew the
entire temperature field.  Alpha geometry therefore retains the any-hit
invocation and accepts the traversal cost.

### 4.3 Two Visible Estimators, and the Band Alias Picks the Sampled One

The visible band lives in one binary twice.  `vis_fused` sweeps 32 fixed
wavelengths down one geometric path and sums them against the CIE observer—a
deterministic Riemann sum with no variance in wavelength at all.  `vis_hero`
draws one wavelength at the first surface a primary ray reaches, derives three
more by rotating it through the band,

```
λⱼ = 400 + mod(λ_h − 400 + j·95, 380),   j = 0…3
```

and carries the quartet to the end of the path in a `float4` radiance
(`Payload::radiance` grew from 36 to 40 bytes, well inside the 64 the RT cores
allow).  `spectral.mode = "VIS"` resolves to `vis_hero`: every other band alias
has one estimator to go to, and asking for a band rather than for an estimator
is asking to render it.  `vis_fused` stays reachable by name because it has a
job—it is the reference the sampled mode is measured against, and the mode to
ask for when an answer has to be repeatable rather than converged.

**Why a rotation rather than four independent draws.**  The rotation is a
measure-preserving action on the band, so the set of four does not depend on
which of them was drawn—any member would have produced the same set—and the
balance heuristic over those four ways of arriving collapses to one scalar
shared by all of them.  There is no per-wavelength weight to carry, which is
why four payload components suffice and why the matching functions are applied
in exactly one place: the vertex that drew the four, after the surface, the
indirect correction, any medium and any glass have all had their say.  The
predecessor design—splitting the band into 8 groups with a ray each—failed the
convergence check below and was reverted.

**Why not four wavelengths everywhere.**  Evaluating the BSDF and querying the
light at 32 wavelengths per path costs roughly 32× on the spectral side, while
on a non-dispersive surface the samples are highly correlated.  The quartet
spends the budget on paths instead.  Three pieces had to become four: the
environment bounce, whose geometry, roulette and lobe stay scalar while its
material weights and the sky they are measured against became a `float4`
evaluated per wavelength (the average of a product is not the product of the
averages); the miss shader, which returns four sky radiances to a quartet and
keeps the deterministic grid for a primary ray, since the sky is analytic and
four samples of it would be worse than 32 evaluations for the same cost in
rays, which is none; and refraction.

**Dispersion collapses the quartet.**  A dispersive interface sends four
wavelengths four ways, so the quartet collapses onto its hero rather than
redrawing, and what comes back is scaled so that the root's own division leaves
exactly the single-wavelength estimator.  Only the wavelength that was bent to
that direction could have generated it, so that technique's share is one.
`heroLambda` is therefore a sign as well as a value: zero for a ray that has
drawn nothing, positive for a quartet, negative for one collapsed by a
refraction.  The assumption underneath all of it is that transport is diagonal
in λ—what arrives at a wavelength leaves at that wavelength—and §5.5 is the one
term that is not.

**Measured against the deterministic grid.**  On a Cornell box the error falls
as 1/√spp, 0.0209 to 0.0053 over sixteen times the samples, with a fixed offset
of −0.17% left over that belongs to the *grid*: a 32-point Riemann sum of a
D65-shaped illuminant over 400–780 nm reads X 0.19% and Z 0.60% high against a
quarter-nanometre reference, while the sampled estimator is unbiased for the
integral itself.

**The check asks three questions whose answers converge**, because its
predecessor compared two renders against a fixed 1×10⁻⁴ that Monte Carlo noise
met from below—1.04×10⁻⁴ at 1024 spp, 6.3×10⁻⁵ after a fix that made the render
more correct, 4.2×10⁻⁵ at 4096—so it passed or failed by how long it ran.
`check_hero_wavelength.py` now measures:

| Check | What it holds | Result |
|---|---|---|
| Switching | `vis_hero` against a `vis_fused` reference at 8192 spp on a scene with no glass; the error must fall faster than 2× per 16× samples, and the signed luminance error at 1024 spp under 0.3% | 5.83× ; +0.011% |
| Collapse | The quartet refracting by its hero wavelength with a *constant* n, so the geometry is identical on both sides and only the spectral bookkeeping around the collapse is under test | error × √spp reads 0.1009 / 0.1013 / 0.1020 across the sweep |
| Dispersion | Signal against a floor the renderer measures: two dispersion values 0.2% apart differ by 4×10⁻⁷–7×10⁻⁷; tripling the dispersion moves the render 1.2×10⁻⁴–3.1×10⁻⁴ | ratio 290 and 431 against a bound of 50, improving with sample count |

Luminance and not the channels, because per channel the reference is the
biased one in blue (R −0.13%, G +0.10%, B −0.50%), and holding all three to a
bound would be holding the sampled mode to the reference's quadrature error.
The atmosphere table is read at the nearest bin rather than interpolated, and
both places that do it say why: the baker fills each entry with the mean over
its bin, so the table is a piecewise-constant τ, and interpolating between bin
averages would build a function neither mode integrates.

**`vis_fused` is no longer bit-identical to its previous self.**  1.1% of a
Cornell box's pixels move by one unit in the last place, at most 3×10⁻⁸
absolute, where the compiler reassociates arithmetic now written over vectors.
Every gate whose answer is exact still reads 0.0000%.

### 4.4 Next-Event Estimation and Multiple Importance Sampling

The renderer combines BSDF sampling with next-event estimation (NEE)
of emissive geometry, weighted by the power heuristic.

**Why both?**  Pure BSDF sampling has explosive variance on small bright
lights (hard to hit); pure light-source sampling has explosive variance
on near-specular surfaces (the hit has near-zero BSDF contribution).
MIS takes the lower envelope of both strategies' variances.
`check_nee_mis.py` verifies that the two sampling paths estimate the same
image—the most common MIS implementation error is a PDF domain mismatch
that produces a plausible but biased result.

Emitter selection uses a luminance-weighted CDF with binary search.

### 4.5 Owen-Scrambled Sobol Stratification

The first bounce uses a padded, Owen-scrambled Sobol sequence (following
PBRT-v4's `PaddedSobolSampler`, Burley 2020) across seven sample slots:

| Slot | Dimensionality | Purpose |
|---|---|---|
| `SAMPLE_SLOT_JITTER` | 2-D | Sub-pixel position |
| `SAMPLE_SLOT_LAMBDA` | 1-D | Wavelength |
| `SAMPLE_SLOT_LIGHT_PICK` | 1-D | Emitter selection |
| `SAMPLE_SLOT_LIGHT_UV` | 2-D | Point on emitter |
| `SAMPLE_SLOT_LOBE` | 1-D | Diffuse/specular lobe |
| `SAMPLE_SLOT_DIRECTION` | 2-D | Bounce direction |
| `SAMPLE_SLOT_FLUOR_LAMBDA` | 1-D | Fluorescence excitation wavelength (§5.5); a scene without fluorescence draws nothing from it, so its sequence is untouched |

The underlying Sobol sequence is 2-D (dimensions 0 and 1), reused per
slot with independent Owen-scramble seeds—hence "seven slots," not seven Sobol
dimensions.  Pre-generated Sobol direction numbers carry no run-time cost.

**Why Sobol rather than PCG white noise?**  The first bounce is low-dimensional
and high-impact; stratifying it directly reduces integration error.
Measured on `cornell_box_vis` at 512² against a 16,384-sample reference of the
same scene at a different seed, the spp required to reach 2% RMSE drops from
3,547 (white noise) to 493—a **7.2× sampling efficiency gain** at zero
additional cost.  The two arms are one build apart:
`QUANTILOOM_UNSTRATIFIED_FIRST_BOUNCE` routes the six padded Sobol slots to the
PCG stream the deeper bounces already use, and nothing else differs.  Each rung
of the ladder gives its own estimate—the arms' convergence constants `k` in
`RMSE = k/√spp` differ by 2.63× to 2.76×, and cost scales as `k²`, so 6.9× to
7.6×; 7.2× is the value at the 2% threshold and sits in the lower half of that
spread.

**This number has moved three times, and each move was a renderer defect rather
than a re-tuning.**  It read 1.74× while the Cornell scene shaded from base
colours instead of its bound curves; 7.1× after those curves reached the shader;
6.6× after they were resampled over the render band rather than their own span
(§6.5); and 7.2× after a bound emission spectrum stopped skipping its MIS
de-weighting (§5.4), which had been rendering every indirectly lit surface in
that scene at 2×.  Removing the doubling *raised* the gain rather than shrinking
it, because both arms' convergence constants moved and the ratio moved with
them.  Two harness defects had to be fixed to measure any of it: flipping the
CMake option left every built `.spv` newer than its source so nothing
recompiled, and the measurement never built the target that copies the shaders
next to the executable—so the first attempt returned bit-identical numbers from
both arms.

**Why Owen scrambling rather than plain Sobol?**  Unscrambled Sobol has
structural correlations in higher dimensions that produce visible regular
patterns.  Owen scrambling randomises the sequence while preserving its
low-discrepancy properties and keeping the estimator unbiased.

### 4.6 Adaptive Sampling: A Data-Rejected Design

`sampling.hlsli` retains the complete design notes for a provably unbiased
per-pixel convergence-stopping mechanism.  Each pixel's stopping decision
depended only on raster-order-earlier pixels, so the dependency graph was
acyclic and the estimator remained unbiased.

**Why it was ultimately not adopted.**  The mechanism is *structurally*
unprofitable: in the test scenes, low-variance pixels (sky, background) are
also the cheapest pixels—stopping them saves little work.  The truly
expensive pixels are precisely the high-variance, multi-bounce ones that
the stopping criterion would keep running.  Measured on a sea-surface
VIS scene at 384 × 384:

| Configuration | Time | RMSE |
|---|---:|---:|
| 2% threshold adaptive | 12.7 s | 1.331% |
| Uniform 2048 spp | 11.8 s | 1.330% |

Uniform sampling was faster *and* more accurate at every tested time budget.

*An earlier version of this section quoted a "1.85× theoretical ceiling, of
which the stratification captures 94%".  That claim is withdrawn rather than
rescaled.*  No artefact in either repository supports the ceiling, and it set an
adaptive-allocation bound against a stratification gain as though the two
competed for one pool of headroom.  They do not: stratification places samples
better, allocation spends more of them where the variance is, and neither
consumes the other's margin.  What the measurement above supports is the
structural argument, and nothing more.

**What a renewed attempt would require.**  A per-pixel *cost* AOV, not
merely a variance AOV.  The stopping criterion optimised "where variance is
high," but the correct objective is "where variance reduction per unit time
is greatest."  This AOV does not yet exist, so the path is closed.

The implementation was built, measured, and removed based on data—a complete
negative result.

---

## 5. RGB-to-Spectrum Conventions

An RGB triple has no spectrum; the renderer must choose a convention to
expand it.

### 5.1 Reflectance: Jakob–Hanika Sigmoid Model

Reflectance is a quadratic in wavelength passed through a sigmoid:

```
R(λ) = s(c₀ t² + c₁ t + c₂),   t = (λ − 380) / 400
s(x) = 1/2 + x / (2√(1 + x²))
```

Each colour maps to three coefficients, stored in a 64³ look-up table
(9.0 MB) on GPU binding 25.

**Why not a sum of three Gaussians (one per sRGB primary)?**  Measured with
this repository's own observer functions and D65 illuminant, the Gaussian
mapping has a mean error of **21 CIELab units** (35 on saturated colours)—
roughly 10–15 just-noticeable differences.  It is also unbounded, requiring
`clamp(…, 0.0, 1.5)` to prevent reflectances above unity (a surface
reflecting more light than it receives).

**Why not a free-form fit to the three constraints?**  Three constraints and
infinitely many degrees of freedom yield a spiky metamer: colour-matched
under D65, but ringing under any other illuminant (e.g. the ASTM G-173
solar spectrum).  The sigmoid-of-quadratic shape guarantees two properties:
`s` maps the result to (0, 1) (bounded, no clamp needed), and a quadratic
through a sigmoid is smooth and at most bimodal—matching the shape of real
reflectance curves.

**Why 64³ rather than smaller?**  The fitting itself is exact (CIELab
residual driven to zero within the sRGB gamut); the only error is from
table interpolation:

| Resolution | Coefficient memory | Mean ΔE | p99 | Worst |
|---:|---:|---:|---:|---:|
| 32 | 1.1 MB | 0.098 | 0.453 | 3.49 |
| 48 | 3.8 MB | 0.043 | 0.201 | 2.05 |
| **64** | **9.0 MB** | **0.024** | **0.112** | **1.23** |

Measured by `colour_lab --lut-sweep`: 200k colours uniform in the unit cube
plus the eight corners, CIE76 in Lab against the colour the fit targets.

One JND is approximately 2.3.  The 48³ table is the smallest whose worst
colour is within one JND; 64³ is the reference implementation's
specification.  At 9 MB, GPU memory is not a concern.

### 5.2 Illuminants: Sigmoid Chromaticity × D65 Amplitude

```
L(λ) = scale · s(c; λ) · d65(λ),   scale = 2 · max(rgb)
```

This is the convention for an authored RGB triple, and it is a *fallback*: a
lamp that has measured or standard data binds it instead (§5.4).

**Why not a flat spectrum?**  A flat spectrum is an equal-energy illuminant E,
not D65.  E integrates to linear sRGB (1.205, 0.948, 0.909), so a nominally
white light renders warm.

**Why not a downstream white-balance correction?**  A fixed pair of
coefficients (E's G/R and G/B: 0.7872 and 1.0437) applied to the final
radiance is correct for the single scene it was fitted to and wrong for
everything else: a scene lit by the true D65 spectrum loses 16.6% red and
gains 10.6% blue; a scene lit by measured ASTM G-173 sunlight is
unnecessarily white-balanced by ~3% per channel.  A convention should be
fixed where it is introduced, not compensated downstream with a
constant—this is a recurring design principle throughout the system.
End-to-end measurement on the Cornell box emissive panel (author values
[15, 15, 12]):

| | Ratio | Deviation |
|---|---|---:|
| Author values | 1 : 1.000 : 0.800 | — |
| Flat spectrum + downstream WB | 1 : 0.946 : 0.821 | 3.57% |
| **Sigmoid × D65** | 1 : 0.997 : 0.800 | **0.22%** |

**Why the factor of 2?**  Emitters are not bound by reflectance ≤ 1, so
amplitude must be separated from chromaticity.  `2 · max(rgb)` pushes the
triple below 0.5, well within the sigmoid's well-conditioned interior, so
the fitting error is independent of amplitude—scaling the triple from 1 to
5,000 keeps the error at 0.10%.

**Why no extra upload channel?**  The D65 curve rides in the `.w` channel
of the CIE colour-matching-function buffer (binding 19), sampled by the
same callers at the same wavelengths.  Widening to `float4` keeps a 16-byte
stride—zero extra bandwidth, zero extra binding.

### 5.3 Out-of-Band Prohibition

**Why RGB upsampling is forbidden outside the visible band.**  The quadratic
grows beyond [380, 780] nm; for warm saturated colours (`c₀ > 0`) the
sigmoid saturates to 1.  The `SheenChair` asset's mango-coloured velvet
yields reflectance 0.9998 at 1.2 µm and 1.0000 at 10 µm: in the thermal
infrared, a wall becomes a mirror, its emissivity drops to zero, and it
ceases to self-emit.  `EvaluateRgbSpectrum` therefore clamps λ to
[380, 780]; out-of-band callers are blocked by the `allowRgbUpsample` rule,
and `quality.fail_on_srgb_upsample = true` promotes violations to hard
failures.  This is a trap unique to cross-band systems: a purely visible
renderer never encounters it.

### 5.4 Emission: Binding a Spectrum Instead of a Colour

The sun could always state its composition—`lighting.solar_lut` binds
ASTM G-173, and `sun_radiance` is only the fallback for when nothing is
bound.  A light *inside* the scene could not: its spectrum came solely from a
glTF `emissiveFactor` expanded by §5.2's formula, which nothing measured and
which is defined only on [380, 780] nm.  That is the same category of
invention §5.3 forbids outright on the reflectance side.

`[material_overrides."<name>"] emissive_curve = "…"` is the way out.  It takes
a built-in token (`d65`, `illuminant_a`, `halogen`, `cie_f1`…`cie_f12`,
`cie_f3.1`…`cie_f3.15`, `blackbody_<T>k`, `equal_energy`) or a path to a
two-column table.  The CIE tables ship under `assets/luts/` with their DOIs and
licence metadata *and* are compiled in, so a token cannot resolve to a file
that is missing or edited; illuminant A is computed from CIE 015:2018 eq. 4.1,
reproducing CIE's own table to 5×10⁻⁶.
`assets/configs/cornell_box_lamp_spectrum.toml` is the worked example.  Five
decisions carry the feature:

**Zero outside the curve's span, never clamped.**  `EvaluateEmissionCurve`
differs from `EvaluateSpectralCurve` in exactly this, and the difference is the
point.  Constant extrapolation is defensible for a reflectance—bounded in
[0, 1], and a surface reflecting 0.4 at 780 nm plausibly reflects about that at
800 nm.  It is indefensible for an emission: the CIE fluorescent tables stop at
780 nm, and holding FL7's last value flat across SWIR furnishes a triphosphor
lamp with near-infrared output it does not have.  The host warns whenever a
bound spectrum falls short of the render band.

**The infrared bands read a bound curve and nothing else.**  There is
deliberately no RGB fallback at 10 µm, where the visible fit has no domain.  A
lamp therefore appears in a thermal band only if someone bound data for it, and
the resolver counts the materials that emit with nothing bound: they contribute
*nothing*, which is the hardest kind of missing data to notice, because the
render is merely dark.

**One lamp, every mode.**  When a curve is bound, `ResolveEmissionSpectrum`
overwrites `emissiveFactor` with the linear sRGB the curve integrates to—which
is what keeps the RGB preview, the emitter-sampling CDF (`EmissiveTriangleGPU`
carries the curve index in what used to be `_pad0`) and the spectral bands from
describing three different lamps.  It happens only where the CIE observer has
support; in a thermal band the authored triple is left alone, because
integrating an 8–12 µm curve against the observer returns a large, confident,
meaningless colour.  One resolver serves both hosts—
`ExternalRenderContext::SetMaterialEmissionSpectrum` calls the same function
with the band the viewport is showing—for the same reason `ResolveSolarLut` was
split out: two copies of the levelling rules is a lamp that renders differently
in Studio than from the CLI.

**`emissive_scale = "match_luminance"` is the default; `"absolute"` is the
calibrated case.**  Every built-in but the blackbody family is a *relative*
distribution normalised to 100 at 560 nm, so the level has to come from
somewhere: the material's own `emissive` triple.  Swapping lamps then changes
the room's colour without re-exposing the render.  `absolute` takes the curve as
spectral radiance in W·m⁻²·sr⁻¹·nm⁻¹ and is *required* outside the visible,
where luminance is undefined—asking for `match_luminance` there is an error
rather than a guess.

**Emission is band-averaged onto the 64-point grid; reflectance is
point-sampled.**  A reflectance is smooth; a fluorescent lamp is mostly mercury
lines, and point-sampling a line spectrum either hits a line or misses it—CIE
FL11 lands 4.9% wrong in green, with the sign decided by nothing more
principled than where the grid falls.  Averaging each sample over its own bin
conserves the energy and brings it to 0.2%.  It does not fix the *estimator*,
which point-samples 32 wavelengths and leaves FL11 ~14% high in blue; the host
detects that case—band-averaging and point-sampling the same curve disagree
exactly when it is not band-limited—and says so.

**MIS applies to the curve path too.**  The de-weighting was originally written
as `emissive *= EmissiveMisWeight(…)`, that is, against the RGB triple, which a
material with a bound curve never reads.  The curve path therefore kept the
emitter's whole radiance from the BSDF strategy while NEE collected it a second
time, and since the light-sampling share is close to unity for a small bright
panel, everything the lamp lit came out at almost exactly twice its correct
value.  The directly viewed panel was unaffected—a camera ray carries weight
1—so the failure looked like a brighter room rather than a broken image, and a
second renderer found it (§12.3).  `BoundEmissionRadiance` now takes the weight
as a **required** parameter rather than a defaulted one, because forgetting it
is silent, and `check_nee_mis.py` sets the emission representation itself and
runs its A/B once for each: reproducing the defect measures 50.10% for the curve
path and 2.13% for RGB, so a check that read the config would have covered the
broken path only by coincidence.

Verified against two quantities the renderer does not get to choose.  D65 bound
as an emitter comes back neutral—[14.87, 14.84, 14.83] rendered against
[14.79, 14.78, 14.78] predicted—which it must, since D65 is the white point sRGB
is defined against.  And a `blackbody_3000k` panel renders 1.948
W·m⁻²·sr⁻¹·nm⁻¹ at 10 µm against Planck's 1.935.  Scenes without an
`emissive_curve` are bit-identical: the GPU material took a padding word, so
every offset is unchanged and the emitter struct is still 64 bytes.

**`KHR_materials_emissive_strength` is folded, not carried.**  glTF clamps
`emissiveFactor` to [0, 1], so an emitter brighter than nominal white has
nowhere to put the excess but this extension's scalar, and the loader was
ignoring it—`assets/models/prism_spectroscope.gltf` carries a strength of 60
on its slit and had rendered sixty times too dark since the day it was
committed, looking merely dim.  The strength multiplies into
`Material::emissiveFactor` at load rather than living beside it, because that
triple is already the one HDR scale every consumer reads: the emissive term in
`closesthit`, the host-side NEE CDF, `EmissiveMisWeight`, the has-an-emitter
test the timeline uses, and the `[[materials]] emissive` override.  The host
CDF and the shader MIS weight have to agree exactly or the estimator is biased,
so a field of its own would oblige each of those to remember the same product,
and whichever forgot would be a bias visible only as a function of light-source
size.  `MaterialDataCPU` is also 656 bytes with every padding word spent.  The
fold composes with §5.4 for free: `match_luminance` levels a bound curve
against the folded triple, so `d65` bound to a strength-16 panel comes out
sixteen times `d65` bound to a strength-1 one, and under `absolute` the
strength is ignored, which is what absolute means.  A negative strength is
clamped to zero rather than propagated into a CDF that cannot be sampled.
Measured: five cubes sharing one factor and differing only in strength read
1.000 / 2.000 / 4.000 / 8.000 / 16.000 in median luminance, and the
spectroscope rendered against a copy with the extension stripped is brighter by
a median per-pixel ratio of 60.0000.

### 5.5 Fluorescence: The One Term That Is Not Diagonal in λ

Transport is diagonal in wavelength everywhere else—what arrives at λ leaves at
λ—which is what lets four wavelengths share one geometric path (§4.3).
Fluorescence is the term that is not, and it is why the payload could not simply
carry more wavelengths.

**Rank one, because that is what published data supports.**  A material
carries an excitation shape `ex(λ)`, an emission shape `em(λ)` and a quantum
yield, on `MaterialDataCPU`'s three former padding words (offsets 44, 312 and
316; the struct is still 656 bytes and the UV array still starts at 320).  What
the surface re-radiates is

```
yield · em(λ_out) · ∫ ex(λ_in) E(λ_in) dλ_in / π
```

with `ex` the dimensionless share of arriving light the fluorescent channel
takes and `em` normalised on the way in to unit area over the band.  That
normalisation is not tidying: a spectrofluorimeter reports counts, so a level
read off a published curve is a property of the instrument, and fixing the area
at 1 leaves the yield as the only number that says how much comes back.  Both a
TOML's five `fluorescence_*` keys and a glTF's `QUANTILOOM_materials_fluorescence`
reach one resolver, which refuses a yield outside [0, 1], an excitation outside
it (a table in percent, and the message says so), and a pair that lands entirely
outside the band—there being no wavelength for the absorbed light to come back
at.  It warns about an emission peak below the excitation peak (two files
swapped) and about reflected plus re-emitted exceeding one.

**One sample of the excitation integral per vertex, and no ray for it.**  On a
sampling slot of its own (§4.5),

```
M ≈ ex(λ_f) E(λ_f) / p(λ_f)
```

with `E` the irradiance already established at that vertex—the sun through the
shadow ray already traced, the sky dome, the emitter the light sample found.
The scalar then enters the shared wavelength loop as `yield · em(λ) · M / π`,
which is what makes the deterministic sweep, the quartet and a collapsed hero
ray agree without three implementations of the same rule.  The emitter term
uses the light sample without its MIS weight: nothing competes to estimate the
excitation integral, since the bounce carries the outgoing wavelengths and not
the absorbed one, and a power heuristic against a technique that does not
exist would discard the share it assigns to nobody.

**No `emissiveFactor` is written**, on either path.  A fluorescent surface
emits only what something else lit it with, and the emitter-sampling table
collects triangles by `luminance(emissiveFactor)`; a triple there would have
next-event estimation aim at a light that is dark on its own.

**The gate no diagonal renderer can pass.**  Every other check in the tree
would still be green with the wavelength coupling deleted.
`check_fluorescence.py`, the illumination suite's `fluor` arm, puts two
illuminants of equal power over the visible band—the second with twice as much
of it below 500 nm—in front of a dye that absorbs on [400, 500] nm and emits
on [550, 650] nm.  At 600 nm the second illuminant carries 0.643 of the first,
and the dye's contribution moves 1.90×: a factor a transport bounded by what
arrives in the emission band cannot produce, because the light got there from
somewhere else.  Three more checks, each failing differently: linearity in the
yield (same seed, per pixel, worst 9.5×10⁻⁶ of the mean); agreement between the
two visible estimators (0.069470 against 0.069467, 0.0038% apart, which catches
the term being added where only one of them reaches); and energy—a surface that
absorbs everything and gives it back evenly must return exactly the irradiance
it received, `(E_sun cos θ + E_sky)/π = 0.413803`, and reads 0.413817 and
0.413777.  That last check is what found the emission normalisation
integrating a 64-point grid as 64 steps instead of 63, so every fluorescent
surface returned 1.6% too little light.  The fixtures are synthetic and say so:
shapes chosen to make a transfer between two bands unambiguous, not
measurements of any dye.

---

## 6. Material Model

### 6.1 Cook–Torrance PBR and glTF 2.0 Extensions

The renderer implements Cook–Torrance with GGX visible-normal sampling
(VNDF).  Each glTF extension threads through the full pipeline: **data
model → glTF parsing → TOML key → spectral curve resolution → the MIS
four-tuple** (BSDF sample, BSDF PDF, light sample, light PDF), with a
dedicated validation scene for each:

| Extension | Implementation Notes |
|---|---|
| `KHR_materials_specular` | Specular strength/chromaticity modulating dielectric F₀ |
| `KHR_materials_anisotropy` | Anisotropic GGX, threaded through the full MIS four-tuple |
| `KHR_materials_clearcoat` | Clear-coat layer with independent normal and roughness |
| `KHR_materials_diffuse_transmission` | Diffuse-transmission lobe (thin leaves, paper) |
| `KHR_materials_sheen` | Charlie distribution lobe with directional albedo |
| `KHR_texture_transform` | Per-texture-slot UV transform |
| `KHR_materials_variants` | Load-time variant selection |
| `KHR_materials_ior` / `transmission` / `volume` / `dispersion` | Index of refraction, transmission, volume absorption, Cauchy dispersion |
| `KHR_materials_emissive_strength` | HDR emitter scale, folded into `emissiveFactor` at load so the NEE CDF, the MIS weight and the TOML `emissive` override read one value |

**Why the full MIS four-tuple, not just BSDF evaluation?**  Modifying only
the evaluation without updating the PDF breaks the MIS weights—the result
is biased and changes with light-source size.  Anisotropic GGX is
especially sensitive because its PDF depends on the tangent direction.

**Why sheen is stripped in MWIR/LWIR but layered in NIR/SWIR.**  Sheen's
colour is RGB and requires the upsampling from §5.1, which is illegal
outside the visible band.  NIR/SWIR are still reflection-dominated, so
treating sheen as a layer on top of the base is a defensible approximation.
MWIR/LWIR are self-emission-dominated; a fabricated sheen reflectance
would directly contaminate the emissivity, so it is removed rather than
approximated.

### 6.2 NMF Spectral Basis

**Why NMF basis vectors rather than per-wavelength storage.**  A material's
reflectance over 400–12,000 nm is thousands of floats; multiplied by the
number of scene materials, this exceeds the GPU's resident budget—and most
degrees of freedom are redundant (real reflectances lie in a low-dimensional
subspace).  Non-negative matrix factorisation (rather than PCA) is used
because reflectance is non-negative, and each NMF component can be
interpreted as an endmember—directly supporting the spectral unmixing
described in §6.3.  The cost is reconstruction error; every basis therefore
records per-band **measured coverage**, and any quality figure must be cited
alongside its coverage (see §13, Known Limitations).

Spectral basis coefficients are uploaded via `SpectralBasisLoader`; GPU-side
query goes through `spectral_query.hlsli`.  Complex refractive-index data
(for physical Fresnel on conductors) is served by `SpectralIO`.

### 6.3 Spectral Unmixing (NNLS)

`SpectralUnmixer` decomposes base-colour textures into endmember weight maps
at run time: for each texel, solve
`argmin ‖Σ wᵢ cᵢ − texel‖, w ≥ 0`
(Lawson–Hanson active-set NNLS, specialised for k ≤ 4, 3 rows), and write
the result into an RGBA texture attached to the scene.  **Measured spectra
retain their measured shape; only the proportions vary across the surface.**

**Why unmix rather than treating the texture as a reflectance modulator?**
The latter uses a procedurally generated base-colour map—whose values have
arbitrary units—as a scaling factor on the spectral reflectance curve,
stripping the curve of its absolute value.  "Binding a measured curve"
would no longer mean the surface has that curve's reflectance.

**Why global scaling rather than per-texel normalisation?**  Per-texel
normalisation forces every texel's weight sum to 1, erasing brightness
variation—the very thing the mechanism exists to recover.  Global scaling
(mean sum = 1) separates two concerns: **level from measurement, variation
from texture.**  Measured on a grass ground plane:

| | Large-area mean | Same-area σ |
|---|---:|---:|
| Flat (no unmixing) | 0.0649 | 0.0018 |
| Global-scaled unmixing | 0.0640 (−1.4%) | 0.0188 |

Without this scaling, the same scene would be 36% darker overall because
the procedural base colour is darker than measured *Spartina*.

**The anchor has to be solved against the clamp.**  `BuildUnmixWeightTextures`
solved for that mean *before* `EncodeWeight` truncated anything above the
ceiling, so the mean it achieved was not the mean it solved for and every texel
past the ceiling lost its excess.  The four KV-2 tank materials rendered at
0.68–0.80 of their measured band reflectance, while the desert they stand on,
having no base-colour texture and therefore no weight map, rendered at exactly
1.00—an asymmetry sitting between the two objects a figure invites the reader
to compare.  The scale is now a fixed point rather than a closed form (raising
it pushes more texels into the ceiling, so the correction feeds back; the
clipped population grows monotonically with the scale, and it converges in a
few passes), and the ceiling `kWeightScale` moved from 2 to 6.  These
distributions are heavily right-skewed—a base colour is mostly dark body with a
thin tail of highlights and markings—so with the mean at 1 the median sits at
0.36–0.62 and the 99th percentile at 5–9:

| Ceiling | Texels clipped (hull / track / turret / wheels) | Quantum vs median texel |
|---:|---|---:|
| 2.0 | 18.5 / 30.8 / 23.3 / 22.0% | 1.3% |
| 4.0 | 4.8 / 6.0 / 1.9 / 9.9% | 2.6% |
| **6.0** | **0.1 / 0.8 / 1.3 / 4.0%** | **3.8%** |
| 8.0 | 0.0 / 0.1 / 1.1 / 0.9% | 5.1% |

Truncation of the bright tail against quantisation everywhere; 6 is where the
first has essentially stopped and the second is still under 4% of a typical
texel.  `kWeightScale` had three definitions—the header, the test's own copy,
and the shader mirror that cannot include a C++ header—and the test went on
decoding against 2.0 after the writer moved to 6.0.  It now has one definition
and one documented mirror.

**Failure path.**  No base colour, unreadable texture, or `unmix = off`
leaves the material without a weight texture; the shader reads it as
"first curve, flat."  Unmixing failure is never worse than not unmixing.

### 6.4 Participating Media

`closesthit.rchit` implements a **single-scattering** volumetric path:
Beer–Lambert transmittance plus Henyey–Greenstein (HG) directional
scattering plus single-scattering from light sources.  Material properties
(`volume_density`, `scattering_coeff`, `absorption_coeff`, `phase_g`)
flow through `ConfigResolve` and `MaterialGpuData` (offset 160–172) to
the shader.

In non-RGB modes the per-channel σ values are averaged to a scalar because
`Material` carries only RGB coefficients, not spectral σ curves.  An extinction
coefficient is unbounded and has no Jakob–Hanika fit, so averaging is the only
honest reduction until the data model supports spectral extinction—the code
marks the insertion point.  What the medium did gain is the sun's own spectrum
at each of the four quartet wavelengths, because that is where the colour of a
hazy day comes from and the scalar branch could only sample it at the camera
wavelength.

**A volume attenuation colour is read as the spectrum it stands for.**  The
upsampling is legitimate there and is not everywhere in this block: what
`KHR_materials_volume`'s attenuation names is the fraction surviving one
attenuation distance, bounded in [0, 1] by construction, which is exactly the
quantity the sigmoid fit is defined for.  Which reading a ray gets depends on
what it carries: four wavelengths get four transmittances, one wavelength gets
one at that wavelength, an RGB preview keeps its three channels, and a band
outside the observer takes the average because there is no fit out there to
read.  A ray carrying one wavelength used to take component 0 of the RGB
triple, so a red-absorbing glass dimmed every wavelength by its red figure—the
visible band's own indirect bounce did this on every path through glass.  A
cyan-tinted prism now transmits with a red-to-blue spread of 0.0131 under the
quartet and 0.0134 under the deterministic grid, where the grid used to read
0.0003 and only dimmed.

**Underneath it, nothing ever exited glass.**  `entering` was computed from a
normal that had already been turned to face the ray, so it was true at every
hit: volume absorption was unreachable in every mode, and a ray leaving glass
refracted with air-to-glass indices and could not reach total internal
reflection.  It now reads which face was hit, recorded before the normal is
turned, and the normal handed to `Refract` stops being flipped with it.  The
correction changes every transmissive render; all four gates stayed green
through it, including the exact ones.

`volumetric.hlsli` also contains a complete **Delta-Tracking**
implementation (`DeltaTrackingHomogeneous`, `MAX_VOLUME_STEPS = 128`) and
HG phase-function sampling (`SampleHenyeyGreenstein`).  The delta-tracking
loop has no call site; multi-scattering (clouds, dense fog) is therefore
not available.  The single-scattering branch and the HG evaluation function
*are* active.  The Rayleigh, Mie, and HG formulas are independently
verified in `scripts/physics-audit/harness.py`.

### 6.5 Binding a Measured Curve

A material may carry a measured spectral reflectance, from the config's
`[spectral_curves]` or from the glTF's
`QUANTILOOM_material_ir.reflectanceCurve`—an entry in the former overriding the
latter, which is how a scene author corrects an asset they cannot edit.  Three
properties of that path were each established by a defect that no internal gate
could see, because the furnace cavities are grey and a grey curve is flat.

**A declared curve must actually reach the shader.**  The glTF-declared curves
were read into `Material::irReflectanceCurve`, logged by every render, and never
registered in the map both hosts use to hand a curve index to the shader—so the
shader upsampled the base colour instead.  The base colour is the arbitrary one:
construction concrete measures 0.254 at 550 nm against a glTF `baseColorFactor`
of 0.65, marble 0.792 against 0.85, asphalt 0.072 against 0.15.  In a closed
room that ratio compounds with every bounce, and the Cornell box rendered 36%
bright on the frame mean and close to 3× bright in its dimmest regions.
Separately, a `[material_overrides]` block that said nothing about infrared
computed `ρ = 1 − ε − τ` from two keys defaulting to zero and overwrote the
loaded curve with a flat perfect reflector—so setting `roughness` on a material
carrying a measured curve silently turned it into an IR mirror.  The derivation
now happens only when the config said something to derive it from.

**Resample over the band being rendered, not over the curve's own span.**
`SpectralCurveGPU` holds 64 uniform samples.  Spreading them across the source
curve's full range—300 to 12,500 nm for a cross-band asset, because one glTF
serves both the visible and the thermal configuration—gives a 194 nm step, so
the entire visible band holds two interior grid points.  The Cornell olive paint
measures 0.2175 at 550 nm and reached the shader as 0.127, a 41.6% error, and
its wall rendered grey.  Measured across all five Cornell curves and all five
bands, that grid was wrong by 10–120% *in every band*: concrete 35% out in SWIR
and 30% in MWIR, marble 54% in SWIR and 39% in LWIR.  `FromCPUBand` resamples
over the render band instead—the grid stays uniform and the struct keeps its
272-byte layout; only the start and the step change, 194 nm to 6.03 nm for a
visible render.  Two alternatives were measured and rejected: uniform in
wavenumber fixes the visible and leaves LWIR one sample (up to 199% wrong), and
splitting the 64 slots equally across bands is 3–8× worse than clipping and
would need a non-uniform grid this struct cannot express.  Partial coverage is
clipped and warned about rather than rejected—a curve covering part of a band is
still the best information about that material, and dropping it would fall the
material back to upsampling its base colour, which is the failure this whole
path exists to avoid.

**A bound curve overrides `ir_emissivity`, and the config is now told so.**  The
shader derives emissivity from the curve as `ε(λ) = 1 − ρ(λ) − τ` and never
reads `ir_emissivity` at all; a config setting both describes two different
surfaces, and the one it wrote down is the one discarded.  The check is
Planck-weighted over the **render** band—it has to follow the camera, because
the question is what the shader will use—and warns past 0.05.  The solver's band
average immediately above it stays hard-coded to LWIR on purpose: a surface
energy balance is a long-wave calculation whatever band the camera looks in.
The check discriminates rather than nags: the KV-2 gallery scene declares
`ir_emissivity = 0.92` for its painted hull and binds an ECOSTRESS sample giving
0.387 over MWIR—a single coat of alkyd paint on an aluminium substrate, opaque
in LWIR where it gives 0.90 and the declaration is right, and not in MWIR where
the metal shows through—so it fires for hull, track and turret in MWIR and is
silent in LWIR.

### 6.6 Emissivity and Reflectance Are Hemispherical

`common.hlsli` carried a directional law—`ε(θ) = ε₀ cos^0.7 θ` for dielectrics,
a saturating Hagen–Rubens boost for metals, and `ρ(θ) = 1 − ε(θ) − τ`—whose
exponents were tuned constants ("typical: 0.5–1.0") rather than Fresnel.  The
exponents were not the fatal half.  `ρ` was a function of the **view**
direction and was then used as the albedo of Lambertian lobes gathering light
from every *other* direction: an isotropic sky, and a sun at its own incidence
angle.  A Lambertian lobe whose albedo depends on the outgoing direction is not
reciprocal, and its directional-hemispherical albedo is not the quantity that
was measured—integrated over the hemisphere, desert sand returned 0.365 against
a measured 0.143.  An isothermal 300 K desert rendered at 324 K in MWIR, and the
322 K tank standing on it came out *darker* than the sand, 0.82× where the
measured emissivities and prescribed temperatures put it at 1.83×.  The thermal
ordering of the scene was inverted.

Both functions are deleted rather than left unused—with a tombstone, because the
shape of the mistake invites reintroduction—and all eleven call sites in NIR,
SWIR, MWIR and LWIR use the hemispherical quantities, on the bound-curve, the
*n,k*, and the scalar-fallback paths alike.  Three things made the change safe
to apply everywhere rather than only where the defect was observed: the SINGLE
branch had always refused to apply the law and said why ("adding a second
convention here would be a third answer rather than agreement with the
second"); the CPU thermal solver has always used a scalar hemispherical
emissivity, so removing the shader's law makes GPU and CPU agree rather than
differ; and nothing measured it—two unit tests asserted the monotonicity of the
fabricated law *itself*, and a third carried a 15% tolerance whose comment named
the approximation it existed to tolerate.  That tolerance is now 1×10⁻¹².

Directional emissivity is a real effect.  When it is wanted it belongs in a
specular lobe driven by measured *n,k* through `FresnelConductor` (`pbr.hlsli`),
which is exact, reciprocal and already checked against
`physics-audit/harness.py`—not as a scale on a diffuse albedo.  The gate that
was missing, `check_view_independence.py`, was added with the fix and shown red
before it.

Related, and found alongside: the infrared bands now read the **textured**
metallic value.  `GetEffectiveIREmissivity` reached inside `MaterialData` for
`metallicFactor` while the visible band has always shaded from
`factor × metallicRoughness.bg`, so a material whose texture says dielectric
over most of its area but whose factor is glTF's default 1.0 was a dielectric in
VIS and a metal in LWIR.

---

## 7. Environment Maps

### 7.1 The Four-Condition Invariant

`LightingParams::enableEnvironmentMap` is 1 on the GPU if and only if all
four conditions hold:

1. The configuration did not disable it.
2. The configuration **names** a map.
3. The map **loaded successfully**.
4. The mode is **RGB**.

**Why "names a map" is a separate condition.**
`renderer.environment_map_enabled` defaults to true—it expresses intent, not
state.  Following it alone, a scene naming no map would be lit by the
fallback cubemap: 256 × 256 of sky blue.  In a set of 15 test scenes, this
invented sky was **46–84% of the signal**.

**Why RGB only.**  The VIS_FUSED and SINGLE branches push the sampled RGB
through `ConvertLinearRGBToIlluminantSpectrum` and use the result as
spectral radiance density per nanometre—roughly 12× an ASTM G-173 sky.
SWIR / NIR / MWIR / LWIR never reference the cubemap; they trace
`TraceEnvBounceResidual` against the analytic sky or Planck downwelling
and are correct as they stand.  A spectral scene is lit by
`lighting.solar_lut` and the analytic sky; naming a map in one earns a
warning rather than a silently mis-lit render.

### 7.2 The Fallback Cubemap

`EnvironmentCubemap::Fallback` builds one black texel
(`kFallbackParams{1, 1}`).  It exists only because the pipeline declares
binding 10 with `descriptorCount 1` and no partially-bound flag—*something*
valid must always be written.  Black means a bug that samples the fallback
darkens visibly instead of silently adding light to a measurement.

### 7.3 Load Failure Corrects the Flag

Both hosts assume at resolve time that a named map will load, then fix the
flag where the fallback is actually chosen:

| Host | Location |
|---|---|
| CLI / offline | `OfflineRenderer::Impl::BuildPipeline`, in the `useFallback` lambda—zeroes the flag and re-uploads the lighting buffer |
| Interactive | `ExternalRenderContext::LoadEnvironmentMap`—sets the flag on success, clears it plus `hasCustomEnvMap` on failure, uploads either way |

`Impl::UploadLightingParams` is the sole writer of the buffer and masks the
flag with `hasCustomEnvMap`, so a host raising it via `SetLightingParams`
with nothing loaded cannot reach the shader.

IBL prefiltering uses `EnvironmentPrefilter` (specular cubemap convolution)
and `BRDFLutGenerator` (split-sum LUT).

---

## 8. Thermal Radiation and Temperature Field Solver

Four independently toggleable capabilities, all off by default:

| Configuration section | Capability |
|---|---|
| `[[materials]] temperature_texture` | Per-texel temperature field instead of a scalar |
| `[thermography]` | Invert the render into apparent temperature (`<output>_tapp.exr`) and report NETD |
| `[atmosphere] sky_model = "clear_sky"` | Zenith-colder-than-horizon sky from air temperature and humidity |
| `[thermal]` | Compute temperatures from a surface energy balance instead of prescribed values |

**Why all off by default.**  A scene without thermal-property data, forced
through an energy balance, would produce a precisely computed *wrong*
temperature.  The default lets the configuration's scalar temperature take
effect.  Naming a thermal conductivity is the switch that opts a material
into the solver.

Inside `[thermal]`, a further set of keys changes what the solve *carries*
rather than what it computes, each off or empty by default and each costing
nothing when unused: `convection_model` (§8.1), `lateral_conduction` (§8.1),
`sun_memory_lags` (§8.2), `parameter_sensitivities` (§8.3), `sun_correction`
and `dump_elements` (§8.12).  On a material, `internal_heat_w_m2`, `interior_bc`
and `shell` (§8.1, §8.9) say what is behind a face.  The interactive host
exposes the same set through `ThermalSolveParams`, which is the only thermal
struct the SDK exports and therefore the only way a front end reaches the
solve.

### 8.1 Surface Energy Balance: Six Fluxes

Six fluxes at the exposed face, three of which are geometric quantities
computed once per trajectory, not specified by any configuration key:

| Flux | Expression | Off when |
|---|---|---|
| Direct sun | `α_s E_dni cos θ vᵢ`, `vᵢ` from the shadow dispatch | No sun |
| **Diffuse sky** | `α_s E_diff Gᵢ`, `Gᵢ` = sky fraction + one bounce | `diffuse_irradiance_w_m2 = 0` |
| **Reflected sun** | `α_s E_dni Rᵢₖ`, baked per sun column | Scene is a single plane |
| Long-wave exchange | `ε σ (Σ Fᵢⱼ Tⱼ⁴ + sᵢ T_sky⁴ − Tᵢ⁴)` | `ir_emissivity = 0` |
| Convection | `h (T_air − Tᵢ)`, half-implicit | `convection_h_w_m2k = 0` |
| **Evaporation** | `f_wet (h/cₚ) Lᵥ (q_sat(Tᵢ) − RH · q_sat(T_air))`, Magnus–Tetens | `wetness_factor = 0` (default) |

**Short-wave gains share the long-wave CSR view factors.**  Radiosity and
radiative exchange are the same integral over the same hemisphere, in two
wavelength bands.  Sharing the CSR view-factor matrix makes `R` and `G`
cost one matrix traversal rather than a second geometric precompute.
Neither depends on temperature, so both are baked into the
`SunVisibilityTable` when the timeline is built; each time step reads them.
`R` has one column per sun sample, interpolated on the same indices as the
visibility it was baked from; `G` has a single column.
(`thermal/ShortwaveGains.hpp`)

**Why evaporation is the only implicit flux.**  Its slope in temperature is
several times the linearised radiative term.  Left explicit, it oscillates
at a one-minute step.  Newton linearisation places it in `diag[0]` and
`rhs[0]` beside the half-implicit convection.  The remaining terms are
either mild enough to be explicit (radiative, ~`4εσT³`) or coupled to other
elements via view factors—implicit treatment would require a global matrix.

**Why humidity is not duplicated into `[thermal]`.**  A constant-forcing run
reads `[atmosphere] relative_humidity`.  A scene with both
`[thermal].relative_humidity` and `[atmosphere].relative_humidity` would be
a scene with two atmospheres.  The forcing CSV's optional eighth column
overrides humidity per time step.

**Why the convective coefficient can vary through the day.**  Measured against
a SURFRAD station, one constant fitted to the daytime signal left the nights
0.55–1.76 K too warm, and none of the pre-registered candidates—latent heat,
soil moisture, lateral conduction—explains that: evaporation is a daytime sink,
and lateral conduction does not switch on at sunset.  The sweep that fitted `h`
says what it is.  Day RMSE has a sharp minimum at `h = 20` (5.84 K at 10,
1.18 K at 20, 4.00 K at 30) while the night bias climbs monotonically across the
same sweep, +0.38 K at 10 to +0.90 K at 30.  At night the ground sits below the
air, so convection is a *source* rather than a sink, and a coefficient sized for
a well-mixed afternoon pours in heat that a stable nocturnal boundary layer
withholds.  One constant cannot serve both halves of a day.  Driving it from the
station's own measured wind through McAdams' `h = 5.7 + 3.8u`—with no parameter
fitted anywhere—halves the night-time RMSE, 0.98–1.99 K to 0.48–1.16 K, at a
cost in the day, 0.91–1.66 K to 1.95–2.49 K.  That comparison was made with
McAdams computed outside the renderer and written into the ninth forcing
column, which left the correlation in a Python script and the night
uncorrected.

**`[thermal] convection_model` moves the law into the solver.**  `constant` is
the material's own number all day and what every existing scene gets; `wind` is
McAdams off a tenth forcing column carrying the wind speed; `stability` corrects
that for how the air is layered.  The stability correction runs the direction
the measurement asks for.  At night the ground is colder than the air, so
convection is a *source*, and the bias to remove is over-warming—so `h` has to
get *smaller* there, not larger.  A cold surface under still air is stably
stratified, the densest air already at the bottom with nothing to overturn, and
the Louis form `h / (1 + 10 Ri)` over a bulk Richardson number across
`convection_reference_height_m` says so.  On the unstable side a surface hotter
than the air raises plumes off itself, and free convection `C |ΔT|^(1/3)` with
`C = 1.52` is a floor under the wind law rather than a damping of it.  (A
`max()` of the two, the obvious first shape, can only raise `h` and would make
the night worse.)  Both branches are temperature-dependent, so both are
Newton-linearised into `diag[0]` beside the latent term—`4h/3` for free
convection, `h/(1 + 10 Ri)` for the damping—and under the constant and wind laws
that slope is written to be exactly zero rather than algebraically zero, which
is what keeps every existing scene bit-identical.  The constants are
`convection_wind_a`, `convection_wind_b`, `convection_free_c` and
`convection_stable_damping`.  The law is held by the stepper rather than by the
forcing, since it says how the balance is modelled rather than what the weather
is doing; `IThermalStepper::Convection()` reports it, and a host that was handed
a stepper that does not evaluate the law asked for falls back to the CPU one
(§8.6).  `e2_surfrad.py` gained the fourth arm this makes possible—measured sky,
wind in the tenth column, the solver applying its own stability law—so the
night bias can be measured against the wind-driven and constant arms rather
than argued about.  The arm exists; its numbers have not been re-measured in
either repository, and this document does not quote any.

**Three ways to say `h`, and the ninth column wins.**  A file carrying a
measured coefficient is stating what the correlations are estimating.  Absent
that, the tenth column feeds the wind law; absent both, `convection_model`
decides.

**Forcing CSV.**  Ten columns, the last four optional with defaults:

```
time_h, air_k, dni, sun_azimuth_deg, sun_elevation_deg, sky_k,
diffuse_w_m2, relative_humidity, convection_h_w_m2k, wind_speed_m_s
```

A file written before any of the four existed keeps its meaning exactly, and a
zero in the ninth means the same as an absent column: use the material's own
coefficient.  `SampleForcing` interpolates field by field, which is how the
first attempt at the ninth column appeared to do nothing—the endpoints carried
it, because they return a whole row, and every step between them read the struct
default.  `test_thermal_convection_forcing.cpp` now interpolates two rows
differing in *every* field and checks every field at the midpoint, written to
fail for the next member somebody adds as well as for this one.

**Heat can cross the edge between two triangles.**  Every element had its own
one-dimensional column and nothing joining it to its neighbours, so the
temperature field could only have edges where the mesh had edges.  That is very
nearly true for dry sand at an hour's timescale—heat diffuses about 3 cm in
one—and it is not true for a metal panel at any timescale.  `[thermal]
lateral_conduction` adds the term.  The adjacency comes from `BuildThermalMesh`,
the only place that still has the indices: edges are welded *within one object*
by a quantised endpoint key, so a mesh split across primitives joins up and two
objects that merely touch do not.  Heat does cross a contact, but through a
conductance nobody supplied, and inventing one is worse than leaving it out.
The conductance `g_ij = k_harmonic · w_ij / d_ij` is per metre of slab depth,
which is what makes the term one rate per element rather than one per node: a
node's cell height scales its capacity and its lateral conductance alike and
cancels.  The harmonic mean is what two conductors in series have.  The term is
explicit and reads a snapshot of the whole field, so the answer does not depend
on element order, and the tangents (§8.2, §8.3) are carried through it on the
same terms—a neighbour stepping into shade cools this element too, and leaving
that out would make `dT/dv` describe a column the solver is no longer stepping.
Explicit means a stability limit rather than an accuracy one, so the solve
reports the shortest lateral time constant and warns past twice it.  The
conductances ride in `ExchangeGeometry` beside the view factors—the same
question asked of contact rather than of sight—which is what lets every stepper
reach them without a signature change.  Checked against the error function: a
step along a chain of square elements is one-dimensional diffusion with the
material's own α, and the profile after twenty hours is within 1 K of `erfc` on
a 100 K step.

**What is behind a face.**  A back face could be insulated or pinned at a
room's temperature; neither describes a panel with an engine behind it, and an
engine behind a panel is most of what an infrared signature is—a shaded surface
warmer than everything around it cannot be explained by anything falling on it.
`internal_heat_w_m2` is a flux entering the back node, and `interior_bc =
"ambient"` lets that face convect to `interior_temperature_k` through
`interior_convection_h_w_m2k` instead of being insulated: a fuselage skin over a
bay, a sign, a fence.  The two compose, and the closed form they produce is
what the tests check—Fourier's law fixes `T_back − T_front = q d / k` whatever
the front is exchanging with, and with both faces open the source divides
between them by a series-parallel network rather than evenly, since it enters
at the back node.  Under `interior_bc = "fixed"` the back node is held, so a
flux there is absorbed by whatever is doing the holding; the config says so
rather than letting a scene carry a number that never reaches the answer.  A
thin plate whose back face sees the sky through its *own* hemisphere is not
this; it is a shell (§8.9).

### 8.2 Shadow Resolution: The Sun-Sensitivity Tangent dT/dv

The solver operates on one element per triangle (`BuildThermalMesh`);
`RunSunVisibility` decides from each element's **centroid** whether the sun
arrives.  The shader reads one temperature per `PrimitiveIndex()`.  The
temperature field can therefore only have edges where the mesh has edges:
on a 120 m desert ground tessellated 201 × 201, each triangle is 0.6 m,
and a 0.7 m sphere casts a triangle-shaped shadow—in a band where a
shadowed sand element is 40 K below a sunlit one.

The physics itself is not that coarse.  Dry sand diffuses heat ~3 cm in one
hour, and the model gives every element an independent 1-D column with **no
lateral conduction** unless `lateral_conduction` is asked for (§8.1)—the shadow edge it describes is as sharp as the
geometry.  Only the discretisation was coarse.

**Refining the mesh is impractical.**  Element count enters the view-factor
matrix as a square term and the per-step solve as a linear term, and every
element subdivided out of a 0.6 m triangle carries redundant physics—its
temperature is fully determined by the same 1-D column and a visibility
scalar.  The coarsening sweep below measures what refinement alone would
cost: 3,750× the element count of a 401² grid, which is 3.6 orders of
magnitude rather than the 2–3 this section used to estimate.

**The adopted path: ship the derivative alongside the value.**  Beside each
element's temperature the solver carries `dT/dv`—how far that temperature
would move per unit change in the element's own sun visibility.  The shader
traces its own ray to the sun and evaluates:

```
T(x) = T_element + (v(x) − v_element) · dT/dv
```

This reproduces the solved temperature at the element mean and resolves the
shadow edge at whatever resolution the ray tracer has.
`renderer/ThermalSunResponse.hpp` holds the GPU layout: one `float4` per
record on binding 26, index 0 a header carrying the solver's sun direction
(owned by the forcing CSV, not necessarily `[lighting] sun_direction`) plus
the on/off flag, then `1 + thermalElementBase + PrimitiveIndex()` per
element.

**dT/dv is a state, not a formula.**  Two tempting closed-form estimates
are badly wrong for anything with thermal inertia.  For sand at 11:00:

| Estimate | Value |
|---|---:|
| Steady-state response `α E cos θ / (h + 4εσT³)` | ~31 K |
| Single-step response | ~5 K |
| **Truth (3 h of sun)** | **27.9 K** |
| **Truth (12 h)** | **30.3 K** |

The tangent is therefore integrated as **the tangent of the trajectory**:
the same tridiagonal matrix as the temperature, a second right-hand side,
one extra elimination pass.  It inherits the slab's thickness, node count,
boundary condition, and full history—no closed form can do this, because
thermal inertia is inherently a function of history.
`ThermalState::sunSensitivity_K` carries it; both `CpuCrankNicolsonStepper`
and `thermal_step.comp.hlsl` step it.  It is **empty by default**: a caller
that only wants bulk temperatures sizes nothing and pays nothing.

**Cost and accuracy:**

| | |
|---|---|
| Extra rays | One per thermal hit where \|dT/dv\| ≥ 0.1 K—none at night, none indoors, none without a solver |
| Extra state | Doubles `ThermalState` and the timeline's checkpoints |
| Extra solver time | One elimination pass per element per step |
| Full-amplitude accuracy | 1.0 K on a 28.9 K contrast (3.5%), because radiative admittance changes by a third across the span |

Against a 0.6 m triangle that is fully lit or fully dark, 3.5% linear-
interpolation error is not in the same order of magnitude—this is the
entire reason the trade-off holds.  Measured across four tessellations of the
same 120 m ground against a pointwise reference (one 1-D column per sample
point, which is exact because there is no lateral conduction), the edge-band
RMSE within ±2 m of the shadow is:

| Mesh | Triangle edge | Elements | Per-triangle | With dT/dv |
|---|---:|---:|---:|---:|
| 51² | 2.40 m | 5,000 | 11.615 K | 0.552 K |
| 101² | 1.20 m | 20,000 | 8.740 K | 0.500 K |
| 201² | 0.60 m | 80,000 | 5.884 K | 0.464 K |
| 401² | 0.30 m | 320,000 | 4.018 K | 0.438 K |

The uncorrected error falls as the edge to the power 0.52—halving the triangle
halves the *width* of the misassigned band and leaves its amplitude at the full
contrast—while the corrected error is flat across a factor of eight in edge
length.  Extrapolating the uncorrected line to where refinement alone would
match the correction gives a 4.9 mm triangle: 3,750× the element count of the
401² grid, about 1.2 billion elements.

**The tangent assumes the shadow is stationary, and that is a condition rather
than a caveat.**  `dT/dv` is the derivative with respect to a visibility *held
for the whole trajectory*, while the shader applies it to the instantaneous
difference `v(x) − v_element`.  The two coincide only while the shadow does not
move over the surface's thermal memory.  Repeating the 201² measurement under
the scene's real moving sun gives an edge-band RMSE of 3.43 K with the
correction against 3.50 K without it—the benefit is gone, because an element
shaded only briefly is pulled down as though it had been shaded all day.
Stationary is not a contrived case: it is a wall's north face, a courtyard, a
vehicle's own underside, anything self-shadowed by fixed geometry, and any
scene evaluated at one time of day from a converged state.  A compact object
under a moving sun is the case a single scalar does not cover, because a pixel
shaded at noon may have been lit at ten, and the ground under it is still warm.

**`sun_memory_lags`: the shadow's history, one tangent per hour of it.**  The
solver already parameterises the sun's history as the K columns of its
visibility table, so `{dT/dv_k}` is the complete first-order linearisation of
the trajectory in that history—each one another right-hand side through the
same elimination, sourced only while its column is being interpolated and
decaying with the slab's own memory after.  `[thermal] sun_memory_lags = M`
gives the M most recent columns a tangent each.  They are a *decomposition*
rather than an addition: with a slot per column they sum to the whole-day
tangent, and the buffer ships the *remainder* in the present-sun record, so a
window too small for the day loses the ability to trace those hours where the
sun actually was and loses no energy.  That is what makes zero slots
byte-identical to the solver before it, and what makes the sliding window's
eviction safe rather than lossy.  Binding 26 grows a stride: `header.w` is
`1 + M`, the past sun directions follow the header, and each element gets
`1 + M` records; with `M = 0` the addressing reduces to `1 + element`.  The
shader traces one occlusion ray per column whose sensitivity clears a tenth of
a kelvin, toward where that column's sun was, and sums the terms.  Each tangent
is pinned by a centred finite difference of two whole runs perturbed in that
column alone.  The default is zero, and deliberately: what the columns are
worth is a measurement, `e4_compare.py --memory-lags` is the third arm built to
make it against the same pointwise reference, and it has not been run.  The
element dump carries a `(dT/dv_k, v_k)` pair per column with the hour and sun
direction of each, so the correction can be evaluated outside the renderer
exactly as `closesthit.rchit` sums it.

**The tangent is local on purpose.**  The neighbours' share of `incoming` is
not differentiated.  Its derivative would be the off-diagonal of a Jacobian
across every mutually visible element—what it would add is the second-order
fact that a colder patch of ground slightly cools its neighbours.  Keeping
it out is what makes `dT/dv` a per-element scalar a shader can apply per
pixel.

**The host, not the shader, decides when the sun is behind an element.**
The shader's geometric normal has been flipped to face the viewer, so it
cannot distinguish a face genuinely turned away from a hit on the *back*
of a sun-facing triangle (which has the same temperature as the front and
does need the correction).  Both hosts zero `dT/dv` for the first case;
the ray is then offset along the sun direction rather than the normal,
which is correct for both.

### 8.3 Parameter Sensitivities: dT/dp

Two things want to know how strongly a surface temperature depends on the
material it was computed for, and neither can get it from the temperature
alone.  A fit does: the convective coefficient is not a property of the soil at
all but of the wind over it, so it is fitted—and fitting it by sweeping costs
one whole trajectory per candidate for a resolution equal to the sweep's own
step.  An uncertainty budget does: what a 0.02 uncertainty in emissivity is
worth in kelvin is `dT/dε · 0.02`, and nothing else answers it.

`[thermal] parameter_sensitivities = ["h", "epsilon", "alpha", "k", "rhoc"]`
carries the tangent of each, by the same construction as `dT/dv`: a second
right-hand side through the same elimination.  Three of them touch only the
flux and ride beside the sun's tangents.  The other two move the matrix—`k`
sets the conduction between nodes and `ρc` the capacity—so their rows carry
`−(∂A/∂p) Tⁿ⁺¹` and cannot be built until the temperature has been
back-substituted.  The elimination is therefore split from its factorisation:
the parameter rows reuse the factors rather than paying for a second sweep, and
the factor recomputed from the modified diagonal is the same float the
factorisation used, so the temperature is bit-identical to before.
`SolveTridiagonal` takes a span of right-hand sides for exactly this reason:
the tangents that follow—one per column of the sun's history, one per material
property—are the same equation with a different source, more right-hand sides
through one elimination rather than more solves.

Every one is checked against a centred finite difference of two whole runs,
which is the definition evaluated the expensive way, and the two matrix-side
ones are what that catches.  `h` is inert where something else decides it—a
forcing column or a wind law is what `h` *is* then, and the material's number
never reaches the balance—rather than reporting the derivative of a number
nobody used.  A derivative is carried beside a trajectory and not inside it:
asking for two of them leaves the mean temperature identical to 1×10⁻⁹, which a
test holds, since otherwise every number a viewport showed would depend on what
its panel happened to have selected.

They reach a study three ways: a `dTd_<name>` column in the element dump, the
same in `thermal_column_tool`, and `e6_fit_h.py`, which is the Gauss–Newton
those columns make possible—`p ← p + (JᵀJ)⁻¹Jᵀr` with `J_i = dT(t_i)/dp`.
Against a synthetic record generated at `h = 18` and started from 20, it lands
on 18.000001 in three iterations with the residual at zero, and reports the
standard error the same Jacobian gives.  What it does not do is make the fitted
number a measurement: a coefficient fitted to one window is a summary of that
window's weather, and the split between the window it was fitted on and the
windows it is tested against is the caller's to declare.  The solve cache
(§8.10) keys on the list and never stores an entry that carried them: they are
a diagnostic the entry format does not hold, and one served back without them
would be a silently incomplete answer.

### 8.4 Time Integration

**Crank–Nicolson, not explicit Euler or fully implicit.**  Explicit Euler
on 1-D conduction has a stability constraint `Δt ≤ Δx²/2α`; for 10-layer
concrete that means second-scale steps and tens of thousands per day.
Fully implicit is unconditionally stable but only first-order accurate—it
smears the diurnal cycle's peak, which is precisely what thermal imaging
cares about.  Crank–Nicolson is second-order and unconditionally stable;
the cost is a tridiagonal solve, which has an O(n) Thomas algorithm.

**Thread-local Thomas, not parallel cyclic reduction.**  Cyclic reduction
parallelises *one large system*; the thermal solver has hundreds of
thousands of *small* systems (10–32 nodes each).  The natural parallel
dimension is elements, not nodes.  One thread per element, system in
registers, is faster than any cross-thread reduction.  The trade-off is a
32-node ceiling—exceeding it falls back to the CPU stepper, which in
practice almost never happens (10 layers through a 0.2 m slab suffices).

**Explicit inter-element coupling (ping-pong).**  Implicit coupling would
require a global matrix spanning all mutually visible elements—no longer
tridiagonal, no longer one thread per element.  Explicit coupling puts the
radiative stability constraint on the time step, but the radiative time
constant is much longer than the conduction one; in practice it does not
bind.

**StepMany batches dispatches.**  A single `ExecuteImmediate` submit costs
enough to dominate the per-step solve.  Scrubbing one time point often
requires hundreds of steps, so the entire batch is submitted at once.

### 8.5 ThermalTimeline

Fixed-grid trajectory: `t_k = startTime_h + k · timestep_s / 3600`.

**Why a fixed step that does not depend on the query time.**  If the step
were chosen by "divide [start, query] evenly," every new query would
produce a new trajectory and nothing could be reused.  A fixed grid lets
every step taken for an earlier query be reused by a later one—the
prerequisite for real-time response when scrubbing the time slider.

**Checkpoints, not full history or pure replay.**  Full history costs
step-count × element-count in memory (1,440 steps × 100 k elements for
one day is unacceptable).  Pure replay costs re-solving from t = 0 on
every backward scrub.  Storing a complete `ThermalState` snapshot every
`checkpointStride_h` simulated hours bounds the backward-scrub cost to
one stride.

**Steady-state initial condition.**  `initial = "steady"` solves for the
equilibrium temperature under the starting forcing—more physical than
starting from an arbitrary number, and it depends only on the initial
forcing, so it is computed once at construction.

### 8.6 Interactive Thermal Path

**GpuThermalStepper** (`renderer/GpuThermalStepper.hpp`) mirrors the CPU
Crank–Nicolson in f32 via `thermal_step.comp.hlsl`.  Each element is one
thread; the Thomas solve is thread-local (max 32 nodes); inter-element
radiative coupling reads from a ping-pong surface buffer.

**The host picks the stepper, and it picks CPU for anything the GPU one does
not carry.**  The list is the GPU stepper's honest limits rather than a
performance preference: no GPU, more than 32 nodes, a convection law other
than the constant one, lateral conduction, `sun_memory_lags > 0`, or a
non-empty `parameter_sensitivities`.  The last two matter more than they look,
and one of them was a bug before it was a rule.  The state is sized by the
timeline's `Desc`, not by the stepper, so a stepper that does not integrate a
tangent still receives the vector and hands it back at zero—and zero is a
derivative, not an absence.  A shading pass would then trace a shadow against
a response of nothing and a what-if preview would predict that no slider
changes anything, with every other check still green.  The host's guard had
covered the convection law and lateral conduction and had not been extended
when the tangents arrived; nothing stopped the GPU stepper running solves it
could not carry.  The interface now says what it carries
(`CarriesLateralConduction`, `CarriesLagSensitivity`,
`CarriesParameterSensitivity`) and the host asks before choosing.

**ThermalPreview** (`renderer/ThermalPreview.hpp`) is the internal subsystem
owned by `ExternalRenderContext::Impl`.  It holds the mesh, exchange
geometry, sun-visibility table, timeline, and stepper, plus dirty flags.
It never touches the pipeline or descriptors—binding 24 (per-element
temperature) is owned by the façade.  Everything the timeline is constructed
with except the descriptor is **held by reference for its lifetime**, so all
dependencies are members, not locals that go out of scope.

**What the viewport can ask a solve.**  Four façade calls, all read-only
against the trajectory except the last, which moves a scalar and re-renders:

| Call | Answers |
|---|---|
| `ThermalElementAt(pick)` | Which element a click landed on.  Says no for a ray that reached the sky, and for geometry whose material declares no conductivity—that one is not in the solve at all, which is a fact about the scene rather than a lookup that failed |
| `GetElementTrajectory(element, from, to, samples)` | That element's temperatures over a stretch of the day, and the six surface fluxes that produced them: absorbed sun, net long wave, convection, evaporation, conduction into the slab, conduction across shared edges.  All W/m², positive into the face, summing to what the surface is storing—which is what makes them an explanation rather than six readings |
| `GetThermalParameterSensitivity(parameter)` | The `dT/dp` field, one value per element, at the hour on screen |
| `SetThermalWhatIf(parameter, step)` | Renders `T + dT/dp · step` instead of `T`—a first-order preview of a slider, followable during a drag when a re-solve is not.  Measured against a real re-solve, half a unit of `h` on a 10 W/m²K material: the tangent predicted −0.660 K and the solve gave −0.642 |

Nothing is re-solved for any of them.  A trajectory replays from checkpoints
and restores the hour the viewport was showing before it returns, which a test
holds it to: a probe is a question about the past, not a request to move.  The
fluxes are decomposed by `EvaluateSurfaceBalance`, the same function the CPU
step builds its right-hand side from—it came out of `Step` with its
accumulation order preserved, so the flux `Step` builds from is bit-for-bit
the one it built before, and 140 thermal tests including the finite-difference
ones say so.  Decomposition is a query rather than an out-parameter on `Step`:
it reads a state and changes nothing, so the hot path is untouched and a
stepper that does not decompose its balance *declines* instead of returning
zeros—an empty flux list means nobody could answer, a list of zeros would mean
no heat moved.  The reference CPU balance answers whichever stepper produced
the trajectory, because the balance is a pure function of the state and the six
numbers do not change with whether the machine has a GPU.

**Binding 27 carries the what-if field**: a step at index 0 and one `dT/dp`
per element after it, so a scene previewing nothing binds a single zero and
the shader's multiply costs nothing and needs no branch.  The step rides in
the buffer rather than in `LightingParams` because that struct has no room
left—both its padding floats are spoken for, and growing it changes the
SDK/Studio pairing hash—which is also why binding 26 carries its own header.
Two debug views read the same derivatives directly:
`DebugVisualizationMode::SunSensitivity` draws `|dT/dv|`, which is what the
shadow-edge correction is worth per triangle and therefore where a coarser
mesh cost something; `ThermalSensitivity` draws `dT/dp` signed.  The
sun-sensitivity view takes a column index in `CameraData`'s last padding word
(the struct stays 80 bytes), so with a shadow memory it can be pointed at one
hour; it used to index `thermalSunResponse[1 + element]`, which is only the
element's own record when the stride is one—so with four lag slots it drew a
quarter of the elements' records, offset, for exactly the scenes it was added
to look at.

**Lazy invalidation.**  Changing the time (`SetThermalTime`) does *not*
invalidate anything—it steps forward or replays from a checkpoint.  These
events set dirty flags that the next `SetThermalTime` resolves:

| Event | Exchange | Sun table | Materials | Timeline |
|---|---|---|---|---|
| Geometry change (Adopt / Rebuild / Refit) | dirty | dirty | — | dirty |
| Material IR curve edit | — | — | dirty | dirty |
| `SetThermalMaterial` / `ClearThermalMaterials` | — | — | dirty | dirty |
| Shell flag or lateral conduction changed | dirty | dirty | — | dirty |
| `SetParams` (rays / topK changed) | dirty | dirty | — | dirty |
| `SetParams` (forcing / sun fields changed) | — | dirty | — | dirty |
| `SetParams` (timestep / layers / initial / anything that sizes the state) | — | — | — | dirty |
| `SetLighting` / `SetSunDirection` (no forcing file) | — | dirty | — | dirty |
| Timeline move (`SetTimelineTime`, §10.3) | — | — | — | — |

The shell flag and lateral conduction rebuild the *geometry* rather than the
material table because the pairing and the adjacency are found while the mesh
is walked.  `sun_correction`, `sun_memory_lags` and `parameter_sensitivities`
size the state, so changing any of them rebuilds the timeline; a parameter
comparison that missed them once reported a tool call as already applied while
the viewport went on showing a trajectory solved under the old ones.

During a gizmo drag, every frame calls `RefitAccelerationStructure`.
Immediately rebuilding the exchange geometry would drop the frame rate to
single digits, and the intermediate temperature fields are unseen.  Setting
a flag is O(1); the cost is deferred to the first scrub after the drag
ends.

**Short-wave gains have no independent dirty flag.**  They depend on
geometry, sun columns, and absorptivities; every event that changes any of
those already marks the timeline dirty.  They are re-baked in
`RebuildTimeline` and are fresh by construction—an extra flag would only
add a place to forget to set it.

### 8.7 Clear-Sky Model

Berdahl & Fromberg (*Solar Energy* 29(4), 1982): dew point from
Magnus–Tetens (WMO coefficients), clear-sky emissivity

```
ε = 0.711 + 0.56 (T_dp / 100) + 0.73 (T_dp / 100)²
```

and effective sky temperature `T_air · ε^(1/4)`.

**Why not an isotropic blackbody sky?**  An upward-looking thermal camera
does not see the air temperature; it sees a partially transparent atmosphere
backed by 3 K.  Using air temperature as sky temperature underestimates
radiative cooling by 10–20 K on a dry, clear night—and nocturnal radiative
cooling is one of the most information-rich phenomena in thermography.

The three intermediate quantities (dew point, emissivity, effective sky
temperature) are exported as `QL_API` wrappers (via `Thermography.hpp`)
so that hosts display them rather than each reimplementing the
correlations—two implementations inevitably diverge.

### 8.8 Thermography Inversion and NETD

A thermal camera does not report radiance: it measures radiance, assumes
(unless told otherwise) that the scene is a blackbody, and displays the
temperature that would produce the measured value.  Two surfaces at the
same temperature but different emissivities therefore read differently.
A measured thermogram cannot be compared with a rendered radiance field
until the render passes through the same arithmetic.

Following Aguerre et al. (*Computer Graphics Forum* 39(6), 2020),
equations 8–11:

```
L = τ [ ε B(T_s) + (1−ε) B(T_refl) ] + (1−τ) B(T_atm)
B(T_s) = [ L/τ − (1−ε) B(T_refl) − ((1−τ)/τ) B(T_atm) ] / ε
```

**Why per-band, not σT⁴?**  σT⁴ is total hemispherical flux, not what a
7–14 µm camera collects.  Using it folds out-of-band tail errors into
every temperature.  The per-band form uses the renderer's own quadrature,
so an isothermal cavity that renders its own blackbody inverts back to its
prescribed temperature—this round-trip is the core assertion of
`test_thermography.cpp`.

**Default parameters: ε = 1, no reflected source, no atmosphere.**  This
yields **apparent temperature**—the quantity a measurement campaign records
when it declines to assume an emissivity.  Setting the default to "some
typical emissivity" would give an uncharacterised scene a better-looking
but assumption-dependent number.

**NETD:**  `NETD = σ_L / (dL/dT)`, where `σ_L = σ_e / responsivity`.

- *Fixed-pattern noise is excluded* because NETD measures what averages
  away between frames; FPN does not.
- *Well-capacity saturation is not included.*  The reported NETD is the
  on-axis, unsaturated value: "the sensitivity that integration time
  would buy," not "what the detector can deliver."  The thermal bands
  place a large DC pedestal on the detector (~300 K background); well
  saturation is easy to hit, and folding it into NETD would hide the
  mismatch rather than flagging it.
- *Responsivity consistency* is checked by assertion: the responsivity
  in the NETD formula must be identical to `GenericSensor::RadianceToElectrons`.

### 8.9 A Panel Exposed on Both Sides

The solver gave every triangle a column with an exposed front and something
behind it.  A car panel, a road sign, a tent and an aircraft skin have nothing
behind them, and an asset models them as two sheets of triangles—so they were
solved as two independent slabs, each insulated against a wall that is not
there.  A panel in the sun came out as hot as if it were bolted to masonry, and
the shaded face sat wherever the initial condition left it, because nothing in
the model could reach it.

`[[materials]] shell = true` pairs the two faces and gives them one column: a
full surface balance at *each* end, one thickness between.  `ThermalMesh`
pairs two triangles of one primitive whose normals oppose and whose centroids
are within a small multiple of the material's own thickness; the lower index
owns the column and is stepped with the back row evaluated through the same
`EvaluateSurfaceBalance` the front uses, with the partner's normal, sky
fraction and view factors.  The higher index is not stepped at all; the
owner's back-node temperature is written into its surface slot, so the
radiative exchange, the render's temperature buffer, a dump and a probe all
read it without knowing a shell is involved.

Four things the pairing refuses, each with a test: it will not cross a
primitive, because two panels a millimetre apart are geometrically
indistinguishable from one shell and the conductance between them is one
nobody supplied; it will not pair faces pointing the same way, which is a
floor and a ceiling; it will not reach further than a few times the material's
thickness; and it counts and logs what it could not pair, because the rule is a
heuristic over geometry nobody authored for it—a material that declared itself
a shell and paired a tenth of its triangles is a modelling problem rather than
a solver one.  Two things the physics does: a symmetric shell under symmetric
forcing comes out isothermal to 1×10⁻⁶, which a back row that was still a
boundary condition could not produce; and a shell in the sun runs cooler than
the same slab against a wall, because it sheds through two faces instead of
one—the difference being zero is exactly what the old behaviour was.

The back face carries no shadow-edge tangent, and that is a claim rather than
an omission: the column holds `dT/dv` for the *owner's* visibility, and the
shading pass would pair that derivative with the partner's own, which is a
different quantity.  The partner's tangent is therefore zero, meaning its
triangle gets one temperature with no sub-triangle correction.  Turning the
flag on is a mesh rebuild rather than a material-table edit, for the same
reason lateral conduction is: the pairing is found while the geometry is
walked.

### 8.10 The Offline Solve Is Cached, and the Key Is the Interesting Part

A batch varies the sensor, and nothing about a sensor reaches the energy
balance.  So a list of 1,290 LWIR renders contains about **15 distinct solves**
and, before this, computed each of them eighty times—61% of a full rebuild's
wall clock.  `thermal/ThermalSolveCache.hpp` stores a solved `ThermalResult`
under a SHA-256 of its inputs.  Measured: a 1,518,656-element scene goes from
169.4 s to 5.2 s, and the render is byte-identical; a 33,884-element one from
5.9 s to 2.7 s.  On by default and invisible from a config—no TOML key, no
`InitParams` field, no exported symbol.  `QUANTILOOM_THERMAL_CACHE=0` switches
it off, `QUANTILOOM_THERMAL_CACHE_DIR` moves it, and there is no eviction; the
management story is deleting the directory.  Every failure resolves to "solve
it again"; nothing in it can fail a render.

**The boundary is the whole solve, not the stepping.**  An entry stands for
the GPU view-factor precompute *as well as* the trajectory—15 s plus 150 s on
the big scene.  Caching only the stepping would give back a tenth of the win,
and the precompute is the part that needs a TLAS.

**The key hashes resolved inputs, never config text.**  That is what makes two
configs differing only in `sensor.*` one entry.  It covers the built
`ThermalMesh`, the merged materials, the `[thermal]` scalars, the forcing CSV's
*contents*, `[lighting] sun_direction`, the stepper's `Name()`, the library
version, the GPU, and—from schema 4—every geometry epoch's start hour and
elements (§10.4), so a static scene hashes exactly as it did and a truck two
metres further along in epoch three is a different solve.  Miss a field and the
cache serves another scene's temperature field while the render exits 0.  That
is why the digest is 256 bits (the repository had no hash of any kind before;
SHA-256 was added and checked against the FIPS 180-4 vectors, with
length-prefixed framing so that concatenating heterogeneous fields cannot
alias), and why `BuildSolvedMaterials` was hoisted out of the solve: the
emissivity the solve uses is the Planck band average of the material's curve,
not the number in the TOML, and a key built from the config would be blind to
0.4 K.

**The GPU is in the key because the precompute is ray-traced GPU work.**  It is
deterministic for a fixed binary on a fixed driver—Hammersley, a stateless
coverage hash, no atomics—but BVH construction and intersection are a vendor's
business.  Keying on device identity makes a driver update a miss rather than a
wrong answer.  The GPU stepper is built before the lookup even though a hit
wastes it, because deferring it would mean naming the stepper in the key before
knowing whether it starts, and owing a second key when it does not.

**A hit still prints the gate line.**  `LogThermalSolveSummary` is the only
place that emits `Thermal: N elements (M solved)`, and downstream reads the
element count off it to catch a scene whose subject fell out of the solve; a
hit that printed nothing would make every render look clean.  Anything new
logs `Thermal cache:` or `Thermal stepper:` so nothing can mistake it for the
summary.  The entry digest covers the header as well as the arrays, because
the header carries `participatingElements` and the temperature range the
summary is printed from, and digesting only the arrays left those as the one
part of an entry where bit rot would be read back as a measurement.  Naming
`thermal.dump_elements` opts out in both directions: the dump needs the
exchange's sky fractions, which an entry does not carry.

`kKeySchemaVersion` is at **4** and `kCacheFormatVersion` at **2**.  The first
bumps when a new input joins the key—the convection law and its constants,
`lateral_conduction`, `sun_memory_lags`, `parameter_sensitivities`,
`internal_heat_w_m2`, the interior boundary, the shell flag and the epochs all
did—and the second when the stored result grows an array, which the lag
tangents did.  Each new field wants an `EveryMaterialFieldChangesIt` or
`EveryConfigScalarChangesIt` case beside it in `test_thermal_solve_cache.cpp`,
which is what makes a forgotten one a failing test rather than a wrong render.
The library version is in the key too, so a release bump empties the cache—
which is the point of a version pin, since half a sweep rendered by one build
and half by the next is not distinguishable from the outputs afterwards.  (A
test that mutated the version to "0.2.7" to prove the key moved went red the
day the library reached 0.2.7; the fixture now defaults to a version the
library will not have.)

**`QUANTILOOM_THERMAL_GPU_STEPPER=1`** runs the offline trajectory on
`GpuThermalStepper` (32-node cap, else it falls back and says so).  Off by
default, and not judgeable by byte equality: f32 and a different reduction
order give different floats.  Against the 0.2.5 baseline at `nominal_LWIR`, on
a DN range of about 34,000, DamagedHelmet moved 85.8% of pixels (mean 6.5 DN,
p99 17, max 23) and CesiumMilkTruck 35.7% (mean 0.4 DN, p99 1, max 2)—an order
of magnitude apart, so a tolerance argued from one scene is wrong for the
other.  Stating one per band is the work before this defaults on; what it
already buys is the wait while authoring, where 165 s is per material edit.
The stepper's name is in the cache key, so GPU and CPU results live at
different entries and cannot be served for each other.

### 8.11 Single Decode Entry Point

A surface temperature has exactly one decode entry point—
`GetSurfaceTemperatureK` in `closesthit.rchit`—and all call sites go
through it: two sampling paths (SWIR and general MWIR/LWIR) and two debug
views (`DEBUG_MODE_TEMPERATURE` and `DEBUG_MODE_IR_EMISSION`), for four
total call sites.  Decode order: solver → temperature texture → material
scalar.  When the solver provides a temperature, the per-pixel sun
correction from §8.2 is applied **inside** this function—placing it outside
would let any new sampling site forget it, producing a rendering bug
(triangle-shaped shadow edge) that looks like a solver bug.

### 8.12 Verification

| Verified quantity | Method | Location |
|---|---|---|
| Planck / Wien / Stefan–Boltzmann | Closed-form + independent Python | `test_blackbody.cpp`, `test_blackbody_physics.cpp`, `harness.py` |
| 1-D conduction | Closed-form (semi-infinite body, steady gradient) | `test_thermal_conduction.cpp` |
| Lateral conduction | `erfc` profile of a 100 K step along a chain of elements after 20 h, within 1 K | `test_thermal_lateral.cpp` |
| Back-face source and boundary | `T_back − T_front = q d / k`; series-parallel split with both faces open | `test_thermal_conduction.cpp` |
| Shell pairing | Refusal rules; symmetric shell isothermal to 1×10⁻⁶; a shell cooler than the same slab against a wall | `test_thermal_shell.cpp` |
| View factors | Analytic view factors | `test_thermal_exchange_gpu.cpp` |
| Short-wave gains | Analytic configurations | `test_shortwave_gains.cpp` |
| Clear-sky model | Original correlations | `test_sky_thermal.cpp` |
| Timeline / checkpoints | Replay consistency; batches split at epoch boundaries | `test_thermal_timeline.cpp` |
| Epoch planning | Candidates, stride, minimum-move filter; a 0.2 m slab still 4 K above equilibrium a day after a truck parks | `test_thermal_epochs.cpp` |
| GPU vs CPU stepper | 24 h trajectory step-by-step comparison | `test_thermal_step_gpu.cpp` |
| dT/dv and the lag tangents | Central differences from two full runs, one per perturbed column; full-shadow trajectory | `test_thermal_sun_sensitivity.cpp` |
| dT/dp | Central differences of two whole runs per parameter; non-interference of the mean to 1×10⁻⁹ | `test_thermal_parameter_sensitivity.cpp` |
| Interactive invalidation, probe, what-if | Dirty-flag state machine; the hour is restored after a probe; the tangent predicts a re-solve; exchange runs are counted across scrubs | `test_thermal_preview.cpp`, `test_timeline_context.cpp` |
| Forcing interpolation | Two rows differing in every field, checked at the midpoint | `test_thermal_convection_forcing.cpp` |
| Solve cache | Every material field and config scalar moves the key; a flipped header byte refuses to load | `test_thermal_solve_cache.cpp`, `test_sha256.cpp` |
| Element dump | Unsolved rows marked, absent tangents blank, material properties as the solve saw them | `test_thermal_element_dump.cpp` |
| Energy balance against a real site | SURFRAD station forcing; RMSE and bias reported per half-day; four forcing arms | `scripts/experiments/e2_surfrad.py` |
| Parameter fit | Gauss–Newton on the tangent lands on a synthetic `h = 18` from 20 in three iterations | `scripts/experiments/e6_fit_h.py` |
| Temperature-emissivity separation | NEM/ratio/MMD run on a rendered LWIR cube of a plate with a known reststrahlen emissivity | `scripts/experiments/e9_tes.py` |
| Temperature texture | Sampling and decode | `test_temperature_texture.cpp` |
| Thermography inversion + NETD | Isothermal-cavity round-trip; responsivity consistency | `test_thermography.cpp` |

**TES is a thing to validate rather than to trust**, and `e9_tes.py` is the
scene that can: N bands give N measurements and N + 1 unknowns, so every
separation method adds a constraint from outside the measurement, and ASTER's
NEM/ratio/MMD adds an empirical regression between a spectrum's spread and its
minimum.  On a plate with a quartz-like reststrahlen band it recovers 307.5 K
against a true 310.0 and an emissivity RMSE of 0.043, which is what a spread at
the edge of the regression's calibration costs; on the grey control it recovers
308.8 K and pulls a true 0.960 emissivity up to 0.978–0.984, which is the MMD
regression's own grey-body failure.  Both are the method's known behaviours,
reproduced rather than tuned away.  Two defects had to be fixed before it
could run: single-wavelength mode had no Planck term at all, so a 310 K plate
in the dark rendered exactly zero at 10 µm and a thermal hyperspectral cube
came back a cube of zeros with the run exiting 0; and a cube ignored
`renderer.resolution` and `[camera]`, coming out 1280 × 720 from a default
camera while the log printed the resolution the config asked for.

**Two `[thermal]` keys exist for measurement rather than for rendering.**
`sun_correction = false` gives the uncorrected temperature field—one constant
per triangle—which is what the correction has to be compared against; it sizes
the tangent out of `ThermalState` rather than suppressing it at the shader, so
the solve neither carries nor pays for it.  `dump_elements` writes the solve one
row per element: the centroid and normal the solve used, the temperature it
reached, the (dT/dv, v) pair the shader would apply—one pair per carried sun
column—and a `dTd_<name>` column per parameter sensitivity.  An unsolved
element is marked rather than omitted, because a reader joins the file against
its own traversal of the mesh and a dropped row shifts every index after it;
an absent tangent leaves the column blank rather than 0, since 0 is the claim
that the temperature does not move with the sun while blank is not having
asked.  The dump carries the material properties **as the solve saw them**, and
that is the part worth keeping—a material bound to a measured spectrum has its
long-wave emissivity replaced by the Planck-weighted band average of that
curve, so a config that says 0.90 can be solved at 0.9329, and reproducing an
element's trajectory from the config instead of from the dump is a 0.4 K error
that looks exactly like a result.

Both reach the interactive host through `ThermalSolveParams`, and the two are
not the same kind of thing.  `sunCorrection` is a solve parameter and rebuilds
the timeline when it moves.  `dumpElementsFile` is a path and *not* a trigger:
offline, one run is one dump and writing from inside the solve is right, but a
viewport re-solves on every scrub of the hour slider, and the same arrangement
would turn dragging a slider into hundreds of writes.  The write is
`ExternalRenderContext::DumpThermalElements()`, and both paths go through one
writer, `thermal::DumpThermalElements`, so the format cannot acquire a dialect;
they nearly did, because the interactive field left `v` empty when no tangent
was carried while the offline writer emitted it either way.  The dump now
samples visibility itself—a property of the geometry and the sun that an
element has whether or not anyone asked how its temperature responds to it.

`thermal_column_tool` drives one 1-D column through the same `ThermalTimeline` a
scene element gets, with no scene around it.  It agrees with the renderer's own
solve, for a fully lit element under an unoccluded sky, to 0.0001 K in
temperature and 0.00004 K in dT/dv—which is the precondition for using it as the
pointwise reference in a mesh-resolution study, since any residual would
otherwise be indistinguishable from the discretisation error being measured.

Example configurations: `thermal_solver_lwir.toml` (0.2 m concrete slab,
adiabatic back, clear sky, noon 900 W/m² DNI, 10 layers, 60 s step,
midnight steady-state start), `thermal_texture_lwir.toml`,
`thermography_lwir.toml`, `clearsky_lwir.toml`, `thermal_sequence/`
(batch rendering at several times of day).

---

## 9. Post-Processing Pipeline

### 9.1 Sensor Effect Chain

Two implementations: the offline chain runs `GenericSensor` on the CPU;
the interactive chain runs six compute-shader passes on the GPU.

**Why two chains?**  The offline path requires deterministic, bit-exact
results—CPU floating-point reduction order is reproducible and there is no
performance pressure.  The interactive path must finish in single-digit
milliseconds; round-tripping the full frame to the CPU and back would alone
exceed the budget.  Coefficient consistency (e.g. responsivity) between the
two paths is enforced by unit tests.

**Offline (CPU) sensor model (`GenericSensor`):**

- **Optics:** Focal length, F-number, pixel pitch, cos⁴ natural vignetting
  (with telecentric lens bypass).
- **Detector:** Quantum efficiency, well capacity, Poisson photon noise,
  read-out noise, dark-current noise—each independently switchable.
- **Fixed-pattern noise:** PRNU (gain non-uniformity, multiplicative) and
  DSNU (dark-signal non-uniformity, additive) modelled separately, and
  neither independent per pixel.  PRNU is generated per **column** and
  smoothed along the array with an 8-pixel kernel, DSNU per **row** with a
  5-pixel one, each carrying a further per-pixel component at a tenth of its
  amplitude and renormalised to the requested sigma.  That follows the readout
  architecture — column amplifiers, row addressing — and it is what survives a
  correction tuned for independent pixel noise.
- **NUC:** Non-uniformity correction with configurable efficiency
  (typical 95–99%).
- **ADC:** Configurable bit depth (12 / 14 / 16 bit) and gain.
- **PSF blur:** Gaussian convolution with sigma matched to the Airy disk's
  FWHM (factor 0.437, analytically derived from `[2 J₁(x)/x]²`).
- **Deterministic random seed.**  Two of them, both fixed by default and
  separate because separate stages consume them: `renderer.seed` = `0x547C`
  drives the path tracer's sample sequence, `sensor.noise_seed` = `0x548C` the
  sensor chain's temporal and fixed-pattern noise.  A sensor-noise study varies
  the second across frames while holding the first, so that the radiance field
  under measurement does not move.  Either set to zero requests a
  nondeterministic draw.

**Interactive (GPU) sensor pipeline — six compute passes:**

1. PSF blur, horizontal (separable Gaussian)
2. PSF blur, vertical
3. Radiance → electrons
4. Poisson + read-out noise
5. Fixed-pattern noise (PRNU + DSNU)
6. Quantise → radiance

The PSF blur runs in the radiance domain (before electron conversion),
and the separable implementation uses two separate shader modules
(`sensor_psf_blur_horizontal`, `sensor_psf_blur_vertical`).

**Why PRNU and DSNU are separate, not one "FPN strength."**  They have
different physical origins and different signal dependence: PRNU is
multiplicative (grows with signal), DSNU is additive (independent of
signal).  A single parameter cannot be calibrated to match both bright-field
and dark-field behaviour simultaneously—and infrared scenes operate under
a high DC background where the ratio matters.

**Why the fixed pattern is measured with one sensor over N frames.**
`sensor_lab` (§2.2) runs N frames through a *single* `GenericSensor` instance.
Rendering N times cannot separate temporal noise from the fixed pattern: a
fresh sensor with a fresh seed draws fresh FPN maps, so the pattern moves with
the noise.  One instance applied N times keeps the maps fixed while the stream
advances, which is one detector and many frames.

**Why NUC deliberately retains a residual.**  Perfect correction is
equivalent to not modelling FPN at all.  Real systems achieve 95–99% NUC
efficiency; the residual non-uniformity is the persistent "fixed texture"
in thermal images and the disturbance that algorithm robustness must
withstand.

**Why the default seed is fixed.**  Same scene, same parameters, bit-
identical raw DN—quantitative comparison is otherwise meaningless.
Setting the seed to 0 enables per-run variation.  The default serves
reproducibility over realism because the system's primary use is
quantitative validation.

### 9.2 Display Enhancement

`BlitToTarget` is a bare format conversion—there is no tone mapping in the
rendering path.  An LWIR render's radiance sits around 1×10⁻², two orders
of magnitude below 1.0, so **display enhancement is not cosmetic; it is
the only reason an infrared scene is visible at all.**

`clahe.comp.hlsl` (three passes, driven by `DisplayEnhancementParams` in
`include/quantiloom/renderer/DisplayControl.hpp`) composes two independent
stages:

| Stage | Decision | Modes |
|---|---|---|
| **Tone** | Contrast: scalar → [0, 1] | `Linear` (linear AGC, default), `Equalize` (plateau AGC), `Clahe` |
| **Palette** | Colour: scalar → RGB, changing no contrast | `Grey` (white-hot), `GreyInverted` (black-hot), `Ironbow`, `Rainbow`, `Viridis` |

**Why two fields, not one enumeration.**  The two decisions compose; the
Cartesian product would be fifteen names for two choices, and every new
palette would add three enum values.

**Why `Linear` is the default, not the more visually striking CLAHE.**
The distinction is not sharpness but **whether the entire image is mapped
identically.**  Measured on `thermal_solver_lwir` at 3.16 M pixels, sorting
display value by raw radiance and counting inverted pairs:

| Operator | Inverted pairs | Worst drop |
|---|---:|---:|
| `Linear` | 0.000% | 0 |
| `Equalize` | 0.000% | 0 |
| `Clahe` | 1.902% | 17 display levels |

The measurement is made on the radiance field rather than on the sensor
chain's output deliberately: the property under test is that two pixels at one
temperature display alike, and once temporal noise and a 16-bit quantisation
are in the image they no longer *arrive* alike—through the full chain the same
statistic reads 0.285%, because quantisation ties leave an ordered comparison
and flatter the operator.

Two pixels at the same temperature in different CLAHE tiles come out as
different greys.  **Do not read a temperature from a CLAHE view.**  It is
for finding edges.  `Linear` is what a thermal camera calls linear AGC.

**`Equalize` required no new pipeline.**  Pass 2 sums every tile's
histogram, so all tiles share one CDF; pass 3's bilinear blend between
four identical CDFs is that CDF.  `Linear` skips passes 1 and 2
entirely—which is why pass 3 re-binds the descriptor set rather than
relying on pass 1 having done so.

**Achromatic bypass.**  The luminance-preserving path divides by the
pixel's own luminance, and `RGBToLuminance(v, v, v)` is not `v` to the
last bit.  The ratio used to wobble grey pixels around each CDF plateau
(12.9% inverted pairs).  Achromatic pixels now bypass it—zero loss for
every infrared render, which is always monochrome.

**Screenshot vs export.**  `CaptureDisplayImage()` reads the display-
enhanced image; `CaptureScreenshot()` reads the raw accumulation.  The
Studio's "Save Screenshot" gives the false-colour view; "Export Image"
gives physical values.  They serve different purposes; unifying them would
disable one.

### 9.3 Multi-Band Fusion

`postprocess/MultibandFusion.cpp` fuses VIS, SWIR, and MWIR renderings
into a single enhanced-display image.  Application: camouflaged objects
indistinguishable in visible light but clearly identifiable in thermal
infrared; fusion superimposes each band's high-response regions.  Output
is EXR (HDR) + PNG (LDR preview).  **Fusion is qualitative—merged values
carry no physical unit.**

Four algorithms, selected by `[fusion] method`:

| Algorithm | Principle |
|---|---|
| `weighted_average` | `Fused = w₁·VIS + w₂·SWIR + w₃·MWIR` |
| `laplacian_pyramid` (default) | Multi-scale: low-frequency average, high-frequency max-absolute |
| `max_response` | Per-pixel maximum |
| `pseudo_color` | VIS → R, SWIR → G, MWIR → B |

**Why Laplacian pyramid by default.**  Weighted average dilutes each band's
unique details by the weights; max response preserves details but produces
discontinuities at band-switching boundaries.  The pyramid decomposes each
band at every spatial frequency, taking the average at low frequencies and
the maximum absolute value at high frequencies—preserving each band's edges
without seams.  The cost is O(N) multi-scale decomposition and
reconstruction, negligible for offline post-processing.

**Why per-band normalisation is mandatory.**  The three bands' radiance
magnitudes differ by orders of magnitude (VIS ~ O(1), SWIR ~ O(10⁻³),
MWIR ~ O(10⁻²)).  Without normalisation, fusion displays only the
highest-magnitude band.

**Distinction from display enhancement.**  Fusion combines *multiple bands*
into one image; display enhancement (§9.2) maps *one band's* radiance to a
visible range.  The two are serial, not interchangeable.

### 9.4 Hyperspectral Cube Output

The renderer writes spectral cubes in three formats, selected by
`[hyperspectral] output_format` (`io/SpectralCubeIO.cpp`), with
GPU-accelerated spectral reconstruction (`GpuSpectralReconstructor`) and
optional per-band intermediate export (`save_intermediates`).  Adaptive
wavelength grids are handled by `hs_core/AdaptiveGridGenerator`.  The
completion line names the extension the format actually wrote.

| Format | What it writes | Reads it |
|---|---|---|
| `envi` (default) | BSQ / BIL / BIP raw plus `.hdr` | Every remote-sensing tool |
| `exr_spectral` | One f32 channel per band, `S0.<wavelength>nm` with a comma for the decimal separator (a dot is OpenEXR's layer separator), `spectralLayoutVersion` and `emissiveUnits` in the header—the layout of Fichet, Pacanowski & Wilkie 2021 | ART, Mitsuba, the spectral-exr tools |
| `geotiff` | A hand-written baseline TIFF: classic little-endian, 32-bit float, uncompressed, one plane per band, wavelengths in GDAL's metadata tag 42112 twice over, as a band description so `gdalinfo` prints "9333.5 nm" and as a number so a reader gets it back without parsing a sentence | QGIS, GDAL, ENVI, `tifffile` |

**Why ENVI, not HDF5.**  HDF5 is a heavy dependency (build time,
distribution size, Windows link complexity), and its hierarchical metadata
capability is over-specified for "one cube plus one header."  ENVI is the
remote-sensing de facto standard; every downstream analysis tool reads it
directly.

**Why the EXR is a single part.**  Both EXR paths were stubs for a long time,
and the note above them said the multipart OpenEXR API had brought a shutdown
crash through a static-initialisation order it shares with VMA.  The layout
that has since become the interchange format wants a single part anyway, so
both go through `ImageIO`, which the render output already uses, and the crash
has nothing to attach to.  `emissiveUnits` says `W.m^-2.sr^-1.nm^-1` rather
than the layout's usual band integral: a cube holds a spectral radiance
density, and the two differ by a bandwidth.  Two things the reader has to get
right: a channel list is a name-keyed map, so the order a file hands back is
alphabetical—`S0.1000nm` arrives before `S0.400nm`—and the band order comes
from the parsed wavelengths instead; and two bands at one wavelength would
collide into one channel, losing a band with no error anywhere, so the writer
refuses.  The two EXR tests that used to skip are live, which took the suite's
SKIP baseline to 8.

**Why the TIFF is written by hand and is not georeferenced.**  `WriteGeoTIFF`
logged "not yet implemented", wrote a `.raw` and a `.hdr` beside the path it
was given, and returned true—a caller that asked for `foo.tif` got no `foo.tif`
and no error.  libtiff and GDAL are both large dependencies whose value here
would be the georeferencing, and a rendered scene has no coordinate reference
system, no tie point and no pixel scale; a file claiming one would be claiming
something false.  What is wanted is a TIFF that opens with one band per
wavelength and the wavelengths attached, and that is a baseline TIFF plus
GDAL's own metadata tag.  Classic TIFF addresses with 32 bits, so a cube past
4 GB is refused rather than truncated; BigTIFF is a different format, not a
larger offset.  The reader takes both planar configurations, since this writer
makes one and most other tools make the other, and says which of the several
ways a TIFF can be unreadable a given one is.  Verified against `tifffile`, an
independent implementation.  One bug worth naming: the metadata tag was first
written with a count of zero, because `metadataBytes.size()` and
`std::move(metadataBytes)` were arguments to one call—unsequenced, the move ran
first and the size was read off an emptied vector.  It compiled, it wrote a
file, and the only symptom was wavelengths that came back as 0, 1, 2, 3.

A cube in a thermal band is rendered band by band through the single-wavelength
mode, which until this round carried no Planck term at all (§8.12).

---

## 10. A Scene Can Have a Clock, and Several Models

Until 0.3.2 a config named one scene file and placed it once.  It can now
name `[[models]]`, each with a rest pose and a trajectory, against a
`[timeline]` that says what a second is.  `assets/configs/timeline_demo.toml`
is the scene this was built for: a block drives across the desert for four
seconds and parks, while ten seconds of clock walk ten hours of sun.

### 10.1 Seconds Are Canonical, and Several Files Become One Scene

Seconds are canonical; a tick is the frame grid those seconds are sampled on,
and `ticks_per_second` relates them—the job USD gives `timeCodesPerSecond` and
Minecraft gives its twenty ticks a second.  It is a **double**, because a
timeline that spans a month wants a tick every hundred and fifty seconds and
0.006667 is a legal answer (`seconds_per_tick` says the same thing the other
way round).  Every time-valued key also accepts a unit: `"15s"`, `"90min"`,
`"36h"`, `"2.5d"`.  `end_s = 2592000` is a number nobody checks.

Merging several files into one `Scene` is not concatenating four vectors—every
index in the second one moves.  `scene/SceneMerge.cpp` keeps the single list
of where a texture index lives, prefixes node names with their model
(`name = "car/Wheel_FL"` in a `[[nodes]]` entry), and leaves material names
bare unless two files bring the same one; the later model's is then renamed
`<model>/<name>` and the log says so.  Such a material is addressed with a
quoted key wherever TOML needs one—`[material_overrides."block/Material"]`—and
the slash needs no quoting in `Config::Get` paths, which split on dots only.
A scene made entirely of `[[models]]` loads through every public entry point,
including the ones nobody hands a resolved config to: `LoadSceneFromConfig`
resolves for itself when nobody resolved for it, here rather than in each
caller, because what those entries mean is `ResolveRenderConfig`'s to say.

### 10.2 Three Grammars, Because Three Kinds of Author Write Them

One spec is in exactly one form, and `ConfigResolve.cpp` carries the full
syntax in a comment beside where it is read.

- **Keyframes** (`[[models.motion.keys]]`, or `keys_file` for a CSV of them),
  with `step` / `linear` / `cubic` interpolation and `hold` / `loop` outside
  the authored range.  What an exporter writes, and what glTF, USD and DIRSIG
  all agree on.  Cubic is Catmull–Rom on position and an eased slerp on
  rotation—deliberately not squad, which would need tangent quaternions nobody
  authors.
- **Piecewise expressions** (`[[models.motion.segments]]`), in `t` (global
  seconds) and `s` (seconds into the segment).  What a person writes when the
  path is a formula.  ExprTk compiles them, behind `scene/MotionExpr.cpp`, the
  only translation unit that includes the header—it costs most of a minute and
  `/bigobj` to build.
- **Parametric engines** (`[models.motion.location]` / `.orientation`):
  straight line, circle, spin, along_velocity, look_at.  The handful of motions
  common enough in sensor simulation that spelling them either other way is
  busywork.

### 10.3 The Composition, and Why `[[nodes]]` Records a Rest Pose

```
world(t) = M_model(t) · R_model · M_node(t) · R_node
```

`R_model` is the `[[models]]` rest pose and `R_node` is what the file placed
the node at.  The model's track moves the whole model in world space; a node's
own `[nodes.motion]` moves it within its model, which is what makes a wheel
spin about its axle rather than about the world origin.  So `[[nodes]]` writes
the **rest** pose for a node the clock moves—where it would stand with every
trajectory at the identity.  A pose meaning "where it is at `time_s`" would be
wrong at every other instant.

None of this touches `Scene`, `SceneNode` or `Material`: they are the layout
contract with Studio and cannot grow a field.  The animation is a side table
keyed by node index (`renderer/TimelineState.hpp`).

**Scrubbing is not a geometry edit.**  `SetTimelineTime` moves every animated
node, refits the acceleration structure in place, and—when the two are
mapped—re-solves the thermal field at the hour that second corresponds to.  It
is deliberately *not* `RefitAccelerationStructure`: that one invalidates the
thermal preview's geometry, which is right for a gizmo drag (a rest pose
changed, so the exchange is stale) and wrong for a scrub (the trajectory the
exchange was computed from has not changed at all).  A private refit path is
what guarantees a scrub costs no exchange precompute, and a test watches the
exchange-run counter across four scrubs.  The emitter list is rebuilt only when
one of the things that moved actually emits.  A gizmo drag at t = 30 goes the
other way: `SetNodeTransform` solves the composition for `R_node`, so the edit
is at every tick rather than at the one it was made on, and
`GetNodeRestTransform` is what a host writes into `[[nodes]]`.

**The thermal hour follows the clock.**  With a `[timeline]`, `thermal.time_h`
stops meaning "the hour to render" and means "the hour at `start_s`";
`thermal_time_scale` says how fast the clock runs from there (3600 in the demo,
8640 watches a day in ten seconds).  `TimelineInfo` reports both, and
`SetTimelineThermalMapping` changes them at runtime.  `thermal_geometry =
"reference"` freezes the geometry at `thermal_reference_s` and solves once,
which is right when the motion does not matter thermally; `"epochs"` is the
next section.

### 10.4 Thermal Geometry Epochs: One Solve Across a Geometry That Moves

A truck drives across the sand and parks.  What happens to the ground?
Freezing the geometry says it was always there or never; re-solving per frame
throws away the history that makes a thermal answer worth having.  Neither
produces the thing an infrared scene is usually about: the ground it left
warming back up while the ground it parked on cools, with the transient in
between.

An epoch schedule is the third answer.  The trajectory is cut into spans that
stand still, each with its own `ExchangeGeometry` and `SunVisibilityTable`
(`ThermalGeometrySchedule` in `thermal/ThermalTypes.hpp`), and **one**
`ThermalState` walks through all of them; `ThermalTimeline::StepRange` ends a
batch at the next checkpoint or the next epoch boundary, whichever is first, so
a step is never integrated against two worlds.  The test says what this costs
and buys: a 0.2 m adiabatic-backed concrete slab is still 4 K above its new
equilibrium a full day after the truck parked, and takes ten to get within a
fifth of a kelvin.  That curve is the feature.

- **Planning** (`thermal/ThermalEpochs.cpp`, `PlanEpochTimes`) needs no GPU.
  Candidates are the start, every keyframe and segment boundary, and a
  `thermal_epoch_stride_s` grid while something is moving; a candidate at
  which no node's `PoseDisplacement` since the last kept boundary exceeds
  `thermal_epoch_min_move_m` is dropped, so a convoy that parks for six hours
  costs one epoch for those six hours.  The stride defaults to one thermal
  timestep in clock seconds, because a finer division is one the stepper cannot
  tell from no division at all; with a large `thermal_time_scale` it is the
  minimum move that decides how many epochs survive.  The demo plans five.
- **Building** (`renderer/ThermalEpochBuilder.cpp`) is shared between the
  interactive context and the offline renderer, because a second implementation
  would be a second set of view factors.  The host walks the scene to each
  epoch (a TLAS refit), meshes it, runs the exchange precompute and the sun
  columns for that span, then restores the clock.  Contacts, shell pairing and
  instance bases come from epoch zero, because rigid motion changes none of
  them.  Each epoch keeps only the forcing rows inside its span plus one either
  side for interpolation—the demo subsets 169 columns to the 93 its epochs
  need, and a month-long timeline is four thousand—while `sun_memory_lags > 0`
  keeps the whole table with a warning, since the lag indexes columns.
- **Telling someone about it.**  The build is K refits, K exchange precomputes
  and K sets of sun columns, synchronous on the calling thread, and it happens
  on the first hour asked for after the plan changes—a config applied, a gizmo
  drag finished—and never on a scrub.  `SetThermalEpochProgressCallback` fires
  once per epoch just before it is measured, with the index and the count, so a
  host can say "building epoch 3 of 24" instead of freezing.  It reports; it
  dirties nothing.
- **The cache** (§8.10) hashes every epoch's start hour and elements, so a
  static scene's entries survive.

Offline, `RunThermalSolve` is build-then-ask: a sequence builds the mesh,
chooses a stepper, relaxes to steady state and measures the world once, then
steps forward per frame instead of replaying from zero.

### 10.5 Rendering a Sequence

```
Quantiloom.exe sequence <config.toml> --from-tick A --to-tick B --every N \
    --output "out/frame_{tick:05}.exr"
```

`batch` gives every configuration its own renderer, which is right for a list
of different scenes and wrong for a list of instants of the same one: a
hundred ticks of a truck driving past would build the same acceleration
structure a hundred times and restart the same trajectory a hundred times.
`sequence` uses one renderer for the run.  The scene is loaded once, the
geometry schedule measured once, and the thermal trajectory **stepped forward**
between frames rather than replayed—so rendering a day in order costs about
what rendering its last hour costs.  Measured on the demo: the frame at t = 3 s
costs 2.6 s of stepping, the one at t = 6 s costs 0.2 s.  `--dry-run` lists the
frames without building a device.  Each frame logs its tick, second, simulated
hour and epoch, and the sampling seed is `renderer.seed` mixed with the tick, so
two frames of one sequence get different patterns while a re-run of either gets
the same one it got before.  `RenderJob`'s output stage is `WriteFrameOutputs`,
shared with the sequence, so a frame from either path gets the same
thermography map, the same sensor chain and the same PNG stretch.

---

## 11. Quantiloom-Qt: The SDK's Production Consumer

Separate repository.  Qt 6 Widgets desktop application, **35,035 lines of
C++ across 94 source files**, versioned alongside the core SDK—both at
**0.3.2**.  **The application contains no physics**—it drives the SDK
exclusively through `ExternalRenderContext`.

### 11.1 SDK Integration

- Links `Quantiloom::libQuantiloom` and `Quantiloom::libSpectraForge`;
  zero references to `quantiloom_core` anywhere in the repository—
  validating the two-target rule (§3.1) in a real system integration.
- **`SdkGuard`:** At configure time, CMake stamps the SDK DLL's SHA-256;
  at run time, the loaded DLL is hashed and compared.  Mismatch → startup
  refusal.  Version skew otherwise manifests as difficult-to-diagnose
  run-time anomalies.
- When a required header or symbol is missing, the correct action is to
  return to the core repository and deliberately widen the SDK API (public
  header + `QL_API` + golden update), not to work around it in the front
  end.

### 11.2 Interactive Progressive Rendering

Not "configure offline, render, view result."  Transform/camera drags
trigger incremental TLAS refit (full rebuild on mouse release) with
temporary render-resolution reduction to maintain interactive frame rate.
GPU ray-query point selection.  Parameter changes funnel through a single
dispatch function that calls `resetAccumulation()` to restart progressive
accumulation.

**`ReprocessAccumulated`:** Sensor, display-enhancement, and other pure-
display parameters reuse already-accumulated samples without re-tracing.
Changing a palette should not discard minutes of accumulation.

### 11.3 Interactive Thermal

`ThermalPanel` exposes a time slider backed by `ThermalPreview` +
`GpuThermalStepper` + checkpoint replay.  The lazy-invalidation rules
from §8.6 apply: geometry edits during a drag set O(1) flags; the cost is
deferred to the first time-slider scrub after the drag finalises.

**A quantitative document survives a save.**  `ConfigManager` writes its TOML
by enumerating keys, so every `[thermal]` key the SDK reads is one Studio has
to read and write too, or a config opened and saved here comes back describing
a different experiment, silently, with no diagnostic anywhere.  That happened
twice: first `sun_correction` and `dump_elements`, then twelve more—the
convection law and its five constants, `lateral_conduction`, `sun_memory_lags`,
`parameter_sensitivities`, and on a material `internal_heat_w_m2`,
`interior_convection_h_w_m2k`, `shell` and the five fluorescence keys.  The
convection constants are written only when the law is not the constant one and
the free-convection three only under `stability`, because five numbers in
every document would suggest they had been chosen; `shell = false` on every
material would read as a claim that had been considered and rejected; the
fluorescence pair is written all or none, since the core refuses the halves
rather than defaulting them.  `[timeline]` and `[[models]]` round-trip as
*text*: Studio cannot author a trajectory and has no business having a second
grammar for one, so the one key it owns inside `[timeline]` is `time_s`.  Reader
and writer are checked symmetric on every `[thermal]` key and all 27 material
keys, and the four converters between those words and the SDK's enums live in
one header (`ThermalNames.hpp`) because a TOML key and an MCP tool argument are
the same string.

**Three things the viewport does with derivatives the solve already carries.**
The click that selects a node also aims a **probe**: `ThermalElementAt` turns
the pick into an element and `GetElementTrajectory` replays that element's
whole day at quarter-hour steps, which `TrajectoryPlotWidget` draws as
temperatures above and the six surface fluxes below—two charts, because kelvin
near 300 and watts per square metre either side of zero on one axis leaves the
temperatures a flat line and the fluxes unreadable, and what makes the pair
worth reading together is that a rise has a term below it that went positive.
Every path leaves the panel saying something—a ray that reached the sky, a
material naming no conductivity, a solve that is off—because a chart that
silently keeps the previous element is worse than an empty one.  A **what-if
slider** renders `T + dT/dp · step` so the viewport follows a drag a re-solve
could not; the first ask about a parameter adds it to
`parameterSensitivities` and costs one re-solve, every move after it a
re-render.  The step is a fraction of a representative magnitude rather than of
a particular material's number, since one slider covers a whole scene.  And
**two debug views**, "Response to Shade" and "Response to a Parameter", with a
spin box that points the first at one remembered hour when the solve carries a
shadow memory—shown only for that view, since a control that does nothing
beside every other mode is worse than no control.  Choosing which of a view's
numbers to look at is a way of looking, not a change to the document, so it
goes through neither a dispatcher nor the undo stack.

**A render can be held against a measurement without leaving.**
`ComparisonPanel` loads a reference EXR and reports bias, RMSE, the 95th
percentile and the worst pixel against the current frame, per channel, by
whatever name the file gave that channel—a comparison with a measurement is
usually about one band rather than about a colour—with a difference image
beside them, red where the render is brighter and blue where darker, scaled by
the 99th percentile so one firefly cannot flatten the frame to grey.  Two
things it refuses: it does not resample a mismatched resolution, because what a
resampling does to a radiance is a decision about the measurement; and it reads
the frame when asked rather than continuously, because statistics against a
still-accumulating image are statistics about the noise.

**The element dump is a File action**, beside the cube and the sequence,
because what it writes is a file the viewport cannot show.  It writes the hour
on screen through `DumpThermalElements`, refuses when the solve is off and says
why, and proposes the document's own `dump_elements` path as the filename.

### 11.4 The Transport

A document with a `[timeline]` gets one, along the bottom under the viewport:
go to start, step, play/pause, step, go to end, a slider over the tick grid, and
a readout saying which tick, which second, which thermal hour and which of the
geometry epochs that hour falls in.  Every entry has a menu item with a
shortcut—Ctrl+Space for play, since bare Space is the gizmo's.  Two playback
modes, and the second is the one that matters: real time steps on a timer;
step-when-converged waits for each frame to reach its sample target first,
because a path-traced sequence watched in real time is a sequence of noise.
The speed multiplier goes to ×100 because a timeline can span a month.

Moving the transport is deliberately not an edit: no undo command, no modified
flag.  Playback would push twenty commands a second, and where you are looking
is not something you changed.  A save still records the tick.  The panel reads
nothing for itself—everything it shows comes from `TimelineInfo`, which the SDK
resolved, since `end_s = "36h"` and a fractional tick rate are the core's to
interpret.  With a clock, `ThermalPanel` disables the hour slider the transport
now owns and says what hour it is showing, and the window's status bar writes
"Building thermal epoch 3 of 24…" from the SDK's progress callback (§10.4) and
repaints itself by hand, the event loop being busy with the build.  The scene
tree shows a node's own name when it carries a `<model>/` prefix, because the
ground and the block in the demo are both a mesh called "Cube".

### 11.5 Hyperspectral Export and Sequence Rendering

Both instantiate `OfflineRenderer` directly in a `QThread`—the same class the
CLI uses, linked as a library rather than invoked as a subprocess.  The cube
dialog offers all three formats of §9.4; the format travels as a protocol
string and the core appends the extension.  `SequenceRenderDialog` has a
Timeline mode: one `OfflineRenderer` for the whole run with the clock moved
between frames, so the scene, the acceleration structure and the thermal
schedule are paid for once, and its manifest export writes the `sequence`
command that does the same thing on the command line.  Its older per-frame mode
varies only `ir_temperature_k`, which the solver overwrites, so every frame
used to re-run an identical solve—up to 165 s each on a large scene; from SDK
0.2.7 they collapse to one through the solve cache (§8.10), and an hour-long
sequence finishing in minutes is the cache working rather than frames being
skipped.

### 11.6 MCP Server

Embeds the SDK's `mcp::Server` module, listening on `127.0.0.1:8767`.
Exposes **30 GUI-specific tools** (e.g. `ql_set_thermal`,
`ql_get_thermal_status`, `ql_get_timeline`, `ql_set_timeline_time`,
`ql_set_display_enhancement`, `ql_capture_composited`, `ql_read_pixel`,
`ql_undo`, `ql_redo`).  `ql_set_thermal` accepts all twenty-three fields of a
solve, not the eleven it once did—the twelve it lacked were the ones that
decide what the solve *carries*—and `ql_get_thermal_status` reports them, since
a caller that set one had no other way to confirm it took.  `ql_set_spectral`
accepts and describes both visible estimators.  Agent-driven edits enter the
undo stack.  This is the same SDK MCP module hosted differently from the CLI's
`serve`; the two never call each other.

### 11.7 UI Surface

- **16 dockable panels:** Atmospheric, Camera, Comparison, DebugVisualization,
  DisplayEnhancement, Lighting, MaterialEditor, Properties, RenderSettings,
  SceneTree, Sensor, SpectralConfig, SpectralLibrary, SpectralMaterialGen,
  Thermal, Timeline.
- **4 workspace presets:** Layout, Environment & Spectral, Material Prep,
  Debug.
- **Undo/redo stack.**
- **Both visible estimators in the mode list:** VIS renders by drawing four
  wavelengths a path, VIS 32 sweeps a fixed grid and has no variance in
  wavelength, so its frame is the one to compare against; switching between
  them answers a question the frame alone cannot—whether what looks wrong is
  the scene or the noise.  The configuration writer knows the identifier, so
  saving a scene that asks for the sampled mode no longer writes back the
  deterministic one.
- **Emission spectrum picker:** the Material Editor's Emissive group carries a
  Spectrum dropdown populated from the core's own `BuiltinEmissionSpectra()`,
  so a build of Studio can neither offer a token the renderer would reject nor
  withhold one it would take (§5.4).  It is a dropdown rather than another
  spinbox because an RGB triple is not a lamp, and the note beneath it says
  which bands the chosen lamp actually covers—emission being zero outside a
  curve's measured span rather than held flat.  The three emissive spin boxes
  are capped at 1×10⁶ rather than 100, because `emissiveFactor` is an HDR
  radiance scale and the core now folds `KHR_materials_emissive_strength` into
  it (§5.4); a box that showed the clamp while the material held the truth
  would have written the clamped channels back over the real ones the moment
  any one of the three was nudged.
- **The spectral library browses five bands.**  The mapping from viewport mode
  to reconstruction band used to be written out by hand and got as far as
  SWIR; MWIR and LWIR fell through to the visible branch, so a material
  assigned while the viewport rendered MWIR was reconstructed over 380–780 nm
  and bound anyway—survivable while a bound curve only decided what a surface
  reflects, and not once §6.5 made it decide how the object radiates.  It is a
  dispatcher now, for the reason the `apply*` functions are.
- **10 themes:** Blender Dark, Classic, Windows 11, Windows XP, Windows 7,
  Neutral Grey, High Contrast, Solarized Light, Green Phosphor, Print
  Friendly.
- **Chinese/English bilingual i18n** with run-time switching; the Chinese
  catalogue carries no backlog.
- **4 dialogs:** Preferences, Help, HyperspectralExport, SequenceRender.

**Four shell rules:**

1. The menu bar is the complete directory—panels are shortcuts, never the
   only entry point.
2. One setting, one dispatch function—menu item, toolbar control, and panel
   control all call it.  "Two paths doing slightly different things" is the
   bug class this rule eliminates.
3. No user-visible string is set only once (run-time language switching).
4. No colour is set only once (run-time theme switching).

---

## 12. Verification and Validation

### 12.1 Formula-Level Verification

`scripts/physics-audit/harness.py` is a stdlib-only Python reference
implementation that independently verifies:

| Formula | Harness function |
|---|---|
| Planck blackbody spectral radiance | `planck_blackbody` |
| Wien displacement law | `wien_peak_wavelength` |
| Stefan–Boltzmann total radiance | `stefan_boltzmann_radiance` |
| Fresnel (conductor, exact) | `fresnel_exact_conductor` |
| Fresnel (dielectric, exact) | `fresnel_exact_dielectric` |
| Schlick Fresnel approximation | `schlick_fresnel` |
| Normal-incidence F₀ | `fresnel_f0` |
| GGX distribution and Smith G₁ | `ggx_distribution`, `ggx_smith_g1` |
| GGX energy integral | `ggx_energy_integral` |
| Rayleigh phase function | `rayleigh_phase` |
| Henyey–Greenstein phase function | `henyey_greenstein` |
| Koschmieder visibility | `koschmieder_visibility` |
| Optical depth (exponential) | `optical_depth_exponential` |
| Band-average radiance and inversion | `band_average_radiance`, `invert_band_average_radiance` |
| Dew point, clear-sky emissivity, effective sky temperature | `dew_point_c`, `clear_sky_emissivity`, `effective_sky_temperature_k` |
| Clear-sky downwelling radiance | `clear_sky_radiance` |
| FLIR surface-temperature inversion | `flir_surface_temperature` |

### 12.2 Render Self-Consistency Checks

`scripts/render-tests/` contains:

| Script | Checks |
|---|---|
| `check_furnace.py` | Isothermal cavity vs Planck integral (8 cases) |
| `check_hero_wavelength.py` | The sampled visible estimator against the deterministic one: switching convergence, the dispersive collapse with a constant n, and dispersion as signal against a measured floor (§4.3) |
| `check_nee_mis.py` | Light sampling on vs off, once per emission representation (RGB triple and bound curve) |
| `check_fluorescence.py` | A transfer between two wavelength bands, which no diagonal renderer can pass; linearity, estimator agreement, energy (§5.5) |
| `check_endmember_mix.py` | Endmember mixing linearity |
| `check_dispersion.py` | The RGB path and the two n(λ) sources |
| `check_sky_equiv.py` | Open sky vs closed-form radiance in SWIR; in the visible, the deterministic estimator with zero spread and the sampled one on its mean |
| `check_color_bleed.py` | Cornell box indirect illumination / colour bleed |
| `check_shadow.py` | Shadow / occlusion per band |
| `check_view_independence.py` | The same surface at 0° and 60° off its normal against one closed form—catches a view-dependent albedo (§6.6) |
| `measure_convergence.py` | Convergence-rate measurement |

Orchestrators: `run_furnace_suite.{sh,ps1}` (furnace gate),
`run_illumination_suite.{sh,ps1}` (illumination gate, nine arms).  The checkers
themselves are shared by both build paths; only the orchestration is written
twice (§3.2), and the PowerShell twin's output is line for line the bash
twin's—illumination 0.0000% on every exact arm, `open:vis` 0.0984%, `mis`
2.13%, `hero` 5.83×, `fluor` 1.9007.

### 12.3 Cross-Renderer Validation

`scripts/validation/` runs four experiments against Mitsuba 3.  Three of the
four measure each renderer against a closed form **independently** rather than
against the other: with no third number a disagreement is uninterpretable, since
there is no way to say which of the two moved.  The pixel-by-pixel comparison is
reserved for the one scene that has no closed form.  Figures below are relative
deviations from the reference, except E3, which reports the ratio between the
two renderers.

| # | Scene | Reference | Result |
|---|---|---|---|
| E1 | Lambertian plane under a uniform sky | `L = ρE/π` | Quantiloom 2.7×10⁻⁸ (band) and 4.0×10⁻⁸ (single λ); Mitsuba 1.2×10⁻⁵–2.6×10⁻⁵ over a 256–4096 spp ladder |
| E2 | The same plane with a collimated sun | `L = ρ(E_sun cos θ + E_sky)/π` | Quantiloom 1.8×10⁻⁷ and 2.4×10⁻⁸; Mitsuba 1.5×10⁻⁵–2.8×10⁻⁵ |
| E3 | Cornell box at 450 / 550 / 650 nm | Each other | Mean radiance ratio 1.0009 / 1.0014 / 1.0017; median pixel 0.43–0.48% |
| E4 | Isothermal 300 K LWIR cavity | Planck band average | Mitsuba 7.7×10⁻⁷; Quantiloom 1.8×10⁻⁸ against its own 16-point quadrature and 2.83×10⁻⁴ against the exact integral |

**Why Quantiloom is exact rather than converged in E1 and E2.**  Every band
keeps its analytic ambient terms and traces one ray carrying only the residual.
With nothing above the plane every such ray escapes and the residual is zero to
the bit, so those frames have no variance at any spp—the spread column reads
0.00×10⁰.  A nonzero spread would mean the bounce had been added on *top* of the
analytic sky instead of as a correction to it.

**Why E3 compares single wavelengths, not a fused band.**  VIS_FUSED integrates
32 wavelengths against the CIE observer and returns linear sRGB; matching that
in Mitsuba would mean matching two colour pipelines as well as two transports,
and a disagreement would have three places to hide.  At one wavelength both
renderers point-evaluate the same reflectance file, and the comparison is about
transport alone.  The three wavelengths are chosen where the five wall curves
are well separated, so colour bleeding is measured rather than assumed.

**What E4 separates that the furnace gate cannot.**  The renderer's 16-point
rule differs from the exact Planck integral by 2.831×10⁻⁴, so the entire
Quantiloom-vs-exact gap is quadrature rather than transport.  The gate has
always certified self-consistency rather than agreement with nature; this puts a
number on the difference, which is what an external anchor is for.  The
experiment had been expected to be impossible—Mitsuba's spectral variant was
believed bounded to the visible—but that bound applies to the CIE tables, the
RGB upsampling and the *default* wavelength sampling; a film with an explicit
spectral response makes the sensor importance-sample that response's own
support, and the stock wheel reaches 8–12 µm.

`mitsuba_common.py` writes down once, with the reason attached, every convention
the two could disagree about: `fov_axis` defaults to `x` in Mitsuba and `y` in
Quantiloom's configs; Mitsuba's default reconstruction filter is a Gaussian,
which is a different image from a box even at convergence; `specfilm` defaults
to float16 storage, which is three decimal digits; a float or RGB emitter value
in a spectral variant is silently wrapped in D65, so an "equal-energy" emitter
is nothing of the sort; and `directional`'s `direction` is where the light
*goes*, the opposite of the surface-to-sun vector Quantiloom's configs carry.

Three defects reached the repository through this path, and no internal gate
could have found any of them: the reflectance curves that never reached the
shader, the 64-sample grid spread over the wrong wavelength range (both §6.5),
and the missing MIS weight on the bound-emission path (§5.4).

### 12.4 What Is Not Verified

- **No CI.**  Tests are a manual gate inside the build scripts.
- **The cross-renderer set covers transport, not the whole system.**  The sensor
  chain, the thermography inversion and the surface energy balance have no
  second implementation; they are checked against closed forms, against
  `harness.py`, and—for the energy balance—against a SURFRAD station.
- **Three measurements have their instrument and not their number.**  The
  stability convection law has its fourth SURFRAD arm (§8.1), the shadow
  memory has its third `e4` arm (§8.2), and the offline GPU stepper has a
  per-scene DN distribution but no per-band tolerance (§8.10).  Each is
  reported here as built, not as measured.

---

## 13. Known Limitations

### Validation

- No continuous integration; testing is a build-script gate.
- Cross-renderer comparison covers transport only (§12.4); the sensor chain and
  the thermography inversion have no second implementation.
- The stratification figure (§4.5) has been re-measured three times, each time
  because a renderer defect invalidated the previous one; the current 7.2×
  describes the current build and the two earlier numbers described builds that
  no longer exist.

### Data

- **MWIR/LWIR measured spectral coverage is ~55%** (ECOSTRESS is the best
  source; the remainder is edge-clamped extrapolation).  Counter-
  intuitively, extrapolated bands show *lower* reconstruction error because
  a straight line is trivially fitted—every basis records per-band coverage,
  and any quality figure must be cited alongside its coverage.
- **Participating media lack spectral σ.**  `Material` carries only RGB
  extinction coefficients; non-RGB modes average them.  A volume *attenuation*
  colour is now read as a spectrum (§6.4), because it is bounded and the fit
  exists for it; an extinction coefficient is not and has no fit.
- **The solar LUT stops at 4 µm.**  ASTM G-173 ends there while MWIR runs to
  5 µm, and the held-flat tail overstates that band's solar irradiance by
  roughly a quarter.  The resolver warns.

### Models

- **Multi-scattering is unavailable.**  `DeltaTrackingHomogeneous` and
  `SampleHenyeyGreenstein` exist in `volumetric.hlsli` but the delta-
  tracking loop has no call site.  Single scattering *is* active.
- **Lateral conduction is optional, explicit and CPU-only** (§8.1).  Off, each
  element is an independent 1-D column—acceptable for sand (~3 cm diffusion per
  hour) on a 0.6 m mesh, not for a metal sheet.  On, the term is explicit with a
  stability limit the solve reports, and it crosses no contact between objects,
  because the conductance there is one nobody supplied.
- **dT/dv is a local tangent** (neighbour Jacobian off-diagonals omitted);
  full-amplitude error is 1.0 K / 28.9 K (3.5%).
- **A single dT/dv is worth an order of magnitude on a stationary shadow and
  nothing on a swept one** (§8.2): under a moving sun the edge-band RMSE is
  3.43 K corrected against 3.50 K uncorrected.  `sun_memory_lags` carries the
  per-hour tangents that address this, defaults to zero, and has not been
  measured against the pointwise reference.
- **Inter-element radiative coupling is explicit;** the time step is subject
  to a stability constraint.
- **The GPU stepper carries less than the CPU one**: 32 nodes, the constant
  convection law only, no lateral conduction, no lag or parameter tangents.
  The host falls back to CPU for any of these (§8.6), so a scene that asks for
  them is slower rather than wrong.  Offline, the GPU stepper is off by default
  because no per-band tolerance has been stated for the f32 difference (§8.10).
- **View factors are top-K truncated** (CSR, each row retains only the
  largest K entries).
- **The stability convection law is implemented and not yet validated
  against the station** (§8.1).  The Louis damping and the free-convection
  floor are standard forms with literature constants; whether they remove the
  1.8 K night bias is what the fourth `e2` arm exists to measure.
- **Shell pairing is a heuristic over geometry nobody authored for it**
  (§8.9).  It refuses to cross a primitive, to pair co-facing triangles, or to
  reach beyond a few thicknesses, and counts what it could not pair; a material
  that declared itself a shell and paired a tenth of its triangles is a
  modelling problem the log reports rather than a solver one.
- **Epoch boundaries are planned from pose displacement, not from thermal
  effect** (§10.4).  A node that moves less than `thermal_epoch_min_move_m`
  between candidates keeps its epoch, whatever it shades.
- **Temperature-field spatial resolution is bounded by the thermal mesh.**
  dT/dv corrects only the sun-visibility degree of freedom; other sources
  of spatial variation (e.g. neighbour temperature gradients) remain at
  mesh resolution.
- **A spectral cube TIFF is not georeferenced** (§9.4), because a rendered
  scene has no coordinate reference system; and a cube past 4 GB is refused,
  since classic TIFF addresses with 32 bits.

---

## 14. Key Design Decisions

| Decision | Adopted | Rejected | Primary Rationale |
|---|---|---|---|
| Reflectance representation | NMF basis vectors | Per-wavelength storage | GPU memory/bandwidth; non-negative basis also enables unmixing |
| RGB → reflectance | Jakob–Hanika sigmoid | Sum of three Gaussians / free-form fit | Accuracy (21 → 0.025 ΔE), bounded, no metamer spikes |
| Upsampling table resolution | 64³ (9 MB) | 32³ / 48³ | Accuracy-first; GPU memory is not a constraint |
| Out-of-band upsampling | Hard prohibition (configurable failure) | Extrapolation / fade-to-zero | Extrapolation drives thermal-IR emissivity to zero—catastrophic |
| RGB light-source convention | Sigmoid × D65 | Flat spectrum / downstream white balance | Fix the convention at the source; 3.57% → 0.22% |
| Lamp spectrum | Bind a measured or standard curve (`emissive_curve`) | RGB `emissiveFactor` alone | An RGB triple is not a lamp, and outside the visible the fit has no domain |
| Emission out of span | Zero | Held flat at the endpoint, as reflectance is | Flat extrapolation gives a triphosphor lamp SWIR output it does not have |
| Emission discretisation | Band-averaged onto the 64-point grid | Point-sampled, as reflectance is | A line spectrum is either hit or missed: FL11 4.9% → 0.2% |
| Emission level | `match_luminance` default; `absolute` required outside the visible | One convention everywhere | The built-ins are relative distributions; luminance is undefined at 10 µm |
| Measured-curve grid | Resample over the render band | Uniform over the curve's own span / uniform in wavenumber | A 194 nm step left the visible two interior points; 10–120% error in every band |
| IR emissivity and reflectance | Hemispherical | Tuned directional law `ε₀ cos^0.7 θ` | A view-dependent Lambertian albedo is not reciprocal and inverted a scene's thermal ordering |
| Visible estimator | A sampled quartet (`vis_hero`) beside the deterministic 32-point sweep (`vis_fused`), both in one binary | One or the other | The sweep has no variance in wavelength and is the reference; the quartet follows n(λ) and spends the budget on paths; the checks compare them inside one render session |
| Band alias | `VIS` resolves to `vis_hero` | To the deterministic mode | Asking for a band rather than an estimator is asking to render it; every other alias has one estimator to go to |
| Quartet construction | Rotate one drawn wavelength through the band | Four independent draws / 8 groups with a ray each | A measure-preserving rotation collapses the balance heuristic to one scalar; the 8-group design failed the convergence check |
| Dispersion in the quartet | Collapse onto the hero wavelength, scaled so the root's division leaves the single-wavelength estimator | Redraw / trace four rays | Only the wavelength bent to that direction could have generated it, so its technique share is one |
| Fluorescence | Rank-one (excitation, emission, yield), one excitation sample per vertex on its own slot, no `emissiveFactor` | More wavelengths in the payload / an emitter entry | Published data is rank-one; the term is the only one off the λ diagonal; a triple in the NEE table would aim light sampling at a surface dark on its own |
| Emissive strength | Fold `KHR_materials_emissive_strength` into `emissiveFactor` at load | A field of its own | Five consumers must agree exactly or MIS is biased as a function of light size; the material struct has no padding left |
| Volume attenuation colour | Upsampled as a spectrum | Averaged like σ | It is bounded in [0, 1], which is the quantity the fit is defined for; σ is unbounded and has no fit |
| Light-source estimation | NEE + BSDF, MIS combined | Either alone | Lower envelope of both strategies' variances |
| First-bounce sampling | Owen-scrambled Sobol | PCG white noise / plain Sobol | 7.2× efficiency; scrambling removes visible structure |
| Adaptive sampling | Not adopted (built, measured, removed) | Per-pixel variance stopping | Measured slower and no more accurate at every time budget; the earlier "ceiling" claim is withdrawn (§4.6) |
| Time discretisation | Crank–Nicolson | Explicit Euler / fully implicit | Unconditionally stable + second-order (preserves diurnal peak) |
| Tridiagonal solver | Per-element thread-local Thomas | Parallel cyclic reduction | Parallel dimension is elements, not nodes; 32-node cap |
| Inter-element radiation | Explicit (ping-pong) | Implicit global matrix | Implicit destroys tridiagonal structure and per-element threading |
| Evaporation | Implicit (Newton-linearised) | Explicit | Slope too steep; explicit oscillates at minute-scale steps |
| Shadow resolution | dT/dv tangent, per-pixel shader correction | Mesh refinement | Refinement costs 3.6 orders of magnitude in element count for redundant DOF (measured) |
| dT/dv computation | Trajectory tangent (same matrix, second RHS) | Steady-state / single-step closed form | Closed forms yield 31 K / 5 K; truth 27.9–30.3 K |
| dT/dv scope | Local (no neighbour Jacobian) | Full Jacobian | Keeps dT/dv a per-element scalar the shader can apply per pixel |
| Short-wave gains | Reuse long-wave CSR view factors | Second geometric precompute | Same hemisphere integral, two bands |
| Trajectory storage | Fixed-step grid + checkpoints | Full history / pure replay | Memory vs backward-scrub latency trade-off |
| Thermal invalidation | Lazy dirty flags | Immediate rebuild | Gizmo drag would drop to single-digit FPS |
| Temperature inversion | Per-band | Total-flux σT⁴ | σT⁴ folds out-of-band tail errors into every temperature |
| NETD noise terms | Exclude FPN, exclude well saturation | Include all | Matches NETD's definition; keeps well-saturation mismatch visible |
| Display enhancement | Tone × palette (two stages) | Single enumeration | Two orthogonal decisions; product enumeration is unmaintainable |
| Default tone operator | `Linear` | `Clahe` | Globally monotonic—the only operator from which temperature can be read (Clahe: 1.902% inverted pairs) |
| Sensor chain | Offline CPU + interactive GPU (6 passes) | Unified single chain | Offline needs determinism; interactive needs frame-rate; round-trip exceeds budget |
| FPN modelling | PRNU / DSNU separate | Single intensity parameter | Multiplicative and additive cannot be calibrated with one parameter |
| NUC | Retain 95–99% residual | Perfect correction | Perfect correction ≡ not modelling FPN |
| Random seed | Default fixed | Default random | Primary use is quantitative validation |
| Endmember weights | Global scaling | Per-texel normalisation | Normalisation erases the brightness variation this mechanism recovers |
| Weight ceiling | 6, anchored against the clamp | 2, anchored against the solve | 18–31% of texels truncated; surfaces rendered at 0.68–0.80 of measurement |
| Convective coefficient | A law in the solver (`constant` / `wind` / `stability`), overridden by a measured column | One constant per material / a correlation in a Python script | One constant over-warms the night by up to 1.8 K while fitting the day; a stable nocturnal layer needs `h` smaller, which a `max()` of two laws cannot give |
| Lateral conduction | Optional, explicit, harmonic-mean conductance per metre of depth, welded within one object | Implicit global matrix / conductance across contacts | Explicit keeps one thread per element and the tangents ride through it; a contact conductance is one nobody supplied |
| Back face | `adiabatic` / `fixed` / `ambient`, plus a source flux at the back node | Adiabatic only | An engine behind a panel is most of what an infrared signature is |
| Thin panel | `shell = true` pairs opposed faces of one primitive into one column | Two independent slabs | A panel in the sun is otherwise as hot as if bolted to masonry; a symmetric shell must come out isothermal |
| Shadow history | One tangent per remembered sun column, shipped as a decomposition with the remainder in the present record | A per-element scalar only | Zero slots is byte-identical; a window too small loses no energy |
| Material derivatives | Trajectory tangents `dT/dp` through the same elimination, matrix-side rows built after back-substitution | Sweeps | One trajectory per Gauss–Newton step instead of one per candidate; the temperature stays bit-identical |
| Stepper choice | The host asks what a stepper carries and falls back to CPU | Run whatever was handed over | A tangent nobody integrated comes back as zero, which is a derivative rather than an absence |
| Probe fluxes | `EvaluateSurfaceBalance` as a read-only query by the CPU reference balance | An out-parameter on `Step` / GPU decomposition | The balance is a pure function of the state, so the six numbers do not depend on which machine solved it; a stepper that cannot decompose declines rather than returning zeros |
| What-if step | In binding 27's buffer | In `LightingParams` | The struct has no padding left, and growing it changes the SDK/Studio pairing hash |
| Solve cache key | SHA-256 of resolved inputs, GPU and library version included | A hash of the config text / 64 bits | Two configs differing in `sensor.*` are one entry; a collision serves another scene's field with exit code 0; a driver update must miss rather than mislead |
| Cache boundary | The whole solve, view factors included | The stepping only | The precompute is most of the wall clock and the part that needs a TLAS |
| Moving geometry | One `ThermalState` across piecewise-static epochs, each with its own exchange | Freeze at a reference / re-solve per frame | The ground a truck left keeps its history; a frozen scene says it was always there or never |
| Epoch planning | Trajectory discontinuities plus a stride while moving, filtered by minimum displacement | Every tick | A convoy parked six hours costs one epoch; a division finer than a timestep is one the stepper cannot see |
| Time base | Seconds canonical, `ticks_per_second` a double, every time key takes a unit | Ticks canonical / integer rate | A month wants a tick every 150 s; `end_s = 2592000` is a number nobody checks |
| Node pose under a clock | `[[nodes]]` records the rest pose; a gizmo drag solves the composition for it | Record the pose at `time_s` | A pose at one instant is wrong at every other |
| Motion grammars | Keyframes, piecewise expressions, parametric engines | One grammar | Three kinds of author write them; ExprTk behind one translation unit because its header costs a minute and `/bigobj` |
| Scrubbing | A private refit path that dirties no thermal geometry | `RefitAccelerationStructure` | The trajectory the exchange was computed from has not changed; a scrub must cost no precompute |
| Sequence rendering | One renderer, trajectory stepped forward between frames | `batch` with a renderer per frame | A day in order costs about what its last hour costs |
| Spectral cube EXR | Single-part, Fichet et al. 2021 layout, through `ImageIO` | Multipart OpenEXR | The interchange layout wants one part, and the multipart API brought a shutdown crash |
| Spectral cube TIFF | Hand-written baseline TIFF with GDAL's metadata tag | libtiff / GDAL | Their value is georeferencing, and a rendered scene has none to claim |
| Environment map | Four-condition invariant, RGB mode only | Follow single switch | Invented sky was 46–84% of signal |
| Fallback cubemap | One black texel | Blue sky | Erroneous sampling darkens visibly instead of silently adding light |
| Cube format | ENVI, spectral EXR, baseline TIFF | HDF5 | Zero added dependency + remote-sensing downstream compatibility |
| Core linking | `quantiloom_core` vs `libQuantiloom`—one or the other | Mixed | Two copies of global state is a correctness bug |
| ABI expansion | Golden-baseline gate | Ad-hoc `QL_API` addition | Every contract expansion is reviewable |
| Windows gates | PowerShell reimplementations + a stamp file | Shell scripts through Git Bash | A build agent has MSVC and CMake, not bash; two scripts cannot enforce an order between themselves |
| Cross-renderer validation | Three closed-form anchors + one pixel comparison | Renderer against renderer throughout | With no third number, a disagreement cannot be attributed to either side |
| Per-frame override | `material_overrides` table | Inline `[[materials]]` | Array replacement deletes other materials |
| Alpha geometry | Retain any-hit | Declare opaque | Opaque turns a grille into a solid wall in both rendering and view factors |

---

## References

- Jakob, W. & Hanika, J. (2019). A Low-Dimensional Function Space for
  Efficient Spectral Upsampling. *Computer Graphics Forum*, 38(2).
- Aguerre, J. P. et al. (2020). Physically Based Simulation and Rendering
  of Urban Thermography. *Computer Graphics Forum*, 39(6).
- Berdahl, P. & Fromberg, R. (1982). The thermal radiance of clear skies.
  *Solar Energy*, 29(4).
- Veach, E. & Guibas, L. J. (1995). Optimally Combining Sampling
  Techniques for Monte Carlo Rendering. *SIGGRAPH '95*.
- Wilkie, A. et al. (2014). Hero Wavelength Spectral Sampling.
  *Eurographics Symposium on Rendering*.
- Owen, A. B. (1997). Scrambled Net Variance for Integrals of Smooth
  Functions. *Annals of Statistics*, 25(4).
- Burley, B. (2020). Practical Hash-Based Owen Scrambling.
  *Journal of Computer Graphics Techniques*, 9(4).
- Fichet, A., Pacanowski, R. & Wilkie, A. (2021). An OpenEXR Layout for
  Spectral Images. *Journal of Computer Graphics Techniques*, 10(3).
- McAdams, W. H. (1954). *Heat Transmission*, 3rd ed.  McGraw-Hill.
- Louis, J.-F. (1979). A Parametric Model of Vertical Eddy Fluxes in the
  Atmosphere. *Boundary-Layer Meteorology*, 17(2).
- Gillespie, A. et al. (1998). A Temperature and Emissivity Separation
  Algorithm for ASTER Images. *IEEE Transactions on Geoscience and Remote
  Sensing*, 36(4).
- ASTM G-173 (Standard Tables for Reference Solar Spectral Irradiances).
- ISO 20473 (Optics and Photonics—Spectral Bands).
- FIPS PUB 180-4 (Secure Hash Standard).
