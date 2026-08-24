# Quantiloom: A Spectral Path Tracer with Coupled Surface Energy Balance

> Targeting Quantiloom v0.2.3 and Quantiloom-Qt v0.2.3.
> This document describes the system as implemented, the rationale behind
> every key design choice, and the verification strategy that guards them.

---

## 1. Overview

Quantiloom is an end-to-end spectral infrared imaging simulation system.
Its pipeline runs:

> Spectral path tracing (physical Fresnel, measured spectral reflectance,
> Jakob–Hanika upsampling, Hero-Wavelength dispersion) → surface energy
> balance temperature field solver (1-D conduction, six-flux coupling,
> view-factor long-wave/short-wave exchange) → sensor effect chain
> (PSF / QE / noise / FPN / NUC / ADC) → thermography inversion (NETD) /
> multi-band fusion / hyperspectral cube output.

The core is a C++20 / HLSL shared library on Vulkan ray tracing, consumed
by three front ends: a CLI (single scene, batch, MCP server), a Qt 6
desktop application with interactive progressive rendering, and an MCP
service for agent-driven workflows.  The complete system—SDK plus GUI,
excluding tests—is approximately 100,500 lines of code.

---

## 2. Codebase Scale

### 2.1 Top-Level Metrics

| Module | Lines | Files |
|---|---:|---:|
| Core C++ source (`src/`, excluding shaders) | 48,736 | — |
| Public headers (`include/quantiloom/`) | 7,572 | 31 |
| Shaders (HLSL: `.rgen` / `.rchit` / `.rmiss` / `.rahit` / `.comp.hlsl` / `.comp` / `.rayq.hlsl` / `.hlsli`) | 12,701 | 27 |
| **Core SDK subtotal** | **~69,000** | — |
| Unit tests (GoogleTest; 1,249 cases / 109 suites, < 8 s) | 28,236 | 71 |
| Quantiloom-Qt GUI (separate repository) | 31,522 | 87 |
| **System total (core SDK + GUI, excluding tests)** | **~100,500** | — |
| **Including tests** | **~128,800** | — |

### 2.2 Core Library Modules (`src/libQuantiloom/`)

| Module | Files | Lines | Responsibility |
|---|---:|---:|---|
| `renderer/` | 47 | 21,760 | Vulkan abstraction, acceleration structures, textures, offline/interactive dual hosts, GPU thermal stepper, spectral unmixing |
| `io/` | 14 | 7,996 | glTF / USD / EXR / ENVI / spectral library I/O |
| `hs_core/` | 13 | 4,803 | Hyperspectral configuration, adaptive wavelength grid |
| `core/` | 15 | 3,383 | Fundamental types, configuration, logging, CIE / D65 / RGB-to-spectrum tables |
| `thermal/` | 14 | 2,076 | Surface energy balance solver, timeline, short-wave gains, Crank–Nicolson stepper |
| `atmos/` | 12 | 1,610 | Atmospheric LUT, ResMLP neural surrogate |
| `scene/` | 7 | 1,545 | Camera, material, mesh, scene editing |
| `postprocess/` | 3 | 1,169 | Sensor effect chain, multi-band fusion, thermography inversion |
| `mcp/` | 11 | 1,126 | MCP service module |

Other modules: `src/app/` (CLI, 1,869 lines), `src/libSpectraForge/`
(IR material generation, 820 lines), `src/tools/` (`fusion_tool` +
`QLTrans` MODTRAN wrapper, 579 lines).

### 2.3 Principal Shaders

| File | Lines |
|---|---:|
| `closesthit.rchit` | 4,707 |
| `common.hlsli` | 1,430 |
| `pbr.hlsli` | 1,131 |
| `SpectralConversion.hlsli` | 609 |
| `clahe.comp.hlsl` | 533 |
| `miss.rmiss` | 440 |
| `volumetric.hlsli` | 403 |
| `thermal_step.comp.hlsl` | 333 |
| `sampling.hlsli` | 300 |
| `blackbody.hlsli` | 270 |
| `thermal_exchange.rayq.hlsl` | 249 |

### 2.4 Test Distribution

`test_renderer/` 24 files; `test_core/` 16; `test_io/` 7; `test_hs_core/` 7;
`test_postprocess/` 6; `test_scene/` 5; `test_mcp/` 2; `test_app/` 1;
and 3 top-level test utilities.
The baseline includes 10 SKIPPED tests by design: 8 BC7 texture compression
cases (disabled after measurement showed a net quality loss) and 2 EXR
multipart cases (unimplemented).

### 2.5 DLL Exports

The shared library exports 244 symbols (`docs/abi/exports.golden`).  Every
new export must pass through a reviewed three-step process: move the header
into `include/quantiloom/`, annotate with `QL_API`, and update the golden
baseline.  A build-time ABI gate rejects any change that skips a step.

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

1. **Unit test suite** — 1,249 tests in < 8 s.
2. **ABI gate** — `scripts/check_exports.sh` diffs the current export list
   against `docs/abi/exports.golden`.
3. **Furnace rendering gate** — `run_furnace_suite.sh` renders 8 isothermal
   cavity configurations (5 LWIR, 3 MWIR) and verifies each against its
   own Planck integral.
4. **Illumination gate** — `run_illumination_suite.sh` checks occlusion,
   open-sky equivalence, and indirect lighting (Cornell box).

Only after all four pass does the script install the SDK into the sibling
`Quantiloom-SDK` tree, which the Qt front end links against.

### 3.3 CLI Modes

| Mode | Invocation | Description |
|---|---|---|
| Single scene | `Quantiloom.exe <config.toml>` | Render one TOML scene |
| Batch | `batch <list.txt>` | Render a manifest of scenes in one GPU session |
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
merge by key, which is the semantics per-frame overrides require.

### 3.4 Cross-Validation by Construction

The same `RenderCore` is consumed by CLI, GUI, and unit tests.  Running two
host paths side by side has exposed bugs including: vertically mirrored
environment maps, mip chains that were never sampled, IR emissivity that
ignored the rendering wavelength, a sensor-chain input-unit mismatch of
~7×10⁴ (LWIR rendered near-black), and a specular path where reflected
radiance exceeded incident radiance by 11,000×.

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

Eight rendering modes are available, expressed by the `SpectralMode` enum:
`RGB`, `VIS_Fused`, `NIR_Fused`, `SWIR_Fused`, `MWIR_Fused`, `LWIR_Fused`,
`Single`, and `Multispectral`.

### 4.2 Alpha Mode and the Any-Hit Stage

The `alphaMode` property (`OPAQUE` / `MASK` / `BLEND`) is threaded from
the glTF loader through acceleration-structure flags to the any-hit shader.
Declaring a geometry as opaque lets the driver skip the any-hit stage, but
for alpha-tested or blended geometry that would turn a mesh grille into a
solid wall—visually wrong in the render, and worse in the thermal solver,
where view factors would treat it as a continuous surface and skew the
entire temperature field.  Alpha geometry therefore retains the any-hit
invocation and accepts the traversal cost.

### 4.3 Hero-Wavelength Spectral Sampling

Each path carries a single sampled wavelength (`Payload::heroLambda` in
`common.hlsli`).  The VIS_FUSED mode traces one hero wavelength per path,
letting the spectral dimension and the spatial dimensions be integrated
together by Monte Carlo.

**Why not a fixed N-wavelength integration (e.g. 32)?**  Evaluating the
BSDF and querying the light source at N wavelengths per path costs roughly
N×, while on non-dispersive surfaces the N samples are highly correlated.
Hero-Wavelength spends the same time budget on an order of magnitude more
paths, trading spectral noise for spatial coverage.  The cost is chromatic
noise, controlled by `check_hero_wavelength.py`, which sweeps deterministic
wavelengths and verifies consistency against the fused result.

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
PBRT-v4's `PaddedSobolSampler`, Burley 2020) across six sample slots:

| Slot | Dimensionality | Purpose |
|---|---|---|
| `SAMPLE_SLOT_JITTER` | 2-D | Sub-pixel position |
| `SAMPLE_SLOT_LAMBDA` | 1-D | Wavelength |
| `SAMPLE_SLOT_LIGHT_PICK` | 1-D | Emitter selection |
| `SAMPLE_SLOT_LIGHT_UV` | 2-D | Point on emitter |
| `SAMPLE_SLOT_LOBE` | 1-D | Diffuse/specular lobe |
| `SAMPLE_SLOT_DIRECTION` | 2-D | Bounce direction |

The underlying Sobol sequence is 2-D (dimensions 0 and 1), reused per
slot with independent Owen-scramble seeds—hence "six slots," not six Sobol
dimensions.  Pre-generated Sobol direction numbers carry no run-time cost.

**Why Sobol rather than PCG white noise?**  The first bounce is low-dimensional
and high-impact; stratifying it directly reduces integration error.
Measured on `cornell_box_vis`, the spp required to reach 2% RMSE drops from
6,598 (white noise) to 3,773—a **1.74× sampling efficiency gain** at zero
additional cost.

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
The theoretical ceiling—L₂-optimal allocation versus uniform—is only
**1.85×**, and the Sobol stratification's 1.74× already captures nearly
all of it.

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
table interpolation.  Measured on 20,000 random colours plus gamut vertices:

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
alongside its coverage (see §12, Known Limitations).

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
`Material` carries only RGB coefficients, not spectral σ curves.  Averaging
is the only honest reduction until the data model supports spectral
extinction—the code marks the insertion point.

`volumetric.hlsli` also contains a complete **Delta-Tracking**
implementation (`DeltaTrackingHomogeneous`, `MAX_VOLUME_STEPS = 128`) and
HG phase-function sampling (`SampleHenyeyGreenstein`).  The delta-tracking
loop has no call site; multi-scattering (clouds, dense fog) is therefore
not available.  The single-scattering branch and the HG evaluation function
*are* active.  The Rayleigh, Mie, and HG formulas are independently
verified in `scripts/physics-audit/harness.py`.

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

**Forcing CSV.**  Eight columns, the last two optional with defaults:

```
time_h, air_k, dni, sun_azimuth_deg, sun_elevation_deg, sky_k, diffuse_w_m2, relative_humidity
```

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
lateral conduction**—the shadow edge it describes is as sharp as the
geometry.  Only the discretisation was coarse.

**Refining the mesh is impractical.**  Element count enters the view-factor
matrix as a square term and the per-step solve as a linear term.
Subdividing 0.6 m triangles to resolve a 0.7 m penumbra requires 2–3 orders
of magnitude more elements, all carrying redundant physics—their
temperatures are fully determined by the same 1-D column and a visibility
scalar.

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
| Extra rays | One per thermal hit where |dT/dv| ≥ 0.1 K—none at night, none indoors, none without a solver |
| Extra state | Doubles `ThermalState` and the timeline's checkpoints |
| Extra solver time | One elimination pass per element per step |
| Full-amplitude accuracy | 1.0 K on a 28.9 K contrast (3.5%), because radiative admittance changes by a third across the span |

Against a 0.6 m triangle that is fully lit or fully dark, 3.5% linear-
interpolation error is not in the same order of magnitude—this is the
entire reason the trade-off holds.

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

### 8.3 Time Integration

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

### 8.4 ThermalTimeline

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

### 8.5 Interactive Thermal Path

**GpuThermalStepper** (`renderer/GpuThermalStepper.hpp`) mirrors the CPU
Crank–Nicolson in f32 via `thermal_step.comp.hlsl`.  Each element is one
thread; the Thomas solve is thread-local (max 32 nodes); inter-element
radiative coupling reads from a ping-pong surface buffer.

**ThermalPreview** (`renderer/ThermalPreview.hpp`) is the internal subsystem
owned by `ExternalRenderContext::Impl`.  It holds the mesh, exchange
geometry, sun-visibility table, timeline, and stepper, plus dirty flags.
It never touches the pipeline or descriptors—binding 24 (per-element
temperature) is owned by the façade.  Everything the timeline is constructed
with except the descriptor is **held by reference for its lifetime**, so all
dependencies are members, not locals that go out of scope.

**Lazy invalidation.**  Changing the time (`SetThermalTime`) does *not*
invalidate anything—it steps forward or replays from a checkpoint.
Geometry changes, material IR-curve edits, `SetThermalMaterial`,
and different fields of `SetParams` each set specific dirty flags
(exchange, sun table, materials, timeline), resolved by the next
`SetThermalTime`.

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

### 8.6 Clear-Sky Model

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

### 8.7 Thermography Inversion and NETD

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

### 8.8 Single Decode Entry Point

A surface temperature has exactly one decode entry point—
`GetSurfaceTemperatureK` in `closesthit.rchit`—and all call sites go
through it: two sampling paths (SWIR and general MWIR/LWIR) and two debug
views (`DEBUG_MODE_TEMPERATURE` and `DEBUG_MODE_IR_EMISSION`), for four
total call sites.  Decode order: solver → temperature texture → material
scalar.  When the solver provides a temperature, the per-pixel sun
correction from §8.2 is applied **inside** this function—placing it outside
would let any new sampling site forget it, producing a rendering bug
(triangle-shaped shadow edge) that looks like a solver bug.

### 8.9 Verification

| Verified quantity | Method | Location |
|---|---|---|
| Planck / Wien / Stefan–Boltzmann | Closed-form + independent Python | `test_blackbody.cpp`, `test_blackbody_physics.cpp`, `harness.py` |
| 1-D conduction | Closed-form (semi-infinite body, steady gradient) | `test_thermal_conduction.cpp` |
| View factors | Analytic view factors | `test_thermal_exchange_gpu.cpp` |
| Short-wave gains | Analytic configurations | `test_shortwave_gains.cpp` |
| Clear-sky model | Original correlations | `test_sky_thermal.cpp` |
| Timeline / checkpoints | Replay consistency | `test_thermal_timeline.cpp` |
| GPU vs CPU stepper | 24 h trajectory step-by-step comparison | `test_thermal_step_gpu.cpp` |
| dT/dv | Central differences from two full runs; full-shadow trajectory | `test_thermal_sun_sensitivity.cpp` |
| Interactive invalidation | Dirty-flag state machine | `test_thermal_preview.cpp` |
| Temperature texture | Sampling and decode | `test_temperature_texture.cpp` |
| Thermography inversion + NETD | Isothermal-cavity round-trip; responsivity consistency | `test_thermography.cpp` |

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
  DSNU (dark-signal non-uniformity, additive) modelled separately.
- **NUC:** Non-uniformity correction with configurable efficiency
  (typical 95–99%).
- **ADC:** Configurable bit depth (12 / 14 / 16 bit) and gain.
- **PSF blur:** Gaussian convolution with sigma matched to the Airy disk's
  FWHM (factor 0.437, analytically derived from `[2 J₁(x)/x]²`).
- **Deterministic random seed** (`renderer.seed` / `sensor.noise_seed`;
  default fixed at `0x5EED`).

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

The renderer writes ENVI cubes in BSQ, BIL, or BIP interleave
(`io/SpectralCubeIO.cpp`), with GPU-accelerated spectral reconstruction
(`GpuSpectralReconstructor`) and optional per-band intermediate export
(`save_intermediates`).  Adaptive wavelength grids are handled by
`hs_core/AdaptiveGridGenerator`.

**Why ENVI, not HDF5.**  HDF5 is a heavy dependency (build time,
distribution size, Windows link complexity), and its hierarchical metadata
capability is over-specified for "one cube plus one header."  ENVI is the
remote-sensing de facto standard; every downstream analysis tool reads it
directly.

> *Note:* GeoTIFF and EXR output paths for spectral cubes are declared in
> the interface but are currently stubs (log a message and fall back to raw
> binary).  Only ENVI is fully functional.

---

## 10. Quantiloom-Qt: The SDK's Production Consumer

Separate repository.  Qt 6 Widgets desktop application, **31,522 lines of
C++ across 87 source files**, versioned in lockstep with the core SDK at
**0.2.3**.  **The application contains no physics**—it drives the SDK
exclusively through `ExternalRenderContext`.

### 10.1 SDK Integration

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

### 10.2 Interactive Progressive Rendering

Not "configure offline, render, view result."  Transform/camera drags
trigger incremental TLAS refit (full rebuild on mouse release) with
temporary render-resolution reduction to maintain interactive frame rate.
GPU ray-query point selection.  Parameter changes funnel through a single
dispatch function that calls `resetAccumulation()` to restart progressive
accumulation.

**`ReprocessAccumulated`:** Sensor, display-enhancement, and other pure-
display parameters reuse already-accumulated samples without re-tracing.
Changing a palette should not discard minutes of accumulation.

### 10.3 Interactive Thermal

`ThermalPanel` exposes a time slider backed by `ThermalPreview` +
`GpuThermalStepper` + checkpoint replay.  The lazy-invalidation rules
from §8.5 apply: geometry edits during a drag set O(1) flags; the cost is
deferred to the first time-slider scrub after the drag finalises.

### 10.4 Hyperspectral Export

Instantiates `OfflineRenderer` directly in a `QThread`—the same class the
CLI's `batch` command uses, linked as a library rather than invoked as a
subprocess.

### 10.5 MCP Server

Embeds the SDK's `mcp::Server` module, listening on `127.0.0.1:8767`.
Exposes **28 GUI-specific tools** (e.g. `ql_set_thermal`,
`ql_get_thermal_status`, `ql_set_display_enhancement`,
`ql_capture_composited`, `ql_read_pixel`, `ql_undo`, `ql_redo`).
Agent-driven edits enter the undo stack.  This is the same SDK MCP module
hosted differently from the CLI's `serve`; the two never call each other.

### 10.6 UI Surface

- **14 dockable panels:** Atmospheric, Camera, DebugVisualization,
  DisplayEnhancement, Lighting, MaterialEditor, Properties, RenderSettings,
  SceneTree, Sensor, SpectralConfig, SpectralLibrary, SpectralMaterialGen,
  Thermal.
- **4 workspace presets:** Layout, Environment & Spectral, Material Prep,
  Debug.
- **Undo/redo stack.**
- **10 themes:** Blender Dark, Classic, Windows 11, Windows XP, Windows 7,
  Neutral Grey, High Contrast, Solarized Light, Green Phosphor, Print
  Friendly.
- **Chinese/English bilingual i18n** with run-time switching.
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

## 11. Verification and Validation

### 11.1 Formula-Level Verification

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

### 11.2 Render Self-Consistency Checks

`scripts/render-tests/` contains:

| Script | Checks |
|---|---|
| `check_furnace.py` | Isothermal cavity vs Planck integral (8 cases) |
| `check_hero_wavelength.py` | Hero-wavelength sweep vs fused result |
| `check_nee_mis.py` | NEE-only vs BSDF-only consistency |
| `check_endmember_mix.py` | Endmember mixing linearity |
| `check_dispersion.py` | Dispersion rendering correctness |
| `check_sky_equiv.py` | Open sky vs closed-form radiance |
| `check_color_bleed.py` | Cornell box indirect illumination / colour bleed |
| `check_shadow.py` | Shadow / occlusion per band |
| `measure_convergence.py` | Convergence-rate measurement |

Orchestrators: `run_furnace_suite.sh` (furnace gate),
`run_illumination_suite.sh` (illumination gate).

### 11.3 What Is Not Verified

- **No cross-renderer ground-truth comparison.**  `scripts/validation/` is
  an empty placeholder.  Neither PBRT-v4 nor Mitsuba 3 comparisons exist.
- **No CI.**  Tests are a manual gate inside the build script.

---

## 12. Known Limitations

### Validation

- No cross-renderer ground-truth comparison (see §11.3).
- No continuous integration; testing is a build-script gate.

### Data

- **MWIR/LWIR measured spectral coverage is ~55%** (ECOSTRESS is the best
  source; the remainder is edge-clamped extrapolation).  Counter-
  intuitively, extrapolated bands show *lower* reconstruction error because
  a straight line is trivially fitted—every basis records per-band coverage,
  and any quality figure must be cited alongside its coverage.
- **Participating media lack spectral σ.**  `Material` carries only RGB
  coefficients; non-RGB modes average them.

### Models

- **Multi-scattering is unavailable.**  `DeltaTrackingHomogeneous` and
  `SampleHenyeyGreenstein` exist in `volumetric.hlsli` but the delta-
  tracking loop has no call site.  Single scattering *is* active.
- **The thermal model has no lateral conduction.**  Each element is an
  independent 1-D column.  Acceptable for sand (~3 cm diffusion per hour)
  on a 0.6 m mesh; not valid for metal sheets or small-scale structures.
- **dT/dv is a local tangent** (neighbour Jacobian off-diagonals omitted);
  full-amplitude error is 1.0 K / 28.9 K (3.5%).
- **Inter-element radiative coupling is explicit;** the time step is subject
  to a stability constraint.
- **GPU stepper limited to 32 nodes;** exceeding this falls back to CPU.
- **View factors are top-K truncated** (CSR, each row retains only the
  largest K entries).
- **Temperature-field spatial resolution is bounded by the thermal mesh.**
  dT/dv corrects only the sun-visibility degree of freedom; other sources
  of spatial variation (e.g. neighbour temperature gradients) remain at
  mesh resolution.
- **Hyperspectral cube GeoTIFF and EXR output** are declared but not
  implemented; only ENVI is functional.

---

## 13. Key Design Decisions

| Decision | Adopted | Rejected | Primary Rationale |
|---|---|---|---|
| Reflectance representation | NMF basis vectors | Per-wavelength storage | GPU memory/bandwidth; non-negative basis also enables unmixing |
| RGB → reflectance | Jakob–Hanika sigmoid | Sum of three Gaussians / free-form fit | Accuracy (21 → 0.025 ΔE), bounded, no metamer spikes |
| Upsampling table resolution | 64³ (9 MB) | 32³ / 48³ | Accuracy-first; GPU memory is not a constraint |
| Out-of-band upsampling | Hard prohibition (configurable failure) | Extrapolation / fade-to-zero | Extrapolation drives thermal-IR emissivity to zero—catastrophic |
| RGB light-source convention | Sigmoid × D65 | Flat spectrum / downstream white balance | Fix the convention at the source; 3.57% → 0.22% |
| Spectral path sampling | Hero-Wavelength | Fixed 32-wavelength integration | Order-of-magnitude more paths at the same budget |
| Light-source estimation | NEE + BSDF, MIS combined | Either alone | Lower envelope of both strategies' variances |
| First-bounce sampling | Owen-scrambled Sobol | PCG white noise / plain Sobol | 1.74× efficiency; scrambling removes visible structure |
| Adaptive sampling | Not adopted (built, measured, removed) | Per-pixel variance stopping | Measured slower and no more accurate; theoretical ceiling already captured by stratification |
| Time discretisation | Crank–Nicolson | Explicit Euler / fully implicit | Unconditionally stable + second-order (preserves diurnal peak) |
| Tridiagonal solver | Per-element thread-local Thomas | Parallel cyclic reduction | Parallel dimension is elements, not nodes; 32-node cap |
| Inter-element radiation | Explicit (ping-pong) | Implicit global matrix | Implicit destroys tridiagonal structure and per-element threading |
| Evaporation | Implicit (Newton-linearised) | Explicit | Slope too steep; explicit oscillates at minute-scale steps |
| Shadow resolution | dT/dv tangent, per-pixel shader correction | Mesh refinement | Refinement costs 2–3 orders of magnitude for redundant DOF |
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
| Environment map | Four-condition invariant, RGB mode only | Follow single switch | Invented sky was 46–84% of signal |
| Fallback cubemap | One black texel | Blue sky | Erroneous sampling darkens visibly instead of silently adding light |
| Cube format | ENVI + (GeoTIFF/EXR stubs) | HDF5 | Zero added dependency + remote-sensing downstream compatibility |
| Core linking | `quantiloom_core` vs `libQuantiloom`—one or the other | Mixed | Two copies of global state is a correctness bug |
| ABI expansion | Golden-baseline gate | Ad-hoc `QL_API` addition | Every contract expansion is reviewable |
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
- ASTM G-173 (Standard Tables for Reference Solar Spectral Irradiances).
- ISO 20473 (Optics and Photonics—Spectral Bands).
