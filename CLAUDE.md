# Quantiloom

Spectral path tracer on Vulkan ray tracing. Renders VIS/SWIR/MWIR/LWIR bands and
hyperspectral cubes from TOML scene configs. C++20 + HLSL, one CMake project.

## Build: WSL2 shell, Windows toolchain

Development happens in WSL2; the product is a Windows binary. Only `.exe` tools are
available — there is no Linux `cmake`/`ctest` on this machine, and Linux cmake cannot
drive the Visual Studio generator. Run everything from the repo root.

| Change | Command |
|---|---|
| Full build + test gate + ABI gate + SDK install | `./build_wsl.sh` |
| C++ only | `cmake.exe --build build --config Release -j` (~76 s) |
| HLSL only | `cmd.exe /c "src\shaders\compile_shaders.bat" </dev/null` (~1 s), then rebuild |
| CMakeLists / new file | `./build_wsl.sh` (re-runs configure) |

`./build_wsl.sh` wipes and reinstalls `../Quantiloom-SDK/windows_amd64` (currently
`H:/Quantiloom-SDK/windows_amd64`; derived from this checkout, override with
`QUANTILOOM_SDK_ROOT`), which the
Quantiloom-Qt frontend links against. It takes minutes — use a 600000 ms timeout and
never interrupt it mid-install.

Four gates run before the install, so nothing red reaches the SDK: the test
suite, `scripts/check_exports.sh` against `docs/abi/*.golden`, the furnace
cavities, and the illumination suite. The export gate prints what to do when it
trips; `src/libQuantiloom/CLAUDE.md` has the rule it enforces.

**The Windows path is `./build_windows.ps1` then `./install_windows.ps1`, and it
must stay equivalent.** Both paths install the same SDK to the same prefix and
Quantiloom-Qt cannot tell which produced it, so a gate added to one belongs in
the other. The gates are *ported*, not wrapped: a Windows build machine has MSVC
and CMake and need not have bash or WSL, so `check_exports.ps1` and the two
`run_*_suite.ps1` are PowerShell reimplementations of their `.sh` twins. What is
shared rather than duplicated: the furnace cavity list (`furnace_cases.txt`, read
by both runners) and every checker in `scripts/render-tests/*.py`, which own all
the thresholds. Those stay Python because they are the measurement — Planck
integrals and RMSE over EXR images — and Python is portable in a way bash is not.
`build_windows.ps1` writes `build/.gates-passed.json`, and `install_windows.ps1`
refuses without it (`-Force` overrides), since two scripts cannot enforce an
order between themselves the way one script's `set -e` does.

## Tests

1279 tests run in ~9 s, and the binary reruns without rebuilding. They link the
objects, not the DLL, so internal code is testable without being exported.

```bash
./build/tests/Release/libquantiloom_tests.exe --gtest_brief=1          # all (122 lines, not 2086)
./build/tests/Release/libquantiloom_tests.exe --gtest_filter='Foo*'    # one suite (~0.04 s)
```

`ctest` registers a single aggregate test here, so `ctest -R` cannot select a case —
always use `--gtest_filter`. 10 SKIPPED is the normal baseline: 8 BC7 (deliberately
off, see `build_wsl.sh`) and 2 EXR multipart (unimplemented). A test needing an asset
must build its path from `QUANTILOOM_SOURCE_ROOT`, never a relative or absolute one —
those resolve against the caller's cwd and skip on miss, so a wrong path reads as
"no test data" rather than as a failure.

## Repo map

| Path | What |
|---|---|
| `include/quantiloom/` | **Public** headers — the only ones the SDK installs, 7 modules |
| `src/libQuantiloom/` | Core library sources and **internal** headers |
| `src/libQuantiloom/thermal/` | Surface energy balance: what sets a temperature, rather than what a config says it is |
| `src/shaders/` | HLSL → SPIR-V, ray tracing + compute |
| `src/app/` | CLI: `Quantiloom.exe <config.toml>`, `batch <list.txt>`, `serve` |
| `src/libSpectraForge/` | IR material generation library |
| `src/tools/` | `fusion_tool`, `QLTrans` (MODTRAN wrapper) |
| `tests/` | GoogleTest; directories mirror libQuantiloom module names |
| `docs/abi/` | Reviewed export baselines — the ABI gate's reference |
| `assets/configs/` | TOML scene configs — the CLI's only input |
| `scripts/` | Python/PowerShell tooling (spectral baking, LUT gen, physics audit) |

## Environment maps light RGB previews, and nothing else

An environment map is an RGB image in arbitrary units. That is fine for a
preview and wrong for a measurement, so exactly one mode samples one.

**The invariant.** `LightingParams::enableEnvironmentMap` is 1 on the GPU if and
only if all four hold: the config did not turn it off, it **names** a map, the
map **loaded**, and the mode is **RGB**. Anything less and the shader must not
sample binding 10.

- *Names one*, because there is no image-based lighting without an image. The
  flag used to follow `renderer.environment_map_enabled` alone, which defaults
  to true, so a scene naming no map was lit by the fallback cubemap — 256×256 of
  sky blue. In `paper-exp`'s 15 scenes that invented sky was 46–84% of the
  signal. The key still defaults to true; it is intent, one of four conditions,
  not the switch.
- *RGB*, because `closesthit.rchit`'s VIS_FUSED and SINGLE branches push the
  sampled RGB through `ConvertLinearRGBToIlluminantSpectrum` and use the result
  as spectral radiance density per nanometre — roughly 12× an ASTM G-173 sky.
  The same branches zero the traced specular residual (`qSpec`) when the flag is
  set. SWIR/NIR/MWIR/LWIR never referenced the cubemap at all; they take
  `TraceEnvBounceResidual` against the analytic sky or Planck downwelling, and
  are correct as they stand. A spectral scene is lit by `lighting.solar_lut` and
  the analytic sky, and naming a map in one earns a warning rather than a
  silently mis-lit render.

**The placeholder is not a sky.** `EnvironmentCubemap::Fallback` builds one black
texel (`kFallbackParams{1, 1}`). It exists only because the pipeline declares
binding 10 with `descriptorCount 1` and no partially-bound flag, so *something*
valid must be written there even for a scene with no environment. Black means a
bug that samples it anyway darkens visibly instead of quietly adding light to a
measurement.

**Load failure corrects the flag; it does not substitute a sky.** Both hosts
assume at resolve time that a named map will load, so both fix it where the
placeholder is actually chosen:

| Path | Where |
|---|---|
| CLI / offline | `OfflineRenderer::Impl::BuildPipeline`, in the `useFallback` lambda — zeroes the flag and re-uploads the lighting buffer, which was uploaded before `BuildPipeline` ran |
| Interactive | `ExternalRenderContext::LoadEnvironmentMap` — sets the flag on success, clears it plus `hasCustomEnvMap` on failure, uploads either way |

`Impl::UploadLightingParams` is the single writer of that buffer and masks the
flag with `hasCustomEnvMap`, so a host raising it through `SetLightingParams`
with nothing loaded cannot reach the shader. `HasEnvironmentMap()` reports load
state, not lighting state — a map can be loaded with IBL turned off.

## A lamp can be a spectrum, and outside the visible it must be

The sun has been able to say what it is made of for a long time:
`lighting.solar_lut` binds ASTM G-173, and `sun_radiance` is only the fallback
for when nothing is bound. A light *inside* the scene had no such path. Its
spectrum could only come from a glTF `emissiveFactor`, expanded as

```
L(λ) = 2·max(rgb) · sigmoid(c; λ) · D65(λ)
```

which is a computer-graphics construct — nothing measured it, and it is defined
only on 380–780 nm. That is the same category of invention `§ An RGB colour has
one spectral meaning` forbids outright for reflectance.

`[material_overrides."<name>"] emissive_curve = "..."` is the way out. It takes
a built-in token (`d65`, `illuminant_a`, `halogen`, `cie_f1`…`cie_f12`,
`cie_f3.1`…`cie_f3.15`, `blackbody_<T>k`, `equal_energy`) or a path;
`assets/luts/README.md` has the table and `assets/configs/cornell_box_lamp_spectrum.toml`
is the worked example. Five things about it are load-bearing:

- **Zero outside the curve's span, never clamped.** `EvaluateEmissionCurve` in
  `spectral_query.hlsli` differs from `EvaluateSpectralCurve` in exactly this,
  and the difference is the point. Constant extrapolation is defensible for a
  reflectance — bounded in [0,1], and a surface that reflects 0.4 at 780 nm
  plausibly reflects about that at 800. It is indefensible for an emission: the
  CIE fluorescent tables stop at 780 nm, and holding FL7's last value flat
  across SWIR furnishes a triphosphor lamp with near-infrared output it does not
  have. The host warns whenever a bound spectrum falls short of the band.
- **The IR bands read a bound curve and nothing else.** SWIR, NIR, MWIR and LWIR
  previously ignored `emissiveFactor` entirely, and they still do — there is
  deliberately no RGB fallback there, because expanding a triple at 10 µm runs
  the 380–780 nm fit far outside its domain. So a lamp appears in a thermal band
  only when someone bound data for it, and the quantitative gate now counts the
  materials that emit with nothing bound: they contribute *nothing*, which is
  the hardest kind of missing data to notice, since the render is merely dark.
- **One lamp, every mode.** When a curve is bound, `ResolveMaterialSpectra`
  overwrites `emissiveFactor` with the linear sRGB the curve integrates to. That
  is what keeps the RGB preview, the emitter-sampling CDF (`EmissiveTriangleGPU`
  carries the curve index in what used to be `_pad0`) and the spectral bands from
  describing three different lamps. It only happens where the CIE observer has
  support: in a thermal band the authored triple is left alone, because
  `SpectralCurve::Evaluate` clamps and integrating an 8–12 µm curve against the
  observer returns a large, confident, meaningless colour.
- **`match_luminance` is the default and `absolute` is the calibrated case.**
  Every built-in but the blackbody family is a *relative* distribution
  normalised to 100 at 560 nm, so the level has to come from somewhere: the
  material's own `emissive` triple. Swapping lamps then changes the room's
  colour without re-exposing the render. `absolute` takes the curve as spectral
  radiance in W·m⁻²·sr⁻¹·nm⁻¹ and is *required* outside the visible, where
  luminance is undefined — asking for `match_luminance` there is an error rather
  than a guess. Checked against Planck: a `blackbody_3000k` panel renders 1.948
  W·m⁻²·sr⁻¹·nm⁻¹ at 10 µm against the closed form's 1.935.
- **Emission is band-averaged onto the 64-point grid, reflectance is
  point-sampled.** A reflectance is smooth; a fluorescent lamp is mostly mercury
  lines, and point-sampling a line spectrum either hits a line or misses it —
  CIE FL11 lands 4.9% wrong in green that way, with the sign decided by nothing
  more principled than where the grid falls. Averaging each sample over its own
  bin conserves the energy and brings it to 0.2%. It does not fix the
  *estimator*, which point-samples 32 wavelengths and leaves FL11 about 14% high
  in blue; the host detects that case — band-averaging and point-sampling the
  same curve disagree exactly when it is not band-limited — and says so.

The host-side twin of the estimator's normalisation is
`EmissionSpectrumToRenderedLinearSrgb`, which is `SpectralIrradianceToLinearSrgb`
divided by `CieLuminanceIntegral()`. Using the un-normalised form is a 106×
error that scales every wavelength equally, so it changes nothing about the
colour and reads as an exposure mistake rather than a unit one. It cost a
debugging round already.

## Thermography

Four things a thermal scene can now do that it could not, each independent of
the others and each off by default:

| Section | What it does |
|---|---|
| `[[materials]] temperature_texture` | a temperature field per texel instead of one number |
| `[thermography]` | invert the render into the temperature a camera would report, as `<output>_tapp.exr`, plus a NETD |
| `[atmosphere] sky_model = "clear_sky"` | a sky that is colder overhead than at the horizon, from air temperature and humidity |
| `[thermal]` | compute the temperatures from a surface energy balance instead of being told them |

`assets/configs/thermal_*.toml` is one worked example of each. The physics is
checked against closed forms in `tests/test_core/test_blackbody.cpp`,
`test_sky_thermal.cpp` and `test_thermal_conduction.cpp`, and against analytic
view factors in `tests/test_renderer/test_thermal_exchange_gpu.cpp`; the
reference implementations live in `scripts/physics-audit/harness.py`.

The one invariant worth knowing before touching any of it: a surface
temperature has exactly one decode, `GetSurfaceTemperatureK` in
`closesthit.rchit`, and all four sampling sites plus both debug views go
through it. Solver, then texture, then the material's own scalar — and, when
the solver answered, the per-pixel sun correction below, which lives inside
that one function precisely so that no site can forget it.

### A shadow is not the size of a triangle

The solver runs on one element per triangle: `BuildThermalMesh` makes them,
`RunSunVisibility` decides from each **centroid** whether the sun arrives, and
the shader reads one temperature per `PrimitiveIndex()`. So the temperature
field it produces can only have edges where the mesh has edges. On a 120 m
desert ground tessellated 201 × 201 that is a 0.6 m triangle, and a 0.7 m
sphere casts a shadow shaped like a triangle — in a band where a shadowed sand
element runs 40 K below a lit one, so it is the most visible thing in the
frame.

Nothing about the physics is that coarse. Dry sand diffuses heat about 3 cm in
an hour, and the model gives every element an independent one-dimensional
column with **no lateral conduction at all** — so the field it describes has a
shadow edge as sharp as the geometry's. Only the discretisation was coarse.

The fix is to ship the derivative alongside the value. Beside each element's
temperature the solver carries `dT/dv`: how far that temperature would move per
unit of the element's own sun visibility. The shader traces its own ray to the
sun and evaluates

```
T(x) = T_element + (v(x) - v_element) * dT/dv
```

which reproduces the solved value at the element's mean and resolves the edge
at whatever resolution the ray tracer has. `renderer/ThermalSunResponse.hpp`
holds the layout: one `float4` per record on binding 26, index 0 a header
carrying the solve's sun direction (the forcing CSV owns it and it need not be
`[lighting] sun_direction`) plus the flag that switches the whole thing off,
then `1 + thermalElementBase + PrimitiveIndex()` per element.

**`dT/dv` is a state, not a formula, and that is the whole design.** The two
formulas it is tempting to use are both badly wrong for anything with thermal
inertia. For sand at 11:00 the steady response `α E cosθ / (h + 4εσT³)` is
about 31 K and the single-step response about 5 K; the truth after three hours
of sun is 27.9 K and after twelve is 30.3 K. So it is integrated as the
**tangent of the trajectory** — the same tridiagonal matrix as the temperature,
a second right-hand side, one extra elimination pass — and inherits the slab's
thickness, node count, boundary condition and history for free.
`ThermalState::sunSensitivity_K` carries it, `CpuCrankNicolsonStepper` and
`thermal_step.comp.hlsl` both step it, and it is **empty by default**: a caller
that only wants bulk temperatures sizes nothing and pays nothing.

What it costs and what it is worth:

| | |
|---|---|
| Extra rays | one per thermal hit where \|dT/dv\| ≥ 0.1 K — so none at night, none indoors, none in a scene with no solve |
| Extra state | doubles `ThermalState`, and therefore the timeline's checkpoints |
| Extra solver time | one elimination pass per element per step |
| Accuracy at full amplitude | 1.0 K on a 28.9 K contrast (3.5%), because the radiative admittance moves by a third across that span. Against a 0.6 m triangle that is fully lit or fully dark, it is not close |

Two traps if you touch it:

- **The tangent is local on purpose.** The neighbours' share of `incoming` is
  not differentiated. Its derivative is the off-diagonal of a Jacobian over
  every element that sees this one, and all it would add is the second-order
  fact that a colder patch of ground slightly cools its neighbours. Keeping it
  out is what makes `dT/dv` a per-element number a shader can apply per pixel.
- **The host, not the shader, decides that the sun is behind an element.** The
  shader's geometric normal has been flipped to face the viewer, so it cannot
  tell a face genuinely turned away from a hit on the *back* of a sun-facing
  triangle — which has the same temperature as the front and does want the
  correction. Both hosts zero `dT/dv` for the first case; the ray is then
  offset along the sun rather than along the normal, which is right for both.

### What the balance is made of

Six fluxes at the exposed face, three of which no config asks for directly —
they are geometry, computed once per trajectory:

| Flux | Where it comes from | Off when |
|---|---|---|
| Direct sun | `α_s E_dni cosθ v_i`, `v_i` from the shadow dispatch | no sun |
| **Diffuse sky** | `α_s E_diff G_i`, `G_i` the sky fraction plus one bounce | `diffuse_irradiance_w_m2 = 0` |
| **Reflected sun** | `α_s E_dni R_ik`, baked per sun column | scene is a single plane |
| Long wave | `ε σ (Σ F_ij T_j⁴ + s_i T_sky⁴ − T_i⁴)` | `ir_emissivity = 0` |
| Convection | `h (T_air − T_i)`, half implicit | `convection_h_w_m2k = 0` |
| **Evaporation** | `f_wet (h/c_p) L_v (q_sat(T_i) − RH q_sat(T_air))`, Magnus | `wetness_factor = 0` (the default) |

`R` and `G` come from `thermal/ShortwaveGains.hpp`, gathered with the **same
CSR view factors as the long wave** — radiosity and radiative exchange are one
integral over one hemisphere, in two bands — so they cost a matrix pass rather
than a second precompute. Neither depends on temperature, so both are baked
into the `SunVisibilityTable` when the timeline is built and a step just reads
them. `R` has a column per sun sample and is interpolated on the same indices
as the visibility it was baked from; `G` is one column.

Evaporation is the only term that is **not** explicit. Its slope in
temperature is several times the linearised radiative one, so it is Newton-
linearised into `diag[0]` and `rhs[0]` beside the convection. Left explicit it
oscillates at a minute per step.

The forcing CSV is eight columns, the last two optional and defaulted:
`time_h, air_k, dni, sun_azimuth_deg, sun_elevation_deg, sky_k,
diffuse_w_m2, relative_humidity`. A file written before those existed keeps
its meaning exactly. A constant-forcing run reads `[thermal]
diffuse_irradiance_w_m2` and `[atmosphere] relative_humidity` — the humidity
is deliberately *not* duplicated into `[thermal]`, since a scene with two of
them would be a scene with two atmospheres.

### Getting a thermogram onto a screen

`BlitToTarget` is a bare format conversion — there is no tone mapping in the
present path. An LWIR render's radiance sits around 1e-2, two orders below 1.0,
so **display enhancement is not a beautifier; it is the only reason an infrared
scene is visible at all**. `clahe.comp.hlsl` (three passes, driven by
`DisplayEnhancementParams` in `include/quantiloom/renderer/DisplayControl.hpp`)
is two independent stages:

| Stage | What it decides | Modes |
|---|---|---|
| **Tone** | contrast: a scalar → [0,1] | `Linear`, `Equalize`, `Clahe` |
| **Palette** | colour: that scalar → RGB, changing no contrast | `Grey`, `GreyInverted`, `Ironbow`, `Rainbow`, `Viridis` |

They are separate fields because they compose; enumerating the product would be
a dozen names for two decisions.

The property that separates the tone operators is not sharpness, it is whether
the whole image is mapped the same way. Measured on `thermal_solver_lwir` at
3.16M pixels, sorting display value by raw radiance and counting inverted pairs:

| | inverted pairs | worst drop |
|---|---:|---:|
| `Linear` | 0.000% | 0 |
| `Equalize` | 0.000% | 0 |
| `Clahe` | 1.902% | 17 display levels |

`Equalize` inverts nothing because a global CDF is a monotone map; `Clahe` is
tile-local, so it is not. That is the whole difference between them, and it is
why `Equalize` is safe to read an *ordering* off and `Clahe` is not. Neither is
safe to read a *value* off without publishing the CDF, which is what keeps
`Linear` the default for anything carrying a colour bar.

(This table read 50.6% and 45 levels until it was re-measured over a 2e7 pair
sample against a numpy reimplementation of the three passes that can be checked
line by line against the shader; a second scene gives 2.02%. The old figure is
not reproducible from anything in either repository. The qualitative claim is
unchanged: 17 display levels is still enough that a temperature must not be read
off one.)

So **do not read a temperature off a `Clahe` view**: two pixels at one
temperature in different tiles come out as different greys. It is for finding an
edge. `Linear` is the default and is what a thermal camera calls linear AGC.

Two traps worth knowing before touching any of it:

- `Equalize` needed no new pipeline. Pass 2 sums every tile's histogram, so all
  tiles end up with one CDF, and pass 3's bilinear blend between four identical
  CDFs is that CDF. `Linear` skips passes 1 and 2 entirely — which is why pass 3
  binds the descriptor set again rather than relying on pass 1 having done it.
- The luminance-preserving path divides by the pixel's own luminance, and
  `RGBToLuminance(v,v,v)` is not `v` to the last bit. That ratio used to wobble
  grey pixels around each CDF plateau — 12.9% inverted pairs. Achromatic pixels
  now bypass it, which is every infrared render.

`CaptureDisplayImage()` reads this image and `CaptureScreenshot()` reads the raw
accumulation, deliberately: Studio's "Save Screenshot" gives the false-colour
view, "Export Image" gives physical values. Neither is a bug to be tidied.

### Interactive thermal path

The solver runs in the viewport, not only offline. The architecture:

**ThermalTimeline** (`thermal/ThermalTimeline.hpp`) is a fixed-grid trajectory:
`t_k = startTime_h + k * timestep_s / 3600`. The timestep does not depend on
the end time, so every step taken for an earlier query is reused for a later
one. Checkpoints (full `ThermalState` snapshots) are stored every
`checkpointStride_h` simulated hours; scrubbing backwards replays from the
nearest checkpoint, not from t=0. The steady-state initial condition is
computed once at construction.

**GpuThermalStepper** (`renderer/GpuThermalStepper.hpp`) mirrors the CPU
Crank-Nicolson math in f32 via `thermal_step.comp.hlsl`. Each element is one
thread; the Thomas solve is thread-local (max 32 nodes); inter-element
radiative coupling reads from a ping-pong surface buffer. `StepMany`
dispatches entire batches in a single `ExecuteImmediate`. Falls back to the
CPU stepper when no GPU is available or `nodeCount > 32`.

**ThermalPreview** (`renderer/ThermalPreview.hpp`) is the internal subsystem
class owned by `ExternalRenderContext::Impl`. It holds the mesh, exchange
geometry, sun visibility table, timeline and stepper, plus dirty flags. It
never touches the pipeline or descriptors — the facade owns binding 24.
Everything the timeline is constructed with except the `Desc` is **held by
reference for its lifetime**, so all of it has to be a member — a local copy
is read after it goes out of scope.

**SunVisibilityTable** (`thermal/ThermalTypes.hpp`) holds K columns of per-
element sun visibility at different times of day, the sun direction each was
taken at, and the short-wave gains baked from them. Built by
`ThermalExchangePrecompute::RunSunVisibility` (K dispatches in one submit,
hemisphere rays skipped) whenever the forcing file has more than one row;
constant forcing gets a single column from the exchange. Each solver step
interpolates between the two nearest columns.

### Invalidation rules

Changing time (`SetThermalTime`) does NOT invalidate anything — it steps
forward from (or replays from) a checkpoint. These events set dirty flags
that the next `SetThermalTime` resolves lazily:

| Event | Exchange | Sun table | Materials | Timeline |
|---|---|---|---|---|
| Geometry change (Adopt/Rebuild/Refit) | dirty | dirty | — | dirty |
| Material IR curve edit | — | — | dirty | dirty |
| SetThermalMaterial / ClearThermalMaterials | — | — | dirty | dirty |
| SetParams (rays/topK changed) | dirty | dirty | — | dirty |
| SetParams (forcing/sun fields changed) | — | dirty | — | dirty |
| SetParams (timestep/layers/initial changed) | — | — | — | dirty |
| SetLighting/SetSunDirection (no forcing file) | — | dirty | — | dirty |

A `RefitAccelerationStructure` during a gizmo drag only sets the flag (O(1)
per frame). The cost is deferred to the first scrub after the drag finalizes.

The short-wave gains have **no flag of their own**. They depend on the
geometry, the sun columns and the absorptivities, and every event that
changes one of those already marks the timeline dirty — so they are re-baked
in `RebuildTimeline` and are fresh by construction.

### The offline solve is cached, and the key is the interesting part

A batch varies the sensor, and nothing about a sensor reaches the energy
balance. So a list of 1290 LWIR renders contains about **15 distinct solves**
and, before this, computed each of them eighty times — 61% of a full rebuild's
wall clock. `thermal/ThermalSolveCache.hpp` stores a solved `ThermalResult`
under a SHA-256 of its inputs. Measured: ABeautifulGame's `nominal_LWIR`
(1518656 elements) goes from 169.4 s to 5.2 s, and the render is byte-identical.

On by default and invisible from a config — no TOML key, no `InitParams` field,
no exported symbol. `QUANTILOOM_THERMAL_CACHE=0` switches it off,
`QUANTILOOM_THERMAL_CACHE_DIR` moves it (default
`<platform cache dir>/Quantiloom/cache/thermal`). There is no eviction; the
management story is deleting the directory.

Four things to know before touching it:

- **The boundary is `RunThermalSolver()`, not `RunThermalSolve`.** An entry
  stands for the GPU view-factor precompute *as well as* the trajectory — 15 s
  plus 150 s on the big scene. Caching only the stepping gives back a tenth of
  the win, and the precompute is the part that needs a TLAS.
- **The key hashes resolved inputs, never config text.** That is what makes two
  configs differing only in `sensor.*` one entry. It covers the built
  `ThermalMesh`, the merged materials, the `[thermal]` scalars, the forcing
  CSV's *contents*, `[lighting] sun_direction`, the stepper's `Name()`, the
  library version and the GPU. Miss a field and the cache serves another
  scene's temperature field while the render exits 0 — which is why it is 256
  bits and why `BuildSolvedMaterials` was hoisted: the emissivity the solve uses
  is the Planck band average of the material's curve, not the number in the
  TOML, and a key built from the config would be blind to 0.4 K.
- **The GPU is in the key because the precompute is ray-traced GPU work.** It is
  deterministic for a fixed binary on a fixed driver — Hammersley, a stateless
  coverage hash, no atomics — but BVH construction and intersection are a
  vendor's business. Keying on device identity makes a driver update a miss
  rather than a wrong answer.
- **A hit still prints the gate line**, through `LogThermalSolveSummary`, which
  is now the only place that emits it. Downstream reads `solved/elements` off
  that line to catch a scene where the subject fell out of the solve; a hit that
  printed nothing would make every render look clean. Anything new logs
  `Thermal cache:` or `Thermal stepper:` so nothing can mistake it for the
  summary. What a hit *does* skip are the advisory warnings — "no view factors",
  the timestep-versus-time-constant note — so a cold and a hot log differ there.

Naming `thermal.dump_elements` opts out in both directions: the dump needs the
exchange's sky fractions, which an entry does not carry.

`QUANTILOOM_THERMAL_GPU_STEPPER=1` runs the offline trajectory on
`GpuThermalStepper` (`kMaxNodes = 32`, else it falls back and says so). **Off by
default, and not judgeable by byte equality** — f32 and a different reduction
order give different floats. Against the 0.2.5 baseline at `nominal_LWIR`, on a
DN range of about 34000: DamagedHelmet moved 85.8% of pixels (mean 6.5 DN, p99
17, max 23), CesiumMilkTruck 35.7% (mean 0.4 DN, p99 1, max 2). An order of
magnitude apart, so a tolerance argued from one scene is wrong for the other.
Stating one per band is the work before this defaults on; what it already buys
is the wait while authoring, where 165 s is per material edit.

The core is compiled once into `quantiloom_core` (an OBJECT library) and consumed
two ways. **A new target links one or the other, never both** — two copies of the
library's global state (the spdlog logger, static caches) in one process is a
correctness bug:

| Link | Who | Sees |
|---|---|---|
| `quantiloom_core` | `libquantiloom_tests`, `fusion_tool` | everything, `QL_API` irrelevant |
| `libQuantiloom` | CLI, `libSpectraForge`, Quantiloom-Qt | only `QL_API`, only `include/quantiloom/` |

## Conventions

- Commits: Conventional Commits — `feat:`, `fix(shaders):`, `chore:`.
- **No Claude Code session link in a commit message.** No `Claude-Session:` trailer,
  no `https://claude.ai/code/...` URL, in the subject, the body or a trailer. Same for
  PR descriptions.
- clang-tidy runs through clangd via `.clang-tidy` — warnings surface as you edit,
  there is no separate lint command. No formatter is configured; match nearby style.
