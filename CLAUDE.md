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

Two gates run before the install, so neither a red suite nor an unreviewed ABI
change can reach the SDK: the test suite, then `scripts/check_exports.sh` against
`docs/abi/*.golden`. The export gate prints what to do when it trips; `src/libQuantiloom/CLAUDE.md`
has the rule it enforces.

## Tests

1125 tests run in ~4 s, and the binary reruns without rebuilding. They link the
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
through it. Solver, then texture, then the material's own scalar.

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
| `Clahe` | 50.6% | 0.176 — 45 display levels |

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
