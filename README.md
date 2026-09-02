# Quantiloom

Spectral path tracer built on Vulkan ray tracing. Renders a scene as VIS, NIR,
SWIR, MWIR or LWIR imagery, or as a hyperspectral cube, from a single TOML scene
description. Materials carry measured spectra rather than RGB albedo, and the
atmosphere comes from neural networks trained on MODTRAN.

Windows/Vulkan only; see **Building** for why.

## Spectral modes

One mode per render, set by `spectral.mode` in the scene TOML.

| `spectral.mode` | Output | Range |
|---|---|---|
| `rgb` | RGB (default, no spectral integration — fastest) | 650 / 550 / 450 nm |
| `vis_hero` | Visible, four sampled wavelengths a path → CIE XYZ → sRGB. Also spelled `VIS` | 400 – 780 nm |
| `vis_fused` | The same band by a deterministic 32-wavelength sweep | 400 – 780 nm |
| `nir_fused` | Near IR, reflected solar | 930 – 1200 nm |
| `swir_fused` | Short-wave IR | 1400 – 2400 nm |
| `mwir_fused` | Mid-wave IR, thermal | 3000 – 5000 nm |
| `lwir_fused` | Long-wave IR, thermal | 8000 – 12000 nm |
| `single` | One wavelength, greyscale EXR | any covered λ |
| `multispectral` | Hyperspectral cube | configurable grid |

The two visible modes estimate one integral and live in one binary. `vis_hero`
draws a wavelength per path and rotates it into a quartet, so it follows
n(lambda) through a dispersive interface and costs four radiances rather than
thirty-two; `vis_fused` sweeps a fixed grid, so it has no variance in wavelength
at all. That makes the second the reference the first is measured against
(`scripts/render-tests/check_hero_wavelength.py`, the illumination suite's
`hero` arm) and the mode to ask for when an answer has to be repeatable rather
than converged.

`GetFusedBandInfo()` in `core/Types.hpp` is the authority for these ranges; the
shader constants in `common.hlsli` are asserted against it by a unit test. Do
not read them off the `WAVELENGTH_*` constants in the same header — those are the
ISO 20473 band taxonomy, which is a different thing and does not match.

## What it does

**Spectral materials.** Reflectance is reconstructed from an NMF basis rather
than stored per wavelength. `scripts/spectral-baker/` bakes a basis plus a
material weight database from USGS splib07a, RefractiveIndex.info, or ECOSTRESS.
Each band records what fraction of it the source library actually measured —
bands the source never reaches are not emitted at all, so a fit is never
reported over extrapolated data.

**Atmosphere.** A ResMLP surrogate trained on MODTRAN supplies transmittance,
path radiance and downwelling radiance per band. `AtmosphereBaker` evaluates the
network into a LUT at startup; the shaders sample the LUT. Weights ship as
safetensors in `assets/atmos_models/`.

**Thermal IR.** Planck emission with metallic/roughness-derived emissivity, so
MWIR and LWIR renders carry a real emission term rather than reflected light
only.

**Volumetrics.** Rayleigh and Mie scattering, with unbiased delta tracking
(Woodcock) for heterogeneous media.

**Sensor chain.** Optical PSF, quantum efficiency, shot and read noise, dark
current, fixed pattern noise (PRNU/DSNU) with optional NUC, vignetting, and ADC
quantisation to raw DN.

**Reproducibility.** Sampling and sensor noise are both seeded and fixed by
default (`renderer.seed`, `sensor.noise_seed`), so two renders of one scene are
byte-identical. Set either to 0 for varying noise.

**Assets.** glTF 2.0 and OpenUSD scenes, EXR environment maps with prefiltered
IBL.

## Building

The product is a Windows binary. Development happens in WSL2 driving the Windows
toolchain, so every command invokes an `.exe`; there is no Linux build.

```bash
./build_wsl.sh                                  # configure, build, run tests, install SDK
cmake.exe --build build --config Release -j     # incremental C++ only, ~76 s
cmd.exe /c "src\shaders\compile_shaders.bat" </dev/null   # shaders only, ~1 s
```

`build_wsl.sh` configures with the `Visual Studio 18 2026` generator — multi-config,
so `--config` is required on every build — and installs to
`../Quantiloom-SDK/windows_amd64` (currently `H:/Quantiloom-SDK/windows_amd64`; the
prefix is derived from this checkout's location, override with `QUANTILOOM_SDK_ROOT`),
which the Quantiloom-Qt frontend links against.

Requires the Vulkan SDK with ray tracing support. OpenUSD is optional
(`-DQUANTILOOM_USE_OPENUSD=ON -DUSD_ROOT=...`; `build_wsl.sh` enables it).

## Running

```bash
./build/src/app/Release/Quantiloom.exe assets/configs/cornell_box_vis.toml
./build/tests/Release/libquantiloom_tests.exe --gtest_brief=1
```

The CLI takes exactly one argument: a scene TOML. 874 tests run in ~2 s; 10 are
skipped by design (8 BC7, disabled after measuring it as a net loss; 2 EXR
multipart, unimplemented).

`CLAUDE.md` carries the same commands plus the project conventions;
`docs/CLAUDE_CODE_SETUP.md` covers the per-developer clangd setup.

## Folder structure

```txt
Quantiloom/
│
├── assets/
│   ├── configs/            # TOML scene configs — the CLI's only input
│   ├── models/             # 3D models (glTF, USD)
│   ├── materials/          # Material and temperature data (CSV)
│   ├── maps/               # Environment maps (EXR)
│   ├── luts/               # Atmosphere LUTs
│   ├── atmos_models/       # Atmosphere network weights (safetensors)
│   ├── scenes/             # Scene assets
│   ├── spectral/           # NMF basis + material databases (see scripts/spectral-baker)
│   ├── usgs/               # USGS splib07a spectral library
│   └── refractiveindex/    # RefractiveIndex.info database
│
├── build/                  # (In .gitignore) CMake build tree
├── build-cdb/              # (In .gitignore) Ninja tree, exists only to emit compile_commands.json
│
├── docs/                   # Documentation
│   └── CLAUDE_CODE_SETUP.md
│
├── examples/               # (Placeholder) API usage examples
│
├── scripts/                # Automation and tooling (Python / shell)
│   ├── spectral-baker/     # Spectral library → NMF basis + material weights
│   ├── physics-audit/      # Physics validation harness
│   ├── render-tests/       # Render regression checks
│   ├── hooks/              # Claude Code PostToolUse hooks
│   ├── utils/              # Scene and LUT generators
│   └── validation/         # (Empty placeholder)
│
├── src/                    # Core source code
│   │
│   ├── libQuantiloom/      # (Built as SHARED library, Quantiloom.dll) HS-core API
│   │   ├── core/           # Basic tools (Log, Config, Math, Types)
│   │   ├── scene/          # Scene and asset management
│   │   ├── renderer/       # Vulkan abstraction (PSO, Buffer, TLAS/BLAS)
│   │   ├── hs_core/        # Hyperspectral cube: adaptive grid, reconstruction
│   │   ├── io/             # I/O utilities (ImageIO, SpectralIO, glTF/USD loaders)
│   │   ├── atmos/          # Neural-network atmosphere (ResMLP, safetensors)
│   │   └── postprocess/    # Sensor and noise chain
│   │
│   ├── libSpectraForge/    # (Built as SHARED library) IR material generation
│   │
│   ├── shaders/            # HLSL ray tracing shaders (compiled to SPIR-V)
│   │   ├── raygen.rgen     # Ray generation shader
│   │   ├── closesthit.rchit# Closest hit (PBR + spectral)
│   │   ├── miss.rmiss      # Miss shader (sky/atmosphere)
│   │   └── *.hlsli         # Shader headers (PBR, blackbody, spectral, volumetric)
│   │
│   ├── tools/              # Standalone tools
│   │   └── QLTrans/        # MODTRAN wrapper
│   │
│   └── app/                # (Built as executable) Main application
│       └── main.cpp        # Main program entry
│
├── tests/                  # C++ unit tests (GTest), one binary
│   ├── test_core/          # Tests for core/
│   ├── test_scene/         # Tests for scene/
│   ├── test_renderer/      # Tests for renderer/
│   ├── test_hs_core/       # Tests for hs_core/
│   ├── test_io/            # Tests for io/
│   └── test_postprocess/   # Tests for postprocess/
│
├── tools/                  # Development tools
│   ├── setup_dxc.sh        # DXC shader compiler setup (Linux)
│   └── setup_dxc.ps1       # DXC shader compiler setup (Windows)
│
├── vendor/                 # Vendored third-party sources
├── reference-only/         # Reference renderers, not built
│
├── .cpm_cache/             # (In .gitignore) CPM dependency cache
│
├── .gitignore
├── .gitattributes
├── CMakeLists.txt          # Top-level CMake (configures all subdirectories)
├── CLAUDE.md               # Build/test/convention reference
├── build_wsl.sh            # Build script (WSL2 shell, Windows toolchain)
├── build_windows.ps1       # Windows build script
├── LICENSE
└── README.md
```

## Dependencies (via CPM)

Fetched by [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) and cached in `.cpm_cache/`:

| Package | Version | Purpose |
|---|---|---|
| spdlog | 1.14.1 | Logging |
| tomlplusplus | 3.4.0 | TOML config parsing |
| tinygltf | 3.0.0 | glTF model loading |
| nlohmann_json | 3.11.3 | JSON (material databases) |
| Imath | 3.2.2 | Math types for OpenEXR |
| openexr | 3.4.3 | EXR image I/O |
| VulkanMemoryAllocator | 3.4.0 | Vulkan memory management |
| glm | 1.0.2 | Graphics mathematics |
| googletest | — | Unit tests (`QUANTILOOM_BUILD_TESTS=ON`) |

Found externally, not via CPM: the **Vulkan SDK** (required) and **OpenUSD**
(optional).

HDF5 is **not** a dependency — hyperspectral cube I/O uses EXR and ENVI instead
(`CMakeLists.txt:187`).

## Known limits

Stated here so results are not over-read:

- **MWIR/LWIR spectral coverage is partial.** Across the three source libraries,
  the best real-measurement coverage of those two bands is about 55%
  (ECOSTRESS); the rest is edge-clamp extrapolation. Their reconstruction error
  therefore reads *better* than the bands backed by full measurements, because a
  straight line is trivial to fit. Every basis records per-band coverage — quote
  it alongside any quality figure taken from this tool.
- **No CI.** The test suite is a manual gate inside `build_wsl.sh`.
- **No reference-renderer cross-validation.** `scripts/physics-audit/` checks
  formulas against an independent Python implementation, but nothing compares
  whole renders against PBRT-v4 or Mitsuba; `scripts/validation/` is an empty
  placeholder.
