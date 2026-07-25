# Quantiloom

Quantiloom is a unified spectral path tracing system designed for advanced imaging simulation across arbitrary spectral bands. It is built from the ground up on Vulkan Ray Tracing and centered around the `HS-core`, a single, unified kernel.

The project's methodology is "Validation-Gated", ensuring physical consistency by benchmarking all outputs against independent reference renderers like PBRT-v4 and/or Mitsuba3.

## Key Features

Quantiloom supports two spectral configurations from a single, unified codebase:

### 1. Band Mode (Discrete Multispectral)

* Defines discrete spectral bands for interactive, multi-band previews.
* Performance is measured in seconds-per-frame.
* Uses efficient band-pass integral estimation with Mixed Importance Sampling (MIS).
* Employs a `LUT-fast` model for atmosphere (sky/sun lookup) without path-traced volumetric scattering.

### 2. Continuous Mode (Full Spectral Range)

* A high-fidelity, per-wavelength ("point-render") configuration for generating high-spectral-resolution data cubes.
* Designed for quantitative comparison and validation against models like MODTRAN.
* Features a full, physically-based volume rendering pipeline, including Rayleigh and Mie scattering.
* Implements unbiased Delta-Tracking (Woodcock tracking) to support heterogeneous media like clouds or fog.

## Technology Stack

* Core (Host): C++20
* GPU Shaders: HLSL, compiled to SPIR-V by the Vulkan SDK's DXC
* Validation & Tooling: Python
* Build System: CMake

## Building

The product is a Windows binary. Development happens in WSL2 driving the Windows
toolchain, so all commands invoke `.exe` tools; there is no Linux build.

```bash
./build_wsl.sh                                  # configure, build, run tests, install SDK
cmake.exe --build build --config Release -j     # incremental C++ only, ~76 s
cmd.exe /c "src\shaders\compile_shaders.bat" </dev/null   # shaders only, ~1 s
```

`build_wsl.sh` configures with the `Visual Studio 18 2026` generator (multi-config,
so `--config` is required on every build) and installs to
`D:/Quantiloom-SDK/windows_amd64`, which the Quantiloom-Qt frontend links against.

Run the renderer and the tests:

```bash
./build/src/app/Release/Quantiloom.exe assets/configs/cornell_box_vis.toml
./build/tests/Release/libquantiloom_tests.exe --gtest_brief=1
```

`CLAUDE.md` has the same table plus the conventions; `docs/CLAUDE_CODE_SETUP.md`
covers the per-developer clangd setup.

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
│   ├── atmos_models/       # Atmosphere network weights
│   ├── scenes/             # Scene assets
│   ├── spectral/           # NMF basis + material databases (see scripts/spectral-baker)
│   ├── usgs/               # USGS splib07a spectral library
│   └── refractiveindex/    # RefractiveIndex.info database
│
├── build/                  # (In .gitignore) CMake build tree
├── build-cdb/              # (In .gitignore) Ninja tree, exists only to emit compile_commands.json
│
├── docs/                   # Documentation
│   ├── CLAUDE_CODE_SETUP.md
│   └── TODO.md
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
│   │   ├── hs_core/        # HS-core algorithms (MIS, Delta-Tracking)
│   │   ├── io/             # I/O utilities (ImageIO, SpectralIO, glTF/USD loaders)
│   │   ├── atmos/          # Neural-network atmosphere model
│   │   └── postprocess/    # Sensor and noise chain
│   │
│   ├── libSpectraForge/    # (Built as SHARED library) IR material generation
│   │
│   ├── shaders/            # HLSL ray tracing shaders (compiled to SPIR-V)
│   │   ├── raygen.rgen     # Ray generation shader
│   │   ├── closesthit.rchit# Closest hit (PBR + spectral)
│   │   ├── miss.rmiss      # Miss shader (sky/atmosphere)
│   │   └── *.hlsli         # Shader headers (PBR, blackbody, spectral)
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

### Dependencies (via CPM)

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
(optional, `-DQUANTILOOM_USE_OPENUSD=ON -DUSD_ROOT=...`; `build_wsl.sh` enables it).

HDF5 is **not** a dependency — hyperspectral cube I/O uses EXR and ENVI instead
(`CMakeLists.txt:187`).


