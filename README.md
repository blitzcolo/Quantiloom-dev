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

* Core (Host): C++
* GPU Shaders: HLSL (later can be compiled to SPIR-V for Vulkan)
* Validation & Tooling: Python (for V&V Benchmark Suite)
* Build System: CMake

## Folder structure

```txt
Quantiloom/
│
├── assets/
│   ├── models/             # 3D models (GLTF)
│   ├── materials/          # Materials and temperatures data (HDF5, CSV)
│   ├── luts/               # MODTRAN LUTs (HDF5, SKYLUT6)
│   └── configs/            # TOML configuration files
│
├── build/                  # (In .gitignore) CMake build files (Windows)
│
├── docs/                   # Documentation
│   └── DATA_PREPARATION.md # Data preparation guide
│
├── examples/               # (Placeholder) API usage examples
│
├── out/                    # (In .gitignore) CMake build output (Linux)
│
├── scripts/                # Automation and validation scripts
│   ├── validation/         # Validation benchmark suite (Python)
│   │   ├── run_benchmark.py
│   │   ├── compare_pbrt.py # Compare with PBRT-v4/Mitsuba3
│   │   └── plot_results.py
│   └── utils/              # Asset conversion, LUT generation tools
│
├── src/                    # Core source code
│   │
│   ├── libQuantiloom/      # (Built as static library) HS-core API
│   │   ├── core/           # Basic tools (Log, Config, Math, Types)
│   │   ├── scene/          # Scene and asset management
│   │   ├── renderer/       # Vulkan abstraction (PSO, Buffer, TLAS/BLAS)
│   │   ├── hs_core/        # HS-core algorithms (MIS, Delta-Tracking)
│   │   ├── io/             # I/O utilities (ImageIO, SpectralIO)
│   │   └── postprocess/    # Sensor and noise chain
│   │
│   ├── shaders/            # HLSL ray tracing shaders (compiled to SPIR-V)
│   │   ├── raygen.rgen     # Ray generation shader
│   │   ├── closesthit.rchit# Closest hit (PBR + spectral)
│   │   ├── miss.rmiss      # Miss shader (sky/atmosphere)
│   │   └── *.hlsli         # Shader headers (PBR, blackbody, spectral)
│   │
│   └── app/                # (Built as executable) Main application
│       └── main.cpp        # Main program entry
│
├── tests/                  # C++ unit tests (GTest)
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
├── .cpm_cache/             # (In .gitignore) CPM dependency cache
│
├── .gitignore
├── .gitattributes
├── CMakeLists.txt          # Top-level CMake (configures all subdirectories)
├── build_linux.sh          # Linux build script
├── build_windows.ps1       # Windows build script
├── LICENSE
└── README.md
```

### Dependencies (via CPM)

Dependencies are managed by [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) and cached in `.cpm_cache/`:
- **spdlog**: Logging
- **tomlplusplus**: TOML config parsing
- **tinygltf**: glTF model loading
- **hdf5**: HDF5 spectral data I/O
- **openexr**: EXR image I/O
- **VulkanMemoryAllocator**: Vulkan memory management


