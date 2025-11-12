# Quantiloom

Quantiloom is a unified spectral path tracing system designed for advanced imaging simulation across arbitrary spectral bands. It is built from the ground up on Vulkan Ray Tracing and centered around the `HS-core`, a single, unified kernel.

The project's methodology is "Validation-Gated", ensuring physical consistency by benchmarking all outputs against independent reference renderers like PBRT-v4 and/or Mitsuba3.

## Key Features

Quantiloom operates in two distinct modes from a single, configurable codebase:

### 1. Quantiloom MS-RT (Multispectral Fast Preview)

* Provides interactive, multi-band previews intended for rapid visualization.
* Performance is measured in seconds-per-frame.
* Uses efficient band-pass integral estimation with Mixed Importance Sampling (MIS).
* Employs a `LUT-fast` model for atmosphere (sky/sun lookup) without path-traced volumetric scattering.

### 2. Quantiloom HS-OFF (Hyperspectral Offline)

* A high-fidelity, per-wavelength ("point-render") mode for generating high-spectral-resolution data cubes.
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
│   ├── scenes/             # 3D Scene objects (OBJ, GLTF, FBX, ...)
│   ├── materials/          # Materials and temperatures data (HDF5, CSV)
│   ├── luts/               # MODTRAN LUTs (HDF5, SKYLUT6)
│   └── configs/            # TOML configuration files
│
├── build/                  # (In .gitignore) CMake generated build files
│
├── docs/                   # Documentation (including API documentation)
│
├── examples/               # Small projects demonstrating API usage
│   ├── 01_load_scene/      # Demonstrates how to load a scene
│   ├── 02_run_ms_rt/       # Demonstrates how to run MS-RT preview
│   └── 03_run_hs_off/      # Demonstrates how to run HS-OFF per-band point rendering
│
├── scripts/                # Automation and validation scripts
│   ├── validation/         # Validation benchmark suite (Python)
│   │   ├── run_benchmark.py
│   │   ├── compare_pbrt.py   # Compare with PBRT-v4/Mitsuba3
│   │   └── plot_results.py
│   └── utils/              # Asset conversion, LUT generation, etc. tool scripts
│
├── src/                    # Core source code
│   │
│   ├── libQuantiloom/      # (Built as a library) HS-core API core
│   │   ├── core/           # Basic tools (Log, Config, Math)
│   │   ├── scene/          # Scene and asset management
│   │   ├── renderer/       # Vulkan abstraction layer (PSO, Buffer, TLAS/BLAS)
│   │   ├── hs_core/        # HS-core algorithms (MIS, Delta-Tracking)
│   │   └── postprocess/    # Sensor and noise chain
│   │
│   └── app/                # (Built as an executable) Main application
│       ├── RendererMS.cpp  # MS-RT mode implementation
│       ├── RendererHS.cpp  # HS-OFF mode implementation
│       └── main.cpp        # Main program entry
│
├── tests/                  # C++ unit tests (GTest/Catch2)
│   ├── test_core/          # Test functions in core/
│   └── test_scene/         # Test functions in scene/
│
├── vendor/                 # Third-party dependency libraries
│   ├── assimp/
│   ├── tinygltf/
│   ├── hdf5/
│   ├── openexr/
│   ├── tomlplusplus/
│   └── ...                 # PBRT-v4/Mitsuba3 (as a V&V reference)
│
├── .gitignore
├── .gitattributes
├── CMakeLists.txt          # Top-level CMake (configures all subdirectories)
├── LICENSE
└── README.md
```
