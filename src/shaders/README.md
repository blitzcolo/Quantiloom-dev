# Quantiloom Shaders

This directory contains HLSL ray tracing shaders for Quantiloom.

## Shader Files

### Ray Tracing Pipeline
| File | Type | Description |
|------|------|-------------|
| `raygen.rgen` | Ray Generation | Camera ray, then the whole path: every vertex is shaded here (`path_driver.hlsli`) |
| `closesthit.rchit` | Closest Hit | Writes the hit record -- where the ray landed -- and nothing else |
| `anyhit.rahit` | Any Hit | Alpha coverage (MASK/BLEND) and fusion-interface re-hit rejection |
| `miss.rmiss` | Miss | Marks the ray as missed (`hit.t = -1`) |
| `shadow_miss.rmiss` | Miss | Clears `isShadowed` for a visibility ray |

Every ray is traced from raygen, so the pipeline's recursion depth is 1.
`src/shaders/AGENTS.md` ("The integrator runs in raygen") has the rules.

### Header Files
| File | Description |
|------|-------------|
| `common.hlsli` | Shared types, spectral modes, the payload and `PathVertex` |
| `path_driver.hlsli` | The path loop raygen runs: record traces, the frame stack, descent and fold |
| `shade_vertex.hlsli` | Surface shading, cut at its children (`ShadeVertexA`, `Advance`, `CompleteChild`, `FoldVertex`) |
| `shade_miss.hlsli` | Sky evaluation for a ray that missed (`ShadeMiss`) |
| `fusion_transport.hlsli` | Fusion-export interfaces (`BeginFusionInterface` / `CompleteFusionInterface`) |
| `ql_trace.hlsli` | The two trace sites: record rays and visibility rays |
| `pbr.hlsli` | Cook-Torrance PBR functions (GGX, Fresnel) |
| `blackbody.hlsli` | Planck blackbody radiation for thermal IR (C1_NM = 1.191×10²⁰) |
| `atmosphere_nn.hlsli` | NN atmosphere LUT sampling (MODTRAN surrogate tau / L_path / L_down) |
| `SpectralConversion.hlsli` | CIE XYZ color matching, spectral upsampling |
| `spectral_query.hlsli` | Spectral curve sampling utilities |

## Compilation

Shaders must be compiled to SPIR-V using **DXC** (DirectX Shader Compiler).

### Install DXC

- **Windows**: Download from [Vulkan SDK](https://vulkan.lunarg.com/)
- **Linux**: `sudo apt install dxc` or build from [microsoft/DirectXShaderCompiler](https://github.com/microsoft/DirectXShaderCompiler)
- **macOS**: `brew install dxc`

### Compile Scripts

Use the provided scripts:
```bash
# Linux/macOS
./compile_shaders.sh

# Windows
./compile_shaders.bat
```

### Manual Compilation

```bash
# Ray Generation
dxc -spirv -T lib_6_3 -fspv-target-env=vulkan1.3 -Fo raygen.spv raygen.rgen

# Closest Hit
dxc -spirv -T lib_6_3 -fspv-target-env=vulkan1.3 -Fo closesthit.spv closesthit.rchit

# Miss
dxc -spirv -T lib_6_3 -fspv-target-env=vulkan1.3 -Fo miss.spv miss.rmiss
```

### Output

After compilation:
- `raygen.spv`
- `closesthit.spv`
- `miss.spv`

## Descriptor Set Layout (Set 0)

| Binding | Type | Stage | Description |
|---------|------|-------|-------------|
| 0 | RWTexture2D | Raygen | Output image (RGBA32F) |
| 1 | AccelerationStructure | Raygen | TLAS (scene) |
| 2 | StructuredBuffer | Raygen | LightingParams (sun/sky) |
| 3 | StructuredBuffer<float3> | Raygen, AnyHit | Vertex buffer (positions) |
| 4 | StructuredBuffer<uint> | Raygen, AnyHit | Index buffer (triangle indices) |
| 5 | StructuredBuffer<MaterialData> | Raygen, AnyHit | Material properties |
| 6 | Texture2D[] | Raygen, AnyHit | Bindless texture array |
| 7 | SamplerState[] | Raygen, AnyHit | Bindless sampler array |
| 8 | StructuredBuffer<float2> | Raygen, AnyHit | UV coordinates |
| 9 | StructuredBuffer<float4> | Raygen | Tangent vectors |
| 10-12 | - | Raygen | IBL resources (cubemap, BRDF LUT) |
| 13 | StructuredBuffer<SpectralCurveGPU> | Raygen | Spectral reflectance curves |
| 14 | StructuredBuffer<ComplexRefractiveIndexGPU> | Raygen | Complex refractive index (n,k) |
| 15 | StructuredBuffer<SolarSpectralLUT> | Raygen | Solar spectral irradiance |

"Stage" is who reads the binding. The layout in `RayTracingPipeline.cpp` also
still grants the closest-hit and miss stages that read these before the
integrator moved into raygen; the bits are harmless.

## Payload Structure

raygen shades every path vertex itself (`path_driver.hlsli`), so a traced ray
only reports where it landed. The payload (`common.hlsli`) is 128 bytes; the
per-vertex state lives in `PathVertex`, which no ray carries.

```hlsl
struct Payload {
    HitRecord hit;                   // 96: t (-1 on a miss), barycentrics, instance,
                                     //     primitive and both instance 3x3s
    uint   isShadowed;               //  4: visibility rays; shadow_miss clears it
    uint   fusionPreviousInstance;   //  4: any-hit re-hit rejection at a
    float3 fusionPreviousPosition;   // 12  fusion interface
    float3 fusionPreviousNormal;     // 12
};
```

## Spectral Rendering Modes

Defined in `common.hlsli`:

| Mode | Value | Integration range | Samples |
|------|-------|-------------------|---------|
| `SPECTRAL_MODE_SINGLE` | 0 | one wavelength | 1 |
| `SPECTRAL_MODE_VIS_FUSED` | 1 | 400-780 nm | 32 |
| `SPECTRAL_MODE_MULTISPECTRAL` | 2 | Hyperspectral cube (TBD) | — |
| `SPECTRAL_MODE_MWIR_FUSED` | 3 | 3000-5000 nm | 16 |
| `SPECTRAL_MODE_LWIR_FUSED` | 4 | 8000-12000 nm | 16 |
| `SPECTRAL_MODE_SWIR_FUSED` | 5 | 1400-2400 nm | 16 |
| `SPECTRAL_MODE_NIR_FUSED` | 6 | 930-1200 nm | 16 |
| `SPECTRAL_MODE_RGB` | 7 | 650/550/450 nm | 3 |

Ranges are defined once, as `SPECTRAL_<BAND>_LAMBDA_{MIN,MAX}` in `common.hlsli`,
and must match `GetFusedBandInfo()` in `core/Types.hpp` — `test_types.cpp` asserts
the C++ side. Do not take these from the `WAVELENGTH_*` constants in `Types.hpp`:
those are the ISO band taxonomy (NIR 780-1400 nm, SWIR 1000-2500 nm) rather than
what the renderer integrates, and copying them here is exactly how the SWIR and
NIR rows were wrong before.

## Current Features

- **Cook-Torrance PBR**: Full GGX specular with height-correlated Smith geometry
- **Spectral rendering**: Physically-based spectral upsampling and XYZ integration
- **IR thermal emission**: Planck blackbody radiation for MWIR/LWIR/SWIR modes ⭐ **Corrected (2025-12-25)**
  - Fixed Planck constant C1_NM (was 10000× too small)
  - Validated against 300K blackbody @ 9.6μm (~0.01 W·sr⁻¹·m⁻²·nm⁻¹)
- **Measured material data**: Spectral reflectance curves and complex refractive indices
- **IBL**: Environment map lighting with prefiltered specular
- **Atmospheric model**: Beer-Lambert attenuation + physical Rayleigh/Mie scattering ⭐ **Enhanced (2025-12-25)**
  - Scalar/RGB versions for type-correct wavelength dependence
  - Delta-Tracking volumetric rendering for heterogeneous media
- **Ray differentials**: Texture LOD computation for proper filtering

## Recent Fixes (2025-12-25)

### Critical Physics Corrections
1. **Planck Constant Error**: `C1_NM = 1.191042972e16` → `1.191042972e20` (blackbody.hlsli:51)
   - **Impact**: All MWIR/LWIR thermal calculations now physically correct
   - **Validation**: 24 unit tests added (test_blackbody_physics.cpp)

2. **Atmospheric Scattering Type Confusion**: Split functions into `_Scalar` and `_RGB` versions
   - **Impact**: Eliminated incorrect float3 averaging in single-wavelength rendering
   - **Files**: atmospheric.hlsli (lines 42-86), closesthit.rchit (line 553)

### Verification
Run physics validation tests:
```bash
cd build
cmake --build . --target libquantiloom_tests
ctest --output-on-failure --gtest_filter=BlackbodyPhysicsTest.*
```
All 24 tests should pass ✅
