# Quantiloom Shaders

This directory contains HLSL ray tracing shaders for Quantiloom.

## Shader Files

### Ray Tracing Pipeline
| File | Type | Description |
|------|------|-------------|
| `raygen.rgen` | Ray Generation | Primary ray casting with camera transform |
| `closesthit.rchit` | Closest Hit | PBR shading with spectral rendering |
| `miss.rmiss` | Miss | Sky/atmosphere radiance computation |

### Header Files
| File | Description |
|------|-------------|
| `common.hlsli` | Shared types, spectral modes, payload structures |
| `pbr.hlsli` | Cook-Torrance PBR functions (GGX, Fresnel) |
| `blackbody.hlsli` | Planck blackbody radiation for thermal IR (C1_NM = 1.191×10²⁰) |
| `atmosphere_nn.hlsli` | NN atmosphere LUT sampling (MODTRAN surrogate tau / L_path / L_down) |
| `SpectralConversion.hlsli` | CIE XYZ color matching, spectral upsampling |
| `spectral_query.hlsli` | Spectral curve sampling utilities |

### IBL Compute Shaders
| File | Description |
|------|-------------|
| `ibl_brdf_lut.comp` | BRDF integration LUT generation |
| `ibl_equirect_to_cube.comp` | Equirectangular to cubemap conversion |
| `ibl_prefilter_env.comp` | Environment map prefiltering |

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
| 2 | StructuredBuffer | ClosestHit, Miss | LightingParams (sun/sky) |
| 3 | StructuredBuffer<float3> | ClosestHit | Vertex buffer (positions) |
| 4 | StructuredBuffer<uint> | ClosestHit | Index buffer (triangle indices) |
| 5 | StructuredBuffer<MaterialData> | ClosestHit | Material properties |
| 6 | Texture2D[] | ClosestHit | Bindless texture array |
| 7 | SamplerState[] | ClosestHit | Bindless sampler array |
| 8 | StructuredBuffer<float2> | ClosestHit | UV coordinates |
| 9 | StructuredBuffer<float4> | ClosestHit | Tangent vectors |
| 10-12 | - | ClosestHit | IBL resources (cubemap, BRDF LUT) |
| 13 | StructuredBuffer<SpectralCurveGPU> | ClosestHit | Spectral reflectance curves |
| 14 | StructuredBuffer<ComplexRefractiveIndexGPU> | ClosestHit | Complex refractive index (n,k) |
| 15 | StructuredBuffer<SolarSpectralLUT> | ClosestHit | Solar spectral irradiance |

## Payload Structure

```hlsl
struct Payload {
    float3 radiance;  // Accumulated radiance (W·sr⁻¹·m⁻²)
    float3 dDdx;      // Ray direction differential (∂D/∂x)
    float3 dDdy;      // Ray direction differential (∂D/∂y)
};
// Total: 36 bytes (optimized from 60 bytes)
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
