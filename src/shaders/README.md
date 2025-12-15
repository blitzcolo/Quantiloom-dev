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
| `blackbody.hlsli` | Planck blackbody radiation for thermal IR |
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

| Mode | Value | Description |
|------|-------|-------------|
| `SPECTRAL_MODE_SINGLE` | 0 | Single wavelength (grayscale) |
| `SPECTRAL_MODE_RGB_FUSED` | 1 | 32-wavelength spectral integration |
| `SPECTRAL_MODE_MULTISPECTRAL` | 2 | Hyperspectral cube (TBD) |
| `SPECTRAL_MODE_MWIR_FUSED` | 3 | Mid-wave IR (3000-5000nm) |
| `SPECTRAL_MODE_LWIR_FUSED` | 4 | Long-wave IR (8000-12000nm) |
| `SPECTRAL_MODE_SWIR_FUSED` | 5 | Short-wave IR (1000-2500nm) |

## Current Features

- **Cook-Torrance PBR**: Full GGX specular with height-correlated Smith geometry
- **Spectral rendering**: Physically-based spectral upsampling and XYZ integration
- **IR thermal emission**: Planck blackbody radiation for MWIR/LWIR/SWIR modes
- **Measured material data**: Spectral reflectance curves and complex refractive indices
- **IBL**: Environment map lighting with prefiltered specular
- **Atmospheric model**: Beer-Lambert attenuation with MODTRAN-compatible LUTs
- **Ray differentials**: Texture LOD computation for proper filtering
