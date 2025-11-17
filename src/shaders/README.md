# Quantiloom Shaders

This directory contains HLSL ray tracing shaders for Quantiloom M1.

## Compilation

Shaders must be compiled to SPIR-V using **DXC** (DirectX Shader Compiler).

### Install DXC

- **Windows**: Download from [Vulkan SDK](https://vulkan.lunarg.com/)
- **Linux**: `sudo apt install dxc` or build from [microsoft/DirectXShaderCompiler](https://github.com/microsoft/DirectXShaderCompiler)
- **macOS**: `brew install dxc`

### Compile Shaders

Run the following commands from this directory:

```bash
# Ray Generation
dxc -spirv -T lib_6_3 -fspv-target-env=vulkan1.3 -Fo raygen.spv raygen.rgen

# Closest Hit
dxc -spirv -T lib_6_3 -fspv-target-env=vulkan1.3 -Fo closesthit.spv closesthit.rchit

# Miss
dxc -spirv -T lib_6_3 -fspv-target-env=vulkan1.3 -Fo miss.spv miss.rmiss
```

### Output

After compilation, you should have:
- `raygen.spv`
- `closesthit.spv`
- `miss.spv`

These SPIR-V files are loaded by the `RayTracingPipeline` class at runtime.

## Shader Layout

### Descriptor Set 0

| Binding | Type | Stage | Description |
|---------|------|-------|-------------|
| 0 | RWTexture2D | Raygen | Output image (RGBA32F) |
| 1 | AccelerationStructure | Raygen | TLAS (scene) |
| 2 | StructuredBuffer | ClosestHit, Miss | LUT data (sun/sky) |

### Payload

```hlsl
struct Payload {
    float3 radiance;  // Accumulated radiance (W·sr⁻¹·m⁻²)
};
```

## M1 Simplifications

- **No recursion**: `maxRayRecursionDepth = 1`
- **No shadow rays**: Direct lighting only
- **Hardcoded camera**: Pinhole camera at origin
- **Hardcoded material**: Gray Lambert (albedo = 0.8)
- **Single LUT entry**: No wavelength or angle dependence

These will be expanded in M2+.
