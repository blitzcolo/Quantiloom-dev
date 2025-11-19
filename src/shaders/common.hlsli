// ============================================================================
// Quantiloom - Common Shader Definitions
// ============================================================================
// Shared types and structures for ray tracing shaders
//
// SPECTRAL RENDERING NOTES:
// - wavelength_nm in CameraData specifies current wavelength (nanometers)
// - For single-wavelength mode: render at one λ, output as grayscale RGB
// - For multi-wavelength mode (M2+): render multiple λ separately
// ============================================================================

#ifndef QUANTILOOM_COMMON_HLSLI
#define QUANTILOOM_COMMON_HLSLI

// ============================================================================
// Ray Payload
// ============================================================================
// Carries radiance information through the ray tracing pipeline
// ============================================================================

struct Payload {
    float3 radiance;  // Accumulated radiance (W·sr⁻¹·m⁻²)
};

// ============================================================================
// LUT Data Structure
// ============================================================================
// Atmospheric lookup table for spectral rendering
// Must match CPU-side LUT data layout
//
// SPECTRAL MODE:
// - sunRadiance_spectral: Spectral radiance at current wavelength (W·sr⁻¹·m⁻²·nm⁻¹)
// - skyRadiance_spectral: Spectral radiance at current wavelength (W·sr⁻¹·m⁻²·nm⁻¹)
// ============================================================================

struct LUTData {
    float3 sunDirection;        // Normalized sun direction vector (FROM surface TO sun)
    float  sunRadiance_spectral; // Sun spectral radiance at current λ
    float  skyRadiance_spectral; // Sky spectral radiance at current λ
    float  _pad0;
    float  _pad1;
    float  _pad2;
};

// ============================================================================
// Camera Data Structure (Push Constants)
// ============================================================================
// Camera parameters for ray generation
// Must match CPU-side CameraData structure
// ============================================================================

struct CameraData {
    float3 origin;         // Camera position (world space)
    float  fovScale;       // tan(fovY / 2)
    float3 forward;        // Forward vector (normalized)
    float  aspectRatio;    // Width / height
    float3 right;          // Right vector (normalized)
    float  wavelength_nm;  // Current wavelength (nanometers) for spectral rendering
    float3 up;             // Up vector (normalized)
    float  _pad1;
};

// ============================================================================
// Material Data Structure (PBR)
// ============================================================================
// Full glTF 2.0 metallic-roughness PBR material
// Must match CPU-side Material structure (see Material.hpp)
//
// Memory layout (std430 / SSBO):
// - Texture indices: -1 = no texture, >=0 = index into texture array
// - Alpha modes: 0 = Opaque, 1 = Mask, 2 = Blend
// - spectralAlbedo: M1 compatibility for single-wavelength rendering
// ============================================================================

struct MaterialData {
    // Base color (PBR albedo)
    float4 baseColorFactor;          // RGBA [0, 1]
    int    baseColorTextureIndex;    // -1 = no texture
    float  metallicFactor;           // [0, 1] (0 = dielectric, 1 = metal)
    float  roughnessFactor;          // [0, 1] (0 = smooth, 1 = rough)
    int    metallicRoughnessTextureIndex; // -1 = no texture (G=roughness, B=metallic)

    // Normal mapping
    int    normalTextureIndex;       // -1 = no normal map
    float  normalScale;              // Normal intensity [0, inf]

    // Emissive
    float3 emissiveFactor;           // RGB [0, inf] (HDR allowed)
    int    emissiveTextureIndex;     // -1 = no texture

    // Alpha blending
    uint   alphaMode;                // 0=Opaque, 1=Mask, 2=Blend
    float  alphaCutoff;              // Threshold for Mask mode [0, 1]

    // Spectral mode (M1 compatibility)
    float  spectralAlbedo;           // Scalar reflectance at current λ [0, 1]
    float  _pad0;                    // Padding to 16-byte alignment
};

#endif // QUANTILOOM_COMMON_HLSLI
