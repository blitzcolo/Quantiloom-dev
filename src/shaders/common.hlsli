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
// Material Data Structure
// ============================================================================
// Surface material properties for spectral rendering
// Must match CPU-side Material structure (see Material.hpp)
//
// SPECTRAL MODE:
// - albedo_spectral: Spectral reflectance at current wavelength [0, 1]
// ============================================================================

struct MaterialData {
    float albedo_spectral;  // Spectral reflectance at current λ [0, 1]
    float _pad0;
    float _pad1;
    float _pad2;
};

#endif // QUANTILOOM_COMMON_HLSLI
