// ============================================================================
// Quantiloom M1 - Common Shader Definitions
// ============================================================================
// Shared types and structures for ray tracing shaders
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
// Simplified atmospheric lookup table for M1
// Must match CPU-side LUT data layout
// ============================================================================

struct LUTData {
    float3 sunDirection;   // Normalized sun direction vector
    float  _pad0;
    float3 sunRadiance;    // Direct sun radiance (W·sr⁻¹·m⁻²)
    float  _pad1;
    float3 skyRadiance;    // Hemispherical sky radiance (W·sr⁻¹·m⁻²)
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
    float  _pad0;
    float3 up;             // Up vector (normalized)
    float  _pad1;
};

// ============================================================================
// Material Data Structure
// ============================================================================
// Surface material properties
// Must match CPU-side Material structure (see Material.hpp)
// ============================================================================

struct MaterialData {
    float3 albedo;   // Diffuse reflectance [0, 1]
    float  _pad0;    // Align to 16 bytes (vec4 boundary)
};

#endif // QUANTILOOM_COMMON_HLSLI
