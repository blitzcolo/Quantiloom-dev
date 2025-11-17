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

#endif // QUANTILOOM_COMMON_HLSLI
