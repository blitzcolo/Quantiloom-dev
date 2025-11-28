// ============================================================================
// Quantiloom - Common Shader Definitions
// ============================================================================
// Shared types and structures for ray tracing shaders
//
// SPECTRAL RENDERING NOTES:
// - wavelength_nm in CameraData specifies current wavelength (nanometers)
// - spectral_mode determines rendering pipeline: single, RGB, MWIR, LWIR, etc.
// - For multi-wavelength mode (M2+): render multiple λ separately
// ============================================================================

#ifndef QUANTILOOM_COMMON_HLSLI
#define QUANTILOOM_COMMON_HLSLI

// ============================================================================
// Spectral Rendering Modes
// ============================================================================
// Defines different spectral rendering pipelines supported by Quantiloom
// ============================================================================

// IMPORTANT: These values MUST match SpectralMode enum in C++ code!
#define SPECTRAL_MODE_SINGLE       0  // Single wavelength (grayscale output)
#define SPECTRAL_MODE_RGB          1  // RGB rendering with spectral conversions
#define SPECTRAL_MODE_MULTISPECTRAL 2  // Multiple wavelengths (hyperspectral cube) - TBD
#define SPECTRAL_MODE_MWIR_FUSED   3  // Mid-wave IR fusion (3000-5000nm)
#define SPECTRAL_MODE_LWIR_FUSED   4  // Long-wave IR fusion (8000-12000nm)

// ============================================================================
// Ray Payload
// ============================================================================
// Carries radiance information through the ray tracing pipeline
//
// Ray Differentials:
// Used for computing texture LOD (level of detail) for proper filtering
// Tracks how ray direction changes between adjacent pixels
//
// References:
// - "Ray Differentials" in PBRT-v4 §10.1
// - Igehy, "Tracing Ray Differentials" (1999)
// ============================================================================

struct Payload {
    float3 radiance;  // Accumulated radiance (W·sr⁻¹·m⁻²)

    // Ray differentials for texture filtering
    // dDdx: change in ray direction per pixel in X direction
    // dDdy: change in ray direction per pixel in Y direction
    float3 dDdx;  // ∂D/∂x (ray direction differential)
    float3 dDdy;  // ∂D/∂y (ray direction differential)

    // Ray origin differentials (for perspective projection)
    float3 dOdx;  // ∂O/∂x (ray origin differential)
    float3 dOdy;  // ∂O/∂y (ray origin differential)
};

// ============================================================================
// Spectral Curve Data Structure (GPU)
// ============================================================================
// Fixed-size spectral curve for wavelength-dependent material properties
// Enables physically-based spectral path tracing with measured reflectance curves
//
// DESIGN:
// - Fixed-size array (64 samples) for efficient GPU memory layout
// - Supports linear interpolation for continuous wavelength queries
// - Must match CPU-side SpectralCurveGPU structure
//
// SIZE: 64*4 + 64*4 + 4 + 12 = 528 bytes per curve
// ============================================================================

#define MAX_SPECTRAL_SAMPLES 64

struct SpectralCurveGPU {
    float wavelengths[MAX_SPECTRAL_SAMPLES];  // Wavelength in nm (monotonic increasing)
    float values[MAX_SPECTRAL_SAMPLES];       // Spectral values (reflectance, emissivity, etc.)
    uint  numSamples;                         // Actual number of valid samples (0 to MAX_SPECTRAL_SAMPLES)
    uint  _padding[3];                        // Padding for 16-byte alignment
};

// ============================================================================
// LUT Data Structure
// ============================================================================
// Atmospheric lookup table for spectral and RGB rendering
// Must match CPU-side LUT data layout
//
// DUAL MODE SUPPORT:
// - RGB mode: Use sunRadiance_rgb and skyRadiance_rgb (float3)
// - Spectral mode: Use sunRadiance_spectral and skyRadiance_spectral (float)
// ============================================================================

struct LUTData {
    float3 sunDirection;         // Normalized sun direction vector (FROM surface TO sun)
    float  sunRadiance_spectral; // Sun spectral radiance at current λ (W·sr⁻¹·m⁻²·nm⁻¹)

    float3 sunRadiance_rgb;      // Sun RGB radiance (W·sr⁻¹·m⁻²) for RGB mode
    float  skyRadiance_spectral; // Sky spectral radiance at current λ (W·sr⁻¹·m⁻²·nm⁻¹)

    float3 skyRadiance_rgb;      // Sky RGB radiance (W·sr⁻¹·m⁻²) for RGB mode
    float  transmittance;        // Atmospheric transmittance τ(λ) [0, 1] (vertical path)
};

// ============================================================================
// Camera Data Structure
// ============================================================================
// Camera parameters for ray generation
// Must match CPU-side CameraData structure in Camera.hpp
// ============================================================================

struct CameraData {
    float3 origin;         // Camera position (world space)
    float  fovScale;       // tan(fovY / 2)
    float3 forward;        // Forward vector (normalized)
    float  aspectRatio;    // Width / height
    float3 right;          // Right vector (normalized)
    float  wavelength_nm;  // Current wavelength (nanometers) for spectral rendering
    float3 up;             // Up vector (normalized)
    uint   spectral_mode;  // Spectral rendering mode (see SPECTRAL_MODE_* defines)
};

// ============================================================================
// Push Constants for Ray Generation Shader
// ============================================================================
// Extended camera data with accumulation sampling parameters
// Must match CPU-side PushConstantsRayGen structure in Camera.hpp
//
// ACCUMULATION SAMPLING:
// - frameIndex: Frame counter for temporal effects (animation, motion blur)
// - sampleIndex: Current sample index (0 to totalSamples-1) for this pixel
// - totalSamples: Total samples per pixel (spp) for averaging
// - randomSeed: Random seed for this frame/sample (ensures unique jitter)
//
// This structure enables:
// 1. Multi-sample anti-aliasing (MSAA) via subpixel jittering
// 2. Monte Carlo path tracing convergence via multiple samples
// 3. Progressive refinement (each sample improves image quality)
// 4. Future extensions: temporal anti-aliasing, motion blur, etc.
// ============================================================================
//
// NOTE: This structure is defined in raygen.rgen as it is specific to that shader.
// Other shaders (closest hit, miss) do not receive these push constants.
// The definition here serves as documentation for cross-reference with C++ code.

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

    // Spectral mode (M1 compatibility and M2+ full spectral)
    float  spectralAlbedo;           // LEGACY: Scalar reflectance at current λ [0, 1] (M1 fallback)
    int    spectralReflectanceCurveIndex;  // Index into spectralCurves buffer (-1 = use spectralAlbedo)

    // Infrared material properties (evaluated at current wavelength)
    float  irEmissivity;             // IR emissivity ε(λ) [0, 1]
    float  irReflectance;            // IR reflectance ρ(λ) [0, 1]
    float  irTransmittance;          // IR transmittance τ(λ) [0, 1]
    float  irTemperature_K;          // IR surface temperature (K) for blackbody emission (0 = no emission)
};

#endif // QUANTILOOM_COMMON_HLSLI
