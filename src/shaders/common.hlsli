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
#define SPECTRAL_MODE_RGB_FUSED    1  // RGB rendering with 32-wavelength spectral integration
#define SPECTRAL_MODE_MULTISPECTRAL 2  // Multiple wavelengths (hyperspectral cube) - TBD
#define SPECTRAL_MODE_MWIR_FUSED   3  // Mid-wave IR fusion (3000-5000nm)
#define SPECTRAL_MODE_LWIR_FUSED   4  // Long-wave IR fusion (8000-12000nm)

// ============================================================================
// Ray Payload - OPTIMIZED FOR RT CORE PERFORMANCE
// ============================================================================
// Carries radiance information through the ray tracing pipeline
//
// PERFORMANCE OPTIMIZATION:
// - Payload size CRITICAL for RT Core performance (NVIDIA: prefer ≤32 bytes, max 64 bytes)
// - Original size: 5×float3 = 60 bytes (near register pressure threshold)
// - Optimized size: 2×float3 = 24 bytes (50% reduction)
//
// Ray Differentials for Texture Filtering:
// Used for computing texture LOD (level of detail) for proper filtering
// Tracks how ray direction changes between adjacent pixels
//
// DESIGN TRADE-OFF:
// - Removed dOdx/dOdy (ray origin differentials): -24 bytes
// - Primary rays: origin differentials are typically zero (camera at single point)
// - Secondary rays (reflection/refraction): origin differentials enable better filtering
//   but at significant performance cost. For most real-time IR applications, the quality
//   improvement is not worth the 50% payload increase.
// - If high-quality anisotropic filtering is critical for secondary rays, re-enable dOdx/dOdy
//
// References:
// - "Ray Differentials" in PBRT-v4 §10.1
// - Igehy, "Tracing Ray Differentials" (1999)
// - NVIDIA RTX Best Practices: "Keep ray payloads small" (<32 bytes ideal)
// ============================================================================

struct Payload {
    float3 radiance;  // Accumulated radiance (W·sr⁻¹·m⁻²)              // 12 bytes

    // Ray direction differentials for texture filtering (LOD computation)
    // dDdx: change in ray direction per pixel in X direction
    // dDdy: change in ray direction per pixel in Y direction
    float3 dDdx;  // ∂D/∂x (ray direction differential)                 // 12 bytes
    float3 dDdy;  // ∂D/∂y (ray direction differential)                 // 12 bytes

    // TOTAL: 36 bytes (was 60 bytes)
    // Note: If payload exceeds 32 bytes, consider using min16float for differentials
    // to reduce to 24 bytes total (3×float3 = 36 -> 1×float3 + 2×min16float3 = 12+12 = 24)

    // REMOVED for performance (if needed, can be recomputed or approximated):
    // float3 dOdx;  // ∂O/∂x (ray origin differential) - usually ~0 for primary rays
    // float3 dOdy;  // ∂O/∂y (ray origin differential) - usually ~0 for primary rays
};

// ============================================================================
// Spectral Curve Data Structure (GPU) - OPTIMIZED
// ============================================================================
// Fixed-size spectral curve for wavelength-dependent material properties
// Enables physically-based spectral path tracing with measured reflectance curves
//
// PERFORMANCE OPTIMIZATION:
// This structure uses UNIFORM SAMPLING (equally-spaced wavelengths) to eliminate
// the need for binary search and reduce memory bandwidth by 50%.
//
// KEY DESIGN:
// - wavelengths[] array REMOVED - wavelength computed from: λ = start + index × step
// - Enables O(1) lookup instead of O(log N) binary search
// - Reduces memory: 528 bytes -> 272 bytes per curve (48% reduction)
// - Eliminates GPU branch instructions during spectral queries
// - CPU must resample irregular data to uniform grid before upload
//
// SAMPLING PARAMETERS:
// - startWavelength_nm: Starting wavelength (e.g., 360 nm for UV-visible-IR)
// - stepSize_nm: Wavelength spacing (e.g., 5 nm or 10 nm)
// - numSamples: Number of valid samples (typically 64 for 360-1000 nm @ 10nm step)
//
// EXAMPLE CONFIGURATIONS:
// - Visible spectrum: start=380nm, step=5nm, samples=64 -> covers 380-695nm
// - UV-Vis-NIR: start=360nm, step=10nm, samples=64 -> covers 360-990nm
// - Full IR range: start=360nm, step=20nm, samples=64 -> covers 360-1620nm
//
// SIZE: 64*4 + 4 + 4 + 4 + 4 = 272 bytes per curve (was 528 bytes)
// Must match CPU-side SpectralCurveGPU structure
// ============================================================================

#define MAX_SPECTRAL_SAMPLES 64

struct SpectralCurveGPU {
    float values[MAX_SPECTRAL_SAMPLES];  // Spectral values (reflectance, emissivity, etc.) [0, 1] or [0, inf]
    float startWavelength_nm;            // Starting wavelength (nm)
    float stepSize_nm;                   // Wavelength step size (nm)
    uint  numSamples;                    // Actual number of valid samples (0 to MAX_SPECTRAL_SAMPLES)
    uint  _padding;                      // Padding for 16-byte alignment
};

// ============================================================================
// Helper: Query Spectral Curve at Arbitrary Wavelength (O(1) lookup)
// ============================================================================
// Performs fast linear interpolation on uniformly-sampled spectral curve
// Returns: Interpolated spectral value at query_wavelength_nm
// ============================================================================

float SampleSpectralCurve(SpectralCurveGPU curve, float query_wavelength_nm) {
    // Handle empty curve
    if (curve.numSamples == 0) {
        return 0.0;
    }

    // Compute fractional index: (λ - λ₀) / Δλ
    float index_f = (query_wavelength_nm - curve.startWavelength_nm) / curve.stepSize_nm;

    // Clamp to valid range [0, numSamples-1]
    if (index_f < 0.0) {
        return curve.values[0];  // Below range: use first value
    }

    if (index_f >= float(curve.numSamples - 1)) {
        return curve.values[curve.numSamples - 1];  // Above range: use last value
    }

    // Linear interpolation between adjacent samples
    uint  index0 = uint(floor(index_f));
    uint  index1 = index0 + 1;
    float t = frac(index_f);  // Fractional part for interpolation

    return lerp(curve.values[index0], curve.values[index1], t);
}

// ============================================================================
// Complex Refractive Index Data Structure (GPU)
// ============================================================================
// Fixed-size complex refractive index N = n + ik for Fresnel calculations
// Data source: RefractiveIndex.INFO database
//
// PHYSICS:
// - n: Refractive index (determines phase velocity and Snell's refraction)
// - k: Extinction coefficient (determines absorption)
// - For metals: k >> 0 (high absorption)
// - For dielectrics: k ≈ 0 (transparent in certain bands)
//
// SIZE: 64*4 (n) + 64*4 (k) + 4 + 4 + 4 + 4 = 528 bytes per curve
// Must match CPU-side ComplexRefractiveIndexGPU structure
// ============================================================================

struct ComplexRefractiveIndexGPU {
    float n[MAX_SPECTRAL_SAMPLES];       // Refractive index at uniform grid
    float k[MAX_SPECTRAL_SAMPLES];       // Extinction coefficient at uniform grid
    float startWavelength_nm;            // Starting wavelength (nm)
    float stepSize_nm;                   // Wavelength step size (nm)
    uint  numSamples;                    // Number of valid samples
    uint  _padding;                      // 16-byte alignment
};

// ============================================================================
// Helper: Query Complex Refractive Index at Arbitrary Wavelength
// ============================================================================
// Returns: float2(n, k) - refractive index and extinction coefficient
// ============================================================================

float2 SampleComplexRefractiveIndex(ComplexRefractiveIndexGPU cri, float query_wavelength_nm) {
    // Handle empty data
    if (cri.numSamples == 0) {
        return float2(1.0, 0.0);  // Default: air
    }

    // Compute fractional index
    float index_f = (query_wavelength_nm - cri.startWavelength_nm) / cri.stepSize_nm;

    // Clamp to valid range
    if (index_f < 0.0) {
        return float2(cri.n[0], cri.k[0]);
    }

    if (index_f >= float(cri.numSamples - 1)) {
        return float2(cri.n[cri.numSamples - 1], cri.k[cri.numSamples - 1]);
    }

    // Linear interpolation
    uint  i0 = uint(floor(index_f));
    uint  i1 = i0 + 1;
    float t = frac(index_f);

    return float2(
        lerp(cri.n[i0], cri.n[i1], t),
        lerp(cri.k[i0], cri.k[i1], t)
    );
}

// ============================================================================
// Fresnel Equations for Complex Refractive Index (Conductor)
// ============================================================================
// Computes Fresnel reflectance for materials with complex refractive index
// (metals, semiconductors) at arbitrary incidence angle.
//
// PHYSICS:
// For complex N = n + ik, the Fresnel equations become:
//   Rs = |((n1*cos_i - n2*cos_t)/(n1*cos_i + n2*cos_t))|^2
//   Rp = |((n1*cos_t - n2*cos_i)/(n1*cos_t + n2*cos_i))|^2
//   R = (Rs + Rp) / 2 (unpolarized light)
//
// For conductors (metals), we use the simplified formulation from:
// "An Inexpensive BRDF Model for Physically-based Rendering" - Schlick 1994
// Combined with conductor Fresnel from PBRT-v4.
//
// Input:
//   cosTheta: cos(incident angle), dot(N, V) or dot(N, L)
//   n: real part of refractive index (at query wavelength)
//   k: imaginary part (extinction coefficient)
//
// Returns:
//   Fresnel reflectance [0, 1]
// ============================================================================

float FresnelConductor(float cosTheta, float n, float k) {
    // Clamp cosTheta to avoid numerical issues
    cosTheta = saturate(abs(cosTheta));

    float cos2 = cosTheta * cosTheta;
    float sin2 = 1.0 - cos2;

    float n2 = n * n;
    float k2 = k * k;

    float t0 = n2 - k2 - sin2;
    float a2b2 = sqrt(t0 * t0 + 4.0 * n2 * k2);
    float t1 = a2b2 + cos2;
    float a = sqrt(0.5 * (a2b2 + t0));
    float t2 = 2.0 * a * cosTheta;
    float Rs = (t1 - t2) / (t1 + t2);

    float t3 = cos2 * a2b2 + sin2 * sin2;
    float t4 = t2 * sin2;
    float Rp = Rs * (t3 - t4) / (t3 + t4);

    return 0.5 * (Rs + Rp);
}

// ============================================================================
// Fresnel Reflectance at Normal Incidence (F0)
// ============================================================================
// Simplified Fresnel for perpendicular incidence (θ = 0)
// Used for PBR F0 computation
//
// Formula: F0 = [(n-1)² + k²] / [(n+1)² + k²]
//
// Input:
//   n: refractive index
//   k: extinction coefficient
//
// Returns:
//   F0 (normal incidence reflectance) [0, 1]
// ============================================================================

float FresnelF0(float n, float k) {
    float numerator = (n - 1.0) * (n - 1.0) + k * k;
    float denominator = (n + 1.0) * (n + 1.0) + k * k;
    return numerator / denominator;
}

// ============================================================================
// Schlick Fresnel Approximation with Spectral F0
// ============================================================================
// Standard Schlick approximation using computed F0 from n,k
//
// Formula: F = F0 + (1 - F0) * (1 - cosTheta)^5
//
// Input:
//   cosTheta: cos(incident angle)
//   F0: normal incidence reflectance (from FresnelF0)
//
// Returns:
//   Fresnel reflectance [0, 1]
// ============================================================================

float FresnelSchlick(float cosTheta, float F0) {
    float oneMinusCos = 1.0 - saturate(cosTheta);
    float oneMinusCos5 = oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos;
    return F0 + (1.0 - F0) * oneMinusCos5;
}

// RGB version for standard PBR
float3 FresnelSchlickRGB(float cosTheta, float3 F0) {
    float oneMinusCos = 1.0 - saturate(cosTheta);
    float oneMinusCos5 = oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos;
    return F0 + (1.0 - F0) * oneMinusCos5;
}

// ============================================================================
// LUT Data Structure
// ============================================================================
// Atmospheric lookup table for spectral and RGB rendering
// Must match CPU-side LUT data layout
//
// DUAL MODE SUPPORT:
// - RGB mode: Use sunRadiance_rgb and skyRadiance_rgb (float3)
// - Spectral mode: Use sunRadiance_spectral and skyRadiance_spectral (float)
//
// WORLD UNITS:
// - worldUnitsToMeters: Conversion factor from scene units to meters
// - Required for correct Beer-Lambert attenuation calculation
// - Example: if scene uses centimeters, worldUnitsToMeters = 0.01
// ============================================================================

struct LUTData {
    float3 sunDirection;         // Normalized sun direction vector (FROM surface TO sun)
    float  sunRadiance_spectral; // Sun spectral radiance at current λ (W·sr⁻¹·m⁻²·nm⁻¹)

    float3 sunRadiance_rgb;      // Sun RGB radiance (W·sr⁻¹·m⁻²) for RGB mode
    float  skyRadiance_spectral; // Sky spectral radiance at current λ (W·sr⁻¹·m⁻²·nm⁻¹)

    float3 skyRadiance_rgb;      // Sky RGB radiance (W·sr⁻¹·m⁻²) for RGB mode
    float  transmittance;        // Atmospheric transmittance τ(λ) [0, 1] (vertical path)

    float  worldUnitsToMeters;   // Conversion factor: world_units × this = meters
    float3 _padding;             // Padding for 16-byte alignment
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
    float4 baseColorFactor;          // RGBA [0, 1]                          // Offset: 0-16
    int    baseColorTextureIndex;    // -1 = no texture                      // Offset: 16-20
    float  metallicFactor;           // [0, 1] (0 = dielectric, 1 = metal)  // Offset: 20-24
    float  roughnessFactor;          // [0, 1] (0 = smooth, 1 = rough)      // Offset: 24-28
    int    metallicRoughnessTextureIndex; // -1 = no texture (G=roughness, B=metallic) // Offset: 28-32

    // Normal mapping
    int    normalTextureIndex;       // -1 = no normal map                   // Offset: 32-36
    float  normalScale;              // Normal intensity [0, inf]            // Offset: 36-40
    float2 _padding0;                // Explicit padding to align emissiveFactor to 16-byte boundary // Offset: 40-48

    // Emissive (now 16-byte aligned at offset 48)
    float3 emissiveFactor;           // RGB [0, inf] (HDR allowed)           // Offset: 48-60
    int    emissiveTextureIndex;     // -1 = no texture                      // Offset: 60-64

    // Alpha blending
    uint   alphaMode;                // 0=Opaque, 1=Mask, 2=Blend            // Offset: 64-68
    float  alphaCutoff;              // Threshold for Mask mode [0, 1]      // Offset: 68-72

    // Spectral mode (M1 compatibility and M2+ full spectral)
    float  spectralAlbedo;           // LEGACY: Scalar reflectance at current λ [0, 1] (M1 fallback) // Offset: 72-76
    int    spectralReflectanceCurveIndex;  // Index into spectralCurves buffer (-1 = use spectralAlbedo) // Offset: 76-80

    // Infrared material properties (evaluated at current wavelength)
    // NOTE: Based on energy conservation: α + ρ + τ = 1, and Kirchhoff's law: ε = α
    // Therefore: ε + ρ + τ = 1  =>  ρ = 1 - ε - τ
    // We store ε and τ explicitly for flexibility, ρ can be derived if needed
    float  irEmissivity;             // IR emissivity ε(λ) [0, 1]            // Offset: 80-84
    float  irTransmittance;          // IR transmittance τ(λ) [0, 1]        // Offset: 84-88
    float  irTemperature_K;          // IR surface temperature (K) for blackbody emission (0 = no emission) // Offset: 88-92

    // Complex refractive index for specular Fresnel (physical metals)
    // When >= 0, uses measured n,k data from RefractiveIndex.INFO for accurate Fresnel
    // When < 0, uses metallicFactor-based F0 approximation (standard PBR)
    int    complexRefractiveIndexIndex; // Index into complexRefractiveIndex buffer (-1 = use PBR approximation) // Offset: 92-96

    // Note: irReflectance removed - can be computed as: 1.0 - irEmissivity - irTransmittance
    // For opaque materials: irTransmittance = 0, so irReflectance = 1.0 - irEmissivity
};

// ============================================================================
// Helper: Compute IR Reflectance from Energy Conservation
// ============================================================================
// Computes IR reflectance ρ(λ) from emissivity and transmittance using
// energy conservation and Kirchhoff's law.
//
// Physical principles:
//   1. Energy conservation: α + ρ + τ = 1 (absorptance + reflectance + transmittance)
//   2. Kirchhoff's law: ε = α (emissivity equals absorptance at thermal equilibrium)
//   3. Therefore: ε + ρ + τ = 1  =>  ρ = 1 - ε - τ
//
// Input:
//   mat: MaterialData with irEmissivity and irTransmittance
//
// Returns:
//   IR reflectance ρ(λ) [0, 1]
//
// Notes:
//   - For opaque materials (τ = 0): ρ = 1 - ε
//   - For transparent materials: all three components contribute
//   - Result is clamped to [0, 1] to handle potential numerical errors
// ============================================================================

float GetIRReflectance(MaterialData mat) {
    return saturate(1.0 - mat.irEmissivity - mat.irTransmittance);
}

// ============================================================================
// Helper: Validate Energy Conservation for IR Material
// ============================================================================
// Checks if IR material properties satisfy energy conservation: ε + ρ + τ ≤ 1
// Useful for debugging material data on GPU
//
// Returns:
//   true if material is physically valid, false otherwise
// ============================================================================

bool IsIRMaterialValid(MaterialData mat) {
    float sum = mat.irEmissivity + GetIRReflectance(mat) + mat.irTransmittance;
    return (sum >= 0.0 && sum <= 1.001);  // Allow small numerical tolerance
}

#endif // QUANTILOOM_COMMON_HLSLI
