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
#define SPECTRAL_MODE_VIS_FUSED    1  // Visible spectral integration (32-wavelength CIE XYZ)
#define SPECTRAL_MODE_MULTISPECTRAL 2  // Multiple wavelengths (hyperspectral cube) - TBD
#define SPECTRAL_MODE_MWIR_FUSED   3  // Mid-wave IR fusion
#define SPECTRAL_MODE_LWIR_FUSED   4  // Long-wave IR fusion
#define SPECTRAL_MODE_SWIR_FUSED   5  // Short-wave IR fusion
#define SPECTRAL_MODE_NIR_FUSED    6  // Near IR fusion - reflected solar
#define SPECTRAL_MODE_RGB          7  // Fast RGB-only (no spectral integration, default)

// ============================================================================
// Fused-Mode Integration Bands
// ============================================================================
// The wavelength range each *_FUSED mode integrates over. These are the single
// definition for HLSL -- closesthit.rchit and miss.rmiss use them rather than
// repeating literals, which is how the ranges drifted out of sync before.
//
// MUST match GetFusedBandInfo() in core/Types.hpp, which is the authority.
// TypesTest.ShaderBandConstantsMatchGetFusedBandInfo reads this file and
// asserts the two agree, so drift fails the suite rather than the render.
//
// Do NOT confuse these with the WAVELENGTH_{MIN,MAX}_* constants in Types.hpp:
// those are the ISO band taxonomy (e.g. NIR = 780-1400nm) and are not what the
// renderer integrates. NIR_FUSED in particular is narrowed to 930-1200nm to
// stay inside the NN atmosphere's trained coverage.
// ============================================================================

#define SPECTRAL_VIS_LAMBDA_MIN     400.0
#define SPECTRAL_VIS_LAMBDA_MAX     780.0
#define SPECTRAL_NIR_LAMBDA_MIN     930.0
#define SPECTRAL_NIR_LAMBDA_MAX    1200.0
#define SPECTRAL_SWIR_LAMBDA_MIN   1400.0
#define SPECTRAL_SWIR_LAMBDA_MAX   2400.0
#define SPECTRAL_MWIR_LAMBDA_MIN   3000.0
#define SPECTRAL_MWIR_LAMBDA_MAX   5000.0
#define SPECTRAL_LWIR_LAMBDA_MIN   8000.0
#define SPECTRAL_LWIR_LAMBDA_MAX  12000.0

// Width of the visible band, which is 1/pdf for a uniformly sampled hero
// wavelength. Named because it appears once when λ_h is drawn and once when the
// result is weighted, and the two must agree.
#define SPECTRAL_VIS_BANDWIDTH (SPECTRAL_VIS_LAMBDA_MAX - SPECTRAL_VIS_LAMBDA_MIN)


// ============================================================================
// Debug Visualization Modes
// ============================================================================
// IMPORTANT: These values MUST match DebugVisualizationMode enum in C++ code!
// ============================================================================

#define DEBUG_MODE_NONE                    0   // Normal rendering

// Geometry (1-9)
#define DEBUG_MODE_WORLD_POSITION          1   // Hit point world coordinates
#define DEBUG_MODE_GEOMETRIC_NORMAL        2   // Raw geometric normal
#define DEBUG_MODE_SHADED_NORMAL           3   // Final shaded normal (with normal map)
#define DEBUG_MODE_TANGENT                 4   // Tangent vector
#define DEBUG_MODE_UV                      5   // Texture coordinates
#define DEBUG_MODE_MATERIAL_ID             6   // Material index (hashed to color)
#define DEBUG_MODE_TRIANGLE_ID             7   // Triangle index (hashed to color)
#define DEBUG_MODE_BARYCENTRIC             8   // Barycentric coordinates

// Material (10-19)
#define DEBUG_MODE_BASE_COLOR              10  // Albedo RGB
#define DEBUG_MODE_METALLIC                11  // Metallic factor
#define DEBUG_MODE_ROUGHNESS               12  // Roughness factor
#define DEBUG_MODE_NORMAL_MAP_DELTA        13  // Normal map contribution
#define DEBUG_MODE_EMISSIVE                14  // Emissive RGB
#define DEBUG_MODE_ALPHA                   15  // Alpha channel

// Lighting (20-29)
#define DEBUG_MODE_NDOTL                   20  // Dot(Normal, LightDir)
#define DEBUG_MODE_NDOTV                   21  // Dot(Normal, ViewDir)
#define DEBUG_MODE_DIRECT_SUN              22  // Direct sun lighting
#define DEBUG_MODE_DIFFUSE                 23  // Diffuse component (kD * albedo)
#define DEBUG_MODE_ATMOSPHERIC_TRANS       24  // Atmospheric transmittance

// BRDF (30-39)
#define DEBUG_MODE_FRESNEL_F0              30  // Fresnel at normal incidence
#define DEBUG_MODE_FRESNEL                 31  // Fresnel at current angle
#define DEBUG_MODE_BRDF                    32  // Full Cook-Torrance BRDF
#define DEBUG_MODE_SPECULAR_D              33  // GGX distribution term
#define DEBUG_MODE_SPECULAR_G              34  // Geometry/masking term

// IBL (40-49)
#define DEBUG_MODE_REFLECTION_DIR          40  // Reflection direction
#define DEBUG_MODE_PREFILTERED_ENV         41  // Prefiltered environment sample
#define DEBUG_MODE_BRDF_LUT                42  // BRDF LUT sample
#define DEBUG_MODE_IBL_SPECULAR            43  // IBL specular contribution
#define DEBUG_MODE_SKY_AMBIENT             44  // Sky ambient contribution

// Spectral (50-59)
#define DEBUG_MODE_XYZ                     50  // CIE XYZ tristimulus values
#define DEBUG_MODE_BEFORE_CHROMA           51  // RGB before chromaticity correction
#define DEBUG_MODE_SPECTRAL_REFL           52  // Spectral reflectance at 550nm

// IR (60-69)
#define DEBUG_MODE_TEMPERATURE             60  // Surface temperature (colormap)
#define DEBUG_MODE_IR_EMISSIVITY           61  // IR emissivity
#define DEBUG_MODE_IR_EMISSION             62  // Thermal emission component
#define DEBUG_MODE_IR_REFLECTION           63  // IR reflection component

// Geometry Diagnostics (70-79) - For debugging mesh/index corruption
#define DEBUG_MODE_VERTEX_POSITIONS        70  // Hash of 3 vertex positions (R=v0, G=v1, B=v2)
#define DEBUG_MODE_INDEX_VALUES            71  // Triangle indices as colors (normalized by 32)
#define DEBUG_MODE_INSTANCE_ID             72  // Instance index (hashed to color)
#define DEBUG_MODE_PRIMITIVE_ID            73  // PrimitiveIndex() value (R=id/12 for cube)
#define DEBUG_MODE_INDEX_BUFFER_POS        74  // Index buffer read position (debug offset calc)
#define DEBUG_MODE_V0_POSITION             75  // v0 vertex position directly (frac of xyz)
#define DEBUG_MODE_RAW_IDX0                76  // Raw idx0 value and read address
#define DEBUG_MODE_V0_RAW                  77  // v0 position clamped (not frac)

// Transmission (80-89) - For debugging transmission materials
#define DEBUG_MODE_TRANSMISSION            80  // Transmission factor
#define DEBUG_MODE_IOR                     81  // Index of refraction (normalized)
#define DEBUG_MODE_FRESNEL_DIELECTRIC      82  // Dielectric Fresnel reflectance
#define DEBUG_MODE_ATTENUATION             83  // Volume attenuation color
#define DEBUG_MODE_ENERGY_AUDIT            84  // Energy conservation (R=reflected, G=transmitted, B=absorbed)
#define DEBUG_MODE_DISPERSION              85  // Dispersion coefficient visualization
#define DEBUG_MODE_IOR_VARIATION           86  // IOR variation across RGB wavelengths

// Volume/Participating Media (90-99) - For debugging fog, smoke, SSS
#define DEBUG_MODE_VOLUME_DENSITY          90  // Volume density (grayscale)
#define DEBUG_MODE_SCATTERING_COEFF        91  // Scattering coefficient σ_s
#define DEBUG_MODE_ABSORPTION_COEFF        92  // Absorption coefficient σ_a
#define DEBUG_MODE_EXTINCTION_COEFF        93  // Extinction coefficient σ_t = σ_s + σ_a
#define DEBUG_MODE_SINGLE_SCATTER_ALBEDO   94  // Single scattering albedo ω = σ_s / σ_t
#define DEBUG_MODE_PHASE_G                 95  // Henyey-Greenstein g parameter

// ============================================================================
// Specialization Constants
// ============================================================================
// Baked at pipeline creation time to eliminate dead spectral/debug branches.
// Defaults match SPECTRAL_MODE_RGB / debug off for safe fallback.
// ============================================================================

[[vk::constant_id(0)]] const uint SPEC_SPECTRAL_MODE = 7;   // SPECTRAL_MODE_RGB
[[vk::constant_id(1)]] const uint SPEC_DEBUG_ENABLED = 0;   // 0 = off

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

    // Shadow ray result (set by shadow miss shader)
    // 0 = not shadowed (ray reached light), 1 = shadowed (ray hit occluder)
    uint isShadowed;  // Shadow occlusion flag                          // 4 bytes

    // Recursion depth for transmission/reflection rays
    // 0 = primary ray, incremented for each bounce
    // Used to limit recursion and for Russian roulette termination
    uint depth;       // Current recursion depth                         // 4 bytes

    // Random number generator state for stochastic decisions
    // Used for Russian roulette termination and importance sampling
    uint rngState;    // PCG hash state                                  // 4 bytes

    // Hero wavelength, in nm, or 0 for "this ray carries the whole band".
    //
    // Non-zero only past a dispersive refraction. n(λ) sends each wavelength
    // somewhere different, so a ray that has been bent can only be accountable
    // for the one wavelength it was bent for -- there is no single path the
    // band shares any more. The refracting surface samples λ_h uniformly and
    // undoes the pdf when it converts the result back to a colour.
    //
    // "Dispersive" is about the medium, not about glass: anything with n
    // meaningfully above 1 disperses, and the Abbe numbers are comparable --
    // water 1.333/55.7, ice, acrylic, quartz, gemstones. The gate below keys on
    // whether a material carries an Abbe number or an n(λ) table, not on what
    // it is called, so a water surface reaches this path the same way a prism
    // does. That is also why this is the mechanism participating media will need
    // when they arrive.
    //
    // A ray with heroLambda != 0 reports SCALAR spectral radiance in
    // `radiance`, not RGB. The two never mix: only the VIS_FUSED paths read
    // this, and only the surface that sampled λ_h converts back.
    float heroLambda;  // nm, 0 = whole band                             // 4 bytes

    // TOTAL: 52 bytes (under 64-byte RT Core limit)
    //
    // Every site that constructs a Payload must set heroLambda. Left
    // uninitialised it is not a crash -- it silently turns an ordinary ray into
    // a single-wavelength one and the frame loses most of its light.

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
// Solar Spectral LUT Data Structure (GPU)
// ============================================================================
// Full spectral irradiance curves for sun and sky illumination.
// Enables physically-accurate spectral rendering with ASTM G-173 or libRadtran data.
//
// DATA SOURCES:
// - ASTM G-173-03 Reference Air Mass 1.5 Spectra (terrestrial solar)
//   - Direct sun: Column 4 (Direct+circumsolar) - W·m⁻²·nm⁻¹
//   - Diffuse sky: Global - Direct - W·m⁻²·nm⁻¹
// - libRadtran: Custom atmospheric conditions
//
// USAGE:
// - Query sun irradiance: SampleSpectralCurve(solarLUT.sunIrradiance, wavelength_nm)
// - Query sky irradiance: SampleSpectralCurve(solarLUT.skyIrradiance, wavelength_nm)
// - Convert irradiance E to radiance L: L = E / Ω_sun (for sun disk)
//   where Ω_sun ≈ 6.8e-5 sr (angular diameter ~0.53°)
//
// SIZE: 272 + 272 = 544 bytes (must match CPU-side SolarSpectralLUT)
// ============================================================================

struct SolarSpectralLUT {
    SpectralCurveGPU sunIrradiance;   // Direct sun spectral irradiance (W·m⁻²·nm⁻¹)
    SpectralCurveGPU skyIrradiance;   // Diffuse sky spectral irradiance (W·m⁻²·nm⁻¹)
};

// ============================================================================
// Helper: Sample Solar Irradiance at Wavelength
// ============================================================================
// Convenience wrappers for querying sun/sky spectral irradiance
// ============================================================================

float SampleSunIrradiance(SolarSpectralLUT lut, float wavelength_nm) {
    return SampleSpectralCurve(lut.sunIrradiance, wavelength_nm);
}

float SampleSkyIrradiance(SolarSpectralLUT lut, float wavelength_nm) {
    return SampleSpectralCurve(lut.skyIrradiance, wavelength_nm);
}

// ============================================================================
// Helper: Convert Sun Irradiance to Radiance
// ============================================================================
// Converts spectral irradiance E (W·m⁻²·nm⁻¹) to radiance L (W·sr⁻¹·m⁻²·nm⁻¹)
// using the sun's solid angle.
//
// Physics:
//   E = ∫ L · cos(θ) · dΩ ≈ L · Ω_sun  (for small angle θ ≈ 0)
//   L = E / Ω_sun
//
// Sun parameters:
//   - Angular diameter: ~0.533° = 9.3 mrad
//   - Solid angle: Ω = π · (θ/2)² ≈ 6.8e-5 sr
// ============================================================================

static const float SUN_SOLID_ANGLE_SR = 6.8e-5;  // Sun solid angle (steradians)

// REMOVED: SunIrradianceToRadiance(E) = E / SUN_SOLID_ANGLE_SR.
//
// It returned the radiance of the solar disk, which is only meaningful if
// something multiplies the solid angle back when integrating over the source.
// Nothing did: all four callers fed the result straight into either
// BRDF * X * NdotL or rho * X * NdotL, so enabling a solar LUT made the direct
// term 10^4 times too large -- a SWIR render came out at 97 W/m^2/sr/nm where
// the incident irradiance caps reflected radiance at 0.095. Use the irradiance
// for the BRDF form and E/PI for the albedo form; SUN_SOLID_ANGLE_SR stays,
// the MWIR Planck fallback still needs it to turn a disk radiance into one.

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
// Fresnel Equations for Dielectric (Transmission Materials)
// ============================================================================
// Computes exact Fresnel reflectance for dielectric-dielectric interface.
// Used for glass, water, and other transparent materials.
//
// Unlike FresnelSchlick (approximation), this is the exact physical formula
// that handles total internal reflection (TIR) correctly.
//
// PHYSICS:
//   At interface between media with refractive indices n1 and n2:
//   - Rs = reflection of s-polarized light
//   - Rp = reflection of p-polarized light
//   - R = (Rs + Rp) / 2 for unpolarized light
//
//   Total Internal Reflection occurs when:
//   sin(θ_t) = (n1/n2) × sin(θ_i) > 1  →  R = 1 (100% reflection)
//
// Input:
//   cosI: cos(incident angle), typically dot(N, -rayDir) or dot(N, viewDir)
//   n1: refractive index of incident medium (e.g., 1.0 for air)
//   n2: refractive index of transmission medium (e.g., 1.5 for glass)
//
// Returns:
//   Fresnel reflectance [0, 1]
// ============================================================================

float FresnelDielectric(float cosI, float n1, float n2) {
    // Ensure cosI is positive (we handle sign externally based on ray direction)
    cosI = abs(cosI);

    // Compute sin²(θ_i) from cos²(θ_i)
    float sin2I = 1.0 - cosI * cosI;

    // Snell's law: n1 × sin(θ_i) = n2 × sin(θ_t)
    // => sin²(θ_t) = (n1/n2)² × sin²(θ_i)
    float eta = n1 / n2;
    float sin2T = eta * eta * sin2I;

    // Total Internal Reflection check
    if (sin2T > 1.0) {
        return 1.0;  // 100% reflection
    }

    float cosT = sqrt(1.0 - sin2T);

    // Fresnel equations for s and p polarization
    // Rs = ((n1 × cosI - n2 × cosT) / (n1 × cosI + n2 × cosT))²
    // Rp = ((n1 × cosT - n2 × cosI) / (n1 × cosT + n2 × cosI))²
    float n1CosI = n1 * cosI;
    float n2CosT = n2 * cosT;
    float n1CosT = n1 * cosT;
    float n2CosI = n2 * cosI;

    float Rs_num = n1CosI - n2CosT;
    float Rs_den = n1CosI + n2CosT;
    float Rs = (Rs_num * Rs_num) / (Rs_den * Rs_den + 1e-8);

    float Rp_num = n1CosT - n2CosI;
    float Rp_den = n1CosT + n2CosI;
    float Rp = (Rp_num * Rp_num) / (Rp_den * Rp_den + 1e-8);

    // Average for unpolarized light
    return 0.5 * (Rs + Rp);
}

// ============================================================================
// Snell's Law Refraction (Transmission Direction)
// ============================================================================
// Computes the refracted ray direction at a dielectric interface.
//
// PHYSICS:
//   Snell's law: n1 × sin(θ_i) = n2 × sin(θ_t)
//
//   Refracted direction:
//   t = η × i + (η × cos(θ_i) - cos(θ_t)) × n
//   where: η = n1/n2, i = incident direction, n = surface normal
//
// Input:
//   I: incident ray direction (normalized, pointing INTO the surface)
//   N: surface normal (normalized, pointing OUT of the surface)
//   eta: ratio of indices of refraction (n1/n2)
//
// Returns:
//   Refracted direction (normalized), or float3(0) if total internal reflection
// ============================================================================

float3 Refract(float3 I, float3 N, float eta) {
    float cosI = -dot(N, I);
    float sin2T = eta * eta * (1.0 - cosI * cosI);

    // Total Internal Reflection check
    if (sin2T > 1.0) {
        return float3(0.0, 0.0, 0.0);  // TIR - no refraction
    }

    float cosT = sqrt(1.0 - sin2T);

    // Refracted direction
    return eta * I + (eta * cosI - cosT) * N;
}

// ============================================================================
// Beer-Lambert Volume Absorption
// ============================================================================
// Computes transmittance through an absorbing medium using Beer-Lambert law.
//
// PHYSICS:
//   T(d) = exp(-σ × d)
//   where σ = absorption coefficient, d = distance traveled
//
// For colored glass/liquids, the absorption coefficient is derived from:
//   attenuationColor = exp(-σ × attenuationDistance)
//   => σ = -ln(attenuationColor) / attenuationDistance
//
// Input:
//   absorptionColor: color remaining after traveling attenuationDistance
//   distance: actual distance traveled through the medium
//   attenDist: reference distance for absorption color (0 = no attenuation)
//
// Returns:
//   RGB transmittance [0, 1] per channel
// ============================================================================

float3 BeerLambertAbsorption(float3 absorptionColor, float distance, float attenDist) {
    // No attenuation if distance is zero or infinite reference distance
    if (attenDist <= 0.0 || distance <= 0.0) {
        return float3(1.0, 1.0, 1.0);
    }

    // Compute absorption coefficient from attenuation color
    // σ = -ln(color) / attenDist
    // Clamp color to avoid log(0) = -inf
    float3 safeColor = max(absorptionColor, float3(0.001, 0.001, 0.001));
    float3 sigma = -log(safeColor) / attenDist;

    // Beer-Lambert transmittance
    return exp(-sigma * distance);
}

// ============================================================================
// Cauchy Dispersion Formula (Wavelength-Dependent IOR)
// ============================================================================
// This is the formula KHR_materials_dispersion specifies, which is the one the
// glTF ecosystem agrees on:
//
//   n(lambda) = max( n_d + (n_d - 1)/V_d * (523655/lambda^2 - 1.5168), 1 )
//
// with lambda in NANOMETRES. `Material::dispersion` holds 1/V_d, so
// (n_d - 1)/V_d is (n_d - 1)*dispersion.
//
// NOTE the extension's own property is 20/V_d, not 1/V_d -- GltfLoader divides
// by 20 on the way in. Keep the two straight; they differ by 20x and both are
// spelled "dispersion".
//
// Two things about the shape are worth not undoing:
//
//   - 523655 is 0.523655 um^2, i.e. 1/(1/lambda_F^2 - 1/lambda_C^2) over the F
//     and C Fraunhofer lines. Deriving it by hand from lambda_F = 486.1 nm and
//     lambda_C = 656.3 nm gives 0.523454 -- the same number to four figures.
//   - The -1.5168 is NOT decoration. It anchors the curve so that
//     n(587.56 nm) = n_d exactly, which is what makes n_d the d-line index
//     everybody quotes. Without it n_d becomes the asymptotic index at
//     lambda -> infinity and every n comes out high by 1.5168*(n_d - 1)/V_d --
//     0.0122 for BK7, which is 3x BK7's entire F-to-C spread.
//
// Verified for BK7 (n_d = 1.5168, V_d = 64.17): n(587.56) = 1.51680,
// n_F = 1.52243, n_C = 1.51438, giving (n_d - 1)/(n_F - n_C) = 64.15 against
// the catalogue's 64.17.
//
// This replaced `n_d + dispersion*0.01/lambda_um^2`, a scaling with nothing
// behind it that understated the spread by 52*(n_d - 1) -- 27x for BK7. It
// produced a rainbow, but far too narrow a one, and wrong by a factor that
// moved with the material.
//
// LIMITS: a visible-range fit. Inaccurate in the infrared and unable to
// represent anomalous dispersion at all, so it is the fallback, not the
// preference -- see RefractionIOR in pbr.hlsli, which prefers measured n(lambda).
//
// Input:
//   ior_d: refractive index at the d-line (587.56 nm)
//   dispersion: 1 / Abbe number (0 = no dispersion)
//   wavelength_nm: wavelength in nanometers, or 0 for "this mode has none"
// ============================================================================

float CauchyIOR(float ior_d, float dispersion, float wavelength_nm) {
    // Guarded here rather than at every call site: wavelength_nm is 0 in modes
    // that have no single wavelength, and n_d is the right answer there.
    if (dispersion <= 0.0 || wavelength_nm <= 0.0) {
        return ior_d;
    }
    float n = ior_d + (ior_d - 1.0) * dispersion *
                      (523655.0 / (wavelength_nm * wavelength_nm) - 1.5168);
    return max(n, 1.0);
}

// ============================================================================
// Lighting Parameters Structure
// ============================================================================
// Runtime lighting parameters for shading (NOT a precomputed LUT).
// This provides fallback RGB/scalar values when SolarSpectralLUT is unavailable.
//
// Must match CPU-side LightingParams data layout in LightingParams.hpp.
// Size: 80 bytes (16-byte aligned)
//
// DUAL MODE SUPPORT:
// - RGB mode: Use sunRadiance_rgb and skyRadiance_rgb (float3)
// - Spectral mode: Use sunRadiance_spectral and skyRadiance_spectral (float)
//
// NOTE: The actual precomputed spectral LUT is SolarSpectralLUT (binding 15),
// which contains full spectral irradiance curves from MODTRAN/libRadtran.
// This struct provides simple scalar/RGB fallback values.
//
// WORLD UNITS:
// - worldUnitsToMeters: Conversion factor from scene units to meters
// - Required for correct Beer-Lambert attenuation calculation
// - Example: if scene uses centimeters, worldUnitsToMeters = 0.01
//
// SHADOW RAYS:
// - enableShadowRays: 0 = disabled (for debugging GPU crashes), 1 = enabled
// - Can be controlled via config: renderer.enable_shadow_rays = true/false
// ============================================================================

struct LightingParams {
    float3 sunDirection;         // Normalized sun direction vector (FROM surface TO sun)
    float  sunRadiance_spectral; // Sun spectral radiance at current λ (W·sr⁻¹·m⁻²·nm⁻¹) - fallback

    float3 sunRadiance_rgb;      // Sun RGB radiance (W·sr⁻¹·m⁻²) for RGB mode - fallback
    float  skyRadiance_spectral; // Sky spectral radiance at current λ (W·sr⁻¹·m⁻²·nm⁻¹) - fallback

    float3 skyRadiance_rgb;      // Sky RGB radiance (W·sr⁻¹·m⁻²) for RGB mode - fallback
    float  transmittance;        // Atmospheric transmittance τ(λ) [0, 1] (vertical path)

    float  worldUnitsToMeters;   // Conversion factor: world_units × this = meters
    float  atmosphereTemperature_K; // Effective atmosphere temperature (K) for IR downwelling radiation
    float  chromaR_correction;   // VIS_FUSED chromaticity correction for R channel (default: 0.7872)
    float  chromaB_correction;   // VIS_FUSED chromaticity correction for B channel (default: 1.0437)

    uint   enableShadowRays;     // Shadow ray enable flag: 0 = disabled, 1 = enabled (configurable)
    float  _padding[3];          // Padding to 80 bytes (16-byte aligned)
};

// ============================================================================
// Camera Data Structure
// ============================================================================
// Camera parameters for ray generation
// Must match CPU-side CameraData structure in Camera.hpp
// Size: 80 bytes (16-byte aligned)
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
    uint   debug_mode;     // Debug visualization mode (see DEBUG_MODE_* defines)
    uint   _padding[3];    // Padding for 16-byte alignment
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
    uint   doubleSided;              // 0=single-sided (cull backface), 1=double-sided // Offset: 40-44
    float  _padding0;                // Explicit padding to align emissiveFactor to 16-byte boundary // Offset: 44-48

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

    // Temperature texture fields (per-pixel temperature map)
    int    temperatureTextureIndex;      // Index into texture array (-1 = use scalar irTemperature_K) // Offset: 96-100
    float  temperatureScale;             // T(K) = tex.r * scale + offset                             // Offset: 100-104
    float  temperatureOffset;            // Kelvin offset                                              // Offset: 104-108
    int    irEmissivityCurveIndex;       // Index into spectralCurves (-1 = use scalar/heuristic)      // Offset: 108-112

    // ========================================================================
    // Transmission Properties (KHR_materials_transmission + KHR_materials_volume)
    // ========================================================================
    // Physical transparency for glass, water, and other dielectric materials.
    //
    // PHYSICS:
    // - IOR determines refraction angle (Snell's law) and Fresnel reflection ratio
    // - Transmission controls how much light passes through (vs absorbed/reflected)
    // - Attenuation models Beer-Lambert absorption in colored glass/liquids
    //
    // ENERGY CONSERVATION:
    //   F = FresnelDielectric(n1, n2, θ) = reflection ratio
    //   T = 1 - F = transmission ratio (before volume absorption)
    //   Final_transmission = T × exp(-σ × distance)

    float  ior;                          // Index of refraction (1.0=air, 1.33=water, 1.5=glass)    // Offset: 112-116
    float  transmission;                 // Transmission strength [0,1]                              // Offset: 116-120
    int    transmissionTextureIndex;     // Transmission texture (-1 = no texture)                  // Offset: 120-124
    int    irTransmittanceCurveIndex;    // Index into spectralCurves (-1 = use scalar)              // Offset: 124-128

    // Volume attenuation (Beer-Lambert absorption)
    float3 attenuationColor;             // Color at attenuation distance                           // Offset: 128-140
    float  attenuationDistance;          // Distance for attenuation (mm, 0 = no attenuation)       // Offset: 140-144

    float  thicknessFactor;              // Thickness for thin-walled approximation                 // Offset: 144-148
    int    thicknessTextureIndex;        // Thickness texture (-1 = no texture)                     // Offset: 148-152
    float  dispersion;                   // Abbe number reciprocal (0 = no dispersion)              // Offset: 152-156
    float  _padding2;                    // Padding for alignment                                    // Offset: 156-160

    // ========================================================================
    // Participating Media Properties (fog, smoke, SSS)
    // ========================================================================
    // For volume rendering with scattering and absorption.
    //
    // PHYSICS:
    //   σ_t = σ_a + σ_s (extinction = absorption + scattering)
    //   T = exp(-σ_t × d) (Beer-Lambert transmittance)

    float  volumeDensity;                // Medium density multiplier (0 = no volume)               // Offset: 160-164
    float  scatteringCoeff;              // Scattering coefficient σ_s (m⁻¹)                        // Offset: 164-168
    float  absorptionCoeff;              // Absorption coefficient σ_a (m⁻¹)                        // Offset: 168-172
    float  phaseG;                       // Henyey-Greenstein g parameter [-1,1]                    // Offset: 172-176

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
// Helper: Get Effective IR Emissivity (P1 Fix: Metallic-Emissivity Consistency)
// ============================================================================
// Derives IR emissivity from PBR metallic factor when not explicitly set.
//
// Physical basis (Kirchhoff's Law):
//   For opaque materials at thermal equilibrium: ε = α = 1 - ρ
//   Metals have high reflectance ρ → low emissivity ε
//   Dielectrics have lower ρ → higher ε
//
// The relationship between metallicFactor and emissivity:
//   - metallicFactor = 0 (dielectric): ε ≈ 0.90-0.98 (typical for plastics, paint)
//   - metallicFactor = 1 (polished metal): ε ≈ 0.03-0.10 (typical for polished Al, Cu)
//   - Oxidized metals have intermediate values
//
// When irEmissivity < 0 (sentinel value), derive from metallicFactor:
//   ε_derived = 0.95 - 0.90 × metallic
//   → metallic=0: ε=0.95 (matte dielectric)
//   → metallic=1: ε=0.05 (polished metal)
//
// Input:
//   mat: MaterialData with irEmissivity and metallicFactor
//
// Returns:
//   Effective IR emissivity [0, 1]
//
// Reference: "Handbook of Optical Constants" (Palik, 1985)
// ============================================================================

float GetEffectiveIREmissivity(MaterialData mat) {
    // Sentinel value: irEmissivity < 0 means "derive from PBR properties"
    if (mat.irEmissivity < 0.0) {
        // Derive emissivity from metallic factor using physically-based model
        // Base emissivity for dielectric: 0.95 (rough surface, high absorptance)
        // Metal emissivity reduction: up to 0.90 (leaving 0.05 for polished metal)
        // The 0.90 factor accounts for typical metal reflectance ρ ≈ 0.95
        float emissivity_derived = 0.95 - 0.90 * mat.metallicFactor;

        // Apply roughness correction: rough metals have higher emissivity
        // due to micro-cavity effects (multiple reflections increase absorptance)
        // Rough metal: ε increases by up to 2x the polished value
        float roughness_correction = mat.metallicFactor * mat.roughnessFactor * 0.15;
        emissivity_derived += roughness_correction;

        return saturate(emissivity_derived);
    }

    // Use explicitly specified emissivity
    return mat.irEmissivity;
}

// ============================================================================
// Helper: Get Effective IR Reflectance (uses derived emissivity)
// ============================================================================
// Same as GetIRReflectance but uses GetEffectiveIREmissivity for metallic consistency.
// Use this function for IR rendering to ensure Kirchhoff's law compliance.
// ============================================================================

float GetEffectiveIRReflectance(MaterialData mat) {
    float emissivity = GetEffectiveIREmissivity(mat);
    return saturate(1.0 - emissivity - mat.irTransmittance);
}

// ============================================================================
// Angle-Dependent IR Emissivity (Fresnel Effect for Thermal Radiation)
// ============================================================================
// Implements the angular dependence of thermal emissivity:
// - Metals: emissivity INCREASES at grazing angles (Hagen-Rubens effect)
// - Dielectrics: emissivity DECREASES at grazing angles (Fresnel effect)
//
// Physics: By Kirchhoff's law, ε = 1 - ρ. At grazing angles:
// - Metal reflectance stays high but decreases slightly → ε increases
// - Dielectric reflectance increases sharply (Fresnel) → ε decreases
//
// Reference: "Radiative Heat Transfer" (Modest, 3rd ed.), Chapter 3
// ============================================================================

float GetAngleDependentIREmissivity(float baseEmissivity, float NdotV, float metallic) {
    // Clamp NdotV to avoid division issues at grazing angles
    float cosTheta = max(NdotV, 0.01);

    if (metallic > 0.5) {
        // METALS: Hagen-Rubens approximation
        // At grazing angles, emissivity approaches ~1.0 for real metals
        // ε(θ) ≈ ε₀ × (1 + α × (1 - cos(θ)))
        // α controls the strength of the effect (typical: 0.5-1.5)
        const float METAL_GRAZING_ALPHA = 1.0;
        float grazingFactor = 1.0 + METAL_GRAZING_ALPHA * (1.0 - cosTheta);
        return saturate(baseEmissivity * grazingFactor);
    } else {
        // DIELECTRICS: Fresnel-like behavior
        // At grazing angles, reflectance → 1, so emissivity → 0
        // ε(θ) ≈ ε₀ × cos^β(θ)
        // β controls sharpness (typical: 0.5-1.0, higher = sharper transition)
        const float DIELECTRIC_GRAZING_BETA = 0.7;
        float grazingFactor = pow(cosTheta, DIELECTRIC_GRAZING_BETA);
        return baseEmissivity * grazingFactor;
    }
}

// Corresponding reflectance adjustment (energy conservation)
float GetAngleDependentIRReflectance(float baseEmissivity, float transmittance,
                                      float NdotV, float metallic) {
    float angleEmissivity = GetAngleDependentIREmissivity(baseEmissivity, NdotV, metallic);
    return saturate(1.0 - angleEmissivity - transmittance);
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

// ============================================================================
// Instance Geometry Info (Per-TLAS-Instance Offsets)
// ============================================================================
// Provides geometry buffer offsets for multi-BLAS support.
// Each TLAS instance stores its offset into merged global geometry buffers.
// Indexed by InstanceIndex() in closest hit shader.
//
// WHY NEEDED:
// In a multi-BLAS scene, each BLAS has its own local vertex/index/normal buffers.
// We merge all BLAS data into global buffers for efficient shader access.
// InstanceGeometryInfo tells the shader where each instance's data starts.
//
// USAGE IN SHADER (with ByteAddressBuffer for vertex/normal to avoid stride issues):
//   uint instIdx = InstanceIndex();
//   InstanceGeometryInfo geo = instanceGeometryInfo[instIdx];
//   uint idx = indexBuffer[geo.indexOffset + PrimitiveIndex() * 3 + localVertexIdx];
//   float3 v = asfloat(vertexBuffer.Load3((geo.vertexOffset + idx) * 12));  // 12 = sizeof(float3)
//   float3 n = asfloat(normalBuffer.Load3((geo.normalOffset + idx) * 12));
//
// SIZE: 32 bytes (must match CPU-side InstanceGeometryInfo)
// ============================================================================

struct InstanceGeometryInfo {
    uint vertexOffset;    // Offset into global vertex buffer (vertex count)
    uint indexOffset;     // Offset into global index buffer (index count)
    uint normalOffset;    // Offset into global normal buffer (normal count)
    uint uvOffset;        // Offset into global UV buffer (UV count)
    uint tangentOffset;   // Offset into global tangent buffer (tangent count)
    uint materialId;      // Material index (used instead of InstanceID for material lookup)
    uint pad[2];          // Padding for 32-byte alignment
};

// ============================================================================
// Debug Visualization Helper Functions
// ============================================================================

// Hash integer to RGB color (for MaterialID, TriangleID visualization)
// Uses simple integer hash and converts to visually distinct colors
float3 HashToColor(uint id) {
    // Simple integer hash (similar to Wang hash but simpler)
    uint hash = id;
    hash = (hash ^ 61) ^ (hash >> 16);
    hash = hash * 9;
    hash = hash ^ (hash >> 4);
    hash = hash * 0x27d4eb2d;
    hash = hash ^ (hash >> 15);

    // Convert hash to RGB (use different bits for each channel)
    float r = float((hash >> 0) & 0xFF) / 255.0;
    float g = float((hash >> 8) & 0xFF) / 255.0;
    float b = float((hash >> 16) & 0xFF) / 255.0;

    // Boost saturation for better visibility
    float3 color = float3(r, g, b);
    float gray = dot(color, float3(0.299, 0.587, 0.114));
    return lerp(float3(gray, gray, gray), color, 1.5);
}

// Temperature to color (simple hot-cold colormap)
// Maps temperature from minT to maxT into blue->cyan->green->yellow->red
float3 TemperatureToColor(float temp, float minT, float maxT) {
    float t = saturate((temp - minT) / (maxT - minT));

    // Simple rainbow colormap: blue -> cyan -> green -> yellow -> red
    float3 color;
    if (t < 0.25) {
        // Blue to Cyan
        float s = t / 0.25;
        color = lerp(float3(0.0, 0.0, 1.0), float3(0.0, 1.0, 1.0), s);
    } else if (t < 0.5) {
        // Cyan to Green
        float s = (t - 0.25) / 0.25;
        color = lerp(float3(0.0, 1.0, 1.0), float3(0.0, 1.0, 0.0), s);
    } else if (t < 0.75) {
        // Green to Yellow
        float s = (t - 0.5) / 0.25;
        color = lerp(float3(0.0, 1.0, 0.0), float3(1.0, 1.0, 0.0), s);
    } else {
        // Yellow to Red
        float s = (t - 0.75) / 0.25;
        color = lerp(float3(1.0, 1.0, 0.0), float3(1.0, 0.0, 0.0), s);
    }

    return color;
}

#endif // QUANTILOOM_COMMON_HLSLI
