// ============================================================================
// Quantiloom - layout locks of the public structs the shaders read
// ============================================================================
// Six structs a shader reads are also public API: CameraData and the prefix of
// PushConstantsRayGen (include/quantiloom/scene/Camera.hpp), LightingParams
// (renderer/LightingParams.hpp), SpectralCurveGPU, ComplexRefractiveIndexGPU
// and SolarSpectralLUT (core/SpectralData.hpp). The SDK installs
// include/quantiloom/ and nothing else, so their C++ definitions stay there and
// cannot come from a header in src/shaders/.
//
// What this header holds instead is a lock: per struct, the HLSL definition and,
// in the C++ branch beside it, an alias of the public type with its size and the
// offset of every member asserted. The C++ branch is compiled in exactly one
// place, gpu/SharedLayoutChecks.cpp, so no public header includes this one. Two
// definitions checked against each other is weaker than one definition: an edit
// confined to the HLSL text below is invisible to the asserts until someone
// edits the list next to it. A member added, moved or resized on the public side
// fails the C++ build.
//
// A shader asks for one struct as with the other shared headers
// (QL_SHARED_WANT_<Name>, see ql_shared_prelude.hlsli). MAX_SPECTRAL_SAMPLES
// (64) is common.hlsli's, defined before SpectralCurveGPU is asked for; the
// public headers spell the same 64 as a constexpr.
//
// No include guard: a shader includes this once per struct it asks for, and
// every section guards itself (QL_SHARED_HAS_<Name>).
// ============================================================================

#include "ql_shared_prelude.hlsli"

#if defined(__cplusplus)
    #include "core/SpectralData.hpp"
    #include "renderer/LightingParams.hpp"
    #include "scene/Camera.hpp"
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_SpectralCurveGPU)) && !defined(QL_SHARED_HAS_SpectralCurveGPU)
#define QL_SHARED_HAS_SpectralCurveGPU
#if defined(__cplusplus)
// The definition is the public one, include/quantiloom/core/SpectralData.hpp.
QL_SHARED_BEGIN
using SpectralCurveGPU = ::quantiloom::SpectralCurveGPU;
QL_SHARED_LAYOUT(SpectralCurveGPU, 272)
QL_SHARED_OFFSET(SpectralCurveGPU, values, 0)
QL_SHARED_OFFSET(SpectralCurveGPU, startWavelength_nm, 256)
QL_SHARED_OFFSET(SpectralCurveGPU, stepSize_nm, 260)
QL_SHARED_OFFSET(SpectralCurveGPU, numSamples, 264)
QL_SHARED_OFFSET(SpectralCurveGPU, nextExactSegment, 268)
QL_SHARED_END
#else
struct SpectralCurveGPU {
    float values[MAX_SPECTRAL_SAMPLES];  // Spectral values (reflectance, emissivity, etc.) [0, 1] or [0, inf]
    float startWavelength_nm;            // Starting wavelength (nm)
    float stepSize_nm;                   // Wavelength step size (nm)
    uint  numSamples;                    // Actual number of valid samples (0 to MAX_SPECTRAL_SAMPLES)
    uint  nextExactSegment;              // Emission only: next exact segment index+1, 0 ends chain; step=0 packs <=32 (nm,value) pairs
};
#endif
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_ComplexRefractiveIndexGPU)) && !defined(QL_SHARED_HAS_ComplexRefractiveIndexGPU)
#define QL_SHARED_HAS_ComplexRefractiveIndexGPU
#if defined(__cplusplus)
// The definition is the public one, include/quantiloom/core/SpectralData.hpp.
QL_SHARED_BEGIN
using ComplexRefractiveIndexGPU = ::quantiloom::ComplexRefractiveIndexGPU;
QL_SHARED_LAYOUT(ComplexRefractiveIndexGPU, 528)
QL_SHARED_OFFSET(ComplexRefractiveIndexGPU, n, 0)
QL_SHARED_OFFSET(ComplexRefractiveIndexGPU, k, 256)
QL_SHARED_OFFSET(ComplexRefractiveIndexGPU, startWavelength_nm, 512)
QL_SHARED_OFFSET(ComplexRefractiveIndexGPU, stepSize_nm, 516)
QL_SHARED_OFFSET(ComplexRefractiveIndexGPU, numSamples, 520)
QL_SHARED_OFFSET(ComplexRefractiveIndexGPU, _padding, 524)
QL_SHARED_END
#else
struct ComplexRefractiveIndexGPU {
    float n[MAX_SPECTRAL_SAMPLES];       // Refractive index at uniform grid
    float k[MAX_SPECTRAL_SAMPLES];       // Extinction coefficient at uniform grid
    float startWavelength_nm;            // Starting wavelength (nm)
    float stepSize_nm;                   // Wavelength step size (nm)
    uint  numSamples;                    // Number of valid samples
    uint  _padding;                      // 16-byte alignment
};
#endif
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_SolarSpectralLUT)) && !defined(QL_SHARED_HAS_SolarSpectralLUT)
#define QL_SHARED_HAS_SolarSpectralLUT
#if defined(__cplusplus)
// The definition is the public one, include/quantiloom/core/SpectralData.hpp.
QL_SHARED_BEGIN
using SolarSpectralLUT = ::quantiloom::SolarSpectralLUT;
QL_SHARED_LAYOUT(SolarSpectralLUT, 544)
QL_SHARED_OFFSET(SolarSpectralLUT, sunIrradiance, 0)
QL_SHARED_OFFSET(SolarSpectralLUT, skyIrradiance, 272)
QL_SHARED_END
#else
struct SolarSpectralLUT {
    SpectralCurveGPU sunIrradiance;   // Direct sun spectral irradiance (W·m⁻²·nm⁻¹)
    SpectralCurveGPU skyIrradiance;   // Diffuse sky spectral irradiance (W·m⁻²·nm⁻¹)
};
#endif
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_LightingParams)) && !defined(QL_SHARED_HAS_LightingParams)
#define QL_SHARED_HAS_LightingParams
#if defined(__cplusplus)
// The definition is the public one, include/quantiloom/renderer/LightingParams.hpp.
QL_SHARED_BEGIN
using LightingParams = ::quantiloom::LightingParams;
QL_SHARED_LAYOUT(LightingParams, 224)
QL_SHARED_OFFSET(LightingParams, sunDirection, 0)
QL_SHARED_OFFSET(LightingParams, sunRadiance_spectral, 12)
QL_SHARED_OFFSET(LightingParams, sunRadiance_rgb, 16)
QL_SHARED_OFFSET(LightingParams, skyRadiance_spectral, 28)
QL_SHARED_OFFSET(LightingParams, skyRadiance_rgb, 32)
QL_SHARED_OFFSET(LightingParams, skyEmissivityClear, 44)
QL_SHARED_OFFSET(LightingParams, worldUnitsToMeters, 48)
QL_SHARED_OFFSET(LightingParams, atmosphereTemperature_K, 52)
QL_SHARED_OFFSET(LightingParams, chromaR_correction, 56)
QL_SHARED_OFFSET(LightingParams, chromaB_correction, 60)
QL_SHARED_OFFSET(LightingParams, enableShadowRays, 64)
QL_SHARED_OFFSET(LightingParams, enableEnvironmentMap, 68)
QL_SHARED_OFFSET(LightingParams, emissiveTriangleCount, 72)
QL_SHARED_OFFSET(LightingParams, emissiveTotalPower, 76)
QL_SHARED_OFFSET(LightingParams, skyDistribution, 80)
QL_SHARED_OFFSET(LightingParams, skyAerosolOpticalDepth, 84)
QL_SHARED_OFFSET(LightingParams, skyDomeLambdaStart_nm, 88)
QL_SHARED_OFFSET(LightingParams, skyDomeLambdaStep_nm, 92)
QL_SHARED_OFFSET(LightingParams, skyDomeN, 96)
QL_SHARED_OFFSET(LightingParams, skyDomeJ, 160)
QL_SHARED_END
#else
struct LightingParams {
    float3 sunDirection;         // Normalized sun direction vector (FROM surface TO sun)
    float  sunRadiance_spectral; // Sun spectral radiance at current λ (W·sr⁻¹·m⁻²·nm⁻¹) - fallback

    float3 sunRadiance_rgb;      // Sun RGB radiance (W·sr⁻¹·m⁻²) for RGB mode - fallback
    float  skyRadiance_spectral; // Sky spectral radiance at current λ (W·sr⁻¹·m⁻²·nm⁻¹) - fallback

    float3 skyRadiance_rgb;      // Sky RGB radiance (W·sr⁻¹·m⁻²) for RGB mode - fallback
    // Zenith emissivity of a clear sky, [0, 1]. Zero = the thermal sky is one
    // isotropic blackbody at atmosphereTemperature_K, which is what it always
    // was. Above zero the flat-slab law below takes over. Was `transmittance`,
    // deprecated when the NN atmosphere took the view path.
    float  skyEmissivityClear;

    float  worldUnitsToMeters;   // Conversion factor: world_units × this = meters
    float  atmosphereTemperature_K; // Effective atmosphere temperature (K) for IR downwelling radiation
    float  chromaR_correction;   // VIS_FUSED chromaticity correction for R channel (default: 0.7872)
    float  chromaB_correction;   // VIS_FUSED chromaticity correction for B channel (default: 1.0437)

    uint   enableShadowRays;     // Shadow ray enable flag: 0 = disabled, 1 = enabled (configurable)
    uint   enableEnvironmentMap; // IBL from the environment cubemap: 0 = contributes nothing, 1 = lights the scene
    uint   emissiveTriangleCount;// Entries in emissiveTriangles (binding 23); 0 = no light sampling
    float  emissiveTotalPower;   // Sum of luminance(emissive) * area; the area-measure sampling
                                 // density on any emitter is luminance(emissive) / this

    // The visible sky's distribution over direction; see the functions below.
    uint   skyDistribution;        // SKY_DISTRIBUTION_UNIFORM or SKY_DISTRIBUTION_CLEAR
    float  skyAerosolOpticalDepth; // at 500 nm, shapes the clear sky
    float  skyDomeLambdaStart_nm;  // grid of the two tables below, filled by the renderer
    float  skyDomeLambdaStep_nm;
    float  skyDomeN[16];           // cosine integral of the unnormalised clear sky
    float  skyDomeJ[16];           // its multiple-scattering source
};
#endif
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_CameraData)) && !defined(QL_SHARED_HAS_CameraData)
#define QL_SHARED_HAS_CameraData
#if defined(__cplusplus)
// The definition is the public one, include/quantiloom/scene/Camera.hpp.
QL_SHARED_BEGIN
using CameraData = ::quantiloom::CameraData;
QL_SHARED_LAYOUT(CameraData, 80)
QL_SHARED_OFFSET(CameraData, origin, 0)
QL_SHARED_OFFSET(CameraData, fovScale, 12)
QL_SHARED_OFFSET(CameraData, forward, 16)
QL_SHARED_OFFSET(CameraData, aspectRatio, 28)
QL_SHARED_OFFSET(CameraData, right, 32)
QL_SHARED_OFFSET(CameraData, wavelength_nm, 44)
QL_SHARED_OFFSET(CameraData, up, 48)
QL_SHARED_OFFSET(CameraData, spectral_mode, 60)
QL_SHARED_OFFSET(CameraData, debug_mode, 64)
QL_SHARED_OFFSET(CameraData, projection, 68)
QL_SHARED_OFFSET(CameraData, orthoHeight, 72)
QL_SHARED_OFFSET(CameraData, debugParam, 76)
QL_SHARED_END
#else
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
    uint   projection;     // CAMERA_PROJECTION_* below
    float  orthoHeight;    // Film-plane height in world units, orthographic only
    uint   debugParam;     // One number for a debug view; see Camera.hpp
};
#endif
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_PushConstantsRayGen)) && !defined(QL_SHARED_HAS_PushConstantsRayGen)
#define QL_SHARED_HAS_PushConstantsRayGen
#if defined(__cplusplus)
// The definition is the public one, include/quantiloom/scene/Camera.hpp.
// The public struct is the first 108 bytes, up to textureCount, which only
// the trace's own push block carries: RayTracingPipeline::TracePushConstants,
// whose definition asserts the 112 bytes of the HLSL struct below.
QL_SHARED_BEGIN
using PushConstantsRayGen = ::quantiloom::PushConstantsRayGen;
QL_SHARED_LAYOUT(PushConstantsRayGen, 108)
QL_SHARED_OFFSET(PushConstantsRayGen, camera, 0)
QL_SHARED_OFFSET(PushConstantsRayGen, frameIndex, 80)
QL_SHARED_OFFSET(PushConstantsRayGen, sampleIndex, 84)
QL_SHARED_OFFSET(PushConstantsRayGen, totalSamples, 88)
QL_SHARED_OFFSET(PushConstantsRayGen, randomSeed, 92)
QL_SHARED_OFFSET(PushConstantsRayGen, sequenceSeed, 96)
QL_SHARED_OFFSET(PushConstantsRayGen, timeStratum, 100)
QL_SHARED_OFFSET(PushConstantsRayGen, timeStratumCount, 104)
QL_SHARED_END
#else
struct PushConstantsRayGen {
    CameraData camera;
    uint frameIndex;
    uint sampleIndex;
    uint totalSamples;
    uint randomSeed;
    uint sequenceSeed;
    // Temporal stratification of one camera exposure. See the CPU-side
    // twin in include/quantiloom/scene/Camera.hpp; the two must stay identical.
    uint timeStratum;         // layer this dispatch writes, 0-based
    uint timeStratumCount;    // layers in the current exposure, 0/1 = plain
    uint textureCount;       // actual bound descriptors, including the dummy texture
};
#endif
#endif
