// ============================================================================
// Quantiloom - Closest Hit Shader (PBR with Textures)
// ============================================================================
// Computes Cook-Torrance PBR shading with:
// - Texture sampling (base color, metallic-roughness, normal, emissive)
// - Direct sun lighting from LUT with Beer-Lambert atmospheric attenuation
// - Sky ambient lighting (hemispherical integration with cosine-weighted sampling)
//
// SPECTRAL RENDERING:
// - Supports multiple rendering modes: single, RGB, MWIR, LWIR
// - RGB mode: Physically-based spectral upsampling + XYZ integration
// - Single mode: Full spectral fidelity with measured reflectance curves
//
//ATMOSPHERIC MODEL (Enhanced with Delta-Tracking support):
// - Beer-Lambert path attenuation: T(λ, d) = exp(-σ_t(λ) × d)
// - MODTRAN LUT provides σ_t(λ) extinction coefficient (fallback)
// - NN atmosphere LUT provides MODTRAN-surrogate tau / path radiance / L_down
// - Hemispherical sky radiance integration for diffuse ambient
// ============================================================================

#include "common.hlsli"
#include "pbr.hlsli"
#include "SpectralConversion.hlsli"
#include "blackbody.hlsli"
#include "spectral_query.hlsli"
#include "atmosphere_nn.hlsli"
#include "volumetric.hlsli"

// ============================================================================
// Bindings
// ============================================================================

// Acceleration structure for shadow rays (binding 1, shared with raygen)
[[vk::binding(1, 0)]] RaytracingAccelerationStructure scene;

[[vk::binding(2, 0)]] StructuredBuffer<LightingParams> lightingParams;
// Use ByteAddressBuffer for vertex/normal to avoid float3 stride alignment issues
// StructuredBuffer<float3> may use 16-byte stride on some drivers, causing out-of-bounds reads
[[vk::binding(3, 0)]] ByteAddressBuffer vertexBuffer;    // Vertex positions (12 bytes each)
[[vk::binding(4, 0)]] StructuredBuffer<uint> indexBuffer;       // Triangle indices
[[vk::binding(5, 0)]] StructuredBuffer<MaterialData> materials; // Material properties
[[vk::binding(6, 0)]] Texture2D textures[];                     // Bindless texture array
[[vk::binding(7, 0)]] SamplerState samplers[];                  // Bindless sampler array
[[vk::binding(8, 0)]] StructuredBuffer<float2> uvBuffer;        // UV coordinates (optional)
[[vk::binding(9, 0)]] StructuredBuffer<float4> tangentBuffer;   // Tangent vectors (optional)
[[vk::binding(16, 0)]] ByteAddressBuffer normalBuffer;   // Normal vectors (12 bytes each)

// ============================================================================
// NEW (M2+): Spectral Curve Buffer
// ============================================================================
// Buffer of spectral reflectance curves for physically-based spectral rendering
// Indexed by MaterialData::spectralReflectanceCurveIndex
// Binding 13 chosen to avoid conflict with IBL resources (10-12)
// ============================================================================

[[vk::binding(13, 0)]] StructuredBuffer<SpectralCurveGPU> spectralCurves;

// NOTE: Complex refractive index buffer (binding 14) is declared in pbr.hlsli
// because the BRDF functions there read it directly.

// ============================================================================
// NEW (M2+): Solar Spectral LUT Buffer
// ============================================================================
// Contains ASTM G-173 solar irradiance curves for spectral rendering
// Enables physically-accurate wavelength-dependent sun/sky illumination
//
// DATA:
// - sunIrradiance: Direct+circumsolar spectral irradiance (W·m⁻²·nm⁻¹)
// - skyIrradiance: Diffuse sky spectral irradiance (W·m⁻²·nm⁻¹)
//
// USAGE:
// - Query sun irradiance: SampleSunIrradiance(solarLUT, wavelength_nm)
// - Query sky irradiance: SampleSkyIrradiance(solarLUT, wavelength_nm)
// - Convert to radiance: L = E / SUN_SOLID_ANGLE_SR
//
// When solarSpectralLUT[0].sunIrradiance.numSamples == 0, fall back to LightingParams RGB values
// ============================================================================

[[vk::binding(15, 0)]] StructuredBuffer<SolarSpectralLUT> solarSpectralLUT;

// ============================================================================
// NN Atmosphere LUT (Bindings 17 + 20)
// ============================================================================
// Baked from MODTRAN surrogate networks on the CPU (see atmosphere_nn.hlsli).
// Header carries the axis parameterization and data offsets; the data blob
// holds tau / lpath / ldown spectral grids indexed by the spectral loop
// counter. When header.enabled == 0 the atmosphere contributes nothing.
// ============================================================================

[[vk::binding(17, 0)]] StructuredBuffer<AtmosNNHeader> atmosNNHeader;
[[vk::binding(20, 0)]] StructuredBuffer<float> atmosNNData;

// ============================================================================
// Emissive geometry (next-event estimation)
// ============================================================================
// World-space emitters with a cumulative-power CDF. Always bound -- a scene
// with no emissive material gets a single zero entry -- so read
// lightingParams[0].emissiveTriangleCount before touching it.
// ============================================================================

[[vk::binding(23, 0)]] StructuredBuffer<EmissiveTriangleGPU> emissiveTriangles;

// Per-element surface temperatures from the thermal solver (binding 24). One
// float per triangle, indexed by the instance's thermalElementBase plus
// PrimitiveIndex(). A scene with no solve binds a single zero and every
// instance carries the sentinel, so nothing reads it.
[[vk::binding(24, 0)]] StructuredBuffer<float> thermalTemperatures;

// ============================================================================
// Path Depth
// ============================================================================
// One counter for every kind of child ray -- refraction, reflection and the
// environment bounce all advance payload.depth and all stop at the same gate.
// Giving any of them its own budget would break the accounting below.
//
// The pipeline is created with maxPipelineRayRecursionDepth = 10
// (RayTracingPipeline.cpp, kMaxPipelineRayRecursionDepth), and 8 is what fits:
//
//   raygen -> primary                            TraceRay call 1
//   child i, gated on depth < MAX_PATH_DEPTH     calls 2..9   (depth 1..8)
//   the depth-8 hit spawns no child, only its shadow ray      call 10
//
// so a chain reaches exactly 10 and no further. Raising this constant without
// raising the pipeline's limit is undefined behaviour, not a slow render.
// ============================================================================

static const uint MAX_PATH_DEPTH = 8;

// Bounces below this depth are always traced when the surface reflects at all;
// deeper ones survive with probability equal to that reflectance. Two keeps the
// guaranteed levels the fixed `depth < 2` gate used to provide, so nothing that
// converged before converges more slowly now, and the roulette above it is what
// lets a polished cavity keep bouncing while a matte one stops after one.
static const uint BOUNCE_DEPTH_DETERMINISTIC = 2;

// ============================================================================
// RGB Channel Representative Wavelengths (sRGB primaries approximation)
// ============================================================================
// Used for wavelength-dependent calculations in RGB mode (atmospheric scattering)
// These values approximate the effective wavelengths of sRGB display primaries
// ============================================================================

static const float WAVELENGTH_R_NM = 650.0;  // Red channel representative wavelength
static const float WAVELENGTH_G_NM = 550.0;  // Green channel representative wavelength
static const float WAVELENGTH_B_NM = 450.0;  // Blue channel representative wavelength

// ============================================================================
// Instance Geometry Info Buffer (Binding 18)
// ============================================================================
// Per-TLAS-instance geometry offset information for multi-BLAS support.
// When scene has multiple BLAS, each instance's geometry data is merged into
// global buffers. This buffer tells us where each instance's data starts.
//
// USAGE:
//   uint instIdx = InstanceIndex();
//   InstanceGeometryInfo geo = instanceGeometryInfo[instIdx];
//   uint idx = indexBuffer[geo.indexOffset + PrimitiveIndex() * 3 + i];
//   float3 pos = vertexBuffer[geo.vertexOffset + idx];
//   float3 nrm = normalBuffer[geo.normalOffset + idx];
// ============================================================================

[[vk::binding(18, 0)]] StructuredBuffer<InstanceGeometryInfo> instanceGeometryInfo;

// ============================================================================
// CIE 1931 Color Matching Functions LUT (Binding 19)
// ============================================================================
// High-precision CIE XYZ color matching functions for VIS_FUSED mode
// 401 samples covering 380-780nm at 1nm resolution
//
// ACCURACY:
// - LUT version: <0.1% error across full spectrum
// - Analytical version (Wyman et al. 2013): <2% core, 10-20% at edges (380-420nm, 700-780nm)
//
// USAGE:
// - Loaded from assets/luts/CIE_xyz_1931_2deg.csv
// - Used in VIS_FUSED spectral integration for accurate XYZ conversion
// ============================================================================

[[vk::binding(19, 0)]] StructuredBuffer<float3> cieCMF_LUT;

// ============================================================================
// IBL (Image-Based Lighting) Resources
// ============================================================================
// Added for physically-based specular reflections on metallic surfaces
// ============================================================================

[[vk::binding(10, 0)]] TextureCube<float4> prefilteredEnvMap;  // Prefiltered environment cubemap (with mipmaps)
[[vk::binding(11, 0)]] Texture2D<float2> brdfLUT;              // BRDF integration lookup table
[[vk::binding(12, 0)]] SamplerState iblSampler;                // Linear sampler for the BRDF LUT (single level)
// Separate from iblSampler because that one clamps maxLod to 0 -- correct for the
// single-level BRDF LUT, but it silently pinned every environment lookup to mip 0,
// so the prefiltered chain below was never reached and rough metals reflected
// mirror-sharp. This one is unclamped and filters between levels.
[[vk::binding(21, 0)]] SamplerState envSampler;                // Trilinear sampler for the prefiltered environment

// ============================================================================
// Sample sources
// ============================================================================
// The push-constant block itself lives in common.hlsli, shared with raygen and
// miss. This shader reads sampleIndex and sequenceSeed from it.
//
// Where a uniform number comes from depends on how deep the path is.
//
// On the FIRST bounce every decision has a fixed identity -- this draw is
// always the wavelength, that one is always the bounce direction -- so each can
// have its own padded, Owen-scrambled copy of the Sobol' sequence and be
// stratified across the samples of the accumulation round. That is where nearly
// all of the variance is, and where stratifying pays.
//
// DEEPER the identity is gone: whether a vertex picks a lobe, or samples a
// light, or terminates on Russian roulette depends on what the path hit, so
// sample i's third draw at depth 3 is not measuring the same thing as sample
// j's. Stratifying along a dimension whose meaning moves between samples buys
// nothing, so those keep drawing from PCG.
//
// Both are unbiased; they differ only in how the samples are correlated with
// each other.
// ============================================================================

// A stratified draw for a first-bounce decision, PCG otherwise. `slot` is a
// SAMPLE_SLOT_* constant and identifies which padded copy of the sequence this
// decision owns.
float PathSample1D(inout Payload payload, uint slot) {
    if (payload.depth == 0u) {
        return StratifiedSample1D(pushConsts.sampleIndex, DispatchRaysIndex().xy,
                                  slot, pushConsts.sequenceSeed);
    }
    return pcg_float(payload.rngState);
}

float2 PathSample2D(inout Payload payload, uint slot) {
    if (payload.depth == 0u) {
        return StratifiedSample2D(pushConsts.sampleIndex, DispatchRaysIndex().xy,
                                  slot, pushConsts.sequenceSeed);
    }
    return float2(pcg_float(payload.rngState), pcg_float(payload.rngState));
}

// ============================================================================
// Hit Attributes
// ============================================================================
// Barycentric coordinates of hit point within triangle
// ============================================================================

struct HitAttributes {
    [[vk::location(0)]] float2 bary : SV_Barycentrics;  // Barycentric coordinates (b1, b2), where b0 = 1 - b1 - b2
};

// ============================================================================
// Helper Functions
// ============================================================================

// Safe normalize: returns fallback if vector is near-zero to prevent NaN/Inf
// This is CRITICAL for GPU stability - normalize() on zero vectors causes crashes
float3 SafeNormalize(float3 v, float3 fallback) {
    float lenSq = dot(v, v);
    if (lenSq < 1e-8) {
        return fallback;
    }
    return v * rsqrt(lenSq);
}

// Safe normalize with default fallback to up vector
float3 SafeNormalize(float3 v) {
    return SafeNormalize(v, float3(0.0, 1.0, 0.0));
}

// Maximum valid texture index (must match MAX_TEXTURES in RayTracingPipeline.cpp)
// CRITICAL: This bounds check prevents GPU hangs from invalid descriptor access
static const int MAX_TEXTURE_INDEX = 1024;

// Ray tracing has no screen-space derivatives, so the mip level has to be
// stated rather than inferred -- and every fetch here states 0.
//
// There used to be a ray-differential LOD path: the payload carried dD/dx and
// dD/dy, and ComputeUVDifferentialX/Y turned them into a UV footprint. It never
// worked and never ran. Both of those functions returned a hardcoded heuristic
// (`float2(t * length(dDdx), 0) * 0.001`) rather than an actual UV gradient,
// neither had a single caller, and the only path into the LOD computation passed
// literal zeros -- whose log2 is -inf, so the clamp handed back LOD 0 anyway.
// 24 of the payload's 56 bytes existed to feed it.
//
// Filtering the indirect bounces would want this back, and would want it
// computed rather than guessed. Until then LOD 0 is what the renderer does, said
// once, in one place.
float4 SampleTexture(int textureIndex, int samplerIndex, float2 uv, float4 fallback) {
    // Check both lower AND upper bounds to prevent invalid descriptor access
    // Invalid indices (negative or out-of-range) can cause GPU hangs with PARTIALLY_BOUND descriptors
    if (textureIndex < 0 || textureIndex >= MAX_TEXTURE_INDEX) {
        return fallback;
    }
    // Ensure sampler index is also valid (use same index as texture for 1:1 mapping)
    if (samplerIndex < 0 || samplerIndex >= MAX_TEXTURE_INDEX) {
        return fallback;
    }
    return textures[NonUniformResourceIndex(textureIndex)].SampleLevel(
        samplers[NonUniformResourceIndex(samplerIndex)], uv, 0.0
    );
}

// ============================================================================
// Endmember Mixing
// ============================================================================
// rho(lambda, uv) = sum_i w_i(uv) * rho_i(lambda)
//
// A bound spectral curve replaces the base-colour texture, so a measured
// surface renders as one flat reflectance and loses everything the texture
// said about where it varies. Mixing several measured curves per texel gives
// that back without giving up the measurement: the spectra stay exactly what
// was measured, and only how much of each is present varies across the
// surface.
//
// The weights come from a texture the loader unmixes out of the base colour,
// channel i holding w_i / 2. Halved because a single-endmember mixture is
// brightness modulation and needs w > 1 wherever the texture is brighter than
// the curve's own colour, which a UNORM texture cannot otherwise store.
// ============================================================================

// Weights for this texel. No weight texture means the first endmember alone,
// which reproduces the flat single-curve behaviour exactly.
float4 SampleEndmemberWeights(MaterialData material, float2 uv) {
    if (material.weightTextureIndex < 0) {
        return float4(1.0, 0.0, 0.0, 0.0);
    }

    // LOD 0: the weight map is data, and a mip average of it would blend
    // materials that are not adjacent in the mixture.
    float4 w = 2.0 * SampleTexture(material.weightTextureIndex,
                                   material.weightTextureIndex, uv,
                                   float4(0.5, 0.0, 0.0, 0.0));

    // A texel with no material in it at all is a black surface, which is
    // almost always a hole in the unmix rather than a physical black. Fall
    // back to the first curve rather than render nothing.
    if (w.x + w.y + w.z + w.w < 1e-4) {
        return float4(1.0, 0.0, 0.0, 0.0);
    }
    return w;
}

// ============================================================================
// Surface Temperature
// ============================================================================
// One decode for every reader: the emission paths and the debug views must
// agree on what the surface temperature is, or a temperature map renders hot
// while its debug view stays flat.
//
// Three sources, most specific first:
//
//   the thermal solver, per triangle -- an energy balance decided this, and
//     nothing a config says should override it
//   a temperature texture, per texel -- R channel normalised [0, 1], the
//     material carrying the kelvin mapping
//   the material's own scalar
//
// A solved element of exactly zero means the solver ran but skipped this
// surface (no thermal properties, or a degenerate triangle), which falls
// through to the two below it.
float GetSurfaceTemperatureK(MaterialData material, InstanceGeometryInfo geoInfo,
                             uint primitiveIndex, float2 uv) {
    if (geoInfo.thermalElementBase != 0xFFFFFFFFu) {
        const float solved = thermalTemperatures[geoInfo.thermalElementBase + primitiveIndex];
        if (solved > 0.0) {
            return solved;
        }
    }

    float T = material.irTemperature_K;
    if (material.temperatureTextureIndex >= 0) {
        float texTemp = SampleTexture(
            material.temperatureTextureIndex,
            material.temperatureTextureIndex,  // 1:1 texture/sampler mapping
            uv,
            float4(0.5, 0, 0, 0)  // mid-range if the texture is missing
        ).r;
        T = texTemp * material.temperatureScale + material.temperatureOffset;
    }
    return T;
}

// ============================================================================
// Downwelling Sky Radiance
// ============================================================================
// What the whole hemisphere sends back at one wavelength. Every thermal path
// needs it three times -- as the control variate the bounce ray corrects, as
// the surface's reflected term, and as the background behind a transmitting
// surface -- and the three MUST agree: the bounce estimator carries
// (L_in - L_base), so a base computed one way and a reflected term computed
// another turns an open scene's exactly-zero correction into a bias.
//
// Three skies in order of what the scene asked for: the network's measured
// downwelling, the analytic clear sky hemispherically averaged, or the
// isotropic blackbody a scene that named neither has always had.
float IRDownwellingRadiance(AtmosNNHeader atmos, uint atmosIdx, float lambda_nm,
                            float T_atmosphere, float skyEmissivityClear) {
    if (atmos.enabled != 0 && atmos.hasLdown != 0) {
        return SampleAtmosLdown(atmos, atmosNNData, atmosIdx);
    }
    if (skyEmissivityClear > 0.0) {
        return IRClearSkyHemisphericalRadiance(skyEmissivityClear, T_atmosphere, lambda_nm);
    }
    return IRPlanckRadiance(T_atmosphere, lambda_nm);
}

// Mixed reflectance at one wavelength. The weights are NOT normalised -- a
// texel darker than every endmember is a legitimately dimmer patch of the same
// material, which is the whole point -- so the sum is clamped instead.
// The mixture at one wavelength, given weights already sampled.
//
// Split from the texture fetch because the weights do not depend on lambda and
// the caller loops over 32 of them: fetching inside meant re-reading the same
// texel at the same LOD 32 times per hit, 16 in the IR bands.
float EvaluateEndmemberReflectanceW(StructuredBuffer<SpectralCurveGPU> curves,
                                    MaterialData material, float4 w, float lambda) {
    float rho = w.x * EvaluateSpectralCurve(curves, material.spectralReflectanceCurveIndex, lambda);
    if (material.endmemberCurveIndex1 >= 0) {
        rho += w.y * EvaluateSpectralCurve(curves, material.endmemberCurveIndex1, lambda);
    }
    if (material.endmemberCurveIndex2 >= 0) {
        rho += w.z * EvaluateSpectralCurve(curves, material.endmemberCurveIndex2, lambda);
    }
    if (material.endmemberCurveIndex3 >= 0) {
        rho += w.w * EvaluateSpectralCurve(curves, material.endmemberCurveIndex3, lambda);
    }
    return saturate(rho);
}

// Convenience for the single-evaluation callers, which have no loop to hoist
// the fetch out of.
float EvaluateEndmemberReflectance(StructuredBuffer<SpectralCurveGPU> curves,
                                   MaterialData material, float2 uv, float lambda) {
    return EvaluateEndmemberReflectanceW(curves, material,
                                         SampleEndmemberWeights(material, uv), lambda);
}

// Compute TBN matrix for normal mapping (Gram-Schmidt orthogonalization)
// N: geometric normal, T: tangent, handedness: ±1 for bitangent direction
// returns orthonormal TBN matrix with correct bitangent orientation
// FIXED: Apply handedness to bitangent to fix mirrored UV and inverted normal map issues
float3x3 ComputeTBN(float3 N, float3 T, float handedness) {
    // Orthogonalize tangent with respect to normal (Gram-Schmidt)
    // Use SafeNormalize to handle edge case where T is parallel to N
    float3 T_ortho = T - N * dot(N, T);
    T = SafeNormalize(T_ortho, T);  // Fallback to original T if orthogonalized is zero

    // Compute bitangent and NORMALIZE it
    // FIXED: cross(N, T) must be normalized for correct TBN transform
    // FIXED: Apply handedness (tangent.w) to bitangent direction
    // glTF spec: handedness = +1 for right-handed (bitangent = cross(N,T))
    //            handedness = -1 for left-handed (bitangent = -cross(N,T))
    float3 B = SafeNormalize(cross(N, T), cross(N, float3(1.0, 0.0, 0.0)));
    B *= handedness;

    return float3x3(T, B, N);
}

// Transform normal from tangent space to world space
// FIXED: Added handedness parameter for correct bitangent orientation
float3 ApplyNormalMap(float3 tangentNormal, float3 worldNormal, float3 worldTangent, float handedness) {
    // Validate tangent normal before transformation
    // If tangent normal is degenerate, return geometric normal
    float tangentLenSq = dot(tangentNormal, tangentNormal);
    if (tangentLenSq < 1e-8 || !isfinite(tangentLenSq)) {
        return worldNormal;
    }

    // Build TBN matrix with handedness
    float3x3 TBN = ComputeTBN(worldNormal, worldTangent, handedness);

    // Transform tangent-space normal to world space
    float3 normal = mul(tangentNormal, TBN);

    // Use SafeNormalize with geometric normal as fallback
    return SafeNormalize(normal, worldNormal);
}

// ============================================================================
// Physical Fresnel F0 Computation
// ============================================================================
// Computes F0 (normal incidence reflectance) using physical n,k data when
// available, falling back to standard PBR approximation otherwise.
//
// PHYSICAL PATH (complexRefractiveIndexIndex >= 0):
//   - Query measured complex refractive index at current wavelength
//   - Use exact Fresnel equation: F0 = [(n-1)² + k²] / [(n+1)² + k²]
//   - Provides wavelength-dependent specular reflection (gold, copper, etc.)
//
// PBR PATH (complexRefractiveIndexIndex < 0):
//   - Use standard approximation: F0 = lerp(0.04, albedo, metallic)
//   - 0.04 = typical dielectric F0 (glass, plastic)
//   - Albedo used for metals (color tinting at normal incidence)
//
// PARAMETERS:
//   material: MaterialData with complexRefractiveIndexIndex
//   albedo: Base color (used for PBR metallic tinting)
//   metallic: Metalness factor [0, 1]
//   wavelength_nm: Current wavelength for spectral lookup
//
// RETURNS:
//   float3 F0 - normal incidence reflectance (replicated to RGB for non-spectral)
// ============================================================================

float3 ComputePhysicalF0(MaterialData material, float3 albedo, float metallic, float wavelength_nm) {
    if (material.complexRefractiveIndexIndex >= 0) {
        // PHYSICAL PATH: Use measured n,k data from RefractiveIndex.INFO
        float2 nk = SampleComplexRefractiveIndex(complexRefractiveIndices, material.complexRefractiveIndexIndex, wavelength_nm);
        float n = nk.x;
        float k = nk.y;

        // Fresnel at normal incidence: F0 = [(n-1)² + k²] / [(n+1)² + k²]
        float F0_physical = FresnelF0(n, k);

        // For spectral mode, return scalar F0 replicated to RGB
        // For RGB mode, this is an approximation (should sample at R/G/B wavelengths)
        return float3(F0_physical, F0_physical, F0_physical);
    }

    // PBR PATH: Standard approximation
    // Dielectrics: F0 ≈ 0.04 (glass, plastic, water)
    // Metals: F0 = albedo (color tinting from base color)
    return lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
}

// Compute full Fresnel reflectance at arbitrary angle using physical n,k data
// Used for specular highlight computation (not just F0)
float ComputePhysicalFresnel(MaterialData material, float cosTheta, float wavelength_nm) {
    if (material.complexRefractiveIndexIndex >= 0) {
        float2 nk = SampleComplexRefractiveIndex(complexRefractiveIndices, material.complexRefractiveIndexIndex, wavelength_nm);
        return FresnelConductor(cosTheta, nk.x, nk.y);
    }

    // Fallback: Use Schlick approximation with typical dielectric F0
    return FresnelSchlick(cosTheta, 0.04);
}

// ============================================================================
// Environment Bounce: one ray, one wavelength, residual form
// ============================================================================
// Every spectral band owes the same integral -- what the hemisphere sends back
// after reflecting off whatever is out there -- and every one of them already
// computes an analytic answer to it for the special case of a uniform sky.
// This function supplies the difference between the two:
//
//     Corr = rho(lambda_b) * W_dir * ( L_in(lambda_b) - L_base(lambda_b) )
//
// so a branch adds Corr to its existing analytic term and is unbiased, rather
// than deleting that term and rebuilding the whole integral stochastically.
// Writing it as a residual is what makes it usable at low sample counts: where
// the analytic assumption holds -- an unobstructed sky, an isothermal cavity --
// the difference is identically zero, so an open scene renders exactly as
// before and a closed one picks up the interreflection that was missing.
//
// The caller owns the wavelength. It samples lambda_b uniformly over its band
// (pdf = 1/bandwidth, which cancels the 1/bandwidth of a band average), or
// inherits one from payload.heroLambda, and evaluates BOTH rho and L_base
// there. Evaluating them at different wavelengths is the error this replaced.
//
// Russian roulette lives here too. Below BOUNCE_DEPTH_DETERMINISTIC every
// reflecting surface bounces; above it a path survives with probability
// rrSurvive and the weight is divided by it -- so passing the surface's own
// reflectance collapses the weight to 1 and a mirror keeps bouncing at full
// strength while a near-black surface almost never does. Killing a path is
// safe because the caller's analytic term is still paid: the fallback is the
// sky, not zero, so there is no closure to substitute and nothing to bias.
//
// Lobe selection is the caller's, through qSpec:
//
//   qSpec = 0    always cosine-sample, weight wDiffuse
//   qSpec = 1    always GGX-sample,    weight wSpecular * G*VdotH/(NdotV*NdotH)
//   in between   pick stochastically and divide by the selection probability
//
// The thermal and reflective IR bands pass 0 or 1 -- they model one total
// reflectance and the lobe is purely a sampling shape for it. VIS_FUSED passes
// a Fresnel-weighted probability, because there the two lobes carry physically
// different terms (kD*rho against F) and both have to be reachable.
//
// Returns 0 when no bounce was taken, which is the correct correction to a
// base the caller has already added.
// ============================================================================

// ============================================================================
// Next-event estimation: the light, sampled directly
// ============================================================================
// A path finds an emitter two ways. It can bounce and land on one, which is
// what this renderer did exclusively -- and which, in a room lit only by a
// ceiling panel, means almost every sample returns nothing and the image needs
// thousands of them. Or the shading point can pick a point on an emitter and
// ask whether it is visible, which costs one shadow ray and always returns
// something.
//
// Neither is good at everything. Light sampling collapses for a near-mirror,
// where the BRDF is nonzero only in a sliver of directions the light sampler
// almost never picks; BSDF sampling collapses for a small or distant light. So
// both run, and each contribution is scaled by the power heuristic over the two
// densities. The weights sum to one at every direction, so the total is still
// an unbiased estimate of the same integral -- MIS buys variance, never energy.
//
// The two densities must be expressed in the same measure. Light sampling is
// natural in area measure and is converted: p_omega = p_A * d^2 / |cos_light|.
// ============================================================================

struct LightSample {
    float3 wi;          // shading point towards the light
    float  dist;        // distance to the sampled point
    float3 emissive;    // the emitter's RGB emissive factor
    float  pdfSolid;    // density in solid angle measure at the shading point
    bool   valid;
};

// The density with which the BSDF lobes would have produced direction wi.
// The MIXTURE over both lobes, not the chosen one's: this is compared against
// the light's density on both sides of the weight, and both sides have to be
// evaluating the same function of direction.
float BsdfMixturePdf(float3 normal, float3 V, float3 wi, float roughness, float qSpec) {
    roughness = max(roughness, MIN_BOUNCE_ROUGHNESS);
    const float NdotWi = dot(normal, wi);
    if (NdotWi <= 0.0) {
        return 0.0;
    }
    float pdf = (1.0 - qSpec) * NdotWi / PI;
    if (qSpec > 0.0) {
        // The visible-normal density SampleGGXVNDF draws from, which is
        // G1 D / (4 NdotV) -- no half-vector term, the VdotH cancels.
        const float3 H     = normalize(V + wi);
        const float  NdotH = max(dot(normal, H), 0.0);
        const float  NdotV = max(dot(normal, V), 1e-4);
        const float  alpha = roughness * roughness;
        pdf += qSpec * SmithG1_GGX(NdotV, alpha) *
               DistributionGGX(NdotH, alpha) / (4.0 * NdotV);
    }
    return pdf;
}

// The BRDF that TraceEnvBounceResidual is implicitly sampling, evaluated at an
// arbitrary direction.
//
// Light sampling and BSDF sampling only combine into an unbiased estimator if
// they are estimating the same integral, which means evaluating the same f.
// Reaching for CookTorranceBRDF_Spectral here instead looks right and is not:
// that function takes its Fresnel at the half vector, the way direct sun does,
// while the bounce takes it at the view vector, the way the split-sum ambient
// terms it is correcting do. Two different BRDFs, so the MIS weights partition
// one integral while the strategies estimate another. It reads as a scene that
// gets brighter the more grazing the view -- 25% on the side walls of a Cornell
// box, 3% on the back wall, which is what pointed at the cause.
//
// So this inverts the bounce's own weights instead. Given weight = f cos / pdf
// for each lobe:
//
//   cosine lobe: weight = wDiffuse/(1-qSpec), pdf = cos/PI   =>  f = wDiffuse/PI
//   GGX lobe:    weight = wSpecular/qSpec * G VdotH/(NdotV NdotH),
//                pdf    = D NdotH/(4 VdotH)
//                                        =>  f = wSpecular G D / (4 NdotV NdotWi)
//
// The specular term is gated on qSpec because a surface with qSpec == 0 never
// samples that lobe, so it is not part of what the bounce estimates and must
// not be part of what the light sampler estimates either.
float EvalBounceBrdf(float3 normal, float3 V, float3 wi, float NdotV,
                     float roughness, float qSpec,
                     float wDiffuse, float wSpecular) {
    roughness = max(roughness, MIN_BOUNCE_ROUGHNESS);
    const float NdotWi = dot(normal, wi);
    if (NdotWi <= 0.0) {
        return 0.0;
    }
    float f = wDiffuse / PI;
    if (qSpec > 0.0) {
        const float3 H     = normalize(V + wi);
        const float  NdotH = max(dot(normal, H), 0.0);
        const float  alpha = roughness * roughness;
        const float  D     = DistributionGGX(NdotH, alpha);
        const float  G     = GeometrySmith_IBL(NdotV, NdotWi, roughness);
        f += wSpecular * D * G / (4.0 * max(NdotV, 1e-4) * NdotWi);
    }
    return f;
}

// Pick an emitter in proportion to its power, then a point uniformly on it.
//
// Because the triangle is chosen with probability (luminance * area) / total
// and the point uniformly within it, the area-measure density is
// luminance / total -- the area cancels. That is what lets an emitter compute
// its own sampling density later from nothing but its material.
LightSample SampleEmissiveGeometry(float3 hitPos, inout Payload payload) {
    LightSample s;
    s.wi = float3(0.0, 1.0, 0.0);
    s.dist = 0.0;
    s.emissive = float3(0.0, 0.0, 0.0);
    s.pdfSolid = 0.0;
    s.valid = false;

    const uint  count = lightingParams[0].emissiveTriangleCount;
    const float total = lightingParams[0].emissiveTotalPower;
    if (count == 0u || total <= 0.0) {
        return s;
    }

    // First entry whose running power passes the target. Binary search, because
    // the list is as long as the scene's emissive geometry and nothing bounds
    // that: a Cornell box has 2 triangles, a glTF model with an emissive
    // material has as many as the mesh does. This was a linear scan, which on a
    // 15452-triangle emitter cost 190x the whole sample -- 1.13 ms to 214.65 --
    // and on a larger scene ran past the driver's watchdog and took the process
    // with it. cumulativePower is non-decreasing by construction, so the search
    // is exact and the loop is 14 iterations there instead of 7700.
    const float target = PathSample1D(payload, SAMPLE_SLOT_LIGHT_PICK) * total;
    uint lo = 0u;
    uint hi = count - 1u;
    [loop] while (lo < hi) {
        const uint mid = lo + (hi - lo) / 2u;
        if (emissiveTriangles[mid].cumulativePower >= target) {
            hi = mid;
        } else {
            lo = mid + 1u;
        }
    }
    const uint idx = lo;

    const EmissiveTriangleGPU tri = emissiveTriangles[idx];

    // Uniform on the triangle: fold the far half of the unit square back.
    const float2 uv = PathSample2D(payload, SAMPLE_SLOT_LIGHT_UV);
    float u = uv.x;
    float v = uv.y;
    if (u + v > 1.0) {
        u = 1.0 - u;
        v = 1.0 - v;
    }
    const float3 lightPoint = tri.v0 + u * tri.edge1 + v * tri.edge2;

    const float3 toLight = lightPoint - hitPos;
    const float  dist2   = dot(toLight, toLight);
    if (dist2 < 1e-9) {
        return s;   // the shading point is on the emitter
    }
    const float dist = sqrt(dist2);
    const float3 wi = toLight / dist;

    const float3 lightNormal = SafeNormalize(cross(tri.edge1, tri.edge2));
    // Absolute, because emission here is two-sided: the closest-hit shader adds
    // a surface's emissive term whichever face was hit, so a light seen from
    // behind is lit. Taking a signed cosine would make the light sampler
    // disagree with what the BSDF side actually collects.
    const float cosAtLight = abs(dot(lightNormal, -wi));
    if (cosAtLight <= 1e-6) {
        return s;   // edge-on: it subtends nothing and p_omega would diverge
    }

    const float pdfArea = dot(tri.emissive, EMISSIVE_LUMINANCE_WEIGHTS) / total;

    s.wi       = wi;
    s.dist     = dist;
    s.emissive = tri.emissive;
    s.pdfSolid = pdfArea * dist2 / cosAtLight;
    s.valid    = s.pdfSolid > 0.0;
    return s;
}

// Is the sampled point on the emitter visible from the shading point?
bool LightSampleVisible(float3 hitPos, float3 normal, LightSample s) {
    RayDesc shadowRay;
    shadowRay.Origin    = hitPos + normal * 1e-3;
    shadowRay.Direction = s.wi;
    shadowRay.TMin      = 0.001;
    // Stop short of the emitter, or the emitter itself is the occluder.
    shadowRay.TMax      = s.dist * 0.999;

    Payload shadowPayload;
    shadowPayload.radiance    = float3(0.0, 0.0, 0.0);
    shadowPayload.isShadowed  = 1;   // cleared by shadow_miss
    shadowPayload.depth       = 0;
    shadowPayload.rngState    = 0;
    shadowPayload.heroLambda  = 0.0;
    shadowPayload.primaryHitT = -1.0;
    shadowPayload.bsdfPdf     = 0.0;

    TraceRay(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             0xFF, 0, 0, 1, shadowRay, shadowPayload);

    return shadowPayload.isShadowed == 0;
}

// Power heuristic, beta = 2, guarded so that a zero opposing density gives the
// whole contribution to this strategy rather than 0/0.
float PowerHeuristic(float thisPdf, float otherPdf) {
    const float a = thisPdf * thisPdf;
    const float b = otherPdf * otherPdf;
    const float denom = a + b;
    return denom > 0.0 ? a / denom : 0.0;
}

// The weight an emitter's own emission keeps when the path arrived by a BSDF
// bounce. 1 when nothing else could have found it -- no emitters registered, or
// this ray was not a BSDF sample (primary, refracted, shadow).
//
// The density the light sampler would have used is recovered from the
// material alone: p_A = luminance(emissive) / totalPower, converted to solid
// angle with this hit's own distance and grazing angle.
float EmissiveMisWeight(float3 emissive, int emissiveTextureIndex,
                        float3 geometricNormal,
                        float3 rayDir, float hitT, float bsdfPdf) {
    const uint  count = lightingParams[0].emissiveTriangleCount;
    const float total = lightingParams[0].emissiveTotalPower;
    if (bsdfPdf <= 0.0 || count == 0u || total <= 0.0 || hitT <= 0.0) {
        return 1.0;
    }
    // A texture-modulated emitter is not in the list -- see
    // CollectEmissiveTriangles for why -- so nothing sampled it and this
    // surface keeps all of its emission. The two sides read the same field to
    // decide, which is what keeps them from disagreeing.
    if (emissiveTextureIndex >= 0) {
        return 1.0;
    }
    const float cosAtLight = abs(dot(geometricNormal, rayDir));
    if (cosAtLight <= 1e-6) {
        return 1.0;
    }
    const float pdfArea  = dot(emissive, EMISSIVE_LUMINANCE_WEIGHTS) / total;
    const float pdfLight = pdfArea * hitT * hitT / cosAtLight;
    return PowerHeuristic(bsdfPdf, pdfLight);
}

float TraceEnvBounceResidual(float3 hitPos, float3 normal, float3 V, float NdotV,
                             float roughness, float qSpec,
                             float wDiffuse, float wSpecular, float rrSurvive,
                             float L_base_b, float lambda_b, inout Payload payload)
{
    if (payload.depth >= MAX_PATH_DEPTH || rrSurvive <= 0.0) {
        return 0.0;
    }
    // The same floor BsdfMixturePdf and EvalBounceBrdf apply; all three must
    // see one lobe.
    roughness = max(roughness, MIN_BOUNCE_ROUGHNESS);

    float weight = 1.0;
    if (payload.depth >= BOUNCE_DEPTH_DETERMINISTIC) {
        // Never reached at depth 0, so this draw has no stratified slot: it is
        // a deep-path decision and PCG is the right source for it.
        if (pcg_float(payload.rngState) >= rrSurvive) {
            return 0.0;
        }
        weight = 1.0 / rrSurvive;
    }

    // Which lobe to sample from. A rough surface scatters near-uniformly and a
    // cosine distribution fits it; a smooth one concentrates around the mirror
    // direction and needs GGX to find it at all.
    const bool specularLobe = (qSpec >= 1.0)
        ? true
        : (qSpec > 0.0 && PathSample1D(payload, SAMPLE_SLOT_LOBE) < qSpec);

    if (specularLobe) {
        weight *= wSpecular / max(qSpec, 1e-4);
    } else {
        weight *= wDiffuse / max(1.0 - qSpec, 1e-4);
    }
    if (weight == 0.0) {
        return 0.0;
    }

    // One 2D draw feeds whichever lobe was chosen. The two lobes share the slot
    // because they are the same decision -- which way does this path go -- and
    // only one of them is ever taken per bounce.
    const float2 uDir = PathSample2D(payload, SAMPLE_SLOT_DIRECTION);

    float pdf_dir;
    const float3 wi = specularLobe
        ? SampleGGXVNDF(normal, V, roughness * roughness, uDir, pdf_dir)
        : CosineSampleHemisphere(normal, uDir, pdf_dir);

    const float NdotWi = dot(normal, wi);
    // Negated comparisons, so that a NaN fails them. Written the other way a
    // NaN pdf passes every guard -- every comparison against NaN is false --
    // and rides out as a NaN weight to be zeroed much later, as black speckle.
    if (!(NdotWi > 0.0) || !(pdf_dir > 1e-6)) {
        return 0.0;
    }

    // f * cos / pdf, with the reflectance already folded into weight.
    //
    //   cosine lobe: f = 1/PI and pdf = cos/PI, so the whole thing is 1.
    //   GGX lobe:    sampling visible normals gives pdf = G1 D / (4 NdotV), and
    //                f cos = D G / (4 NdotV), so D and NdotV both cancel and
    //                what is left is the masking ratio G / G1.
    //
    // That ratio is at most 1, which is the point of sampling visible normals.
    // Drawing from D alone instead left G VdotH / (NdotV NdotH) here, unbounded
    // because NdotH is floored at 1e-4 and NDF sampling is exactly what puts H
    // far from N -- one sample worth thousands of its neighbours, carried by
    // the running average long after.
    //
    // G is the IBL remap because this is weighed against a split-sum base built
    // with that remap (see GeometrySmith_IBL), while G1 is the exact Smith term
    // the sampler actually drew from. They are deliberately not the same
    // function: one belongs to the BRDF, the other to the density.
    if (specularLobe) {
        const float alpha = roughness * roughness;
        weight *= GeometrySmith_IBL(NdotV, NdotWi, roughness) /
                  max(SmithG1_GGX(NdotV, alpha), 1e-4);
    }

    RayDesc bounceRay;
    bounceRay.Origin    = hitPos + normal * 1e-3;
    bounceRay.Direction = wi;
    bounceRay.TMin      = 0.0;
    bounceRay.TMax      = 1e9;

    Payload child;
    child.radiance    = float3(0.0, 0.0, 0.0);
    child.isShadowed  = 0;
    child.depth       = payload.depth + 1;
    child.rngState    = payload.rngState;
    child.heroLambda  = lambda_b;
    child.primaryHitT = -1.0;
    // If this ray lands on an emitter, that surface has to know how likely the
    // BSDF was to have sent a ray its way, so it can weigh its emission against
    // the light sampling the same vertex did. The mixture density, evaluated at
    // the direction actually taken -- see BsdfMixturePdf.
    child.bsdfPdf     = BsdfMixturePdf(normal, V, wi, roughness, qSpec);
    TraceRay(scene, RAY_FLAG_NONE, 0xFF, 0, 0, 0, bounceRay, child);

    // Carry the child's consumption forward, or the parent's later draws
    // repeat numbers the child already used. With paths this long that
    // correlation is visible.
    payload.rngState = child.rngState;

    // Scalar spectral radiance at lambda_b, by the contract on
    // Payload::heroLambda.
    return weight * (child.radiance.r - L_base_b);
}

// ============================================================================
// Closest Hit Entry Point
// ============================================================================

[shader("closesthit")]
void main(inout Payload payload, in HitAttributes attribs) {
    // Record this ray's hit distance for the depth AOV. Only the depth-0
    // (primary) value survives: raygen snapshots its own payload right after
    // the primary trace, and recursive rays carry separate Payload instances.
    payload.primaryHitT = RayTCurrent();

    // ========================================================================
    // Get instance geometry info for multi-BLAS support
    // ========================================================================
    // InstanceIndex() returns the TLAS instance index
    // instanceGeometryInfo provides offsets into merged global geometry buffers
    // This allows correct geometry access when scene has multiple BLAS
    // ========================================================================

    uint instanceIdx = InstanceIndex();
    InstanceGeometryInfo geoInfo = instanceGeometryInfo[instanceIdx];

    // Material ID is stored in InstanceGeometryInfo (not InstanceID anymore)
    uint materialID = geoInfo.materialId;
    MaterialData material = materials[materialID];

    // ========================================================================
    // Compute geometric normal from triangle vertices
    // ========================================================================

    uint primitiveID = PrimitiveIndex();

    // Read triangle indices with offset into global index buffer
    uint idx0 = indexBuffer[geoInfo.indexOffset + primitiveID * 3 + 0];
    uint idx1 = indexBuffer[geoInfo.indexOffset + primitiveID * 3 + 1];
    uint idx2 = indexBuffer[geoInfo.indexOffset + primitiveID * 3 + 2];

    // Read vertex positions with offset into global vertex buffer
    // Use ByteAddressBuffer.Load3 with explicit byte offset (12 bytes per vec3)
    // This avoids StructuredBuffer<float3> stride alignment issues (some drivers use 16-byte stride)
    float3 v0 = asfloat(vertexBuffer.Load3((geoInfo.vertexOffset + idx0) * 12));
    float3 v1 = asfloat(vertexBuffer.Load3((geoInfo.vertexOffset + idx1) * 12));
    float3 v2 = asfloat(vertexBuffer.Load3((geoInfo.vertexOffset + idx2) * 12));

    // ========================================================================
    // Compute TRUE geometric normal from triangle edges (flat normal)
    // This is the actual surface normal, NOT the interpolated shading normal
    // ========================================================================
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 objectGeometricNormal = SafeNormalize(cross(edge1, edge2), float3(0.0, 1.0, 0.0));

    // Transform geometric normal to world space using inverse-transpose
    // mul(v, M) = v * M = transpose(M) * v, and we want transpose(inverse(ObjectToWorld)) * v
    // WorldToObject = inverse(ObjectToWorld), so mul(v, WorldToObject) = transpose(WorldToObject) * v ✓
    float3x3 normalTransform = (float3x3)WorldToObject3x4();
    float3 worldGeometricNormal = SafeNormalize(mul(objectGeometricNormal, normalTransform), float3(0.0, 1.0, 0.0));

    // Face forward: ensure geometric normal points toward camera (opposite to ray direction)
    float3 rayDir = WorldRayDirection();
    if (dot(worldGeometricNormal, rayDir) > 0.0) {
        worldGeometricNormal = -worldGeometricNormal;
    }

    // Read per-vertex normals with offset into global normal buffer
    // This gives smooth shading (Gouraud/Phong) instead of flat shading
    float3 n0 = asfloat(normalBuffer.Load3((geoInfo.normalOffset + idx0) * 12));
    float3 n1 = asfloat(normalBuffer.Load3((geoInfo.normalOffset + idx1) * 12));
    float3 n2 = asfloat(normalBuffer.Load3((geoInfo.normalOffset + idx2) * 12));

    // Barycentric interpolation: n = n0 * w0 + n1 * w1 + n2 * w2
    // where w0 = (1 - bary.x - bary.y), w1 = bary.x, w2 = bary.y
    float3 objectNormal = n0 * (1.0 - attribs.bary.x - attribs.bary.y)
                        + n1 * attribs.bary.x
                        + n2 * attribs.bary.y;
    objectNormal = SafeNormalize(objectNormal, float3(0.0, 1.0, 0.0));

    // Transform shading normal to world space
    // FIXED: Use SafeNormalize to prevent NaN propagation
    float3 worldNormal = SafeNormalize(mul(objectNormal, normalTransform), float3(0.0, 1.0, 0.0));

    // Ensure smooth shading normal and geometric normal are in the same hemisphere
    // This prevents shading artifacts from normal interpolation across hard edges
    if (dot(worldNormal, worldGeometricNormal) < 0.0) {
        worldNormal = -worldNormal;
    }

    // UV coordinates with offset into global UV buffer
    // Barycentric interpolation: uv = u0 * (1 - b1 - b2) + u1 * b1 + u2 * b2
    float2 uv0 = uvBuffer[geoInfo.uvOffset + idx0];
    float2 uv1 = uvBuffer[geoInfo.uvOffset + idx1];
    float2 uv2 = uvBuffer[geoInfo.uvOffset + idx2];
    float2 uv = uv0 * (1.0 - attribs.bary.x - attribs.bary.y) + uv1 * attribs.bary.x + uv2 * attribs.bary.y;

    // Read tangent from buffer with offset (or fallback to fake tangent)
    float3 worldTangent;
    float worldHandedness = 1.0;  // Default handedness (right-handed)
    if (material.normalTextureIndex >= 0) {  // Only compute tangent if normal map is used
        // Read tangents with offset into global tangent buffer
        float4 tangent4_0 = tangentBuffer[geoInfo.tangentOffset + idx0];
        float4 tangent4_1 = tangentBuffer[geoInfo.tangentOffset + idx1];
        float4 tangent4_2 = tangentBuffer[geoInfo.tangentOffset + idx2];

        // Barycentric interpolation of tangents
        float4 tangent4 = tangent4_0 * (1.0 - attribs.bary.x - attribs.bary.y) +
                          tangent4_1 * attribs.bary.x +
                          tangent4_2 * attribs.bary.y;

        float3 tangent = tangent4.xyz;
        worldHandedness = tangent4.w;  // ±1 for bitangent orientation

        // Transform tangent to world space
        float3x3 objectToWorld = (float3x3)ObjectToWorld3x4();
        float3 tangentWorld = mul(tangent, objectToWorld);

        // Validate tangent - if invalid, fall back to fake tangent
        if (isfinite(dot(tangent, tangent)) && dot(tangent, tangent) > 1e-8) {
            worldTangent = SafeNormalize(tangentWorld, float3(1.0, 0.0, 0.0));
        } else {
            // Fallback: fake tangent (for backward compatibility)
            float3 refVector = abs(worldNormal.y) > 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
            worldTangent = SafeNormalize(cross(worldNormal, refVector), float3(1.0, 0.0, 0.0));
            worldHandedness = 1.0;  // Reset to default for fallback
        }
    } else {
        // No normal map: use fake tangent (doesn't matter since it won't be used)
        float3 refVector = abs(worldNormal.y) > 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
        worldTangent = SafeNormalize(cross(worldNormal, refVector), float3(1.0, 0.0, 0.0));
    }

    // ========================================================================
    // Sample textures
    // ========================================================================

    // Base color texture (glTF: baseColor = baseColorTexture * baseColorFactor)
    // Use white (1,1,1,1) as fallback so multiplication with factor works correctly
    float4 baseColor = SampleTexture(
        material.baseColorTextureIndex,
        material.baseColorTextureIndex,  // Use same index for sampler (1:1 mapping)
        uv,
        float4(1.0, 1.0, 1.0, 1.0)  // White fallback for correct factor multiplication
    );

    // Modulate with base color factor (texture * factor, or 1 * factor if no texture)
    baseColor *= material.baseColorFactor;

    // Metallic-Roughness texture (G=roughness, B=metallic)
    float4 metallicRoughness = SampleTexture(
        material.metallicRoughnessTextureIndex,
        material.metallicRoughnessTextureIndex,
        uv,
        float4(1.0, material.roughnessFactor, material.metallicFactor, 1.0)
    );

    float roughness = material.roughnessFactor * metallicRoughness.g;
    float metallic = material.metallicFactor * metallicRoughness.b;

    // Normal map (tangent space, [0,1] -> [-1,1])
    float3 normal = worldNormal;
    if (material.normalTextureIndex >= 0) {
        float3 tangentNormal = SampleTexture(
            material.normalTextureIndex,
            material.normalTextureIndex,
            uv,
            float4(0.5, 0.5, 1.0, 1.0)  // Default: pointing up in tangent space
        ).xyz;

        // Convert [0,1] to [-1,1]
        tangentNormal = tangentNormal * 2.0 - 1.0;

        // Clamp normalScale to reasonable range to prevent extreme values
        float clampedNormalScale = clamp(material.normalScale, 0.0, 10.0);
        tangentNormal.xy *= clampedNormalScale;

        // FIXED: Use SafeNormalize to handle edge case where tangent normal becomes near-zero
        // This can happen with extreme normalScale values or degenerate texture data
        tangentNormal = SafeNormalize(tangentNormal, float3(0.0, 0.0, 1.0));

        // Transform to world space (pass handedness for correct bitangent orientation)
        normal = ApplyNormalMap(tangentNormal, worldNormal, worldTangent, worldHandedness);
    }

    // CRITICAL: Face-forward correction for shading normal
    // Ensures normal always points toward camera (opposite to ray direction)
    // This prevents BRDF breakdown for thin geometry (grass, leaves) and backface hits
    // Without this, NdotV can be negative -> Fresnel/GGX produce NaN/white spots
    if (dot(normal, rayDir) > 0.0) {
        normal = -normal;
    }

    // Emissive texture
    // NOTE: glTF 2.0 spec allows emissiveFactor to exceed 1.0 (HDR emissive)
    // This is intentional for self-luminous surfaces (e.g., lights, displays, neon signs)
    // No clamping is applied here; emissive can be arbitrarily high for physically-based rendering
    // The final radiance will be clamped in the validation step to prevent NaN/Inf
    // The endmember mixture weights, fetched once. They do not vary with
    // wavelength, and every band below loops over 16 or 32 of those.
    const float4 endmemberW = SampleEndmemberWeights(material, uv);

    float3 emissive = material.emissiveFactor;
    if (material.emissiveTextureIndex >= 0) {
        emissive *= SampleTexture(
            material.emissiveTextureIndex,
            material.emissiveTextureIndex,
            uv,
            float4(1.0, 1.0, 1.0, 1.0)
        ).rgb;
    }

    // If a BSDF-sampled bounce landed here and this surface emits, the vertex
    // that sent the ray also sampled the emitters explicitly, and both would
    // report the same light. Keep the share the power heuristic gives this
    // strategy; the other share is added back at that vertex.
    //
    // Only in the modes that do light sampling. The mode is a specialization
    // constant, so this and the NEE terms below compile out together and the
    // thermal bands -- which spawn bounce rays but sample no lights -- keep
    // whatever emission they find. The density is built from the untextured
    // emissiveFactor, matching the CDF the host built.
    if (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_VIS_FUSED ||
        SPEC_SPECTRAL_MODE == SPECTRAL_MODE_SINGLE) {
        emissive *= EmissiveMisWeight(material.emissiveFactor,
                                      material.emissiveTextureIndex,
                                      worldGeometricNormal,
                                      rayDir, RayTCurrent(), payload.bsdfPdf);
    }

    // ========================================================================
    // Fetch sun/sky lighting data from LUT
    // ========================================================================

    LightingParams lut = lightingParams[0];
    // FIXED: Use SafeNormalize in case LUT data is invalid
    float3 sunDir = SafeNormalize(lut.sunDirection, float3(0.0, 1.0, 0.0));

    // Choose lighting based on spectral mode
    float3 sunRadiance;
    float3 skyRadiance;

    if (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_RGB || SPEC_SPECTRAL_MODE == SPECTRAL_MODE_VIS_FUSED) {
        // RGB and VIS_FUSED modes: Use full RGB lighting
        sunRadiance = lut.sunRadiance_rgb;
        skyRadiance = lut.skyRadiance_rgb;
    } else {
        // Spectral modes: Use scalar spectral radiance (replicate to RGB)
        //
        // Dead in these modes, and kept only because the declarations above are
        // shared. These two feed `radiance` (directSun + skyAmbient + ...),
        // which is consumed by the RGB branch alone -- every spectral branch
        // below builds its own output_radiance from the solar LUT per
        // wavelength. Do not read them as the spectral illuminant; the
        // *_spectral fields are an RGB average in arbitrary units, which is
        // exactly what the spectral paths stopped inventing.
        sunRadiance = float3(lut.sunRadiance_spectral, lut.sunRadiance_spectral, lut.sunRadiance_spectral);
        skyRadiance = float3(lut.skyRadiance_spectral, lut.skyRadiance_spectral, lut.skyRadiance_spectral);
    }

    // ========================================================================
    // NN Atmosphere (MODTRAN surrogate LUT)
    // ========================================================================
    // Composition contract applied per spectral sample at the end of each
    // mode branch:
    //   L_pixel(λ) = τ_view(λ) × L_surface(λ) + L_path(λ)
    // Surface illumination (sun + sky) stays on the binding-15 solar LUT --
    // the NN only supplies view-path transmittance, view-path radiance and
    // thermal downwelling. Only primary rays (depth 0) composite; secondary
    // rays return surface radiance for BRDF integration.
    // ========================================================================

    AtmosNNHeader atmos = atmosNNHeader[0];
    bool atmosEnabled = (atmos.enabled != 0) && (payload.depth == 0);
    float atmosA = 0.0;
    float atmosAz = 0.0;
    if (atmosEnabled) {
        atmosA = AtmosCoordA(atmos, WorldRayDirection(), RayTCurrent());
        atmosAz = AtmosRelAz(atmos, WorldRayDirection());
    }

    // ========================================================================
    // PBR Shading
    // ========================================================================

    // View direction (FROM surface TO camera)
    // FIXED: Use SafeNormalize to handle edge cases
    float3 V = SafeNormalize(-WorldRayDirection(), float3(0.0, 0.0, 1.0));

    // Light direction (FROM surface TO sun)
    float3 L = sunDir;

    // FIXED: Final validation of normal - if still invalid, use geometric normal
    if (!isfinite(dot(normal, normal)) || dot(normal, normal) < 1e-8) {
        normal = worldNormal;
    }

    // Compute PBR BRDF (Cook-Torrance)
    float3 albedo = baseColor.rgb;
    float3 brdf = CookTorranceBRDF(normal, V, L, albedo, metallic, roughness, material.complexRefractiveIndexIndex, pushConsts.camera.wavelength_nm);

    // Direct sun lighting with atmospheric attenuation (Beer-Lambert law)
    // L_out = BRDF * L_sun * τ(λ, d) * (N · L)
    // where τ(λ, d) is atmospheric transmittance computed from Beer-Lambert law
    // NOTE: view-path atmosphere is composited per spectral sample at branch end
    float NdotL = max(dot(normal, L), 0.0);

    // ========================================================================
    // Shadow Ray Tracing
    // ========================================================================
    // Trace a shadow ray toward the sun to determine if this point is occluded
    // Uses geometric normal for self-shadowing check to avoid "terminator problem"
    // ========================================================================

    float shadowFactor = 1.0;  // 1.0 = fully lit, 0.0 = fully shadowed

    // Shadow ray enable flag from LightingParams (configurable via renderer.enable_shadow_rays)
    // Can be disabled for debugging GPU crashes or driver issues
    //
    // LWIR excluded: its branch sets includeSolarReflection = false
    // unconditionally, because solar is under 0.1% of that band, so it is the
    // one mode that traces this ray and then reads nothing from it. The mode is
    // a specialization constant, so the whole block folds away in that variant
    // rather than costing a runtime branch in the others.
    const bool ENABLE_SHADOW_RAYS = (lut.enableShadowRays != 0) &&
                                    (SPEC_SPECTRAL_MODE != SPECTRAL_MODE_LWIR_FUSED);

    // Step 1: Geometric normal check - prevents self-shadowing artifacts on curved surfaces
    // Use worldGeometricNormal (computed earlier) instead of shading normal
    float NdotL_geom = dot(worldGeometricNormal, L);

    if (NdotL_geom <= 0.0) {
        // Surface is physically facing away from the light - self-shadowed
        // No need to trace shadow ray, directly treat as fully shadowed
        shadowFactor = 0.0;
    } else if (ENABLE_SHADOW_RAYS && NdotL > 0.0) {
        // Step 2: Surface faces the light (both geometric and shading normals)
        // Trace shadow ray to check for occlusion by other geometry

        // Compute shadow ray origin with geometric normal offset
        // CRITICAL: Use geometric normal for offset, NOT shading normal
        // Shading normal can push origin inside geometry on curved surfaces
        float3 hitPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
        const float SHADOW_BIAS = 0.001;  // Adjust based on scene scale
        float3 shadowOrigin = hitPos + worldGeometricNormal * SHADOW_BIAS;

        // Create shadow ray descriptor
        RayDesc shadowRay;
        shadowRay.Origin = shadowOrigin;
        shadowRay.Direction = L;  // Direction toward light (sun)
        shadowRay.TMin = 0.0;
        shadowRay.TMax = 1e10;  // Infinite distance (directional light)

        // Initialize shadow payload - assume shadowed (will be cleared by shadow_miss)
        Payload shadowPayload;
        shadowPayload.radiance = float3(0.0, 0.0, 0.0);
        shadowPayload.isShadowed = 1;  // Assume shadowed, shadow_miss will clear this
        shadowPayload.heroLambda = payload.heroLambda;
        shadowPayload.bsdfPdf = 0.0;   // not a BSDF sample; carries no emission

        // Trace shadow ray with optimized flags:
        // - RAY_FLAG_SKIP_CLOSEST_HIT_SHADER: Don't run closest hit, just check occlusion
        // - RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH: Accept first hit and terminate (fast shadow test)
        TraceRay(
            scene,                                              // Acceleration structure
            RAY_FLAG_SKIP_CLOSEST_HIT_SHADER |                  // Skip closest hit shader
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,           // Accept first hit, terminate search
            0xFF,                                               // Instance mask (all instances)
            0,                                                  // SBT hit group offset
            0,                                                  // SBT multiplier
            1,                                                  // Miss index = 1 (shadow_miss)
            shadowRay,
            shadowPayload
        );

        // shadowPayload.isShadowed: 1 = hit occluder (shadowed), 0 = miss (lit)
        shadowFactor = (shadowPayload.isShadowed == 0) ? 1.0 : 0.0;
    }

    // Apply shadow factor to direct sun lighting
    float3 directSun = brdf * sunRadiance * NdotL * shadowFactor;

    // ========================================================================
    // Image-Based Lighting (IBL) - Diffuse and Specular
    // ========================================================================

    // Compute F0 (reflectance at normal incidence) for Fresnel calculations
    // Uses physical n,k data when available for wavelength-accurate metal reflections
    float3 F0 = ComputePhysicalF0(material, albedo, metallic, pushConsts.camera.wavelength_nm);

    // ------------------------------------------------------------------------
    // Sky Radiance Hemispherical Integration (Diffuse Ambient)
    // ------------------------------------------------------------------------
    // Computes diffuse sky lighting by integrating sky radiance over hemisphere:
    //   L_sky = ∫_Ω L_sky(ω) × BRDF(ω) × (N · ω) dω
    //
    // For Lambertian BRDF (f = ρ/π), this simplifies to:
    //   L_sky = (ρ/π) × L_sky × ∫_Ω (N · ω) dω
    //         = (ρ/π) × L_sky × π
    //         = ρ × L_sky
    //
    // For PBR materials with Fresnel term, we account for:
    // - kD = diffuse reflection coefficient (energy not reflected specularly)
    // - Fresnel term reduces diffuse contribution at grazing angles
    // - Metallic materials have no diffuse reflection (kD ≈ 0)
    //
    // PHYSICAL INTERPRETATION:
    // - Sky radiance L_sky(λ) is the average radiance from sky dome
    // - Hemispherical integral ∫(N·ω)dω = π (solid angle of hemisphere)
    // - For Lambertian: outgoing radiance = albedo × incident irradiance
    //
    // NOTE: This is the "ambient" term in traditional graphics, but physically
    // it represents diffuse reflection of scattered sky radiance.
    //
    // SKY MODEL LIMITATION (E/π Lambertian approximation):
    // - This assumes uniform sky dome radiance: L_sky ≈ E_sky / π
    // - Real atmosphere has anisotropic scattering (Rayleigh phase function)
    // - Horizon is typically brighter than zenith (gradient sky model)
    // - Circumsolar region is significantly brighter (sun aureole)
    // - This approximation is accurate for zenith-facing surfaces (~5% error)
    // - For low-angle surfaces facing horizon: up to ~20% error
    // - For quantitative analysis, consider CIE sky models or precomputed LUTs
    // ========================================================================

    // Compute diffuse reflection coefficient (energy conservation with specular)
    float3 kD = (1.0 - FresnelSchlickRGB(max(dot(normal, V), 0.0), F0)) * (1.0 - metallic);

    // Hemispherical integration with Lambertian BRDF
    // Factor of π from hemisphere integral cancels with π in BRDF denominator
    float3 skyAmbient = kD * albedo * skyRadiance;

    // ========================================================================
    // Image-Based Lighting (IBL) Specular Reflection
    // ========================================================================
    // Implements split-sum approximation for physically-based environment reflections
    // References:
    // - "Real Shading in Unreal Engine 4" (Brian Karis, Epic Games, 2013)
    // - glTF 2.0 specification (KHR_lights_punctual + IBL extension)
    // ========================================================================

    float3 iblSpecular = float3(0.0, 0.0, 0.0);

    // Declare IBL variables outside conditional for use in spectral integration
    float3 prefilteredColor = float3(0.0, 0.0, 0.0);
    float2 envBRDF = float2(0.0, 0.0);

    // Only compute IBL for surfaces with non-zero metallic or roughness < 1.0
    // This optimization skips perfectly diffuse surfaces (no specular reflection).
    //
    // enableEnvironmentMap off removes the *map*, not the sky. The two are
    // different illuminants and only the first one double counts: an HDRI
    // alongside an analytic sun states the same light twice, so turning the map
    // off has to actually remove it. The analytic sky does not go away when it
    // does -- skyAmbient above still integrates it over the hemisphere -- and
    // taking the specular lobe to zero here made that sky light Lambertian
    // surfaces and not mirrors. A metal has kD = (1-F)(1-metallic) = 0, so it
    // lost every term but the sun and rendered black under a bright sky.
    //
    // The substitute is the dome the diffuse term already assumes: uniform
    // radiance skyRadiance. Prefiltering a uniform dome returns that radiance at
    // every roughness and direction, so this adds no assumption the diffuse side
    // has not already made, and it is what a reflection ray would fetch -- the
    // miss shader returns exactly lut.skyRadiance_rgb.
    //
    // What it cannot do is reflect the *scene*. This whole block, and the
    // skyAmbient above it, describe an unoccluded uniform dome and nothing else,
    // so a mirror shading from them shows sky and sun and never the ground.
    //
    // For RGB mode -- the interactive preview, and the only consumer of the
    // `radiance` these two feed -- that is the end of the story: opaque surfaces
    // spawn no bounce ray there, and only shadow rays and transmission recurse.
    // Every spectral band instead treats these terms as the analytic base of a
    // residual and traces one ray per hit for the difference; see
    // TraceEnvBounceResidual. They still read envBRDF and prefilteredColor from
    // here, which is why the block runs for them too.
    const bool hasEnvMap = (lut.enableEnvironmentMap != 0);
    if (metallic > 0.01 || roughness < 0.99) {
        // 1. Sample BRDF integration LUT
        //    Inputs: (NdotV, roughness) → Outputs: (scale, bias) for Fresnel term
        //    Independent of the environment, and the LUT is always generated.
        float NdotV_clamped = max(dot(normal, V), 0.0);
        envBRDF = brdfLUT.SampleLevel(iblSampler, float2(NdotV_clamped, roughness), 0.0).rg;

        if (hasEnvMap) {
            // 2. Compute reflection vector R = reflect(-V, N)
            //    This is the direction we would see a perfect mirror reflection
            float3 R = reflect(-V, normal);

            // 3. Select mipmap level based on roughness
            //    Rougher surfaces sample blurrier reflections (higher mip levels)
            //    Query number of mip levels at runtime (shader intrinsic)
            uint width, height, numMips;
            prefilteredEnvMap.GetDimensions(0, width, height, numMips);
            float lod = roughness * float(numMips - 1);

            // 4. Sample prefiltered environment map
            //    SampleLevel = explicit LOD (required in ray tracing shaders)
            prefilteredColor = prefilteredEnvMap.SampleLevel(envSampler, R, lod).rgb;
        } else {
            // Uniform analytic sky dome. Valid for the RGB branch only -- the
            // spectral branches below rebuild this term from the solar LUT per
            // wavelength, because skyRadiance is an RGB average in arbitrary
            // units there (see the sunRadiance/skyRadiance selection above).
            prefilteredColor = skyRadiance;
        }

        // 5. Split-sum approximation
        //    L_ibl = ∫ L(l) * BRDF(l,v) * (n·l) dl
        //          ≈ (∫ L(l) * (n·l) dl) * (∫ BRDF(l,v) * (n·l) dl)
        //          ≈ prefilteredColor * (F0 * envBRDF.x + envBRDF.y)
        //
        //    Where:
        //    - envBRDF.x (scale): multiplies F0 (Fresnel at normal incidence)
        //    - envBRDF.y (bias): constant offset for grazing angles
        //    NOTE: F0 is computed outside this block using physical n,k data when available
        iblSpecular = prefilteredColor * (F0 * envBRDF.x + envBRDF.y);

        // 6. Energy conservation: for metals, reduce diffuse contribution
        //    (already handled by kD term in skyAmbient calculation above)
    }

    // Total outgoing radiance: direct sun + sky ambient + IBL specular + emissive
    float3 radiance = directSun + skyAmbient + iblSpecular + emissive;

    // ========================================================================
    // Spectral Mode Selection: Choose rendering pipeline based on mode
    // ========================================================================

    float3 output_radiance;

    if (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_RGB) {
        // ====================================================================
        // RGB Mode: Fast Pure-RGB Pipeline (No Spectral Integration)
        // ====================================================================
        // Direct RGB rendering without spectral wavelength sampling.
        // Uses standard PBR calculations already computed above.
        //
        // This is the FASTEST mode with no spectral integration overhead.
        // For physically-correct spectral rendering, use VIS_FUSED mode.
        // ====================================================================
        output_radiance = radiance;

        // NN atmosphere composition per RGB channel (iLambda 0/1/2 = R/G/B,
        // baked at 650/550/450 nm through the vis network)
        if (atmosEnabled) {
            [unroll]
            for (uint ch = 0; ch < 3; ++ch) {
                float tau_ch = SampleAtmosTau(atmos, atmosNNData, ch, atmosA);
                float lpath_ch = SampleAtmosLpath(atmos, atmosNNData, ch, atmosA, atmosAz);
                output_radiance[ch] = tau_ch * output_radiance[ch] + lpath_ch;
            }
        }

        // Validation: clamp and sanitize to prevent NaN/Inf
        if (!isfinite(output_radiance.r) || !isfinite(output_radiance.g) || !isfinite(output_radiance.b)) {
            output_radiance = float3(0.0, 0.0, 0.0);
        }
        output_radiance = clamp(output_radiance, 0.0, 1000.0);

    } else if (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_VIS_FUSED) {
        // ====================================================================
        // VIS_Fused Mode: True 32-Wavelength Spectral Integration
        // ====================================================================
        // Physically-correct spectral rendering with full wavelength sampling:
        //   1. Sample 32 wavelengths uniformly across visible spectrum (380-780nm)
        //   2. Compute spectral radiance L(λ) at each wavelength
        //   3. Integrate via CIE XYZ color matching functions
        //   4. Convert XYZ → Linear RGB (sRGB D65)
        //
        // This is the full spectral rendering path for visible light.
        // Performance: ~10-15x slower than RGB mode, but physically accurate.
        //
        // SPECTRAL REFLECTANCE SOURCE (priority order):
        //   1. Measured spectral curve (spectralReflectanceCurveIndex >= 0)
        //   2. RGB texture upsampling via Gaussian basis (fallback)
        //
        // ILLUMINATION SOURCE (priority order):
        //   1. SolarSpectralLUT with measured ASTM G-173 spectra (preferred)
        //   2. RGB values converted to colored spectrum via ConvertLinearRGBToSpectrum
        // ====================================================================

        // Spectral integration parameters
        // NOTE: 400-780 nm (narrowed from 380 to match NN atmosphere coverage)
        const uint   NUM_WAVELENGTH_SAMPLES = 32;
        const float  LAMBDA_MIN_VIS = SPECTRAL_VIS_LAMBDA_MIN;
        const float  LAMBDA_MAX_VIS = SPECTRAL_VIS_LAMBDA_MAX;
        const float  LAMBDA_STEP = (LAMBDA_MAX_VIS - LAMBDA_MIN_VIS) / float(NUM_WAVELENGTH_SAMPLES - 1);


        // Accumulate XYZ tristimulus values
        float3 XYZ_accum = float3(0.0, 0.0, 0.0);

        // Check if we have physical spectral irradiance data
        bool hasSpectralSolarLUT = (solarSpectralLUT[0].sunIrradiance.numSamples > 0);

        // Precompute IBL parameters for spectral integration
        // F0 wavelength-dependent: interpolate RGB F0 based on wavelength for colored metals
        // This preserves gold's yellow tint, copper's red tint, etc.
        // Instead of simple average, we use linear interpolation between RGB wavelengths
        bool useIBL = (metallic > 0.01 || roughness < 0.99);

        // A ray past a dispersive refraction carries one wavelength, not the
        // band: n(λ) bent it for that wavelength alone, so integrating the
        // whole band along this path would be integrating light that never
        // came this way. It reports scalar spectral radiance and the surface
        // that sampled λ_h converts it back.
        const bool  heroRay = (payload.heroLambda > 0.0);
        const uint  sampleCount = heroRay ? 1u : NUM_WAVELENGTH_SAMPLES;
        float heroRadiance = 0.0;

        // ====================================================================
        // Indirect light: the traced correction
        // ====================================================================
        // Until now an opaque surface in this mode spawned no ray at all. Its
        // ambient light was L_ambient + L_ibl below -- an unoccluded uniform sky
        // dome, integrated analytically -- which is exact for a surface with an
        // open sky above it and simply wrong for a surface in a room, where it
        // grants full sky illumination through the ceiling and omits every
        // photon that arrived off a wall.
        //
        // Both errors are fixed by the same ray, carrying the difference
        // between what the hemisphere actually sends back and the uniform dome
        // the analytic terms assumed. Occlusion arrives as a negative
        // correction, interreflection as a positive one, and where the
        // assumption holds the correction is exactly zero, so an open scene
        // renders as it always did -- see scripts/render-tests/check_sky_equiv.py.
        //
        // The estimator is the dispersion path's, at a wavelength sampled
        // uniformly over the band and weighted by cmf(lambda_b) * bandwidth;
        // that equivalence is checked in check_hero_wavelength.py.
        //
        // Two lobes here, unlike the IR bands. There a surface has one total
        // reflectance and the lobe is only a sampling shape; here the diffuse
        // and specular terms are physically distinct -- kD*rho against F -- so
        // the choice is Fresnel-weighted and both must stay reachable, or a
        // metal (kD = 0) would lose all of its indirect light.
        float3 XYZ_bounce = float3(0.0, 0.0, 0.0);
        float  heroBounce = 0.0;

        // Light sampling. The direction and its two densities do not depend on
        // wavelength, so they are settled once here; only the BRDF and the
        // emitter's spectrum are evaluated per lambda inside the loop.
        //
        // visNeeScale is f-independent: cos(theta) * w_MIS / pdf. Zero means
        // there is nothing to add -- no emitters, edge-on, occluded, or below
        // the horizon -- and the loop skips the work.
        LightSample visLight;
        float visNeeScale = 0.0;
        // The bounce's BRDF terms, kept so the loop evaluates the same f the
        // bounce samples. wDiffuse is per-wavelength, so what is carried out is
        // everything except rho: kD_b, and the pieces of the specular lobe.
        float visKD = 0.0, visF = 0.0, visQSpec = 0.0, visNdotV = 0.0;
        {
            // Drawn against a curve shaped like the colour matching functions
            // rather than flat, so that cmf(lambda_b)/pdf below does not swing
            // by 4x with nothing but the draw. See SampleVisibleWavelength.
            // A hero ray has no choice to make -- its wavelength was fixed by
            // the refraction that created it.
            const float lambda_b = heroRay
                ? payload.heroLambda
                : SampleVisibleWavelength(PathSample1D(payload, SAMPLE_SLOT_LAMBDA));

            // Reflectance and Fresnel at lambda_b, by the rules the loop uses.
            const float rho_b = (material.spectralReflectanceCurveIndex >= 0)
                ? EvaluateEndmemberReflectanceW(spectralCurves, material, endmemberW, lambda_b)
                : ConvertLinearRGBToSpectrum(baseColor.rgb, lambda_b);

            float F0_b;
            if (lambda_b < 450.0)      F0_b = F0.b;
            else if (lambda_b < 500.0) F0_b = lerp(F0.b, F0.g, (lambda_b - 450.0) / 50.0);
            else if (lambda_b < 600.0) F0_b = lerp(F0.g, F0.r, (lambda_b - 500.0) / 100.0);
            else                       F0_b = F0.r;

            const float NdotV_b = max(dot(normal, V), 0.0);
            const float F_b     = FresnelSchlick(NdotV_b, F0_b);
            const float kD_b    = (1.0 - F_b) * (1.0 - metallic);

            // Pick the specular lobe about as often as it carries energy. The
            // floor keeps a rough dielectric's specular reachable; metals go to
            // 1 because their diffuse term is identically zero.
            //
            // Except with an environment map bound, where the specular lobe is
            // off entirely. The base it would be corrected against is the
            // prefiltered map, and the ray comes back carrying the analytic sky
            // the miss shader returns; the difference between two different
            // illuminants is not a correction to anything. The diffuse side has
            // no such problem -- L_ambient reflects the analytic sky whether or
            // not a map is loaded, which is the same thing the ray brings back.
            // So an env-mapped scene keeps split-sum specular exactly as before
            // and still gains diffuse interreflection; only a metal, whose
            // diffuse term is identically zero, gets no bounce at all there.
            // Lifting this needs the miss shader to sample the map.
            const float qSpec_b = (useIBL && !hasEnvMap)
                ? clamp(lerp(FresnelSchlick(NdotV_b, (F0.r + F0.g + F0.b) / 3.0), 1.0, metallic),
                        0.05, 1.0)
                : 0.0;

            // Roulette on the surface's own brightness, not on one wavelength:
            // the lobe split already accounts for where the energy goes, and a
            // wavelength-dependent survival would fight it. Capped below 1 so a
            // white room still terminates.
            const float rrSurvive_b =
                clamp(lerp(max(baseColor.r, max(baseColor.g, baseColor.b)), 1.0, metallic),
                      0.0, 0.95);

            // The base both lobes are measured against: the uniform sky dome at
            // lambda_b, which is what L_ambient reflects and -- prefiltering a
            // uniform dome being the identity -- what L_ibl reflects too.
            const float sky_b = hasSpectralSolarLUT
                ? SampleSkyIrradiance(solarSpectralLUT, lambda_b) / PI
                : 0.0;

            const float3 visHitPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

            // Sampled before the bounce so both draw from the same RNG stream
            // in a fixed order; the bounce advances payload.rngState itself.
            visLight = SampleEmissiveGeometry(visHitPos, payload);
            if (visLight.valid) {
                const float NdotWl = dot(normal, visLight.wi);
                if (NdotWl > 0.0 && LightSampleVisible(visHitPos, normal, visLight)) {
                    const float pdfBsdfAtLight =
                        BsdfMixturePdf(normal, V, visLight.wi, roughness, qSpec_b);
                    visNeeScale = NdotWl *
                                  PowerHeuristic(visLight.pdfSolid, pdfBsdfAtLight) /
                                  visLight.pdfSolid;
                    visKD     = kD_b;
                    visF      = F_b;
                    visQSpec  = qSpec_b;
                    visNdotV  = NdotV_b;
                }
            }

            const float  corr_b = TraceEnvBounceResidual(
                visHitPos, normal, V, NdotV_b, roughness, qSpec_b,
                kD_b * rho_b, F_b, rrSurvive_b,
                sky_b, lambda_b, payload);

            if (heroRay) {
                // Scalar: the surface that sampled lambda_h owns the weighting.
                heroBounce = corr_b;
            } else {
                // XYZ = L(lambda_b) * cmf(lambda_b) / pdf. Same estimator and
                // same normalisation as the deterministic grid below, which
                // divides by CIE_Y_INTEGRAL once for both. The pdf is no longer
                // the constant 1/bandwidth, so it is evaluated rather than
                // folded in as a literal.
                float tau_b = 1.0;
                if (atmosEnabled) {
                    const uint atmosIdx_b = (uint)clamp(
                        round((lambda_b - LAMBDA_MIN_VIS) / LAMBDA_STEP),
                        0.0, float(NUM_WAVELENGTH_SAMPLES - 1));
                    tau_b = SampleAtmosTau(atmos, atmosNNData, atmosIdx_b, atmosA);
                }
                XYZ_bounce = corr_b * tau_b / VisibleWavelengthPDF(lambda_b) *
                             SampleCIE_XYZ_LUT(cieCMF_LUT, lambda_b);
            }
        }

        // Loop over wavelengths
        // NOTE: Removed [unroll] to reduce shader compilation time (was 50+ seconds)
        // Modern GPUs handle small loops efficiently without forced unrolling
        for (uint i = 0; i < sampleCount; ++i) {
            float lambda = heroRay ? payload.heroLambda
                                   : (LAMBDA_MIN_VIS + float(i) * LAMBDA_STEP);

            // The atmosphere LUT is baked on the fixed grid, so a hero
            // wavelength has no index of its own -- take the nearest. The
            // grid is 12.3 nm apart and tau/lpath vary slowly across it, but
            // this is an approximation the deterministic path does not make.
            const uint atmosIdx = heroRay
                ? (uint)clamp(round((lambda - LAMBDA_MIN_VIS) / LAMBDA_STEP),
                              0.0, float(NUM_WAVELENGTH_SAMPLES - 1))
                : i;

            // ================================================================
            // Query Sun/Sky Spectral Radiance at Wavelength λ
            // ================================================================
            float sun_radiance_lambda;
            float sky_radiance_lambda;

            if (hasSpectralSolarLUT) {
                // PHYSICAL PATH: Query ASTM G-173 spectral irradiance curves
                // Convert irradiance (W·m⁻²·nm⁻¹) to radiance (W·sr⁻¹·m⁻²·nm⁻¹)
                float sun_irr = SampleSunIrradiance(solarSpectralLUT, lambda);
                float sky_irr = SampleSkyIrradiance(solarSpectralLUT, lambda);

                // Irradiance, not the sun disk's radiance. This feeds
                // BRDF * X * NdotL below, and for a distant source that
                // identity wants E: L_out = BRDF·E·cosθ. Dividing by the
                // sun's solid angle here, with nothing multiplying it back,
                // made the term 1/Ω_sun ≈ 14700x too large whenever a solar
                // LUT was loaded.
                sun_radiance_lambda = sun_irr;
                // Sky: diffuse hemispherical, already in radiance-like units (W·m⁻²·nm⁻¹·sr⁻¹ approximated)
                // For sky dome, we assume uniform sky approximation: L_sky ≈ E_sky / π
                sky_radiance_lambda = sky_irr / PI;
            } else {
                // No curve, no sun. The sun's spectrum is user-supplied data
                // like a reflectance curve, and there is nothing sensible to
                // invent without it: the RGB triple this used to spread across
                // the band is in arbitrary units, so the same scene rendered
                // differently depending on whether its spectrum was present.
                // src/app/main.cpp refuses a spectral render that has a sun and
                // no spectrum, so arriving here means the scene asked for none.
                sun_radiance_lambda = 0.0;
                sky_radiance_lambda = 0.0;
            }

            // 1. Get spectral reflectance at this wavelength
            float rho_lambda;
            if (material.spectralReflectanceCurveIndex >= 0) {
                // Quantitative path: measured spectral curve
                rho_lambda = EvaluateEndmemberReflectanceW(spectralCurves, material, endmemberW, lambda);
            } else {
                // Fallback path: RGB → Spectrum upsampling
                rho_lambda = ConvertLinearRGBToSpectrum(baseColor.rgb, lambda);
            }

            // 2. Compute BRDF at this wavelength (scalar Cook-Torrance)
            float brdf_lambda = CookTorranceBRDF_Spectral(normal, V, L, rho_lambda, metallic, roughness, material.complexRefractiveIndexIndex, lambda);

            // 3. Compute spectral radiance: L(λ) = BRDF(λ) × L_sun(λ) × (N·L) × shadow + kD × ρ(λ)/π × L_sky(λ)
            // shadowFactor is computed in RGB mode block and reused here for consistency
            float L_direct = brdf_lambda * sun_radiance_lambda * NdotL * shadowFactor;

            // Diffuse ambient with energy conservation (consistent with RGB mode)
            // kD = (1 - F) × (1 - metallic) ensures specular+diffuse ≤ 1
            //
            // WAVELENGTH-DEPENDENT F0 (P2 Enhancement):
            // For colored metals (gold, copper), F0 varies with wavelength:
            //   - Gold: high F0 at red (700nm), low at blue (400nm) → yellow reflection
            //   - Copper: higher F0 at red, lower at blue → reddish reflection
            // We interpolate F0.rgb based on wavelength to preserve metal color tint
            //
            // Wavelength-to-RGB mapping (approximate sRGB primary wavelengths):
            //   450nm (blue) → F0.b
            //   550nm (green) → F0.g
            //   650nm (red) → F0.r
            //
            float F0_at_lambda;
            if (lambda < 500.0) {
                // Blue-to-green interpolation (450-500nm)
                float t = clamp((lambda - 450.0) / 50.0, 0.0, 1.0);
                F0_at_lambda = lerp(F0.b, F0.g, t);
            } else if (lambda < 600.0) {
                // Green-to-red interpolation (500-600nm)
                float t = clamp((lambda - 500.0) / 100.0, 0.0, 1.0);
                F0_at_lambda = lerp(F0.g, F0.r, t);
            } else {
                // Red region (600nm+)
                F0_at_lambda = F0.r;
            }
            // Handle edge cases below blue wavelength
            if (lambda < 450.0) {
                F0_at_lambda = F0.b;
            }

            float NdotV_ambient = max(dot(normal, V), 0.0);
            float F_ambient = FresnelSchlick(NdotV_ambient, F0_at_lambda);  // Wavelength-dependent Fresnel
            float kD_lambda = (1.0 - F_ambient) * (1.0 - metallic);
            // Lambertian BRDF = ρ/π, hemisphere integral = π, so π cancels
            // sky_radiance_lambda is already radiance (W·sr⁻¹·m⁻²·nm⁻¹)
            float L_ambient = kD_lambda * rho_lambda * sky_radiance_lambda;

            // 4. Emissive contribution (spectrally integrated)
            // Convert emissive RGB to spectral radiance at this wavelength
            // Use illuminant function (no clamp) to preserve HDR emissive intensity
            float L_emissive = ConvertLinearRGBToIlluminantSpectrum(emissive, lambda);

            // 5. IBL specular contribution (spectrally integrated)
            // Apply Fresnel × BRDF in RGB space first, then convert to spectrum
            // This preserves colored metal reflections (gold, copper)
            float L_ibl = 0.0;
            if (useIBL) {
                if (hasEnvMap) {
                    float3 ibl_rgb = prefilteredColor * (F0 * envBRDF.x + envBRDF.y);
                    L_ibl = ConvertLinearRGBToIlluminantSpectrum(ibl_rgb, lambda);
                } else {
                    // Uniform analytic sky dome, kept on the measured solar LUT:
                    // prefiltering a uniform dome returns its radiance, which at
                    // this wavelength is sky_radiance_lambda. Routing it through
                    // the RGB triple instead would substitute an arbitrary-unit
                    // average for the illuminant this mode exists to integrate.
                    L_ibl = sky_radiance_lambda *
                            (F0_at_lambda * envBRDF.x + envBRDF.y);
                }
            }

            // 4b. Light sampled directly on an emitter, at this wavelength.
            // f * L_e * cos * w / p_omega -- the geometry term and the area-to-
            // solid-angle conversion are both already inside pdfSolid.
            float L_nee = 0.0;
            if (visNeeScale > 0.0) {
                const float brdf_at_light = EvalBounceBrdf(
                    normal, V, visLight.wi, visNdotV, roughness, visQSpec,
                    visKD * rho_lambda, visF);
                L_nee = brdf_at_light *
                        ConvertLinearRGBToIlluminantSpectrum(visLight.emissive, lambda) *
                        visNeeScale;
            }

            float L_lambda = L_direct + L_ambient + L_emissive + L_ibl + L_nee;

            // NN atmosphere composition: L = tau_view(λ)·L_surface(λ) + L_path(λ)
            if (atmosEnabled) {
                float tau_l = SampleAtmosTau(atmos, atmosNNData, atmosIdx, atmosA);
                float lpath_l = SampleAtmosLpath(atmos, atmosNNData, atmosIdx, atmosA, atmosAz);
                L_lambda = tau_l * L_lambda + lpath_l;
            }

            if (heroRay) {
                // Scalar spectral radiance: no CIE weighting and no dλ here.
                // The surface that sampled λ_h applies both, with 1/pdf. The
                // indirect correction is at this same wavelength, so it adds
                // straight in.
                heroRadiance = L_lambda + heroBounce;
                continue;
            }

            // 6. Weight by CIE XYZ color matching functions (High-precision LUT version)
            // Using LUT instead of analytical approximation for <0.1% error (vs 10-20% at edges)
            float3 xyz_cmf = SampleCIE_XYZ_LUT(cieCMF_LUT, lambda);
            float x_bar = xyz_cmf.x;
            float y_bar = xyz_cmf.y;
            float z_bar = xyz_cmf.z;

            // Riemann sum integration: ∫L(λ)×CMF(λ)dλ ≈ Σ L(λᵢ)×CMF(λᵢ)×Δλ
            XYZ_accum.x += L_lambda * x_bar * LAMBDA_STEP;
            XYZ_accum.y += L_lambda * y_bar * LAMBDA_STEP;
            XYZ_accum.z += L_lambda * z_bar * LAMBDA_STEP;
        }

        // The indirect correction, already carrying cmf(lambda_b) and the 1/pdf
        // that the deterministic grid gets from its LAMBDA_STEP. Zero for a hero
        // ray, which took it above instead.
        XYZ_accum += XYZ_bounce;

        // ====================================================================
        // XYZ Normalization for RGB Input Compatibility
        // ====================================================================
        // The Riemann sum XYZ = Σ[L(λ) × CMF(λ) × Δλ] produces values proportional
        // to CIE integrals (∫ȳdλ ≈ 106.9 for Y channel).
        //
        // For NORMALIZED RGB input (0-1 range from config), we must divide by
        // CIE_Y_INTEGRAL to get normalized output:
        //   - Input white (1,1,1) → XYZ_y ≈ 106.9 → after normalization → 1.0 ✓
        //   - Input gray (0.7,0.7,0.7) → XYZ_y ≈ 74.8 → after normalization → 0.7 ✓
        //
        // NOTE: If using physical radiance units (W/sr/m²/nm), comment out this
        // normalization to preserve absolute values.
        // ====================================================================
        XYZ_accum /= CIE_Y_INTEGRAL;

        // XYZ → Linear RGB (sRGB D65)
        output_radiance = ConvertXYZToLinearRGB(XYZ_accum);

        // ====================================================================
        // CHROMATICITY CORRECTION for Equal-Integral CIE CMF Data
        // ====================================================================
        // Our CIE CMF LUT uses EQUAL-INTEGRAL normalization:
        //   ∫x̄(λ)dλ ≈ ∫ȳ(λ)dλ ≈ ∫z̄(λ)dλ ≈ 106.85
        //
        // For FLAT spectrum material (gray colors), XYZ ratio is (1.0 : 1.0 : 1.0)
        // After sRGB matrix, this produces RED-BIASED output:
        //   RGB ∝ (1.27, 1.00, 0.96) due to XYZ→RGB matrix properties
        //
        // This correction neutralizes the chromaticity shift by scaling
        // R and B channels to match G, ensuring flat spectrum → neutral gray.
        //
        // Correction factors (default 0.7872, 1.0437) derived from G/R and G/B ratios
        // ====================================================================
        output_radiance.r *= lut.chromaR_correction;
        output_radiance.b *= lut.chromaB_correction;

        // A hero ray reports its one wavelength's radiance, replacing all of
        // the above. Placed AFTER the chromaticity correction on purpose: that
        // correction scales R and B against G to neutralise a full-band flat
        // spectrum, and applying it to a scalar would split one number into
        // three different ones. This is not a colour and nothing downstream may
        // treat it as one -- only the refracting surface reads it, and it
        // multiplies by cmf(λ_h)/pdf to make a colour.
        if (heroRay) {
            output_radiance = float3(heroRadiance, heroRadiance, heroRadiance);
        }

        // NOTE: IBL is now integrated in the spectral loop above (L_ibl term)
        // No need to add iblSpecular separately

        // Validation: clamp and sanitize to prevent NaN/Inf
        if (!isfinite(output_radiance.r) || !isfinite(output_radiance.g) || !isfinite(output_radiance.b)) {
            output_radiance = float3(0.0, 0.0, 0.0);  // Fallback to black
        }
        // SYMMETRIC, for the reason raygen.rgen gives at its own clamp, plus a
        // second one that applies to every band.
        //
        // A single sample here is out of gamut twice over. The indirect term is
        // one wavelength, and a monochromatic radiance converted to linear sRGB
        // lands outside the triangle with a genuinely negative channel; those
        // excursions are what cancel against other wavelengths' when the
        // samples are averaged. And the correction itself is a residual,
        // L_in - L_base, which is negative wherever the hemisphere is darker
        // than the uniform sky the analytic term assumed -- that is what
        // occlusion IS in this estimator.
        //
        // Rectifying either at zero turns a mean of zero into a mean of
        // (1 - rrSurvive) * rho * sky: the samples that kill the path keep the
        // full analytic term while the ones that would have cancelled it are
        // clipped away. Enclosed and shadowed surfaces read too bright, by more
        // the darker they are. Neither the furnace suite nor check_sky_equiv
        // can see it -- both are built on scenes where the correction is
        // identically zero.
        output_radiance = clamp(output_radiance, -1000.0, 1000.0);

    } else if (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_SINGLE) {
        // ====================================================================
        // Single Wavelength Mode: True Spectral Rendering (Quantitative)
        // ====================================================================
        // For single wavelength, we:
        // 1. Query physical spectral reflectance curve if available
        // 2. Fallback to RGB upsampling if no measured curve
        // 3. Compute scalar PBR BRDF with spectral reflectance
        // 4. Query sun/sky radiance from SolarSpectralLUT or LightingParams fallback
        //
        // SPECTRAL DATA PRIORITY:
        // 1. SolarSpectralLUT (ASTM G-173) → True spectral illumination
        // 2. LightingParams.sunRadiance_spectral → Scalar fallback
        // ====================================================================

        float lambda = pushConsts.camera.wavelength_nm;

        // ================================================================
        // Query Sun/Sky Spectral Radiance at Wavelength λ
        // ================================================================
        float sunRadiance_lambda;
        float skyRadiance_lambda;

        if (solarSpectralLUT[0].sunIrradiance.numSamples > 0) {
            // PHYSICAL PATH: Query ASTM G-173 spectral irradiance curves
            float sun_irr = SampleSunIrradiance(solarSpectralLUT, lambda);
            float sky_irr = SampleSkyIrradiance(solarSpectralLUT, lambda);

            // Irradiance: consumed as BRDF * X * NdotL, same as above
            sunRadiance_lambda = sun_irr;
            skyRadiance_lambda = sky_irr / PI;  // Diffuse sky: L ≈ E / π
        } else {
            // No curve, no sun. The sun's spectrum is user-supplied data
            // like a reflectance curve, and there is nothing sensible to
            // invent without it: the RGB triple this used to spread across
            // the band is in arbitrary units, so the same scene rendered
            // differently depending on whether its spectrum was present.
            // src/app/main.cpp refuses a spectral render that has a sun and
            // no spectrum, so arriving here means the scene asked for none.
            sunRadiance_lambda = 0.0;
            skyRadiance_lambda = 0.0;
        }

        // 1. Query spectral reflectance: prefer measured curve, fallback to RGB upsampling
        float spectralAlbedo;

        if (material.spectralReflectanceCurveIndex >= 0) {
            // QUANTITATIVE PATH: Use physically-measured spectral reflectance curve
            // This enables physically-accurate spectral rendering.
            // One endmember and no weight texture is the single flat curve.
            spectralAlbedo = EvaluateEndmemberReflectanceW(spectralCurves, material, endmemberW, lambda);
        } else {
            // FALLBACK PATH: RGB → Spectrum upsampling (approximate, ~70-80% accuracy)
            // WARNING: This path does NOT guarantee physical accuracy
            // For quantitative rendering, materials MUST have measured spectral curves
            spectralAlbedo = GetSpectralReflectanceFromRGBTexture(
                baseColor.rgb,
                lambda,
                false  // Already in linear space (not sRGB)
            );
        }

        // 2. Compute scalar PBR BRDF with spectral albedo
        //    Uses the same Cook-Torrance model, but with scalar reflectance
        float brdf_scalar = CookTorranceBRDF_Spectral(
            normal,
            V,
            L,
            spectralAlbedo,
            metallic,
            roughness,
            material.complexRefractiveIndexIndex,
            lambda
        );

        // 3. Direct sun lighting: L_out = BRDF * L_sun(λ) * (N · L) * shadow
        //    Use spectral sun radiance at wavelength λ
        //    shadowFactor is computed in RGB mode block and reused here for consistency
        float directSun_scalar = brdf_scalar * sunRadiance_lambda * NdotL * shadowFactor;

        // 4. Sky ambient lighting (scalar)
        //    Use simplified diffuse approximation (same as RGB mode)
        float3 F0_scalar = lerp(float3(0.04, 0.04, 0.04), float3(spectralAlbedo, spectralAlbedo, spectralAlbedo), metallic);
        float3 F_scalar = FresnelSchlickRGB(max(dot(normal, V), 0.0), F0_scalar);
        float kD_scalar = ((1.0 - F_scalar.r) * (1.0 - metallic));  // Use .r since all channels are identical

        // No 1/PI here. skyRadiance_lambda is already E_sky/PI -- the conversion
        // from irradiance to radiance happened when it was sampled -- so the
        // Lambertian identity L_out = rho*E/PI is complete at this point. This
        // used to divide by PI a second time and the term came out PI times too
        // dark. VIS_FUSED does not, and the RGB branch spells the cancellation
        // out: the hemisphere integral's PI cancels the one in the BRDF
        // denominator.
        float skyAmbient_scalar = kD_scalar * spectralAlbedo * skyRadiance_lambda;

        // 5. IBL specular contribution, matching VIS_FUSED.
        //    Was absent entirely, so every specular or metallic surface lost its
        //    environment reflection in the mode labelled "quantitative" -- worth
        //    30% of the frame on a smooth dielectric.
        // Same predicate as VIS_FUSED's, recomputed because that one is local to
        // its branch.
        const bool useIBL_scalar = (metallic > 0.01 || roughness < 0.99);
        float ibl_scalar = 0.0;
        if (useIBL_scalar) {
            if (hasEnvMap) {
                float3 ibl_rgb = prefilteredColor * (F0 * envBRDF.x + envBRDF.y);
                ibl_scalar = ConvertLinearRGBToIlluminantSpectrum(ibl_rgb, lambda);
            } else {
                // Uniform analytic sky dome on the solar LUT, as VIS_FUSED.
                ibl_scalar = skyRadiance_lambda *
                             (F0_scalar.r * envBRDF.x + envBRDF.y);
            }
        }

        // 6. Total spectral radiance (scalar)
        //    Emissive goes through the RGB->spectrum conversion rather than
        //    having its red channel read off as a spectral value, which is what
        //    `emissive.r` did. An emissive colour is a colour; at a wavelength
        //    it has to be evaluated, not indexed.
        float emissive_scalar = ConvertLinearRGBToIlluminantSpectrum(emissive, lambda);
        float radiance_spectral = directSun_scalar + skyAmbient_scalar +
                                  emissive_scalar + ibl_scalar;

        // 7. Indirect light: the traced correction to terms 4 and 5.
        //    This mode is the one labelled quantitative, so it is the last place
        //    an unoccluded uniform sky should stand in for the actual
        //    hemisphere. Same residual as every other band; the only difference
        //    is that no wavelength is sampled, because the whole render is at
        //    one. The child ray carries heroLambda = 0 and arrives back in this
        //    same branch, which evaluates at pushConsts.camera.wavelength_nm -- the same
        //    wavelength, by construction rather than by being told.
        //
        //    Costs roughly one extra ray per hit, roulette-limited. RGB mode is
        //    the one to reach for when that matters.
        {
            const float NdotV_s = max(dot(normal, V), 0.0);

            // Env-map scenes keep split-sum specular, as VIS_FUSED: the
            // prefiltered map and the analytic sky the miss shader returns are
            // different illuminants, and their difference corrects nothing.
            const float qSpec_s = (useIBL_scalar && !hasEnvMap)
                ? clamp(lerp(F_scalar.r, 1.0, metallic), 0.05, 1.0)
                : 0.0;
            const float rrSurvive_s = clamp(lerp(spectralAlbedo, 1.0, metallic), 0.0, 0.95);

            const float3 singleHitPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

            // 7b. Light sampling, as VIS_FUSED. One wavelength, so there is no
            // loop to hoist anything out of: the emitter's spectrum is
            // evaluated here alongside everything else.
            LightSample s = SampleEmissiveGeometry(singleHitPos, payload);
            if (s.valid) {
                const float NdotWl = dot(normal, s.wi);
                if (NdotWl > 0.0 && LightSampleVisible(singleHitPos, normal, s)) {
                    const float pdfBsdfAtLight =
                        BsdfMixturePdf(normal, V, s.wi, roughness, qSpec_s);
                    const float brdf_at_light = EvalBounceBrdf(
                        normal, V, s.wi, NdotV_s, roughness, qSpec_s,
                        kD_scalar * spectralAlbedo, F_scalar.r);
                    radiance_spectral +=
                        brdf_at_light *
                        ConvertLinearRGBToIlluminantSpectrum(s.emissive, lambda) *
                        NdotWl * PowerHeuristic(s.pdfSolid, pdfBsdfAtLight) / s.pdfSolid;
                }
            }

            radiance_spectral += TraceEnvBounceResidual(
                singleHitPos, normal, V, NdotV_s, roughness, qSpec_s,
                kD_scalar * spectralAlbedo, F_scalar.r, rrSurvive_s,
                skyRadiance_lambda, 0.0, payload);
        }

        // NN atmosphere composition (single wavelength: LUT baked with one sample)
        if (atmosEnabled) {
            float tau_l = SampleAtmosTau(atmos, atmosNNData, 0, atmosA);
            float lpath_l = SampleAtmosLpath(atmos, atmosNNData, 0, atmosA, atmosAz);
            radiance_spectral = tau_l * radiance_spectral + lpath_l;
        }

        // Validation. Symmetric: the bounce above is a residual and goes
        // negative under occlusion; see the note at the VIS_FUSED clamp.
        if (!isfinite(radiance_spectral)) {
            radiance_spectral = 0.0;
        }
        radiance_spectral = clamp(radiance_spectral, -1000.0, 1000.0);

        // Output as grayscale (replicate scalar to RGB for display)
        output_radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);

    } else if (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_SWIR_FUSED) {
        // ====================================================================
        // SWIR Fused Mode: Short-Wave IR Band Integration (1000-2500nm)
        // ====================================================================
        // In SWIR band, solar radiation is still significant (unlike MWIR/LWIR).
        // Physics model combines:
        //   1. Reflected solar irradiance (dominant for passive imaging)
        //   2. Minor thermal emission (only for very hot objects T > 500K, Wien peak ~5.8μm)
        //
        // L_total(λ) = ρ(λ) × [L_sun(λ) + L_sky(λ)] + ε(λ) × L_bb(T,λ)
        //
        // For typical outdoor scenes at ambient temperature (~300K), thermal
        // emission in SWIR is negligible (Planck peak at ~10μm, not 1-2.5μm).
        // ====================================================================

        // NOTE: 1400-2400 nm (narrowed from 1000-2500 to match NN atmosphere coverage)
        const float SWIR_LAMBDA_MIN = SPECTRAL_SWIR_LAMBDA_MIN;
        const float SWIR_LAMBDA_MAX = SPECTRAL_SWIR_LAMBDA_MAX;
        const uint  NUM_SWIR_SAMPLES = 16;
        // At 500K, Planck tail at 2.4µm ≈ 3.5e-3 W/sr/m²/nm — comparable to
        // reflected solar; below this, SWIR thermal emission is negligible.
        const float SWIR_EMISSION_MIN_TEMP_K = 500.0;
        const float lambda_step = (SWIR_LAMBDA_MAX - SWIR_LAMBDA_MIN) / float(NUM_SWIR_SAMPLES - 1);

        float radiance_accum = 0.0;

        // Check if we have spectral solar LUT for accurate SWIR illumination
        bool hasSpectralSolarLUT = (solarSpectralLUT[0].sunIrradiance.numSamples > 0);

        // Compute view angle for angle-dependent emissivity
        float NdotV_swir = max(dot(normal, V), 0.0);

        // Material IR properties with angle-dependent correction (P1 fix + angle correction)
        float baseEmissivity_swir = GetEffectiveIREmissivity(material);
        float emissivity = GetAngleDependentIREmissivity(baseEmissivity_swir, NdotV_swir, material.metallicFactor);
        float reflectance = GetAngleDependentIRReflectance(baseEmissivity_swir, material.irTransmittance,
                                                           NdotV_swir, material.metallicFactor);

        // Sample surface temperature from texture or use scalar value
        float T_surface_swir = GetSurfaceTemperatureK(material, geoInfo, PrimitiveIndex(), uv);

        // A ray spawned by an environment bounce carries one wavelength and
        // reports scalar spectral radiance; see Payload::heroLambda.
        const bool heroRay = (payload.heroLambda > 0.0);
        const uint sampleCount = heroRay ? 1u : NUM_SWIR_SAMPLES;
        float heroRadiance = 0.0;

        // ====================================================================
        // Reflected environment: the traced correction
        // ====================================================================
        // The loop's sky term reflects a uniform dome at rho(lambda)*L_sky --
        // right for an unobstructed surface, and blind to every other surface
        // in the scene. This ray carries the difference. SWIR is where that
        // matters most of the reflective bands: soil and dry vegetation sit
        // near rho = 0.3, so interreflection carries several times the weight
        // it does in the visible, and the band had no bounce at all.
        const float lambda_b = heroRay
            ? payload.heroLambda
            : SWIR_LAMBDA_MIN + PathSample1D(payload, SAMPLE_SLOT_LAMBDA) * (SWIR_LAMBDA_MAX - SWIR_LAMBDA_MIN);
        const uint atmosIdx_b = (uint)clamp(round((lambda_b - SWIR_LAMBDA_MIN) / lambda_step),
                                            0.0, float(NUM_SWIR_SAMPLES - 1));

        float rho_b = reflectance;
        if (material.spectralReflectanceCurveIndex >= 0) {
            rho_b = EvaluateEndmemberReflectanceW(spectralCurves, material, endmemberW, lambda_b);
        }

        // The base is the loop's own sky radiance, at lambda_b. Continuous in
        // wavelength, so unlike the thermal bands there is no grid to snap to.
        const float L_base_b = hasSpectralSolarLUT
            ? SampleSkyIrradiance(solarSpectralLUT, lambda_b) / PI
            : 0.0;

        // One total reflectance, so the lobe is only a sampling shape for it:
        // qSpec is 0 or 1 and both weights are rho_b.
        const float3 swirHitPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
        float bounceCorr = TraceEnvBounceResidual(
            swirHitPos, normal, V, NdotV_swir, roughness,
            (roughness > 0.5) ? 0.0 : 1.0, rho_b, rho_b, rho_b,
            L_base_b, lambda_b, payload);

        // NOTE: Removed [unroll] to reduce shader compilation time
        for (uint i = 0; i < sampleCount; ++i) {
            float lambda = heroRay ? payload.heroLambda
                                   : SWIR_LAMBDA_MIN + float(i) * lambda_step;
            uint  atmosIdx = heroRay ? atmosIdx_b : i;

            // 1. Query solar/sky irradiance at this SWIR wavelength
            float sun_radiance_lambda;
            float sky_radiance_lambda;

            if (hasSpectralSolarLUT) {
                float sun_irr = SampleSunIrradiance(solarSpectralLUT, lambda);
                float sky_irr = SampleSkyIrradiance(solarSpectralLUT, lambda);
                // E/PI, matching the sky line below and the MWIR branch:
                // this is multiplied by rho directly, with no BRDF to carry
                // the 1/PI, so L = rho·(E/PI)·cosθ is the Lambertian result.
                sun_radiance_lambda = sun_irr / PI;
                sky_radiance_lambda = sky_irr / PI;
            } else {
                // No curve, no sun. The sun's spectrum is user-supplied data
                // like a reflectance curve, and there is nothing sensible to
                // invent without it: the RGB triple this used to spread across
                // the band is in arbitrary units, so the same scene rendered
                // differently depending on whether its spectrum was present.
                // src/app/main.cpp refuses a spectral render that has a sun and
                // no spectrum, so arriving here means the scene asked for none.
                sun_radiance_lambda = 0.0;
                sky_radiance_lambda = 0.0;
            }

            // 2. Get spectral reflectance at this wavelength
            float rho_lambda;
            if (material.spectralReflectanceCurveIndex >= 0) {
                rho_lambda = EvaluateEndmemberReflectanceW(spectralCurves, material, endmemberW, lambda);
            } else {
                // Fallback: use IR reflectance from energy conservation
                rho_lambda = reflectance;
            }

            // 3. Reflected solar radiance: ρ(λ) × (L_sun(λ) × NdotL × V + L_sky(λ))
            //
            // shadowFactor is the sun-visibility term traced above. It was
            // computed for every mode and then read by RGB, VIS_FUSED and
            // SINGLE only, so in this band the sun shone through walls -- the
            // shadow ray's cost was already paid and its answer thrown away.
            // The sky term stays unoccluded here; the traced bounce is what
            // will occlude it.
            float L_reflected = rho_lambda * (sun_radiance_lambda * NdotL * shadowFactor
                                              + sky_radiance_lambda);

            // 4. Thermal emission (minor in SWIR below threshold)
            float L_emission = 0.0;
            if (T_surface_swir > SWIR_EMISSION_MIN_TEMP_K) {
                float L_blackbody = IRPlanckRadiance(T_surface_swir, lambda);
                L_emission = emissivity * L_blackbody;
            }

            // 5. Total spectral radiance
            float L_lambda = L_reflected + L_emission;

            // NN atmosphere composition: L = tau_view(λ)·L_surface(λ) + L_path(λ)
            if (atmosEnabled) {
                float tau_l = SampleAtmosTau(atmos, atmosNNData, atmosIdx, atmosA);
                float lpath_l = SampleAtmosLpath(atmos, atmosNNData, atmosIdx, atmosA, atmosAz);
                L_lambda = tau_l * L_lambda + lpath_l;
            }

            if (heroRay) {
                // Scalar spectral radiance: no trapezoid weight, no band
                // normalisation. The surface that sampled this wavelength owns
                // both.
                heroRadiance = L_lambda;
                continue;
            }

            // Trapezoid rule. N samples span N-1 intervals, so the two
            // endpoints carry half weight. Summing N full-width rectangles
            // and then dividing by (N-1)*step made every fused band read
            // N/(N-1) high -- 6.7% for the 16-sample bands. Checked against
            // an isothermal cavity, which must render exactly its own
            // blackbody and read 1.066667x it.
            float trapezoidW = (i == 0 || i == NUM_SWIR_SAMPLES - 1) ? 0.5 : 1.0;
            radiance_accum += L_lambda * lambda_step * trapezoidW;
        }

        // Normalize by band width to get AVERAGE spectral radiance (W·sr⁻¹·m⁻²)
        // This allows fair comparison across bands with different bandwidths.
        // For sensor simulation, use radiance_accum (band-integrated radiance) instead.
        float band_width = SWIR_LAMBDA_MAX - SWIR_LAMBDA_MIN;
        float radiance_avg = heroRay ? heroRadiance : (radiance_accum / band_width);

        // Sampling lambda_b uniformly gives pdf = 1/band_width, which cancels
        // the 1/band_width of the band average, so the correction is added with
        // no further weight -- and the same expression serves a hero ray, where
        // neither factor is present. View-path transmittance applies as it does
        // to the loop's terms; L_path does not, being per pixel rather than per
        // light path.
        if (atmosEnabled) {
            bounceCorr *= SampleAtmosTau(atmos, atmosNNData, atmosIdx_b, atmosA);
        }
        radiance_avg += bounceCorr;

        // Validation. Symmetric: bounceCorr is a residual and goes negative
        // under occlusion; see the note at the VIS_FUSED clamp.
        if (!isfinite(radiance_avg)) {
            radiance_avg = 0.0;
        }
        radiance_avg = clamp(radiance_avg, -1e6, 1e6);

        output_radiance = float3(radiance_avg, radiance_avg, radiance_avg);

    } else if (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_NIR_FUSED) {
        // ====================================================================
        // NIR Fused Mode: Near-Infrared Band Integration (780-1400nm)
        // ====================================================================
        // NIR is "reflected infrared" - behaves almost identically to visible light.
        // Solar radiation is the dominant source; thermal emission is negligible
        // for objects below ~600K (327°C).
        //
        // Physics model:
        //   L_total(λ) = ρ(λ) × [L_sun(λ) × cos(θ) + L_sky(λ)]
        //
        // Key properties of NIR (780-1400nm):
        //   - ~46-55% of solar energy is in NIR/SWIR bands
        //   - Can be focused with standard glass optics (unlike thermal IR)
        //   - Commonly used in: vegetation analysis (chlorophyll reflection),
        //     night vision (active illumination), material identification
        //   - Water absorbs strongly at 970nm and 1200nm (moisture detection)
        //
        // Reference: ISO 20473 classifies NIR as IR-A (780nm - 1.4μm)
        // ====================================================================

        // NOTE: 930-1200 nm (narrowed from 780-1400 to match NN atmosphere coverage)
        const float NIR_LAMBDA_MIN = SPECTRAL_NIR_LAMBDA_MIN;
        const float NIR_LAMBDA_MAX = SPECTRAL_NIR_LAMBDA_MAX;
        const uint  NUM_NIR_SAMPLES = 16;
        const float lambda_step = (NIR_LAMBDA_MAX - NIR_LAMBDA_MIN) / float(NUM_NIR_SAMPLES - 1);

        float radiance_accum = 0.0;

        // Check if we have spectral solar LUT for accurate NIR illumination
        bool hasSpectralSolarLUT = (solarSpectralLUT[0].sunIrradiance.numSamples > 0);

        // A ray spawned by an environment bounce carries one wavelength and
        // reports scalar spectral radiance; see Payload::heroLambda.
        const bool heroRay = (payload.heroLambda > 0.0);
        const uint sampleCount = heroRay ? 1u : NUM_NIR_SAMPLES;
        float heroRadiance = 0.0;

        // ====================================================================
        // Reflected environment: the traced correction
        // ====================================================================
        // Same residual as the other bands, against the loop's uniform-sky
        // term. NIR has the highest albedos the renderer deals with -- healthy
        // vegetation reflects about half the light that reaches it -- and
        // interreflection scales as rho/(1-rho), so this is the band where
        // having no bounce at all cost the most.
        const float NdotV_nir = max(dot(normal, V), 0.0);
        const float lambda_b = heroRay
            ? payload.heroLambda
            : NIR_LAMBDA_MIN + PathSample1D(payload, SAMPLE_SLOT_LAMBDA) * (NIR_LAMBDA_MAX - NIR_LAMBDA_MIN);
        const uint atmosIdx_b = (uint)clamp(round((lambda_b - NIR_LAMBDA_MIN) / lambda_step),
                                            0.0, float(NUM_NIR_SAMPLES - 1));

        // Reflectance at lambda_b, by the same rule the loop uses.
        const float rho_b = (material.spectralReflectanceCurveIndex >= 0)
            ? EvaluateEndmemberReflectanceW(spectralCurves, material, endmemberW, lambda_b)
            : ConvertLinearRGBToSpectrum(baseColor.rgb, lambda_b);

        const float L_base_b = hasSpectralSolarLUT
            ? SampleSkyIrradiance(solarSpectralLUT, lambda_b) / PI
            : 0.0;

        const float3 nirHitPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
        float bounceCorr = TraceEnvBounceResidual(
            nirHitPos, normal, V, NdotV_nir, roughness,
            (roughness > 0.5) ? 0.0 : 1.0, rho_b, rho_b, rho_b,
            L_base_b, lambda_b, payload);

        // NOTE: Removed [unroll] to reduce shader compilation time
        for (uint i = 0; i < sampleCount; ++i) {
            float lambda = heroRay ? payload.heroLambda
                                   : NIR_LAMBDA_MIN + float(i) * lambda_step;
            uint  atmosIdx = heroRay ? atmosIdx_b : i;

            // 1. Query solar/sky irradiance at this NIR wavelength
            float sun_radiance_lambda;
            float sky_radiance_lambda;

            if (hasSpectralSolarLUT) {
                float sun_irr = SampleSunIrradiance(solarSpectralLUT, lambda);
                float sky_irr = SampleSkyIrradiance(solarSpectralLUT, lambda);
                // E/PI, matching the sky line below and the MWIR branch:
                // this is multiplied by rho directly, with no BRDF to carry
                // the 1/PI, so L = rho·(E/PI)·cosθ is the Lambertian result.
                sun_radiance_lambda = sun_irr / PI;
                sky_radiance_lambda = sky_irr / PI;
            } else {
                // No curve, no sun. The sun's spectrum is user-supplied data
                // like a reflectance curve, and there is nothing sensible to
                // invent without it: the RGB triple this used to spread across
                // the band is in arbitrary units, so the same scene rendered
                // differently depending on whether its spectrum was present.
                // src/app/main.cpp refuses a spectral render that has a sun and
                // no spectrum, so arriving here means the scene asked for none.
                sun_radiance_lambda = 0.0;
                sky_radiance_lambda = 0.0;
            }

            // 2. Get spectral reflectance at this wavelength
            float rho_lambda;
            if (material.spectralReflectanceCurveIndex >= 0) {
                // Quantitative path: use measured spectral curve
                rho_lambda = EvaluateEndmemberReflectanceW(spectralCurves, material, endmemberW, lambda);
            } else {
                // Fallback: RGB upsampling (NIR is close enough to visible for this to be reasonable)
                // This uses Gaussian basis functions centered at R/G/B wavelengths
                rho_lambda = ConvertLinearRGBToSpectrum(baseColor.rgb, lambda);
            }

            // 3. Reflected solar radiance: ρ(λ) × (L_sun(λ) × NdotL × V + L_sky(λ))
            //
            // shadowFactor as in the SWIR branch: traced for every mode, read
            // by three of them. NIR is the band where losing it is least
            // defensible -- reflected solar is the entire signal here.
            float L_reflected = rho_lambda * (sun_radiance_lambda * NdotL * shadowFactor
                                              + sky_radiance_lambda);

            // Note: Thermal emission is negligible in NIR for T < 600K
            // A 600K object peaks at ~4800nm (Wien's law), far from NIR band
            // Skip thermal calculation for performance

            // NN atmosphere composition: L = tau_view(λ)·L_surface(λ) + L_path(λ)
            if (atmosEnabled) {
                float tau_l = SampleAtmosTau(atmos, atmosNNData, atmosIdx, atmosA);
                float lpath_l = SampleAtmosLpath(atmos, atmosNNData, atmosIdx, atmosA, atmosAz);
                L_reflected = tau_l * L_reflected + lpath_l;
            }

            if (heroRay) {
                // Scalar spectral radiance: no trapezoid weight, no band
                // normalisation. The surface that sampled this wavelength owns
                // both.
                heroRadiance = L_reflected;
                continue;
            }

            // Trapezoid rule. N samples span N-1 intervals, so the two
            // endpoints carry half weight. Summing N full-width rectangles
            // and then dividing by (N-1)*step made every fused band read
            // N/(N-1) high -- 6.7% for the 16-sample bands. Checked against
            // an isothermal cavity, which must render exactly its own
            // blackbody and read 1.066667x it.
            float trapezoidW = (i == 0 || i == NUM_NIR_SAMPLES - 1) ? 0.5 : 1.0;
            radiance_accum += L_reflected * lambda_step * trapezoidW;
        }

        // Normalize by band width to get AVERAGE spectral radiance (W·sr⁻¹·m⁻²)
        // This allows fair comparison across bands with different bandwidths.
        // For sensor simulation, use radiance_accum (band-integrated radiance) instead.
        float band_width = NIR_LAMBDA_MAX - NIR_LAMBDA_MIN;
        float radiance_avg = heroRay ? heroRadiance : (radiance_accum / band_width);

        // pdf = 1/band_width cancels the band average's own factor, so the
        // correction needs no further weight; see the SWIR branch.
        if (atmosEnabled) {
            bounceCorr *= SampleAtmosTau(atmos, atmosNNData, atmosIdx_b, atmosA);
        }
        radiance_avg += bounceCorr;

        // Validation
        if (!isfinite(radiance_avg)) {
            radiance_avg = 0.0;
        }
        // Symmetric: bounceCorr is a residual and goes negative under
        // occlusion; see the note at the VIS_FUSED clamp.
        radiance_avg = clamp(radiance_avg, -1e6, 1e6);

        output_radiance = float3(radiance_avg, radiance_avg, radiance_avg);

    } else if (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_MWIR_FUSED || SPEC_SPECTRAL_MODE == SPECTRAL_MODE_LWIR_FUSED) {
        // ====================================================================
        // MWIR/LWIR Fused Mode: Multi-Wavelength IR Band Integration
        // ====================================================================
        // True spectral integration across infrared bands:
        //   - MWIR: 3000-5000nm (Mid-Wave Infrared)
        //   - LWIR: 8000-12000nm (Long-Wave Infrared)
        //
        // Physics model (updated with solar reflection for MWIR - P2 fix):
        //   L_total(λ) = ε(λ) × L_bb(T,λ) + ρ(λ) × [L_atm↓(λ) + L_sun(λ)×cos(θ)]
        //
        // where:
        //   ε(λ) = emissivity (derived from metallicFactor if not set - P1 fix)
        //   L_bb(T,λ) = Planck blackbody radiance at temperature T
        //   ρ(λ) = reflectance = 1 - ε - τ (Kirchhoff's law)
        //   L_atm↓ = atmospheric downwelling radiance (sky thermal)
        //   L_sun = solar irradiance (significant in MWIR 3-5μm, negligible in LWIR)
        //
        // MWIR Solar Contribution (P2 fix):
        //   At 4μm, solar irradiance ≈ 5 W·m⁻²·μm⁻¹ (AM1.5)
        //   For T=300K surface: thermal emission ≈ 0.24 W·sr⁻¹·m⁻²·μm⁻¹
        //   Solar reflection can contribute 5-20% of total radiance for sunlit surfaces!
        //
        // Output: Single grayscale value (band-integrated radiance)
        // ====================================================================

        // Determine wavelength range and whether to include solar term
        float lambda_min, lambda_max;
        bool includeSolarReflection = false;  // Only for MWIR during daytime

        if (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_MWIR_FUSED) {
            lambda_min = SPECTRAL_MWIR_LAMBDA_MIN;
            lambda_max = SPECTRAL_MWIR_LAMBDA_MAX;
            // MWIR: solar contributes 5-20% for sunlit surfaces (P2 fix)
            // Only include if surface faces the sun AND the sun is visible.
            // shadowFactor was traced for every mode and read by three of
            // them, so this band's solar term used to ignore occluders
            // entirely. Gating here skips the LUT sample on shadowed pixels;
            // the term is also scaled by it below, which is what a partial
            // shadowFactor would need.
            includeSolarReflection = (NdotL > 0.0 && shadowFactor > 0.0);
        } else {  // LWIR
            lambda_min = SPECTRAL_LWIR_LAMBDA_MIN;
            lambda_max = SPECTRAL_LWIR_LAMBDA_MAX;
            // LWIR: solar contribution < 0.1%, skip for performance
            includeSolarReflection = false;
        }

        // Integration parameters
        const uint  NUM_IR_SAMPLES = 16;  // Fewer samples than visible (smoother spectra)
        const float lambda_step = (lambda_max - lambda_min) / float(NUM_IR_SAMPLES - 1);

        // A ray spawned by an environment bounce carries one wavelength, not
        // the band: the surface that spawned it sampled that wavelength and
        // weighs what comes back by it, so integrating the whole band along
        // this path would answer a question nobody asked. It reports scalar
        // spectral radiance, by the contract on Payload::heroLambda.
        const bool heroRay = (payload.heroLambda > 0.0);
        const uint sampleCount = heroRay ? 1u : NUM_IR_SAMPLES;
        float heroRadiance = 0.0;

        // Accumulate band-integrated radiance
        float radiance_accum = 0.0;

        // Compute view angle for angle-dependent emissivity
        float NdotV = max(dot(normal, V), 0.0);

        // Material IR properties with angle-dependent correction (Fresnel effect)
        float baseEmissivity = GetEffectiveIREmissivity(material);
        float emissivity = GetAngleDependentIREmissivity(baseEmissivity, NdotV, material.metallicFactor);
        float reflectance = GetAngleDependentIRReflectance(baseEmissivity, material.irTransmittance,
                                                           NdotV, material.metallicFactor);

        // Sample surface temperature from texture or use scalar value
        float T_surface = GetSurfaceTemperatureK(material, geoInfo, PrimitiveIndex(), uv);

        // Atmospheric downwelling radiation temperature
        float T_atmosphere = lut.atmosphereTemperature_K;

        // Check if we have spectral solar LUT for accurate MWIR illumination
        bool hasSpectralSolarLUT = (solarSpectralLUT[0].sunIrradiance.numSamples > 0);

        // ====================================================================
        // Reflected environment: the traced correction
        // ====================================================================
        // The wavelength loop below reflects the analytic downwelling spectrum
        // from the whole hemisphere at once. That is the base; this is what it
        // gets wrong when the surface can see something other than sky. See
        // TraceEnvBounceResidual.
        //
        // The wavelength is hoisted above the call because the loop needs its
        // LUT index too: a ray that already carries a hero wavelength answers
        // at that wavelength throughout, base and correction alike.
        const float lambda_b = heroRay
            ? payload.heroLambda
            : lambda_min + PathSample1D(payload, SAMPLE_SLOT_LAMBDA) * (lambda_max - lambda_min);

        // The atmosphere LUT is baked on the loop's own sample points, so a
        // sampled wavelength has no index of its own -- take the nearest, as
        // VIS_FUSED does for a hero ray.
        const uint atmosIdx_b = (uint)clamp(round((lambda_b - lambda_min) / lambda_step),
                                            0.0, float(NUM_IR_SAMPLES - 1));

        // Reflectance at lambda_b, by the same rule the loop uses.
        float rho_b = reflectance;
        if (material.spectralReflectanceCurveIndex >= 0) {
            float rho_curve = EvaluateEndmemberReflectanceW(spectralCurves, material, endmemberW, lambda_b);
            float eps_b = saturate(1.0 - rho_curve - material.irTransmittance);
            rho_b = GetAngleDependentIRReflectance(eps_b, material.irTransmittance,
                                                   NdotV, material.metallicFactor);
        }

        const float L_base_b = IRDownwellingRadiance(atmos, atmosIdx_b, lambda_b,
                                                    T_atmosphere, lut.skyEmissivityClear);

        const float3 irHitPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
        float bounceCorr = TraceEnvBounceResidual(
            irHitPos, normal, V, NdotV, roughness,
            (roughness > 0.5) ? 0.0 : 1.0, rho_b, rho_b, rho_b,
            L_base_b, lambda_b, payload);

        // Loop over wavelengths in IR band ([loop]: keep code size bounded).
        // A hero ray runs one iteration at its own wavelength and reports a
        // scalar; see Payload::heroLambda.
        [loop]
        for (uint i = 0; i < sampleCount; ++i) {
            float lambda = heroRay ? payload.heroLambda
                                   : lambda_min + float(i) * lambda_step;
            uint  atmosIdx = heroRay ? atmosIdx_b : i;

            // 0. Emissivity and reflectance at this wavelength.
            // ================================================================
            // Kirchhoff's law in thermal equilibrium: a surface emits exactly
            // as well as it absorbs, so eps = 1 - rho - tau. With a measured
            // reflectance curve bound, that makes emissivity spectral AND
            // spatial for free -- the same endmember weights that vary the
            // reflectance across the surface vary what it radiates, which is
            // the thing a thermal image is actually of.
            //
            // Without a curve these stay the scalars computed outside the
            // loop, so materials that never had one render bit for bit as
            // before.
            float emissivity_l = emissivity;
            float reflectance_l = reflectance;
            if (material.spectralReflectanceCurveIndex >= 0) {
                float rho_l = EvaluateEndmemberReflectanceW(spectralCurves, material, endmemberW, lambda);
                float eps_l = saturate(1.0 - rho_l - material.irTransmittance);
                emissivity_l = GetAngleDependentIREmissivity(eps_l, NdotV, material.metallicFactor);
                reflectance_l = GetAngleDependentIRReflectance(eps_l, material.irTransmittance,
                                                              NdotV, material.metallicFactor);
            }

            // 1. Self-emission: ε(λ) × L_blackbody(T_surface, λ)
            float L_emission = 0.0;
            if (T_surface > 0.0) {
                float L_blackbody = IRPlanckRadiance(T_surface, lambda);
                L_emission = emissivity_l * L_blackbody;
            }

            // 2. Reflected environment, base term.
            // ================================================================
            // The whole hemisphere filled with the analytic downwelling
            // spectrum. Unconditional now: this is not the fallback it used to
            // be when the traced sample was unavailable, it is the control
            // variate the traced sample corrects. Two things follow from that.
            // It is per-wavelength and exact, so a measured reflectance curve
            // multiplies the right spectrum at every lambda rather than a band
            // average. And it is the closure at the end of every path -- a
            // bounce that is killed, or that runs out of depth, leaves this
            // standing, which is why an isothermal cavity still returns its own
            // Planck radiance no matter where the path stops.
            //
            // Which sky, in order: the network's measured downwelling; the
            // analytic clear sky, hemispherically averaged because this term
            // stands for the whole hemisphere rather than one direction; or
            // the isotropic blackbody, which is what a scene that asked for
            // neither gets and renders exactly as it always did.
            float L_down = IRDownwellingRadiance(atmos, atmosIdx, lambda, T_atmosphere,
                                                lut.skyEmissivityClear);
            float L_reflected_atm = reflectance_l * L_down;

            // 3. Reflected solar radiance (P2 fix: MWIR daytime solar contribution)
            float L_reflected_sun = 0.0;
            if (includeSolarReflection && NdotL > 0.0) {
                // Get solar spectral irradiance at this MWIR wavelength
                float sun_irr_lambda = 0.0;

                if (hasSpectralSolarLUT) {
                    // Query ASTM G-173 or similar (if data extends to MWIR)
                    sun_irr_lambda = SampleSunIrradiance(solarSpectralLUT, lambda);
                } else {
                    // No curve, no sun -- as in the other bands. The 5778 K
                    // disk that stood in here was the last invented illuminant
                    // in the renderer: a fixed blackbody shape scaled by an
                    // RGB triple in arbitrary units, in a band where the solar
                    // term is a few percent of the signal and therefore easy
                    // to be wrong about without anyone noticing.
                    sun_irr_lambda = 0.0;
                }

                // Convert irradiance to radiance and apply Lambertian BRDF
                // L_reflected = ρ/π × E_sun × cos(θ) for diffuse surfaces
                // For simplicity, using ρ × (E/π) × NdotL
                // View-path attenuation comes from the NN composition below;
                // sun-path attenuation is folded into the illumination source.
                float sun_radiance_lambda = sun_irr_lambda / PI;
                L_reflected_sun = reflectance_l * sun_radiance_lambda * NdotL * shadowFactor;
            }

            // 4. IR Transmittance: Background radiation through transparent materials
            // ================================================================
            // For IR-transparent materials (ZnSe, Ge, CaF2 windows, thin films):
            //   L_transmitted = τ(λ) × L_background(λ)
            //
            // Energy conservation: ε + ρ + τ = 1 (Kirchhoff's law)
            // This enables rendering of IR optics and windows.
            // ================================================================
            float L_transmitted = 0.0;
            float transmittance = material.irTransmittance;

            if (transmittance > 0.001 && payload.depth < MAX_PATH_DEPTH) {
                // Trace transmission ray to get background radiance
                // For IR, we approximate background as atmospheric thermal emission
                // In a full implementation, would trace through and sample far surface
                float L_background = IRDownwellingRadiance(atmos, atmosIdx, lambda,
                                                          T_atmosphere,
                                                          lut.skyEmissivityClear);
                L_transmitted = transmittance * L_background;

                // Note: For accurate IR window simulation, should trace recursive ray
                // and sample the transmitted scene radiance. This simplified model
                // uses atmospheric background which is valid for outdoor scenes.
            }

            // 5. Total spectral radiance at this wavelength (with transmittance)
            float L_lambda = L_emission + L_reflected_atm + L_reflected_sun + L_transmitted;

            // NN atmosphere composition: L = tau_view(λ)·L_surface(λ) + L_path(λ)
            // (MWIR L_path already merges PTH_THRML + night-gated SOL_SCAT at bake time)
            if (atmosEnabled) {
                float tau_l = SampleAtmosTau(atmos, atmosNNData, atmosIdx, atmosA);
                float lpath_l = SampleAtmosLpath(atmos, atmosNNData, atmosIdx, atmosA, atmosAz);
                L_lambda = tau_l * L_lambda + lpath_l;
            }

            if (heroRay) {
                // Scalar spectral radiance: no trapezoid weight and no band
                // normalisation. The surface that sampled this wavelength
                // applies whatever weighting it owes.
                heroRadiance = L_lambda;
                continue;
            }

            // Accumulate (Riemann sum)
            // Trapezoid rule. N samples span N-1 intervals, so the two
            // endpoints carry half weight. Summing N full-width rectangles
            // and then dividing by (N-1)*step made every fused band read
            // N/(N-1) high -- 6.7% for the 16-sample bands. Checked against
            // an isothermal cavity, which must render exactly its own
            // blackbody and read 1.066667x it.
            float trapezoidW = (i == 0 || i == NUM_IR_SAMPLES - 1) ? 0.5 : 1.0;
            radiance_accum += L_lambda * lambda_step * trapezoidW;
        }

        // Normalize by wavelength range to get AVERAGE spectral radiance (W·sr⁻¹·m⁻²)
        // This allows fair comparison across bands with different bandwidths.
        // For sensor simulation or thermal analysis, use radiance_accum (band-integrated radiance).
        float band_width = lambda_max - lambda_min;
        float radiance_avg = heroRay ? heroRadiance : (radiance_accum / band_width);

        // The bounce correction. Sampling lambda_b uniformly gives pdf =
        // 1/band_width, which cancels the 1/band_width the band average carries,
        // so the estimator is added with no further weight -- and it is the same
        // expression for a hero ray, where neither factor is present. Multiplying
        // by band_width here and dividing again above is the mistake to avoid.
        //
        // The view-path transmittance still applies: the loop's terms each got
        // tau(lambda_i), so this one gets tau(lambda_b). L_path does not -- it is
        // added once per pixel by the loop, not once per light path.
        if (atmosEnabled) {
            bounceCorr *= SampleAtmosTau(atmos, atmosNNData, atmosIdx_b, atmosA);
        }
        radiance_avg += bounceCorr;

        // Validation
        if (!isfinite(radiance_avg)) {
            radiance_avg = 0.0;
        }
        // Symmetric, and wide for IR's dynamic range: bounceCorr is a residual
        // and goes negative under occlusion; see the note at the VIS_FUSED
        // clamp.
        radiance_avg = clamp(radiance_avg, -1e6, 1e6);

        // Output as grayscale (IR images are single-channel)
        output_radiance = float3(radiance_avg, radiance_avg, radiance_avg);

    } else {
        // ====================================================================
        // Fallback: Unknown mode or MULTISPECTRAL (TBD)
        // ====================================================================

        float radiance_spectral = (radiance.r + radiance.g + radiance.b) / 3.0;

        if (!isfinite(radiance_spectral)) {
            radiance_spectral = 0.0;
        }
        radiance_spectral = clamp(radiance_spectral, 0.0, 1000.0);

        output_radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);
    }

    // ========================================================================
    // Debug Visualization Output
    // ========================================================================
    // If debug mode is enabled, output debug visualization instead of normal radiance
    // This allows inspecting intermediate rendering data for debugging pipeline issues
    // ========================================================================

    if (SPEC_DEBUG_ENABLED != 0 && pushConsts.camera.debug_mode != DEBUG_MODE_NONE) {
        float3 debug_output = float3(1.0, 0.0, 1.0);  // Magenta = unhandled mode

        switch (pushConsts.camera.debug_mode) {
            // ----------------------------------------------------------------
            // Geometry Debug (1-9)
            // ----------------------------------------------------------------
            case DEBUG_MODE_WORLD_POSITION: {
                // World position (use frac for visibility, scaled by 0.1)
                float3 worldPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
                debug_output = frac(worldPos * 0.1);
                break;
            }
            case DEBUG_MODE_GEOMETRIC_NORMAL:
                // TRUE geometric normal from triangle edges (flat normal, remap [-1,1] to [0,1])
                // This is NOT the interpolated shading normal - it's the actual surface orientation
                debug_output = worldGeometricNormal * 0.5 + 0.5;
                break;

            case DEBUG_MODE_SHADED_NORMAL:
                // Final shaded normal with normal map (remap [-1,1] to [0,1])
                debug_output = normal * 0.5 + 0.5;
                break;

            case DEBUG_MODE_TANGENT:
                // Tangent vector (remap [-1,1] to [0,1])
                debug_output = worldTangent * 0.5 + 0.5;
                break;

            case DEBUG_MODE_UV:
                // UV coordinates (use frac for tiling visibility)
                debug_output = float3(frac(uv), 0.0);
                break;

            case DEBUG_MODE_MATERIAL_ID:
                // Material index hashed to color
                debug_output = HashToColor(materialID);
                break;

            case DEBUG_MODE_TRIANGLE_ID:
                // Triangle/primitive index hashed to color
                debug_output = HashToColor(PrimitiveIndex());
                break;

            case DEBUG_MODE_BARYCENTRIC:
                // Barycentric coordinates (b0, b1, b2) where b0 = 1 - b1 - b2
                debug_output = float3(1.0 - attribs.bary.x - attribs.bary.y,
                                      attribs.bary.x,
                                      attribs.bary.y);
                break;

            // ----------------------------------------------------------------
            // Material Debug (10-19)
            // ----------------------------------------------------------------
            case DEBUG_MODE_BASE_COLOR:
                // Albedo/base color RGB
                debug_output = baseColor.rgb;
                break;

            case DEBUG_MODE_METALLIC:
                // Metallic factor (grayscale)
                debug_output = float3(metallic, metallic, metallic);
                break;

            case DEBUG_MODE_ROUGHNESS:
                // Roughness factor (grayscale)
                debug_output = float3(roughness, roughness, roughness);
                break;

            case DEBUG_MODE_NORMAL_MAP_DELTA: {
                // Normal map contribution (difference from geometric normal)
                float3 delta = normal - worldNormal;
                debug_output = delta * 0.5 + 0.5;
                break;
            }

            case DEBUG_MODE_EMISSIVE:
                // Emissive RGB (tone map for visibility)
                debug_output = emissive / (1.0 + emissive);  // Simple Reinhard
                break;

            case DEBUG_MODE_ALPHA:
                // Alpha channel (grayscale)
                debug_output = float3(baseColor.a, baseColor.a, baseColor.a);
                break;

            // ----------------------------------------------------------------
            // Lighting Debug (20-29)
            // ----------------------------------------------------------------
            case DEBUG_MODE_NDOTL:
                // N dot L (grayscale)
                debug_output = float3(NdotL, NdotL, NdotL);
                break;

            case DEBUG_MODE_NDOTV: {
                // N dot V (grayscale)
                float NdotV_val = max(dot(normal, V), 0.0);
                debug_output = float3(NdotV_val, NdotV_val, NdotV_val);
                break;
            }

            case DEBUG_MODE_DIRECT_SUN:
                // Direct sun contribution (tone map for visibility)
                debug_output = directSun / (1.0 + directSun);
                break;

            case DEBUG_MODE_DIFFUSE:
                // Diffuse component (kD * albedo)
                debug_output = kD * albedo;
                break;

            case DEBUG_MODE_ATMOSPHERIC_TRANS: {
                // NN atmosphere view-path transmittance (RGB average at 650/550/450)
                float avg_transmittance = 1.0;
                if (atmosEnabled) {
                    avg_transmittance = (SampleAtmosTau(atmos, atmosNNData, 0, atmosA) +
                                         SampleAtmosTau(atmos, atmosNNData, 1, atmosA) +
                                         SampleAtmosTau(atmos, atmosNNData, 2, atmosA)) / 3.0;
                }
                debug_output = float3(avg_transmittance, avg_transmittance, avg_transmittance);
                break;
            }

            // ----------------------------------------------------------------
            // BRDF Debug (30-39)
            // ----------------------------------------------------------------
            case DEBUG_MODE_FRESNEL_F0:
                // Fresnel at normal incidence (F0)
                debug_output = F0;
                break;

            case DEBUG_MODE_FRESNEL: {
                // Fresnel at current viewing angle
                float NdotV_fresnel = max(dot(normal, V), 0.0);
                debug_output = FresnelSchlickRGB(NdotV_fresnel, F0);
                break;
            }

            case DEBUG_MODE_BRDF:
                // Full Cook-Torrance BRDF (scale for visibility)
                debug_output = brdf * 0.1;  // Scale down since BRDF can be large
                break;

            // ----------------------------------------------------------------
            // IBL Debug (40-49)
            // ----------------------------------------------------------------
            case DEBUG_MODE_REFLECTION_DIR: {
                // Reflection direction (remap [-1,1] to [0,1])
                float3 R = reflect(-V, normal);
                debug_output = R * 0.5 + 0.5;
                break;
            }

            case DEBUG_MODE_PREFILTERED_ENV:
                // Prefiltered environment map sample
                debug_output = prefilteredColor;
                break;

            case DEBUG_MODE_BRDF_LUT:
                // BRDF LUT sample (scale, bias, 0)
                debug_output = float3(envBRDF, 0.0);
                break;

            case DEBUG_MODE_IBL_SPECULAR:
                // IBL specular contribution
                debug_output = iblSpecular;
                break;

            case DEBUG_MODE_SKY_AMBIENT:
                // Sky ambient/diffuse contribution
                debug_output = skyAmbient;
                break;

            // ----------------------------------------------------------------
            // Spectral Debug (50-59)
            // ----------------------------------------------------------------
            case DEBUG_MODE_XYZ:
                // For spectral modes, this would show XYZ values
                // For RGB mode, show a placeholder
                debug_output = output_radiance;  // Current output as fallback
                break;

            case DEBUG_MODE_BEFORE_CHROMA:
                // RGB before chromaticity correction (same as XYZ for now)
                debug_output = output_radiance;
                break;

            case DEBUG_MODE_SPECTRAL_REFL: {
                // Spectral reflectance at 550nm (green)
                float refl550 = (albedo.r + albedo.g + albedo.b) / 3.0;  // Approximate
                debug_output = float3(refl550, refl550, refl550);
                break;
            }

            // ----------------------------------------------------------------
            // IR Debug (60-69)
            // ----------------------------------------------------------------
            case DEBUG_MODE_TEMPERATURE: {
                // Surface temperature (colormap 200K - 500K for visibility).
                // Same decode as the emission paths, so a temperature map
                // shows its field here instead of the scalar it overrides.
                float temp_K = GetSurfaceTemperatureK(material, geoInfo, PrimitiveIndex(), uv);
                if (temp_K <= 0.0) temp_K = 300.0;  // Default to room temp
                debug_output = TemperatureToColor(temp_K, 200.0, 500.0);
                break;
            }

            case DEBUG_MODE_IR_EMISSIVITY: {
                // IR emissivity (grayscale)
                float emissivity = GetEffectiveIREmissivity(material);
                debug_output = float3(emissivity, emissivity, emissivity);
                break;
            }

            case DEBUG_MODE_IR_EMISSION: {
                // Thermal emission component (grayscale, scaled)
                float temp_K = GetSurfaceTemperatureK(material, geoInfo, PrimitiveIndex(), uv);
                if (temp_K <= 0.0) temp_K = 300.0;
                float emission = GetEffectiveIREmissivity(material) * IRPlanckRadiance(temp_K, 10000.0);
                emission = emission / (1.0 + emission);  // Tone map
                debug_output = float3(emission, emission, emission);
                break;
            }

            case DEBUG_MODE_IR_REFLECTION: {
                // IR reflection component (grayscale)
                float refl = GetEffectiveIRReflectance(material);
                debug_output = float3(refl, refl, refl);
                break;
            }

            // ================================================================
            // Geometry Diagnostics (70-79) - For debugging mesh/index corruption
            // ================================================================
            // These modes help identify issues where triangles read vertices
            // from wrong locations, causing incorrect geometric normals.
            // ================================================================

            case DEBUG_MODE_VERTEX_POSITIONS: {
                // Hash each vertex position to RGB color
                // Same face should show similar colors (shared vertices)
                float hash0 = frac(v0.x * 12.9898 + v0.y * 78.233 + v0.z * 37.719);
                float hash1 = frac(v1.x * 12.9898 + v1.y * 78.233 + v1.z * 37.719);
                float hash2 = frac(v2.x * 12.9898 + v2.y * 78.233 + v2.z * 37.719);
                debug_output = float3(hash0, hash1, hash2);
                break;
            }

            case DEBUG_MODE_INDEX_VALUES: {
                // Show triangle indices as colors (normalized to 0-1 range)
                // Use 32.0 for small meshes like cube (24 vertices)
                float maxIdx = 32.0;
                debug_output = float3(
                    float(idx0) / maxIdx,
                    float(idx1) / maxIdx,
                    float(idx2) / maxIdx
                );
                break;
            }

            case DEBUG_MODE_INSTANCE_ID: {
                // Instance index hashed to color
                // Useful for verifying TLAS instance mapping
                float h = float(instanceIdx);
                debug_output = float3(
                    frac(h * 0.123456789),
                    frac(h * 0.234567891),
                    frac(h * 0.345678912)
                );
                break;
            }

            case DEBUG_MODE_PRIMITIVE_ID: {
                // Show PrimitiveIndex() directly
                // For 12-triangle cube: each face pair should have consecutive colors
                float pid = float(primitiveID);
                debug_output = float3(
                    pid / 12.0,                    // R: linear gradient 0-1 over 12 triangles
                    frac(pid * 0.5),               // G: alternates for triangle pairs
                    float(primitiveID % 2)         // B: 0 or 1 for even/odd triangles
                );
                break;
            }

            case DEBUG_MODE_INDEX_BUFFER_POS: {
                // Show the actual index buffer position we're reading from
                // This helps debug if indexOffset or primitiveID*3 calculation is wrong
                uint basePos = geoInfo.indexOffset + primitiveID * 3;
                debug_output = float3(
                    float(basePos) / 36.0,         // R: read position (0-35 for cube)
                    float(geoInfo.indexOffset) / 36.0,  // G: offset (should be 0 for single BLAS)
                    float(primitiveID) / 12.0      // B: primitive ID
                );
                break;
            }

            case DEBUG_MODE_V0_POSITION: {
                // Show v0 vertex position directly (use frac to map to 0-1)
                // For cube vertices at +/-1, frac gives 0 or 1-epsilon
                debug_output = float3(
                    frac(v0.x * 0.5 + 0.5),  // Map [-1,1] to [0,1]
                    frac(v0.y * 0.5 + 0.5),
                    frac(v0.z * 0.5 + 0.5)
                );
                break;
            }

            case DEBUG_MODE_RAW_IDX0: {
                // Show raw idx0 value and the actual read address
                // R = idx0 / 32 (for 24-vertex cube, range 0-0.75)
                // G = actual buffer read address (geoInfo.vertexOffset + idx0) / 32
                // B = geoInfo.vertexOffset / 32 (should be 0 for single BLAS)
                uint readAddr = geoInfo.vertexOffset + idx0;
                debug_output = float3(
                    float(idx0) / 32.0,
                    float(readAddr) / 32.0,
                    float(geoInfo.vertexOffset) / 32.0
                );
                break;
            }

            case DEBUG_MODE_V0_RAW: {
                // Show v0 position directly clamped (not frac)
                // This shows the actual sign: negative = 0, positive = 1
                // For cube at ±1: x=-1 gives 0, x=+1 gives 1
                debug_output = float3(
                    saturate(v0.x * 0.5 + 0.5),  // -1→0, 0→0.5, +1→1
                    saturate(v0.y * 0.5 + 0.5),
                    saturate(v0.z * 0.5 + 0.5)
                );
                break;
            }

            // ================================================================
            // Transmission Debug (80-89) - For debugging transmission materials
            // ================================================================
            // These modes help visualize transmission material properties
            // and verify correct Fresnel/refraction calculations.
            // ================================================================

            case DEBUG_MODE_TRANSMISSION: {
                // Transmission factor (grayscale)
                // 0 = opaque, 1 = fully transparent
                float trans = material.transmission;
                debug_output = float3(trans, trans, trans);
                break;
            }

            case DEBUG_MODE_IOR: {
                // Index of refraction (normalized for visualization)
                // Maps IOR range [1.0, 3.0] to [0.0, 1.0]
                // Common values: 1.0=air, 1.33=water, 1.5=glass, 2.4=diamond
                float ior_norm = saturate((material.ior - 1.0) / 2.0);
                debug_output = float3(ior_norm, ior_norm, ior_norm);
                break;
            }

            case DEBUG_MODE_FRESNEL_DIELECTRIC: {
                // Dielectric Fresnel reflectance at current view angle
                // Uses exact Fresnel equations (not Schlick approximation)
                float cosI = abs(dot(normal, V));
                float n1 = 1.0;  // Air
                float n2 = material.ior;
                float F = FresnelDielectric(cosI, n1, n2);
                debug_output = float3(F, F, F);
                break;
            }

            case DEBUG_MODE_ATTENUATION: {
                // Volume attenuation color
                // Shows the color that light becomes after traveling attenuationDistance
                debug_output = material.attenuationColor;
                break;
            }

            case DEBUG_MODE_ENERGY_AUDIT: {
                // Energy conservation visualization
                // R = reflected energy (Fresnel)
                // G = transmitted energy (1 - Fresnel, before absorption)
                // B = would-be absorbed energy (absorption coefficient indicator)
                //
                // For energy conservation: R + G + B should integrate to ~1
                float cosI = abs(dot(normal, V));
                float n1 = 1.0;
                float n2 = material.ior;
                float F = FresnelDielectric(cosI, n1, n2);
                float T = 1.0 - F;

                // Absorption indicator (0 = no absorption, 1 = full absorption at ref distance)
                float absorption_indicator = 0.0;
                if (material.attenuationDistance > 0.0) {
                    float3 atten = BeerLambertAbsorption(material.attenuationColor, material.attenuationDistance, material.attenuationDistance);
                    absorption_indicator = 1.0 - (atten.r + atten.g + atten.b) / 3.0;
                }

                debug_output = float3(F, T * (1.0 - absorption_indicator), T * absorption_indicator);
                break;
            }

            case DEBUG_MODE_DISPERSION: {
                // Dispersion coefficient visualization
                // Shows dispersion strength (0 = no dispersion, 1 = high dispersion)
                // Color indicates: R = low dispersion, G = medium, B = high
                float disp = saturate(material.dispersion * 10.0);  // Scale for visibility
                debug_output = float3(1.0 - disp, 1.0 - abs(disp - 0.5) * 2.0, disp);
                break;
            }

            case DEBUG_MODE_IOR_VARIATION: {
                // IOR variation across RGB wavelengths (Cauchy dispersion)
                // Shows how much IOR differs between R (650nm), G (550nm), B (450nm)
                float ior_R = CauchyIOR(material.ior, material.dispersion, 650.0);
                float ior_G = CauchyIOR(material.ior, material.dispersion, 550.0);
                float ior_B = CauchyIOR(material.ior, material.dispersion, 450.0);

                // Normalize to 0-1 range for visualization
                // IOR typically ranges 1.0 - 2.5
                debug_output = float3(
                    saturate((ior_R - 1.0) / 1.5),
                    saturate((ior_G - 1.0) / 1.5),
                    saturate((ior_B - 1.0) / 1.5)
                );
                break;
            }

            // ================================================================
            // Volume/Participating Media Debug (90-99)
            // ================================================================

            case DEBUG_MODE_VOLUME_DENSITY: {
                // Volume density (grayscale, scaled for visibility)
                float density = saturate(material.volumeDensity * 10.0);
                debug_output = float3(density, density, density);
                break;
            }

            case DEBUG_MODE_SCATTERING_COEFF: {
                // Scattering coefficient σ_s (scaled for visibility)
                float sigma_s = saturate(material.scatteringCoeff * 0.1);
                debug_output = float3(sigma_s, sigma_s, sigma_s);
                break;
            }

            case DEBUG_MODE_ABSORPTION_COEFF: {
                // Absorption coefficient σ_a (scaled for visibility)
                float sigma_a = saturate(material.absorptionCoeff * 0.1);
                debug_output = float3(sigma_a, sigma_a, sigma_a);
                break;
            }

            case DEBUG_MODE_EXTINCTION_COEFF: {
                // Extinction coefficient σ_t = σ_s + σ_a (scaled)
                float sigma_t = saturate((material.scatteringCoeff + material.absorptionCoeff) * 0.1);
                debug_output = float3(sigma_t, sigma_t, sigma_t);
                break;
            }

            case DEBUG_MODE_SINGLE_SCATTER_ALBEDO: {
                // Single scattering albedo ω = σ_s / σ_t
                // 0 = pure absorption, 1 = pure scattering
                float sigma_t = material.scatteringCoeff + material.absorptionCoeff;
                float albedo = (sigma_t > 0.001) ? material.scatteringCoeff / sigma_t : 0.0;
                debug_output = float3(albedo, albedo, albedo);
                break;
            }

            case DEBUG_MODE_PHASE_G: {
                // Henyey-Greenstein g parameter visualization
                // R channel: forward scattering (g > 0)
                // B channel: backward scattering (g < 0)
                // G channel: isotropic (g ≈ 0)
                float g = material.phaseG;
                debug_output = float3(
                    saturate(g),           // R: forward (g > 0)
                    1.0 - abs(g),          // G: isotropic (g ≈ 0)
                    saturate(-g)           // B: backward (g < 0)
                );
                break;
            }

            default:
                // Unknown mode: show magenta error color
                debug_output = float3(1.0, 0.0, 1.0);
                break;
        }

        // Apply debug output and return early
        payload.radiance = debug_output;
        return;
    }

    // ========================================================================
    // Participating Media Handling (Fog, Smoke, Clouds)
    // ========================================================================
    // For materials with volumeDensity > 0, compute volumetric effects:
    //   - Single scattering from light sources
    //   - Transmittance attenuation
    //   - Phase function for directional scattering
    //
    // PHYSICS:
    //   σ_t = σ_s + σ_a (extinction = scattering + absorption)
    //   T(d) = exp(-σ_t × d) (Beer-Lambert transmittance)
    //   L_inscatter = σ_s × phase(θ) × L_sun × visibility
    //
    // This is a simplified single-scattering model. For multiple scattering
    // (clouds, dense fog), use the full delta-tracking in volumetric.hlsli.
    // ========================================================================

    if (material.volumeDensity > 0.0 && material.scatteringCoeff > 0.0) {
        // ====================================================================
        // What "RGB-like" means here, and why the rest of this block asks
        // ====================================================================
        // Below the RGB and VIS_FUSED branches, output_radiance is a single
        // spectral value replicated across three channels -- every fused and
        // single-wavelength branch ends that way. This block used to ignore
        // that: it took the sun from lut.sunRadiance_rgb in every mode, so a
        // SWIR or LWIR render had an RGB triple in arbitrary units standing in
        // for spectral irradiance at 4 um, and it combined that with the
        // medium's per-channel sigma to produce three different answers to a
        // question that has one.
        //
        // The medium's coefficients are RGB because that is all a Material
        // carries -- there is no spectral sigma to sample. Averaging them is
        // the honest reduction until a medium can carry a curve the way a
        // surface reflectance does; that is a data problem, and this is where
        // it plugs in when it is solved.
        const bool isRgbLike = (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_RGB ||
                                SPEC_SPECTRAL_MODE == SPECTRAL_MODE_VIS_FUSED);

        // Create medium properties from material
        MediumProperties medium = CreateMediumFromMaterial(material);

        // Distance traveled through the medium (ray path length)
        float pathLength = RayTCurrent();

        // Compute transmittance using Beer-Lambert law
        float3 volumeTransmittance = BeerLambertTransmittance(pathLength, medium);
        if (!isRgbLike) {
            float t_avg = (volumeTransmittance.r + volumeTransmittance.g +
                           volumeTransmittance.b) / 3.0;
            volumeTransmittance = float3(t_avg, t_avg, t_avg);
        }

        // ====================================================================
        // Single Scattering: In-scattered Light from Sun
        // ====================================================================
        // Compute light scattered toward camera from particles along the ray
        // Using simplified single-scattering approximation
        // ====================================================================

        float3 inScatteredLight = float3(0.0, 0.0, 0.0);

        // Sun direction and irradiance. Irradiance, not the sun disk's
        // radiance: the phase function is normalised over the sphere, so
        // integrating a directional source against it leaves E rather than L
        // -- the same identity the fused surface paths spell out.
        //
        // Shadows the sunRadiance declared at the top of main(), on purpose:
        // that one is the RGB path's, and reading it here is the bug this
        // replaced.
        float3 sunDir = normalize(lut.sunDirection);
        float3 sunRadiance;
        if (isRgbLike) {
            // Derived from the illuminant's own spectrum by the host, so this
            // is the same sun the spectral branches sample, just colour-matched.
            sunRadiance = lut.sunRadiance_rgb;
        } else if (solarSpectralLUT[0].sunIrradiance.numSamples > 0) {
            const float sun_irr =
                SampleSunIrradiance(solarSpectralLUT, pushConsts.camera.wavelength_nm);
            sunRadiance = float3(sun_irr, sun_irr, sun_irr);
        } else {
            // No curve, no sun -- the same refusal as every surface path. An
            // illuminant's spectrum is scene data; there is nothing to invent.
            sunRadiance = float3(0.0, 0.0, 0.0);
        }

        // Phase function: Henyey-Greenstein
        float cosTheta = dot(-WorldRayDirection(), sunDir);
        float phase = HenyeyGreenstein(cosTheta, medium.g);

        // Estimate average scattering along ray (simplified)
        // For proper integration, would need to march along ray
        // Here we use midpoint approximation
        float3 midPoint = WorldRayOrigin() + WorldRayDirection() * (pathLength * 0.5);

        // Shadow test at midpoint for sun visibility
        RayDesc shadowRay;
        shadowRay.Origin = midPoint;
        shadowRay.Direction = sunDir;
        shadowRay.TMin = 0.001;
        shadowRay.TMax = 10000.0;

        Payload shadowPayload;
        shadowPayload.radiance = float3(0.0, 0.0, 0.0);
        shadowPayload.isShadowed = 1;  // Assume shadowed until miss shader says otherwise
        shadowPayload.heroLambda = payload.heroLambda;
        shadowPayload.depth = payload.depth + 1;
        shadowPayload.rngState = payload.rngState;
        shadowPayload.bsdfPdf = 0.0;   // not a BSDF sample

        // Trace shadow ray (uses miss shader index 1 for shadows)
        TraceRay(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
                 0xFF, 0, 0, 1, shadowRay, shadowPayload);

        float sunVisibility = (shadowPayload.isShadowed == 0) ? 1.0 : 0.0;

        // In-scattering: σ_s × phase × L_sun × visibility × (1 - T) / σ_t
        // The (1 - T) / σ_t term integrates scattering over the path
        float3 sigma_t = medium.sigma_t;
        float3 sigma_s = medium.sigma_s;
        float sigma_t_avg = (sigma_t.r + sigma_t.g + sigma_t.b) / 3.0;
        if (!isRgbLike) {
            // See the note at the top of this block: one spectral sample means
            // one sigma, not three.
            const float s_avg = (sigma_s.r + sigma_s.g + sigma_s.b) / 3.0;
            sigma_t = float3(sigma_t_avg, sigma_t_avg, sigma_t_avg);
            sigma_s = float3(s_avg, s_avg, s_avg);
        }

        if (sigma_t_avg > 0.001) {
            float3 oneMinusT = float3(1.0, 1.0, 1.0) - volumeTransmittance;
            inScatteredLight = sigma_s * phase * sunRadiance * sunVisibility * oneMinusT / sigma_t;
        }

        // ====================================================================
        // Apply Volume Effects to Output
        // ====================================================================
        // Final = surface_radiance × transmittance + in_scattered
        // ====================================================================

        output_radiance = output_radiance * volumeTransmittance + inScatteredLight;
    }

    // ========================================================================
    // Transmission Material Handling (Glass, Water, etc.)
    // ========================================================================
    // For materials with transmission > 0, we need to trace additional rays
    // to compute refraction and reflection contributions.
    //
    // PHYSICS:
    //   At a dielectric interface (air/glass):
    //   - Fresnel equations determine reflection vs transmission ratio
    //   - Snell's law determines refraction angle
    //   - Beer-Lambert law models volume absorption (colored glass)
    //
    // ALGORITHM:
    //   1. Check if material has transmission and depth allows recursion
    //   2. Compute Fresnel reflectance F (exact dielectric formula)
    //   3. Use Russian roulette to choose reflection OR refraction (not both)
    //   4. Trace the chosen ray and accumulate contribution
    //   5. Apply Beer-Lambert absorption for refracted rays inside medium
    //
    // ENERGY CONSERVATION:
    //   E_incident = F × E_reflected + (1-F) × E_transmitted
    //   Russian roulette ensures unbiased estimation of both paths
    // ========================================================================

    if (material.transmission > 0.0 && payload.depth < MAX_PATH_DEPTH) {
        // Get hit point and ray direction
        float3 hitPoint = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
        float3 rayDir = WorldRayDirection();

        // Determine if entering or exiting the medium
        // Entering: ray direction and normal point in opposite directions (dot < 0)
        bool entering = dot(rayDir, normal) < 0.0;

        // Use geometric normal facing the ray
        float3 N = entering ? normal : -normal;

        // Compute Fresnel reflectance (exact dielectric formula)
        float cosI = abs(dot(N, -rayDir));

        // Scale transmission contribution by material.transmission factor
        // This allows partial transmission (frosted glass effect)
        float transmissionWeight = material.transmission;

        // Compute reflection direction (same for all wavelengths)
        float3 reflectDir = reflect(rayDir, N);

        // ====================================================================
        // Dispersion Handling: Wavelength-Dependent IOR
        // ====================================================================
        // When material.dispersion > 0, different wavelengths refract at
        // different angles (chromatic dispersion / rainbow effect).
        //
        // For RGB mode: Approximate by tracing 3 rays at R/G/B wavelengths
        // For VIS_FUSED: Use central wavelength (550nm) for single ray
        // For SINGLE: Use pushConsts.camera.wavelength_nm
        //
        // Cauchy formula: n(λ) = n_d + dispersion × 0.01 / λ²
        // ====================================================================

        float3 transmissionRadiance = float3(0.0, 0.0, 0.0);

        // PCG random number for reflection/refraction decision
        uint rngState = payload.rngState;
        rngState = rngState * 747796405u + 2891336453u;
        uint word = ((rngState >> ((rngState >> 28u) + 4u)) ^ rngState) * 277803737u;
        word = (word >> 22u) ^ word;
        float xi = float(word) / 4294967296.0;
        payload.rngState = rngState;

        // Prepare recursive ray template
        RayDesc recursiveRay;
        recursiveRay.Origin = hitPoint;
        recursiveRay.TMin = 0.001;
        recursiveRay.TMax = 10000.0;

        // Prepare recursive payload template
        Payload recursivePayload;
        recursivePayload.isShadowed = 0;
        // A refracted or specularly reflected ray is not a BSDF *sample* in the
        // MIS sense -- the direction was determined, not drawn from a density --
        // so an emitter it lands on contributes its full emission.
        recursivePayload.bsdfPdf = 0.0;
        recursivePayload.depth = payload.depth + 1;
        recursivePayload.rngState = rngState;

        recursivePayload.heroLambda = payload.heroLambda;

        // A material disperses if it carries an Abbe number or a measured
        // n(lambda) table. Nothing here is specific to glass: any medium with n
        // meaningfully above 1 disperses, and water (1.333, Abbe ~56), ice,
        // acrylic and quartz all reach this path the same way a prism does.
        bool materialDisperses = (material.dispersion > 0.001) ||
                                 (material.complexRefractiveIndexIndex >= 0);

        // VIS_FUSED disperses by hero wavelength sampling; only an undivided
        // ray samples one. A ray that already carries a hero wavelength
        // refracts at that wavelength and stays single -- resampling would
        // branch the path count multiplicatively and bias nothing usefully.
        bool heroSplit = materialDisperses &&
                         SPEC_SPECTRAL_MODE == SPECTRAL_MODE_VIS_FUSED &&
                         payload.heroLambda <= 0.0;

        bool hasDispersion = materialDisperses &&
                             (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_RGB);

        if (heroSplit) {
            // ================================================================
            // Hero wavelength: one wavelength, one path, weighted by 1/pdf
            // ================================================================
            // The band cannot follow one path through a dispersive interface,
            // so pick a wavelength and follow that. The estimator is
            //
            //   XYZ = L(λ_h) · cmf(λ_h) / pdf(λ_h)
            //
            // whose expectation is ∫L(λ)cmf(λ)dλ -- the same integral the
            // deterministic 32-point grid approximates. That equivalence is the
            // test: with a CONSTANT n(λ) this must converge to the
            // non-dispersing render, and it does, to 0.29% at 512 spp.
            //
            // λ_h is drawn against the CMF-shaped density rather than flat, for
            // the reason given at SampleVisibleWavelength: uniform draws make
            // the weight cmf/pdf swing with nothing but the draw, and a prism
            // pixel already carries the reflect/refract coin on top of it.
            //
            // Wavelength is where the noise goes. Nothing outside a dispersive
            // refraction samples it, so an ordinary scene is unaffected.
            // ================================================================
            uint heroState = payload.rngState * 747796405u + 2891336453u;
            uint heroWord = ((heroState >> ((heroState >> 28u) + 4u)) ^ heroState)
                            * 277803737u;
            heroWord = (heroWord >> 22u) ^ heroWord;
            payload.rngState = heroState;

            const float u_lambda = float(heroWord) / 4294967296.0;
            const float lambda_h = SampleVisibleWavelength(u_lambda);

            const float ior_h = RefractionIOR(material, lambda_h);
            const float n1 = entering ? 1.0 : ior_h;
            const float n2 = entering ? ior_h : 1.0;

            float F = FresnelDielectric(cosI, n1, n2);
            const float3 refractDir = Refract(rayDir, N, n1 / n2);
            if (length(refractDir) < 0.001) F = 1.0;   // total internal reflection

            recursivePayload.radiance = float3(0.0, 0.0, 0.0);
            recursivePayload.heroLambda = lambda_h;
            recursivePayload.rngState = payload.rngState;
            recursiveRay.Direction = (xi < F) ? reflectDir : refractDir;

            TraceRay(scene, RAY_FLAG_NONE, 0xFF, 0, 0, 0, recursiveRay, recursivePayload);

            // Scalar spectral radiance, by the contract on Payload::heroLambda.
            float L_h = recursivePayload.radiance.r;

            if (!entering && material.attenuationDistance > 0.0) {
                // Beer-Lambert is per-channel RGB data; at one wavelength the
                // honest reduction is the channel average. A medium carrying a
                // spectral absorption curve would be sampled at λ_h instead.
                const float3 atten = BeerLambertAbsorption(
                    material.attenuationColor, RayTCurrent(), material.attenuationDistance);
                L_h *= (atten.r + atten.g + atten.b) / 3.0;
            }

            const float3 cmf = SampleCIE_XYZ_LUT(cieCMF_LUT, lambda_h);
            float3 XYZ_h = L_h * cmf / VisibleWavelengthPDF(lambda_h);
            XYZ_h /= CIE_Y_INTEGRAL;

            float3 heroRgb = ConvertXYZToLinearRGB(XYZ_h);
            heroRgb.r *= lut.chromaR_correction;
            heroRgb.b *= lut.chromaB_correction;
            transmissionRadiance = heroRgb;

        } else if (hasDispersion && SPEC_SPECTRAL_MODE == SPECTRAL_MODE_RGB) {
            // ================================================================
            // RGB Dispersion: Trace 3 separate rays for R, G, B wavelengths
            // ================================================================
            // This produces chromatic aberration / rainbow effects at edges
            // Wavelengths: R=650nm, G=550nm, B=450nm (approximate primaries)
            // ================================================================

            float wavelengths[3] = { 650.0, 550.0, 450.0 };  // nm
            float3 channelRadiance = float3(0.0, 0.0, 0.0);

            // [loop] is REQUIRED: this loop contains a recursive TraceRay.
            // Unrolling it creates multiple static recursive trace call sites,
            // which crashes the device on nested closest-hit invocations
            // (same failure mode as the thermal-IR wavelength loop).
            [loop]
            for (int ch = 0; ch < 3; ch++) {
                float lambda = wavelengths[ch];
                float ior_lambda = RefractionIOR(material, lambda);

                float n1 = entering ? 1.0 : ior_lambda;
                float n2 = entering ? ior_lambda : 1.0;
                float eta = n1 / n2;

                float F = FresnelDielectric(cosI, n1, n2);
                float3 refractDir = Refract(rayDir, N, eta);
                bool hasTIR = (length(refractDir) < 0.001);

                if (hasTIR) F = 1.0;

                recursivePayload.radiance = float3(0.0, 0.0, 0.0);

                // Use same random decision for all channels for consistency
                if (xi < F) {
                    // Reflection
                    recursiveRay.Direction = reflectDir;
                } else {
                    // Refraction with wavelength-specific direction
                    recursiveRay.Direction = refractDir;
                }

                TraceRay(scene, RAY_FLAG_NONE, 0xFF, 0, 0, 0, recursiveRay, recursivePayload);

                // Extract the channel-specific contribution
                float channelValue = (ch == 0) ? recursivePayload.radiance.r :
                                     (ch == 1) ? recursivePayload.radiance.g :
                                                 recursivePayload.radiance.b;

                // Apply Beer-Lambert absorption for this channel
                if (!entering && material.attenuationDistance > 0.0) {
                    float travelDistance = RayTCurrent();
                    float3 volumeAtten = BeerLambertAbsorption(
                        material.attenuationColor, travelDistance, material.attenuationDistance);
                    float attenValue = (ch == 0) ? volumeAtten.r :
                                       (ch == 1) ? volumeAtten.g : volumeAtten.b;
                    channelValue *= attenValue;
                }

                if (ch == 0) channelRadiance.r = channelValue;
                else if (ch == 1) channelRadiance.g = channelValue;
                else channelRadiance.b = channelValue;
            }

            transmissionRadiance = channelRadiance;

        } else {
            // ================================================================
            // Standard Refraction (no dispersion or single wavelength mode)
            // ================================================================

            // The wavelength this ray is accountable for, or 0 where the mode
            // has none and n falls back to the material's n_d.
            //
            //   any band, ray carrying a hero wavelength -> that wavelength
            //   VIS_FUSED undivided, or RGB              -> none
            //   SINGLE and the undivided fused IR bands  -> the render wavelength
            //
            // A hero ray reaching here means the material does not disperse, or
            // it is the second dispersive interface on the same path. Either
            // way it keeps refracting at its own wavelength -- and since the IR
            // bands now spawn hero rays of their own for the environment
            // bounce, a bounce that then passes through a window has to refract
            // at the wavelength it was sampled for, not at pushConsts.camera.wavelength_nm.
            float refractLambda;
            if (payload.heroLambda > 0.0) {
                refractLambda = payload.heroLambda;
            } else if (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_VIS_FUSED ||
                       SPEC_SPECTRAL_MODE == SPECTRAL_MODE_RGB) {
                refractLambda = 0.0;
            } else {
                refractLambda = pushConsts.camera.wavelength_nm;
            }

            float effectiveIOR = RefractionIOR(material, refractLambda);

            float n1 = entering ? 1.0 : effectiveIOR;
            float n2 = entering ? effectiveIOR : 1.0;
            float eta = n1 / n2;

            float F = FresnelDielectric(cosI, n1, n2);
            float3 refractDir = Refract(rayDir, N, eta);
            bool hasTIR = (length(refractDir) < 0.001);

            if (hasTIR) F = 1.0;

            recursivePayload.radiance = float3(0.0, 0.0, 0.0);

            // Single trace call site: pick the direction first, then trace.
            // Duplicating TraceRay per branch multiplies the static recursive
            // call sites for no benefit.
            bool reflected = (xi < F);
            recursiveRay.Direction = reflected ? reflectDir : refractDir;
            TraceRay(scene, RAY_FLAG_NONE, 0xFF, 0, 0, 0, recursiveRay, recursivePayload);

            if (reflected) {
                transmissionRadiance = recursivePayload.radiance;
            } else {
                // Apply Beer-Lambert absorption on the refraction path
                float3 volumeAttenuation = float3(1.0, 1.0, 1.0);
                if (!entering && material.attenuationDistance > 0.0) {
                    float travelDistance = RayTCurrent();
                    volumeAttenuation = BeerLambertAbsorption(
                        material.attenuationColor, travelDistance, material.attenuationDistance);
                }
                transmissionRadiance = recursivePayload.radiance * volumeAttenuation;
            }
        }

        // Blend transmission with surface shading
        output_radiance = lerp(output_radiance, transmissionRadiance, transmissionWeight);
    }

    payload.radiance = output_radiance;
}
