/**
 * @file Material.hpp
 * @brief PBR material properties (glTF 2.0 metallic-roughness workflow) with spectral/IR extensions
 *
 * Provides Material struct implementing:
 * - Full glTF 2.0 PBR metallic-roughness model
 * - Base color (RGB + texture)
 * - Metallic-Roughness workflow (metal vs. dielectric)
 * - Normal mapping for surface detail
 * - Emissive properties (HDR self-emission)
 * - Alpha blending modes (opaque, mask, blend)
 *
 * Spectral rendering extensions:
 * - spectralAlbedo: Scalar reflectance for single-wavelength (legacy)
 * - spectralReflectanceCurveIndex: Index into full spectral curves (quantitative)
 * - SpectralSource tracking: Measured vs. RGB-upsampled (quality gate)
 *
 * Infrared (MWIR/LWIR) extensions:
 * - irEmissivityCurve: Blackbody emission fraction ε(λ)
 * - irReflectanceCurve: Reflected radiance fraction ρ(λ)
 * - irTransmittanceCurve: Transmitted radiance fraction τ(λ)
 * - irTemperature_K: Surface temperature for Planck's law
 * - Kirchhoff's law validation: ε + ρ + τ = 1
 *
 * Quantiloom spectral material system:
 * - quantiloomMaterialType: Database type (e.g., "quantiloom_usgs")
 * - quantiloomMaterialRef: Material name in database
 * - Enables SpectralBaker NMF basis reconstruction
 *
 * @note All texture indices reference Scene::textures array (-1 = no texture)
 * @note Uploaded to GPU via MaterialData buffer (see main.cpp MaterialDataCPU)
 * @note Shader access via bindless descriptor arrays
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "scene/BRDFModels.hpp"
#include <glm/glm.hpp>
#include <string>

// ============================================================================
// Material - PBR material properties (glTF 2.0 metallic-roughness model)
// ============================================================================
    
namespace quantiloom {

/**
 * @struct UvTransform
 * @brief One texture slot's UV transform (glTF KHR_texture_transform)
 *
 * Applied as uv' = Translate(offset) * Rotate(rotation) * Scale(scale) * uv,
 * in that order, which is what the extension specifies. The default is the
 * identity, so a slot whose glTF carried no extension is untouched.
 *
 * Held in the authored form rather than pre-multiplied because that is what a
 * config or an editor sets; ConvertMaterial folds it into a 2x3 affine once,
 * on the way to the GPU.
 */
struct UvTransform {
    glm::vec2 offset{0.0f, 0.0f};
    f32 rotation = 0.0f;  // radians, counter-clockwise about the UV origin
    glm::vec2 scale{1.0f, 1.0f};

    [[nodiscard]] bool IsIdentity() const {
        return offset.x == 0.0f && offset.y == 0.0f && rotation == 0.0f &&
               scale.x == 1.0f && scale.y == 1.0f;
    }
};

/**
 * @struct Material
 * @brief Physically-based material with glTF 2.0 PBR and spectral/IR extensions
 *
 * Combines standard glTF 2.0 PBR with Quantiloom spectral rendering features:
 * - glTF 2.0: base color, metallic, roughness, normal maps, emissive
 * - Spectral: Full wavelength-dependent reflectance curves (400-2500nm)
 * - Infrared: Emissivity, reflectance, transmittance curves (3-12μm)
 * - Quality tracking: Measured vs. RGB-upsampled spectral data
 *
 * Usage example:
 * @code
 * // Create Lambertian material
 * Material mat = Material::CreateLambertian(glm::vec3(0.8f, 0.2f, 0.1f), "RedDiffuse");
 *
 * // glTF-loaded material with textures
 * Material gltfMat;
 * gltfMat.baseColorTextureIndex = 0;  // Index into Scene::textures
 * gltfMat.metallicFactor = 0.0f;
 * gltfMat.roughnessFactor = 0.8f;
 *
 * // Spectral material (quantitative)
 * gltfMat.spectralReflectanceCurveIndex = 5;  // Index into spectral curves buffer
 * gltfMat.spectralSource = Material::SpectralSource::Measured;
 *
 * // IR thermal material
 * gltfMat.irEmissivityCurve = {{3000.0f, 0.9f}, {5000.0f, 0.85f}, ...};
 * gltfMat.irTemperature_K = 300.0f;  // Room temperature
 * @endcode
 *
 * @note For quantitative spectral rendering, use Measured spectral sources only
 * @note RGB-upsampled materials are NOT suitable for scientific analysis
 * @note Infrared materials must satisfy Kirchhoff's law: ε + ρ + τ ≤ 1
 *
 * @see SpectralCurve for spectral reflectance curves
 * @see main.cpp MaterialDataCPU for GPU upload structure
 */
struct QL_API Material {
    // ========================================================================
    // PBR Base Color
    // ========================================================================
    glm::vec4 baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};  // RGBA [0, 1]
    i32 baseColorTextureIndex = -1;  // -1 = no texture

    // ========================================================================
    // Metallic-Roughness
    // ========================================================================
    f32 metallicFactor = 0.0f;   // [0, 1] (0 = dielectric, 1 = metal)
    f32 roughnessFactor = 1.0f;  // [0, 1] (0 = smooth, 1 = rough)
    i32 metallicRoughnessTextureIndex = -1;  // -1 = no texture
    // NOTE: In glTF, this is a combined texture (R=unused, G=roughness, B=metallic)

    // ========================================================================
    // Normal Mapping
    // ========================================================================
    i32 normalTextureIndex = -1;  // -1 = no normal map
    f32 normalScale = 1.0f;       // Normal map intensity [0, inf]

    // ========================================================================
    // Emissive
    // ========================================================================
    glm::vec3 emissiveFactor{0.0f, 0.0f, 0.0f};  // RGB [0, inf] (HDR allowed)
    i32 emissiveTextureIndex = -1;  // -1 = no texture

    // ========================================================================
    // Alpha Mode
    // ========================================================================
    enum class AlphaMode : u32 {
        Opaque = 0,  // Alpha channel ignored
        Mask = 1,    // Binary alpha test (alphaCutoff threshold)
        Blend = 2    // Alpha blending (requires sorted rendering)
    };

    AlphaMode alphaMode = AlphaMode::Opaque;
    f32 alphaCutoff = 0.5f;  // Threshold for AlphaMode::Mask

    // ========================================================================
    // Double-Sided Rendering (glTF 2.0 / USD)
    // ========================================================================
    // When true: disable backface culling, flip normal for backface hits
    // When false: enable hardware backface culling (rays pass through backfaces)
    bool doubleSided = false;

    // ========================================================================
    // Spectral Mode (M1 compatibility and M2+ full spectral)
    // ========================================================================
    // LEGACY (M1): Scalar spectral reflectance for single-wavelength rendering
    // Computed from baseColorFactor during scene loading:
    //   spectralAlbedo = (R + G + B) / 3.0
    // This field is kept for backward compatibility with M1 test scenes
    f32 spectralAlbedo = 0.8f;

    // NEW (M2+): Index into Scene::spectralReflectanceCurves array
    // -1 = no spectral curve (fallback to spectralAlbedo scalar)
    // >=0 = index into spectral curve buffer for physically-based spectral rendering
    // This enables full spectral fidelity for quantitative spectral rendering
    i32 spectralReflectanceCurveIndex = -1;

    // ========================================================================
    // Spectral Data Source Tracking (for quantitative validation)
    // ========================================================================
    // Tracks the origin of spectral data to enforce quality gates
    enum class SpectralSource : u32 {
        Unknown = 0,           // Default: unspecified
        Measured = 1,          // Physically measured spectral data (quantitative)
        RGBUpsampled = 2,      // Upsampled from RGB (sRGB or linear) - NOT quantitative
        Procedural = 3         // Procedurally generated (e.g., metal Fresnel)
    };
    SpectralSource spectralSource = SpectralSource::Unknown;

    // ========================================================================
    // Complex Refractive Index (for physical Fresnel)
    // ========================================================================
    // Index into the CRI buffer (binding 14) for wavelength-dependent Fresnel.
    // -1 = use standard PBR Schlick approximation (backward compatible default)
    // >= 0 = use physical Fresnel with measured n(λ), k(λ) data
    i32 complexRefractiveIndexIndex = -1;

    // ========================================================================
    // Infrared Material Properties (for MWIR/LWIR modes)
    // ========================================================================
    // Spectral curves for infrared rendering (3-12μm wavelength range)
    // Used for quantitative thermal imaging simulation

    // Emissivity curve ε(λ): fraction of blackbody radiation emitted [0, 1]
    // By Kirchhoff's law: ε(λ) = α(λ) = 1 - ρ(λ) - τ(λ) in thermal equilibrium
    Vector<std::pair<f32, f32>> irEmissivityCurve;  // (wavelength_nm, emissivity)

    // Reflectance curve ρ(λ): fraction of incident radiation reflected [0, 1]
    Vector<std::pair<f32, f32>> irReflectanceCurve;  // (wavelength_nm, reflectance)

    // Transmittance curve τ(λ): fraction of incident radiation transmitted [0, 1]
    Vector<std::pair<f32, f32>> irTransmittanceCurve;  // (wavelength_nm, transmittance)

    // Surface temperature (K) for self-emission calculation
    // If <= 0, no thermal emission (or use scene ambient temperature)
    f32 irTemperature_K = 0.0f;

    // Temperature texture for per-pixel temperature field
    // When >= 0, samples texture R channel and computes:
    //   T(K) = texValue * temperatureScale + temperatureOffset
    // When < 0 (default), uses scalar irTemperature_K
    i32 temperatureTextureIndex = -1;
    f32 temperatureScale = 500.0f;    // Default: [0,1] -> [200K, 700K]
    f32 temperatureOffset = 200.0f;

    // Authored temperature map, relative to the config, the way
    // spectralWeightTexturePath is. Mounted into temperatureTextureIndex by
    // MountTemperatureTextures before texture upload. Stored as UNORM8, so the
    // resolution is temperatureScale / 255 kelvin per step -- narrow the range
    // (scale = 60, offset = 270 gives 0.24 K) when the field is subtle.
    String temperatureTexturePath;

    // ========================================================================
    // Transmission Properties (KHR_materials_transmission, KHR_materials_volume)
    // ========================================================================
    // Physical transparency for glass, water, and other dielectric materials.
    // Based on glTF 2.0 KHR_materials_transmission and KHR_materials_volume extensions.
    //
    // PHYSICS:
    // - IOR determines refraction angle (Snell's law) and Fresnel reflection ratio
    // - Transmission controls how much light passes through (vs absorbed/reflected)
    // - Attenuation models Beer-Lambert absorption in colored glass/liquids
    // - Dispersion causes wavelength-dependent refraction (rainbows, prism effects)
    //
    // ENERGY CONSERVATION:
    //   Incident = Reflected + Transmitted + Absorbed
    //   F = Fresnel(n1, n2, θ) = reflection ratio
    //   T = 1 - F = transmission ratio (before volume absorption)
    //   Final_transmission = T × exp(-σ × distance)  where σ = absorption coefficient

    f32 ior = 1.5f;                    // Index of refraction (1.0=air, 1.33=water, 1.5=glass, 2.4=diamond)
    f32 transmission = 0.0f;           // Transmission strength [0,1] (0=opaque, 1=fully transparent)
    i32 transmissionTextureIndex = -1; // Transmission texture index (-1 = no texture)

    // Volume attenuation (KHR_materials_volume)
    glm::vec3 attenuationColor = {1.0f, 1.0f, 1.0f}; // Color after light travels attenuationDistance
    f32 attenuationDistance = 0.0f;    // Distance at which light attenuates to attenuationColor (0 = infinite, no attenuation)
    f32 thicknessFactor = 0.0f;        // Thickness for thin-walled approximation (0 = solid)
    i32 thicknessTextureIndex = -1;    // Thickness texture index (-1 = no texture)

    // Dispersion (wavelength-dependent IOR)
    f32 dispersion = 0.0f;             // Abbe number reciprocal (0 = no dispersion)

    // ========================================================================
    // Participating Media Properties (fog, smoke, subsurface scattering)
    // ========================================================================
    // For volume rendering with scattering and absorption.
    // Uses delta-tracking algorithm (same as atmospheric scattering).
    //
    // PHYSICS:
    //   σ_t = σ_a + σ_s (extinction = absorption + scattering)
    //   Beer-Lambert: T = exp(-σ_t × d)
    //   Single scattering albedo: ω = σ_s / σ_t

    f32 volumeDensity = 0.0f;          // Medium density multiplier (0 = no volume)
    f32 scatteringCoeff = 0.0f;        // Scattering coefficient σ_s (m⁻¹)
    f32 absorptionCoeff = 0.0f;        // Absorption coefficient σ_a (m⁻¹)
    f32 phaseG = 0.0f;                 // Henyey-Greenstein g parameter [-1,1] (0=isotropic, >0=forward)

    // ========================================================================
    // Quantiloom Spectral Material Reference (from glTF extras)
    // ========================================================================
    // When set, this material uses pre-computed spectral data from external databases.
    // The type specifies the data source, and the name references an entry in that database.
    //
    // Example glTF extras:
    //   "extras": {
    //     "quantiloom_material": {
    //       "type": "quantiloom_usgs",
    //       "name": "Aluminum brushed 293K"
    //     }
    //   }
    //
    // Supported types:
    //   - "quantiloom_usgs": SpectralBaker NMF basis from USGS library
    //   - (future) "refractiveindex_info": RefractiveIndex.INFO database
    //   - (future) "custom_csv": Custom CSV spectral curve
    String quantiloomMaterialType;  // e.g., "quantiloom_usgs"
    String quantiloomMaterialRef;   // Material name in the database

    // Check if material has a Quantiloom spectral reference
    [[nodiscard]] bool HasQuantiloomRef() const {
        return !quantiloomMaterialType.empty() && !quantiloomMaterialRef.empty();
    }

    // ========================================================================
    // Endmember Mixing (spatially varying spectral reflectance)
    // ========================================================================
    // A single bound curve replaces the base-colour texture outright, so a
    // measured surface renders as one flat reflectance and loses every bit of
    // spatial detail the texture carried. A mixture restores it:
    //
    //   rho(lambda, uv) = sum_i w_i(uv) * rho_i(lambda)
    //
    // where the rho_i are up to MAX_ENDMEMBERS measured curves (this ref plus
    // quantiloomExtraRefs) and w_i comes from a weight texture, by default
    // unmixed from the base colour at load. With one endmember the mixture
    // degenerates to brightness modulation of that curve, which is why the
    // default applies to existing single-ref materials too.
    //
    // The mixing is evaluated per wavelength on the GPU, never collapsed to a
    // scalar on the CPU: hyperspectral renders loop bands without rebuilding
    // the material buffer, so anything pre-evaluated at one wavelength would
    // be frozen there.
    enum class SpectralUnmixMode : u8 {
        Auto = 0,   // derive weights from the base-colour texture at load
        Texture,    // use the weight texture named below, as authored
        Off         // no weights: the first curve, flat, as before
    };

    // Four: one per channel of an RGBA weight texture, and the most that three
    // colour equations can be asked to resolve.
    static constexpr i32 MAX_ENDMEMBERS = 4;

    Vector<String> quantiloomExtraRefs;  // endmembers 1..3; endmember 0 is the ref above
    SpectralUnmixMode spectralUnmixMode = SpectralUnmixMode::Auto;
    String spectralWeightTexturePath;    // relative to the config, for Texture mode

    // Runtime slots, filled by ResolveMaterialSpectra the way
    // spectralReflectanceCurveIndex is (that one holds endmember 0).
    // -1 means absent, and a weight texture of -1 means w = (1, 0, 0, 0),
    // which reproduces the flat single-curve behaviour exactly.
    i32 endmemberCurveIndex1 = -1;
    i32 endmemberCurveIndex2 = -1;
    i32 endmemberCurveIndex3 = -1;
    i32 weightTextureIndex = -1;

    // Planck-weighted band-averaged emissivity for the LWIR thermal solver.
    // Computed by ConfigResolve from the material's spectral ref (if bound):
    //   eps = 1 - <rho>_Planck(8-12um, 300K) - tau
    // Sentinel -1 means "not computed; fall back to the heuristic".
    f32 bandAveragedIREmissivity = -1.0f;

    // ========================================================================
    // BRDF Model Selection (CPU-side analytical evaluation)
    // ========================================================================
    enum class BRDFModel : uint8_t {
        CookTorrance = 0,  // default GPU path (GGX in pbr.hlsli)
        Lambertian,        // brdf_lambertian
        FiveParam,         // brdf_5p
        KernelDriven,      // brdf_kernel_driven (RossThick-LiTransit)
        Ocean,             // brdf_ocean (Cox-Munk)
        StaylorSuttles,    // brdf_staylor_suttles
        Otterman,          // brdf_otterman
    };
    BRDFModel brdfModel = BRDFModel::CookTorrance;

    // ========================================================================
    // Metadata
    // ========================================================================
    String name;  // Material name (for debugging)

    // ========================================================================
    // Sheen (KHR_materials_sheen) and per-slot UV transforms
    // (KHR_texture_transform)
    // ========================================================================
    // Appended here rather than beside the other KHR extensions above because
    // this header is a layout contract Quantiloom-Qt reads by offset, and an
    // append leaves every field that already existed where it was.
    //
    // PHYSICS:
    // Sheen is the lobe of a microfibre surface -- velvet, felt, brushed cloth.
    // Light scatters off fibres standing away from the surface, so the lobe
    // peaks at grazing angles instead of around the mirror direction, which is
    // the opposite of what a GGX specular does and why no roughness setting on
    // the base lobe reproduces it. Evaluated with the Charlie distribution the
    // glTF specification names.
    //
    // ENERGY:
    // Sheen layers on top of the base BRDF, and the base is scaled by
    //   1 - max(sheenColor) * E_sheen(cos theta_v)
    // to pay for it (the albedo-scaling approximation from the spec). With
    // sheenColorFactor at its default of zero that scale is exactly 1 and every
    // sheen term is exactly 0, so an existing scene renders bit-identically.

    glm::vec3 sheenColorFactor{0.0f, 0.0f, 0.0f};  // [0,1]; 0 = no sheen (glTF default)
    f32 sheenRoughnessFactor = 0.0f;               // [0,1] (glTF default)
    i32 sheenColorTextureIndex = -1;               // RGB channels, sRGB-encoded
    i32 sheenRoughnessTextureIndex = -1;           // ALPHA channel, linear

    // Measured sheen reflectance, resolved by ResolveMaterialSpectra exactly as
    // spectralReflectanceCurveIndex is. This is also the only way sheen reaches
    // the infrared bands: an RGB factor upsampled through the visible Gaussian
    // basis carries no meaning past ~1400nm, so NIR/SWIR/MWIR/LWIR ignore the
    // factor and act only on a bound curve.
    i32 sheenReflectanceCurveIndex = -1;
    String quantiloomSheenRef;  // database name, resolved into the index above

    // Per-slot UV transforms. Per slot rather than per material because the
    // sample assets require it: SheenChair's fabric puts its base colour at
    // scale 7 and its normal map at scale 2 within one material. Identity by
    // default, so an asset without the extension is unaffected.
    UvTransform baseColorUv;
    UvTransform metallicRoughnessUv;
    UvTransform normalUv;
    UvTransform emissiveUv;
    UvTransform sheenColorUv;
    UvTransform sheenRoughnessUv;

    // ========================================================================
    // KHR_materials_specular
    // ========================================================================
    // Appended for the same reason the sheen block above was: this header is a
    // layout contract, and appending leaves every existing offset alone.
    //
    // These do not add a lobe; they reshape the dielectric Fresnel the base
    // already has. F0 becomes
    //   min(f0_ior * specularColorFactor, 1) * specularFactor
    // and F90 becomes specularFactor, where f0_ior = ((ior-1)/(ior+1))^2 -- 0.04
    // at the default ior of 1.5, which is the constant that was hardcoded
    // before. The clamp happens before the factor multiply, not after:
    // SpecularSilkPouf authors specularColorFactor [10, 0.6, 0], and clamping
    // the product instead lets the red channel out at more than it should be.
    //
    // Both defaults are the neutral element, so a material without the
    // extension produces exactly the F0 and F90 it produced before.
    //
    // There is no measured-curve path here and there will not be one: F0/F90
    // machinery exists only in the bands that split a BRDF into diffuse and
    // specular. NIR, SWIR, MWIR and LWIR take their reflectance from a curve or
    // the config directly, so there is nothing for specular to modulate.
    f32 specularFactor = 1.0f;                        // [0,1] (glTF default 1)
    glm::vec3 specularColorFactor{1.0f, 1.0f, 1.0f};  // linear, may exceed 1 (HDR)
    i32 specularTextureIndex = -1;                    // ALPHA channel, linear
    i32 specularColorTextureIndex = -1;               // RGB channels, sRGB-encoded

    // ========================================================================
    // KHR_materials_anisotropy
    // ========================================================================
    // Stretches the existing GGX lobe along a tangent direction rather than
    // adding anything: alpha_t = mix(alpha, 1, strength^2) along the direction,
    // alpha_b = alpha across it. Note alpha_t >= alpha_b always -- this
    // extension only ever roughens one axis, it never sharpens the other.
    //
    // The direction is the material tangent rotated by anisotropyRotation, and
    // rotated again by the texture's RG when one is bound. A primitive with no
    // TANGENT attribute gets an arbitrary (though continuous) frame, which
    // makes the highlight orientation arbitrary -- the loader warns when it
    // sees that combination, because nothing downstream can detect it.
    f32 anisotropyStrength = 0.0f;   // [0,1]; 0 = isotropic (glTF default)
    f32 anisotropyRotation = 0.0f;   // radians, CCW from the tangent
    i32 anisotropyTextureIndex = -1; // RG = direction (x2-1), B = strength, linear

    // ========================================================================
    // KHR_materials_clearcoat
    // ========================================================================
    // An infinitely thin dielectric coat over everything else, with its own
    // normal and roughness. The coat's Fresnel is taken at N.V rather than V.H
    // -- deliberate in the specification, for energy conservation with the
    // simple layering operator -- and the base, including emission, is weighted
    // by 1 - clearcoat * F_c to pay for it.
    //
    // clearcoatNormalTextureIndex absent means the coat is NOT normal mapped,
    // even where the base is: the coat then follows the interpolated vertex
    // normal. ClearCoatTest's BaseNorm_Coated is the case that checks it.
    //
    // MWIR and LWIR act on clearcoatReflectanceCurveIndex alone. The 0.04 a
    // dielectric coat reflects in the visible is a fiction at 10 microns, where
    // real lacquers are strongly absorbing, so a factor is not enough to earn a
    // thermal lobe. Where a curve is bound, those bands carve the coat out of
    // the reflectance that is already there rather than adding to it -- see the
    // sheen carve-out in closesthit.rchit for why adding would make an
    // isothermal cavity emit.
    f32 clearcoatFactor = 0.0f;              // [0,1]; 0 = no coat (glTF default)
    f32 clearcoatRoughnessFactor = 0.0f;     // [0,1] (glTF default)
    f32 clearcoatNormalScale = 1.0f;         // normalTextureInfo.scale on the coat
    i32 clearcoatTextureIndex = -1;          // R channel, linear
    i32 clearcoatRoughnessTextureIndex = -1; // G channel, linear
    i32 clearcoatNormalTextureIndex = -1;    // tangent-space normal map
    i32 clearcoatReflectanceCurveIndex = -1; // measured, the only IR path
    String quantiloomClearcoatRef;           // database name, resolved into the index

    // ========================================================================
    // KHR_materials_diffuse_transmission
    // ========================================================================
    // A Lambertian BTDF on a thin surface: light scattered through into the
    // back hemisphere. The specification mixes it against the diffuse BRDF
    // *inside* the Fresnel mix, so the diffuse reflection is scaled by
    // (1 - diffuseTransmission) and the specular lobe and its Fresnel weight
    // are untouched -- energy moves between the two diffuse halves rather than
    // appearing.
    //
    // This is not the same quantity as irTransmittance, and neither is derived
    // from the other. irTransmittance is a thermal-band property that enters
    // Kirchhoff's law as eps = 1 - rho - tau; feeding a visible-band diffuse
    // transmission into it would give a surface two transmittances and move its
    // emissivity. MWIR and LWIR therefore ignore these fields entirely.
    f32 diffuseTransmissionFactor = 0.0f;                        // [0,1] (glTF default 0)
    glm::vec3 diffuseTransmissionColorFactor{1.0f, 1.0f, 1.0f};  // [0,1] (glTF default)
    i32 diffuseTransmissionTextureIndex = -1;                    // ALPHA channel, linear
    i32 diffuseTransmissionColorTextureIndex = -1;               // RGB channels, sRGB
    i32 diffuseTransmissionColorCurveIndex = -1;                 // measured, the NIR/SWIR path
    String quantiloomDiffuseTransmissionRef;  // database name, resolved into the index

    // Per-slot UV transforms for the eight new texture slots, same rules as the
    // six above: identity by default, so an asset without KHR_texture_transform
    // is unaffected. Order must stay parallel to the UV_SLOT_* constants in
    // MaterialGpuData.hpp and common.hlsli.
    UvTransform specularUv;
    UvTransform specularColorUv;
    UvTransform anisotropyUv;
    UvTransform clearcoatUv;
    UvTransform clearcoatRoughnessUv;
    UvTransform clearcoatNormalUv;
    UvTransform diffuseTransmissionUv;
    UvTransform diffuseTransmissionColorUv;

    // ========================================================================
    // Utilities
    // ========================================================================

    // Check if material is valid
    [[nodiscard]] bool IsValid() const {
        // Base color must be in valid range
        if (baseColorFactor.r < 0.0f || baseColorFactor.g < 0.0f ||
            baseColorFactor.b < 0.0f || baseColorFactor.a < 0.0f) {
            return false;
        }

        // Metallic and roughness must be in [0, 1]
        if (metallicFactor < 0.0f || metallicFactor > 1.0f ||
            roughnessFactor < 0.0f || roughnessFactor > 1.0f) {
            return false;
        }

        // Alpha cutoff must be in [0, 1]
        if (alphaCutoff < 0.0f || alphaCutoff > 1.0f) {
            return false;
        }

        return true;
    }

    // Compute spectral albedo from base color (for M1 mode)
    void ComputeSpectralAlbedo() {
        spectralAlbedo = (baseColorFactor.r + baseColorFactor.g + baseColorFactor.b) / 3.0f;
    }

    // Check if material has any textures
    [[nodiscard]] bool HasTextures() const {
        return baseColorTextureIndex != -1 ||
               metallicRoughnessTextureIndex != -1 ||
               normalTextureIndex != -1 ||
               emissiveTextureIndex != -1;
    }

    // Whether any sheen term can be non-zero. A sheen texture with a zero
    // factor still yields nothing -- glTF multiplies the two -- so the factor
    // alone decides, and a bound curve counts because the infrared bands read
    // it instead of the factor.
    [[nodiscard]] bool HasSheen() const {
        return sheenColorFactor.r > 0.0f || sheenColorFactor.g > 0.0f ||
               sheenColorFactor.b > 0.0f || sheenReflectanceCurveIndex >= 0 ||
               !quantiloomSheenRef.empty();
    }

    // Whether specular deviates from the neutral element. Unlike the other
    // three this one is not "greater than zero": glTF defaults specularFactor
    // to 1 and specularColorFactor to white, and those values reproduce exactly
    // the F0 and F90 the renderer used before the extension existed. A texture
    // counts, since it can only scale these down.
    [[nodiscard]] bool HasSpecular() const {
        return specularFactor != 1.0f || specularColorFactor.r != 1.0f ||
               specularColorFactor.g != 1.0f || specularColorFactor.b != 1.0f ||
               specularTextureIndex >= 0 || specularColorTextureIndex >= 0;
    }

    // Whether the GGX lobe is stretched. The factor alone decides: the
    // specification multiplies it by the texture's blue channel, so a texture
    // cannot rescue a zero factor, and a rotation of a lobe that is not
    // stretched is not observable.
    [[nodiscard]] bool HasAnisotropy() const {
        return anisotropyStrength > 0.0f;
    }

    // Whether a clearcoat lobe can be non-zero. The factor alone decides in the
    // reflective bands; a bound curve counts because MWIR and LWIR read it
    // instead of assuming a dielectric 0.04 that does not hold there.
    [[nodiscard]] bool HasClearcoat() const {
        return clearcoatFactor > 0.0f || clearcoatReflectanceCurveIndex >= 0 ||
               !quantiloomClearcoatRef.empty();
    }

    // Whether light is scattered through the surface. The colour factor
    // defaults to white and only tints what the factor lets through, so the
    // factor alone decides; a bound curve counts because NIR and SWIR act on it
    // rather than on a visible-basis colour.
    [[nodiscard]] bool HasDiffuseTransmission() const {
        return diffuseTransmissionFactor > 0.0f || diffuseTransmissionColorCurveIndex >= 0 ||
               !quantiloomDiffuseTransmissionRef.empty();
    }

    // ========================================================================
    // Infrared Material Property Helpers
    // ========================================================================

    // Get IR emissivity at specific wavelength (linear interpolation)
    // Returns 0.0 if curve is empty or wavelength out of range
    [[nodiscard]] f32 GetIREmissivity(f32 lambda_nm) const;

    // Get IR reflectance at specific wavelength (linear interpolation)
    // Returns a neutral dielectric default if curve is empty (spectralAlbedo is a visible-RGB average
    // and is not physically valid in MWIR/LWIR).
    [[nodiscard]] f32 GetIRReflectance(f32 lambda_nm) const;

    // Get IR transmittance at specific wavelength (linear interpolation)
    // Returns 0.0 if curve is empty (opaque)
    [[nodiscard]] f32 GetIRTransmittance(f32 lambda_nm) const;

    // Validate Kirchhoff's law: ε + ρ + τ ≤ 1 at all wavelengths
    // Returns true if valid, false if energy conservation violated
    [[nodiscard]] bool ValidateIRKirchhoffLaw() const;

    // Check if material has IR data
    [[nodiscard]] bool HasIRData() const {
        return !irEmissivityCurve.empty() ||
               !irReflectanceCurve.empty() ||
               !irTransmittanceCurve.empty();
    }

    // Create simple Lambertian material (for procedural geometry)
    static Material CreateLambertian(const glm::vec3& albedo, const String& name = "Lambertian") {
        Material mat;
        mat.name = name;
        mat.baseColorFactor = glm::vec4(albedo, 1.0f);
        mat.metallicFactor = 0.0f;   // Non-metal
        mat.roughnessFactor = 1.0f;  // Fully rough (Lambertian limit)
        mat.ComputeSpectralAlbedo();
        mat.spectralSource = SpectralSource::RGBUpsampled;  // Mark as upsampled
        return mat;
    }
};

// ============================================================================
// Inline IR Property Helpers Implementation
// ============================================================================

namespace detail {
    // Helper: Linear interpolation for spectral curves
    inline f32 InterpolateSpectralCurve(
        const Vector<std::pair<f32, f32>>& curve,
        f32 lambda_nm,
        f32 fallback = 0.0f)
    {
        if (curve.empty()) {
            return fallback;
        }

        // Clamp to boundaries
        if (lambda_nm <= curve.front().first) {
            return curve.front().second;
        }
        if (lambda_nm >= curve.back().first) {
            return curve.back().second;
        }

        // Binary search for surrounding wavelengths
        usize left = 0;
        usize right = curve.size() - 1;

        while (right - left > 1) {
            if (usize mid = (left + right) / 2; curve[mid].first < lambda_nm) {
                left = mid;
            } else {
                right = mid;
            }
        }

        // Linear interpolation
        f32 lambda0 = curve[left].first;
        f32 lambda1 = curve[right].first;
        f32 value0 = curve[left].second;
        f32 value1 = curve[right].second;

        f32 t = (lambda_nm - lambda0) / (lambda1 - lambda0);
        return value0 * (1.0f - t) + value1 * t;
    }
} // namespace detail

inline f32 Material::GetIREmissivity(f32 lambda_nm) const {
    return detail::InterpolateSpectralCurve(irEmissivityCurve, lambda_nm, 0.0f);
}

inline f32 Material::GetIRReflectance(f32 lambda_nm) const {
    // Fallback to a neutral dielectric reflectance when no IR curve is available.
    // spectralAlbedo is a visible-RGB average and is not physically valid in MWIR/LWIR.
    const f32 DEFAULT_IR_REFLECTANCE = 0.1f;
    return detail::InterpolateSpectralCurve(irReflectanceCurve, lambda_nm, DEFAULT_IR_REFLECTANCE);
}

inline f32 Material::GetIRTransmittance(f32 lambda_nm) const {
    return detail::InterpolateSpectralCurve(irTransmittanceCurve, lambda_nm, 0.0f);
}

// ============================================================================
// Default IR Temperature Injection
// ============================================================================
// Standard glTF/USD assets carry no surface temperature, leaving
// irTemperature_K at 0 and silencing thermal emission in MWIR/LWIR.
// Backfills a scene-wide ambient temperature for materials that have no
// temperature source of their own (neither an explicit scalar nor a
// per-pixel temperature texture). Returns the number of materials modified.
// ============================================================================

inline u32 ApplyDefaultIRTemperature(Vector<Material>& materials, f32 defaultTemperature_K) {
    if (defaultTemperature_K <= 0.0f) {
        return 0;
    }

    u32 modified = 0;
    for (auto& mat : materials) {
        if (mat.irTemperature_K <= 0.0f && mat.temperatureTextureIndex < 0) {
            mat.irTemperature_K = defaultTemperature_K;
            ++modified;
        }
    }
    return modified;
}

} // namespace quantiloom
