// ============================================================================
// Quantiloom - scene structs shared by the ray tracing stages and the host
// ============================================================================
// Emissive triangles, materials and per-instance geometry offsets
// (common.hlsli; renderer/RenderCore.hpp and renderer/MaterialGpuData.hpp) and
// the NN atmosphere header (atmosphere_nn.hlsli; atmos/AtmosNNGpuTypes.hpp).
// Each is defined here once, for both languages. ql_shared_prelude.hlsli has
// the layout rules and how a shader asks for one struct (QL_SHARED_WANT_<Name>);
// the public structs the same shaders read are locked in ql_shared_public.hlsli.
//
// No include guard: a shader includes this once per struct it asks for, and
// every section guards itself (QL_SHARED_HAS_<Name>).
// ============================================================================

#include "ql_shared_prelude.hlsli"

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_EmissiveTriangleGPU)) && !defined(QL_SHARED_HAS_EmissiveTriangleGPU)
#define QL_SHARED_HAS_EmissiveTriangleGPU
QL_SHARED_BEGIN
// One world-space emissive triangle for next-event estimation; the host
// calls it rendercore::EmissiveTriangleGPU too. Its alias in
// renderer/RenderCore.hpp says why the positions are world space and what
// cumulativePower is for.
struct EmissiveTriangleGPU {
    float3 v0;        float cumulativePower;
    float3 edge1;     float area;
    float3 edge2;     int   emissiveCurveIndex;  // -1 = the RGB triple below
    float3 emissive;  float _pad1;
};
QL_SHARED_LAYOUT(EmissiveTriangleGPU, 64)
QL_SHARED_OFFSET(EmissiveTriangleGPU, v0, 0)
QL_SHARED_OFFSET(EmissiveTriangleGPU, cumulativePower, 12)
QL_SHARED_OFFSET(EmissiveTriangleGPU, edge1, 16)
QL_SHARED_OFFSET(EmissiveTriangleGPU, area, 28)
QL_SHARED_OFFSET(EmissiveTriangleGPU, edge2, 32)
QL_SHARED_OFFSET(EmissiveTriangleGPU, emissiveCurveIndex, 44)
QL_SHARED_OFFSET(EmissiveTriangleGPU, emissive, 48)
QL_SHARED_OFFSET(EmissiveTriangleGPU, _pad1, 60)
QL_SHARED_END
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_MaterialData)) && !defined(QL_SHARED_HAS_MaterialData)
#define QL_SHARED_HAS_MaterialData
QL_SHARED_BEGIN
// Full glTF 2.0 metallic-roughness PBR material, one per Scene::materials entry;
// the host calls it MaterialDataCPU. The UV_SLOT_* indices of uvTransformMat and
// uvTransformOffset are common.hlsli's and renderer/MaterialGpuData.hpp's.
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
    // NOT READ by any shader: the path tracer shades both sides by design.
    // A closest-hit shader cannot un-see an intersection traversal already
    // committed, and the any-hit shader reserves IgnoreHit for alpha coverage,
    // so single-sided culling has no place to live on this pipeline -- it
    // would need per-material any-hit programs or geometry/instance flags.
    // glTF's flag is loaded and uploaded for CPU-side bookkeeping and to keep
    // the layout in step with the host Material struct; both sides shade
    // regardless of its value.
    uint   doubleSided;              // loaded, not read; both sides shade // Offset: 40-44
    float  fluorescenceYield;        // Fraction of absorbed light re-emitted, 0 = does not fluoresce // Offset: 44-48

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
    float  attenuationDistance;          // Distance for attenuation (m, 0 = no attenuation)   // Offset: 140-144

    // NOT READ by the shaders. KHR_materials_volume's thicknessFactor
    // approximates the volume path for thin-walled assets that carry no
    // volume geometry; this renderer traces real geometry and Beer-Lamberts
    // over the actual path length through the medium (RayTCurrent, converted
    // to metres by worldUnitsToMeters), which is the quantity thickness
    // approximates. Overwriting the measured path with an authored thickness
    // would discard information, so the fields are loaded and validated but
    // intentionally unused. thicknessTexture would be the G channel per the
    // extension, if a consumer ever wants it.
    float  thicknessFactor;              // Thickness for thin-walled approximation                 // Offset: 144-148
    int    thicknessTextureIndex;        // Thickness texture (-1 = no texture)                     // Offset: 148-152
    float  dispersion;                   // Abbe number reciprocal (0 = no dispersion)              // Offset: 152-156
    int    sheenReflectanceCurveIndex;   // Index into spectralCurves (-1 = use sheenColorFactor)   // Offset: 156-160

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

    // ========================================================================
    // Endmember Mixing (spatially varying spectral reflectance)
    // ========================================================================
    // rho(lambda, uv) = sum_i w_i(uv) * rho_i(lambda)
    //
    // Endmember 0 is spectralReflectanceCurveIndex above; 1..3 are here.
    // The weights come from weightTextureIndex, RGBA channel i holding w_i/2
    // -- halved because a single-endmember mixture is brightness modulation
    // and needs w > 1 for texels brighter than the curve's own colour, which
    // a UNORM texture cannot store otherwise.
    //
    // weightTextureIndex < 0 means w = (1, 0, 0, 0): the first curve, flat,
    // exactly as before this existed.

    int    endmemberCurveIndex1;         // Index into spectralCurves (-1 = unused)                 // Offset: 176-180
    int    endmemberCurveIndex2;         // Index into spectralCurves (-1 = unused)                 // Offset: 180-184
    int    endmemberCurveIndex3;         // Index into spectralCurves (-1 = unused)                 // Offset: 184-188
    int    weightTextureIndex;           // Index into texture array (-1 = single endmember)        // Offset: 188-192

    // ========================================================================
    // Sheen (KHR_materials_sheen)
    // ========================================================================
    // The retroreflective lobe of a microfibre surface -- velvet, felt, cloth.
    // Fibres standing away from the surface scatter light into a lobe that
    // peaks at grazing angles instead of around the mirror direction, which no
    // roughness setting on a GGX lobe reproduces.
    //
    // sheenReflectanceCurveIndex is up at offset 156, in what was padding.
    // Priority, matching how base colour resolves its reflectance:
    //   curve bound  -> EvaluateSpectralCurve, quantitative, every band
    //   otherwise    -> sheenColorFactor upsampled, visible bands only
    // The infrared bands read the curve alone: an RGB factor pushed through the
    // visible Gaussian basis says nothing past ~1400nm.

    float3 sheenColorFactor;             // RGB [0,1], 0 = no sheen (glTF default)                 // Offset: 192-204
    float  sheenRoughnessFactor;         // [0,1] (glTF default 0)                                  // Offset: 204-208
    int    sheenColorTextureIndex;       // RGB channels, sRGB-encoded (-1 = no texture)           // Offset: 208-212
    int    sheenRoughnessTextureIndex;   // ALPHA channel, linear (-1 = no texture)                // Offset: 212-216

    // ========================================================================
    // Anisotropy (KHR_materials_anisotropy)
    // ========================================================================
    // Stretches the existing GGX lobe along a tangent direction; adds nothing.
    //   alpha_t = lerp(alpha, 1, strength^2)   along the direction
    //   alpha_b = alpha                        across it
    // so alpha_t >= alpha_b always: this only ever roughens one axis.
    //
    // The direction is the material tangent rotated by anisotropyRotation, and
    // rotated again by the texture's RG where one is bound. These two took the
    // padding that used to sit here, which is why the texture index is further
    // down at 248 -- every offset before 224 had to stay where it was.

    float  anisotropyStrength;           // [0,1], 0 = isotropic (glTF default)                     // Offset: 216-220
    float  anisotropyRotation;           // radians, CCW from the tangent                           // Offset: 220-224

    // ========================================================================
    // Specular (KHR_materials_specular)
    // ========================================================================
    // Reshapes the dielectric Fresnel rather than adding a lobe:
    //   F0  = min(f0_ior * specularColorFactor, 1) * specularFactor
    //   F90 = specularFactor
    // with f0_ior = ((ior-1)/(ior+1))^2, which is 0.04 at the default ior of
    // 1.5 -- the constant that used to be hardcoded. The clamp is applied
    // before the factor multiply, not after: SpecularSilkPouf authors
    // specularColorFactor [10, 0.6, 0] and the order is visible there.
    //
    // Both defaults are the neutral element, so a material without the
    // extension yields exactly the F0 and F90 it did before. There is no curve
    // index because F0/F90 exist only in the bands that split a BRDF into
    // diffuse and specular; the infrared bands take reflectance directly.

    float3 specularColorFactor;          // linear, may exceed 1 (HDR) (glTF default [1,1,1])       // Offset: 224-236
    float  specularFactor;               // [0,1] (glTF default 1)                                  // Offset: 236-240
    int    specularTextureIndex;         // ALPHA channel, linear (-1 = no texture)                // Offset: 240-244
    int    specularColorTextureIndex;    // RGB channels, sRGB-encoded (-1 = no texture)           // Offset: 244-248
    int    anisotropyTextureIndex;       // RG = direction (x2-1), B = strength, linear            // Offset: 248-252

    // ========================================================================
    // Clearcoat (KHR_materials_clearcoat)
    // ========================================================================
    // An infinitely thin dielectric coat with its own normal and roughness.
    // Its Fresnel is taken at N.V, not V.H -- deliberate in the specification,
    // for energy conservation with the simple layering operator -- and the base
    // including emission is weighted by 1 - clearcoat * F_c to pay for it.
    //
    // clearcoatNormalTextureIndex < 0 means the coat is NOT normal mapped even
    // where the base is; it then follows the interpolated vertex normal.
    //
    // MWIR and LWIR act on clearcoatReflectanceCurveIndex alone: the 0.04 a
    // dielectric coat reflects in the visible is a fiction at 10 microns. Where
    // a curve is bound, those bands carve the coat out of the reflectance
    // already there rather than adding to it -- adding would make an isothermal
    // cavity emit.

    float  clearcoatFactor;              // [0,1], 0 = no coat (glTF default)                       // Offset: 252-256
    float  clearcoatRoughnessFactor;     // [0,1] (glTF default 0)                                  // Offset: 256-260
    float  clearcoatNormalScale;         // normalTextureInfo.scale on the coat                     // Offset: 260-264
    int    clearcoatTextureIndex;        // R channel, linear (-1 = no texture)                     // Offset: 264-268
    int    clearcoatRoughnessTextureIndex; // G channel, linear (-1 = no texture)                   // Offset: 268-272
    int    clearcoatNormalTextureIndex;  // tangent-space normal map (-1 = coat is flat)            // Offset: 272-276
    int    clearcoatReflectanceCurveIndex; // measured, the only infrared path (-1 = none)          // Offset: 276-280

    // ========================================================================
    // Diffuse transmission (KHR_materials_diffuse_transmission)
    // ========================================================================
    // A Lambertian BTDF on a thin surface. The specification mixes it against
    // the diffuse BRDF inside the Fresnel mix, so the diffuse reflection is
    // scaled by (1 - diffuseTransmission) and the specular lobe and its Fresnel
    // weight are untouched -- energy moves between the two diffuse halves
    // rather than appearing.
    //
    // Not the same quantity as irTransmittance, and not derived from it.
    // irTransmittance enters Kirchhoff's law as eps = 1 - rho - tau; feeding a
    // visible-band diffuse transmission into it would give a surface two
    // transmittances and move its emissivity. MWIR and LWIR ignore these.

    float  diffuseTransmissionFactor;    // [0,1] (glTF default 0)                                  // Offset: 280-284
    int    diffuseTransmissionTextureIndex; // ALPHA channel, linear (-1 = no texture)              // Offset: 284-288
    float3 diffuseTransmissionColorFactor;  // [0,1] linear (glTF default [1,1,1])                  // Offset: 288-300
    int    diffuseTransmissionColorTextureIndex; // RGB, sRGB-encoded (-1 = no texture)             // Offset: 300-304
    int    diffuseTransmissionColorCurveIndex;   // measured, the NIR/SWIR path (-1 = none)         // Offset: 304-308

    // Spectral self-emission: index into spectralCurves, -1 = expand
    // emissiveFactor through the RGB->illuminant path instead. Zero outside the
    // curve's own span, never held flat -- see EvaluateEmissionCurve.
    int    emissiveRadianceCurveIndex;   // Index into spectralCurves (-1 = RGB expansion)          // Offset: 308-312
    int    fluorescenceExcitationCurveIndex; // -1 = none; the shape this surface absorbs into the fluorescent channel // Offset: 312-316
    int    fluorescenceEmissionCurveIndex;   // -1 = none; unit area over the band, so a density per nm // Offset: 316-320

    // ========================================================================
    // Per-slot UV transforms (KHR_texture_transform)
    // ========================================================================
    // Pre-multiplied by ConvertMaterial into a 2x3 affine, applied by
    // TransformUV in hit_common.hlsli:
    //
    //   uv' = float2(m.x * uv.x + m.y * uv.y, m.z * uv.x + m.w * uv.y) + offset
    //
    // Indexed by the UV_SLOT_* constants below. Per slot rather than per
    // material because SheenChair's fabric puts its base colour at scale 7 and
    // its normal map at scale 2 within one material.
    //
    // Two arrays rather than one array of a {float4, float2} pair: that pair is
    // 24 bytes, so its float4 would lose 16-byte alignment every other element
    // and HLSL would pad it out of step with the CPU. That alignment is also
    // what the three padding words above are for -- they carry the array start
    // from 308 up to 320.

    float4 uvTransformMat[14];           // (m00, m01, m10, m11) per slot                          // Offset: 320-544
    float2 uvTransformOffset[14];        // translation per slot                                    // Offset: 544-656

    // Note: irReflectance removed - can be computed as: 1.0 - irEmissivity - irTransmittance
    // For opaque materials: irTransmittance = 0, so irReflectance = 1.0 - irEmissivity
};
QL_SHARED_LAYOUT(MaterialData, 656)
QL_SHARED_OFFSET(MaterialData, baseColorFactor, 0)
QL_SHARED_OFFSET(MaterialData, baseColorTextureIndex, 16)
QL_SHARED_OFFSET(MaterialData, metallicFactor, 20)
QL_SHARED_OFFSET(MaterialData, roughnessFactor, 24)
QL_SHARED_OFFSET(MaterialData, metallicRoughnessTextureIndex, 28)
QL_SHARED_OFFSET(MaterialData, normalTextureIndex, 32)
QL_SHARED_OFFSET(MaterialData, normalScale, 36)
QL_SHARED_OFFSET(MaterialData, doubleSided, 40)
QL_SHARED_OFFSET(MaterialData, fluorescenceYield, 44)
QL_SHARED_OFFSET(MaterialData, emissiveFactor, 48)
QL_SHARED_OFFSET(MaterialData, emissiveTextureIndex, 60)
QL_SHARED_OFFSET(MaterialData, alphaMode, 64)
QL_SHARED_OFFSET(MaterialData, alphaCutoff, 68)
QL_SHARED_OFFSET(MaterialData, spectralAlbedo, 72)
QL_SHARED_OFFSET(MaterialData, spectralReflectanceCurveIndex, 76)
QL_SHARED_OFFSET(MaterialData, irEmissivity, 80)
QL_SHARED_OFFSET(MaterialData, irTransmittance, 84)
QL_SHARED_OFFSET(MaterialData, irTemperature_K, 88)
QL_SHARED_OFFSET(MaterialData, complexRefractiveIndexIndex, 92)
QL_SHARED_OFFSET(MaterialData, temperatureTextureIndex, 96)
QL_SHARED_OFFSET(MaterialData, temperatureScale, 100)
QL_SHARED_OFFSET(MaterialData, temperatureOffset, 104)
QL_SHARED_OFFSET(MaterialData, irEmissivityCurveIndex, 108)
QL_SHARED_OFFSET(MaterialData, ior, 112)
QL_SHARED_OFFSET(MaterialData, transmission, 116)
QL_SHARED_OFFSET(MaterialData, transmissionTextureIndex, 120)
QL_SHARED_OFFSET(MaterialData, irTransmittanceCurveIndex, 124)
QL_SHARED_OFFSET(MaterialData, attenuationColor, 128)
QL_SHARED_OFFSET(MaterialData, attenuationDistance, 140)
QL_SHARED_OFFSET(MaterialData, thicknessFactor, 144)
QL_SHARED_OFFSET(MaterialData, thicknessTextureIndex, 148)
QL_SHARED_OFFSET(MaterialData, dispersion, 152)
QL_SHARED_OFFSET(MaterialData, sheenReflectanceCurveIndex, 156)
QL_SHARED_OFFSET(MaterialData, volumeDensity, 160)
QL_SHARED_OFFSET(MaterialData, scatteringCoeff, 164)
QL_SHARED_OFFSET(MaterialData, absorptionCoeff, 168)
QL_SHARED_OFFSET(MaterialData, phaseG, 172)
QL_SHARED_OFFSET(MaterialData, endmemberCurveIndex1, 176)
QL_SHARED_OFFSET(MaterialData, endmemberCurveIndex2, 180)
QL_SHARED_OFFSET(MaterialData, endmemberCurveIndex3, 184)
QL_SHARED_OFFSET(MaterialData, weightTextureIndex, 188)
QL_SHARED_OFFSET(MaterialData, sheenColorFactor, 192)
QL_SHARED_OFFSET(MaterialData, sheenRoughnessFactor, 204)
QL_SHARED_OFFSET(MaterialData, sheenColorTextureIndex, 208)
QL_SHARED_OFFSET(MaterialData, sheenRoughnessTextureIndex, 212)
QL_SHARED_OFFSET(MaterialData, anisotropyStrength, 216)
QL_SHARED_OFFSET(MaterialData, anisotropyRotation, 220)
QL_SHARED_OFFSET(MaterialData, specularColorFactor, 224)
QL_SHARED_OFFSET(MaterialData, specularFactor, 236)
QL_SHARED_OFFSET(MaterialData, specularTextureIndex, 240)
QL_SHARED_OFFSET(MaterialData, specularColorTextureIndex, 244)
QL_SHARED_OFFSET(MaterialData, anisotropyTextureIndex, 248)
QL_SHARED_OFFSET(MaterialData, clearcoatFactor, 252)
QL_SHARED_OFFSET(MaterialData, clearcoatRoughnessFactor, 256)
QL_SHARED_OFFSET(MaterialData, clearcoatNormalScale, 260)
QL_SHARED_OFFSET(MaterialData, clearcoatTextureIndex, 264)
QL_SHARED_OFFSET(MaterialData, clearcoatRoughnessTextureIndex, 268)
QL_SHARED_OFFSET(MaterialData, clearcoatNormalTextureIndex, 272)
QL_SHARED_OFFSET(MaterialData, clearcoatReflectanceCurveIndex, 276)
QL_SHARED_OFFSET(MaterialData, diffuseTransmissionFactor, 280)
QL_SHARED_OFFSET(MaterialData, diffuseTransmissionTextureIndex, 284)
QL_SHARED_OFFSET(MaterialData, diffuseTransmissionColorFactor, 288)
QL_SHARED_OFFSET(MaterialData, diffuseTransmissionColorTextureIndex, 300)
QL_SHARED_OFFSET(MaterialData, diffuseTransmissionColorCurveIndex, 304)
QL_SHARED_OFFSET(MaterialData, emissiveRadianceCurveIndex, 308)
QL_SHARED_OFFSET(MaterialData, fluorescenceExcitationCurveIndex, 312)
QL_SHARED_OFFSET(MaterialData, fluorescenceEmissionCurveIndex, 316)
QL_SHARED_OFFSET(MaterialData, uvTransformMat, 320)
QL_SHARED_OFFSET(MaterialData, uvTransformOffset, 544)
QL_SHARED_END
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_InstanceGeometryInfo)) && !defined(QL_SHARED_HAS_InstanceGeometryInfo)
#define QL_SHARED_HAS_InstanceGeometryInfo
QL_SHARED_BEGIN
// One per TLAS instance, indexed by InstanceIndex(): where that instance's
// slice of each merged global buffer starts.
struct InstanceGeometryInfo {
    uint vertexOffset;    // Offset into global vertex buffer (vertex count)
    uint indexOffset;     // Offset into global index buffer (index count)
    uint normalOffset;    // Offset into global normal buffer (normal count)
    uint uvOffset;        // Offset into global UV buffer (UV count)
    uint tangentOffset;   // Offset into global tangent buffer (tangent count)
    uint materialId;      // Material index (used instead of InstanceID for material lookup)
    // First thermal element of this instance, or 0xFFFFFFFF where the solver
    // did not run. Add PrimitiveIndex() to it and the triangle has the
    // temperature the surface energy balance gave it.
    uint thermalElementBase;
    uint pad;             // Padding for 32-byte alignment
};
QL_SHARED_LAYOUT(InstanceGeometryInfo, 32)
QL_SHARED_OFFSET(InstanceGeometryInfo, vertexOffset, 0)
QL_SHARED_OFFSET(InstanceGeometryInfo, indexOffset, 4)
QL_SHARED_OFFSET(InstanceGeometryInfo, normalOffset, 8)
QL_SHARED_OFFSET(InstanceGeometryInfo, uvOffset, 12)
QL_SHARED_OFFSET(InstanceGeometryInfo, tangentOffset, 16)
QL_SHARED_OFFSET(InstanceGeometryInfo, materialId, 20)
QL_SHARED_OFFSET(InstanceGeometryInfo, thermalElementBase, 24)
QL_SHARED_OFFSET(InstanceGeometryInfo, pad, 28)
QL_SHARED_END
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_AtmosNNHeader)) && !defined(QL_SHARED_HAS_AtmosNNHeader)
#define QL_SHARED_HAS_AtmosNNHeader
QL_SHARED_BEGIN
// The NN atmosphere header at binding 17; the bulk LUT data is the flat float
// buffer at binding 20 (layout in atmosphere_nn.hlsli). The host calls it
// AtmosNNHeaderGPU, and its defaults are the disabled header.
struct AtmosNNHeader {
    uint  enabled QL_DEFAULT(0);       // 0 = no NN atmosphere (passthrough)
    uint  pathMode QL_DEFAULT(0);      // 0 = ground (range axis), 1 = slant (cos zenith axis)
    uint  numLambda QL_DEFAULT(0);     // spectral samples (= render loop sample count)
    uint  countA QL_DEFAULT(0);        // samples along axis a
    uint  countAz QL_DEFAULT(1);       // samples along the relative-azimuth axis (lpath)
    uint  hasLdown QL_DEFAULT(0);
    uint  hasSky QL_DEFAULT(0);        // reserved: sky dome network not trained yet
    uint  thermalBand QL_DEFAULT(0);   // 1 for mwir/lwir
    float aStart QL_DEFAULT(0.0f);     // first a sample
    float aStep QL_DEFAULT(0.0f);      // uniform step in a
    float azStart QL_DEFAULT(0.0f);    // degrees
    float azStep QL_DEFAULT(0.0f);     // degrees
    uint  tauOffset QL_DEFAULT(0);     // float offsets into the binding-20 data blob
    uint  lpathOffset QL_DEFAULT(0);
    uint  ldownOffset QL_DEFAULT(0);
    uint  skyOffset QL_DEFAULT(0);
    float3 sunDirWorld QL_DEFAULT(0.0f, 1.0f, 0.0f);  // unit vector toward the sun
    float worldUnitsToMeters QL_DEFAULT(1.0f);
};
QL_SHARED_LAYOUT(AtmosNNHeader, 80)
QL_SHARED_OFFSET(AtmosNNHeader, enabled, 0)
QL_SHARED_OFFSET(AtmosNNHeader, pathMode, 4)
QL_SHARED_OFFSET(AtmosNNHeader, numLambda, 8)
QL_SHARED_OFFSET(AtmosNNHeader, countA, 12)
QL_SHARED_OFFSET(AtmosNNHeader, countAz, 16)
QL_SHARED_OFFSET(AtmosNNHeader, hasLdown, 20)
QL_SHARED_OFFSET(AtmosNNHeader, hasSky, 24)
QL_SHARED_OFFSET(AtmosNNHeader, thermalBand, 28)
QL_SHARED_OFFSET(AtmosNNHeader, aStart, 32)
QL_SHARED_OFFSET(AtmosNNHeader, aStep, 36)
QL_SHARED_OFFSET(AtmosNNHeader, azStart, 40)
QL_SHARED_OFFSET(AtmosNNHeader, azStep, 44)
QL_SHARED_OFFSET(AtmosNNHeader, tauOffset, 48)
QL_SHARED_OFFSET(AtmosNNHeader, lpathOffset, 52)
QL_SHARED_OFFSET(AtmosNNHeader, ldownOffset, 56)
QL_SHARED_OFFSET(AtmosNNHeader, skyOffset, 60)
QL_SHARED_OFFSET(AtmosNNHeader, sunDirWorld, 64)
QL_SHARED_OFFSET(AtmosNNHeader, worldUnitsToMeters, 76)
QL_SHARED_END
#endif
