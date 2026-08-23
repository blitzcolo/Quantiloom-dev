#pragma once
// MaterialDataCPU — single definition for the CPU-side mirror of the GPU
// MaterialData struct (common.hlsli).  Both main.cpp and
// ExternalRenderContext.cpp include this header; no local copies.

#include "core/Types.hpp"
#include <glm/glm.hpp>
#include <cstddef>

namespace quantiloom {

struct MaterialDataCPU {
    glm::vec4 baseColorFactor;               // offset   0, 16 bytes
    i32 baseColorTextureIndex;               // offset  16,  4
    f32 metallicFactor;                      // offset  20,  4
    f32 roughnessFactor;                     // offset  24,  4
    i32 metallicRoughnessTextureIndex;       // offset  28,  4

    i32 normalTextureIndex;                  // offset  32,  4
    f32 normalScale;                         // offset  36,  4
    u32 doubleSided;                         // offset  40,  4
    f32 _padding0;                           // offset  44,  4

    glm::vec3 emissiveFactor;                // offset  48, 12
    i32 emissiveTextureIndex;                // offset  60,  4

    u32 alphaMode;                           // offset  64,  4
    f32 alphaCutoff;                         // offset  68,  4
    f32 spectralAlbedo;                      // offset  72,  4
    i32 spectralReflectanceCurveIndex;       // offset  76,  4

    f32 irEmissivity;                        // offset  80,  4
    f32 irTransmittance;                     // offset  84,  4
    f32 irTemperature_K;                     // offset  88,  4
    i32 complexRefractiveIndexIndex;         // offset  92,  4

    i32 temperatureTextureIndex;             // offset  96,  4
    f32 temperatureScale;                    // offset 100,  4
    f32 temperatureOffset;                   // offset 104,  4
    i32 irEmissivityCurveIndex;              // offset 108,  4  (was _padding3)

    f32 ior;                                 // offset 112,  4
    f32 transmission;                        // offset 116,  4
    i32 transmissionTextureIndex;            // offset 120,  4
    i32 irTransmittanceCurveIndex;           // offset 124,  4  (was _padding1)

    glm::vec3 attenuationColor;              // offset 128, 12
    f32 attenuationDistance;                 // offset 140,  4

    f32 thicknessFactor;                     // offset 144,  4
    i32 thicknessTextureIndex;               // offset 148,  4
    f32 dispersion;                          // offset 152,  4
    i32 sheenReflectanceCurveIndex;          // offset 156,  4  (was _padding2)

    f32 volumeDensity;                       // offset 160,  4
    f32 scatteringCoeff;                     // offset 164,  4
    f32 absorptionCoeff;                     // offset 168,  4
    f32 phaseG;                              // offset 172,  4

    // Endmember mixing. Endmember 0 is spectralReflectanceCurveIndex above;
    // these are 1..3, and -1 for a weight texture means w = (1, 0, 0, 0),
    // which is the single flat curve this feature generalises.
    i32 endmemberCurveIndex1;                // offset 176,  4
    i32 endmemberCurveIndex2;                // offset 180,  4
    i32 endmemberCurveIndex3;                // offset 184,  4
    i32 weightTextureIndex;                  // offset 188,  4

    // Sheen (KHR_materials_sheen). sheenReflectanceCurveIndex is up at 156,
    // in what used to be padding.
    glm::vec3 sheenColorFactor;              // offset 192, 12
    f32 sheenRoughnessFactor;                // offset 204,  4
    i32 sheenColorTextureIndex;              // offset 208,  4
    i32 sheenRoughnessTextureIndex;          // offset 212,  4

    // Anisotropy (KHR_materials_anisotropy). These two took the padding that
    // used to sit here, which is why they are separated from the texture index
    // down at 248 -- everything before offset 224 keeps the offset it had.
    f32 anisotropyStrength;                  // offset 216,  4  (was _padding2)
    f32 anisotropyRotation;                  // offset 220,  4  (was _padding3)

    // Specular (KHR_materials_specular). Reshapes the dielectric F0/F90; adds
    // no lobe. Both factors default to the neutral element.
    glm::vec3 specularColorFactor;           // offset 224, 12
    f32 specularFactor;                      // offset 236,  4
    i32 specularTextureIndex;                // offset 240,  4
    i32 specularColorTextureIndex;           // offset 244,  4
    i32 anisotropyTextureIndex;              // offset 248,  4

    // Clearcoat (KHR_materials_clearcoat).
    f32 clearcoatFactor;                     // offset 252,  4
    f32 clearcoatRoughnessFactor;            // offset 256,  4
    f32 clearcoatNormalScale;                // offset 260,  4
    i32 clearcoatTextureIndex;               // offset 264,  4
    i32 clearcoatRoughnessTextureIndex;      // offset 268,  4
    i32 clearcoatNormalTextureIndex;         // offset 272,  4
    i32 clearcoatReflectanceCurveIndex;      // offset 276,  4

    // Diffuse transmission (KHR_materials_diffuse_transmission).
    f32 diffuseTransmissionFactor;           // offset 280,  4
    i32 diffuseTransmissionTextureIndex;     // offset 284,  4
    glm::vec3 diffuseTransmissionColorFactor;// offset 288, 12
    i32 diffuseTransmissionColorTextureIndex;// offset 300,  4
    i32 diffuseTransmissionColorCurveIndex;  // offset 304,  4

    // Spectral self-emission. Took _padding2, so sizeof and every offset above
    // are unchanged -- see Material::emissiveRadianceCurveIndex for what it
    // means and why emissiveFactor is rewritten to match when it is set.
    i32 emissiveRadianceCurveIndex;          // offset 308,  4
    f32 _padding3;                           // offset 312,  4
    f32 _padding4;                           // offset 316,  4

    // Per-slot UV transforms (KHR_texture_transform), pre-multiplied by
    // ConvertMaterial into a 2x3 affine:  uv' = M * uv + offset,  with M packed
    // as (m00, m01, m10, m11).  Indexed by the UV_SLOT_* constants below, which
    // common.hlsli mirrors.
    //
    // The matrices and the offsets are two arrays rather than one array of a
    // {float4, float2} pair because that pair is 24 bytes, and a float4 at an
    // odd multiple of 8 would not be 16-byte aligned -- HLSL would pad the
    // element and the two layouts would silently disagree. Which is also why
    // the three words of padding above are there: they carry the array start
    // from 308 up to 320.
    glm::vec4 uvTransformMat[14];            // offset 320, 224
    glm::vec2 uvTransformOffset[14];         // offset 544, 112
};  // 656 bytes total

// UV transform slots. The weight texture deliberately has no slot of its own:
// it is unmixed from the base-colour texture's texels, so it must be sampled
// with the base colour's transform or the mixture reads the wrong texels.
// The temperature texture has none either -- it is a Quantiloom-authored slot
// with no glTF textureInfo to carry the extension, so it is always identity.
enum : i32 {
    UV_SLOT_BASE_COLOR = 0,
    UV_SLOT_METALLIC_ROUGHNESS = 1,
    UV_SLOT_NORMAL = 2,
    UV_SLOT_EMISSIVE = 3,
    UV_SLOT_SHEEN_COLOR = 4,
    UV_SLOT_SHEEN_ROUGHNESS = 5,
    UV_SLOT_SPECULAR = 6,
    UV_SLOT_SPECULAR_COLOR = 7,
    UV_SLOT_ANISOTROPY = 8,
    UV_SLOT_CLEARCOAT = 9,
    UV_SLOT_CLEARCOAT_ROUGHNESS = 10,
    UV_SLOT_CLEARCOAT_NORMAL = 11,
    UV_SLOT_DIFFUSE_TRANSMISSION = 12,
    UV_SLOT_DIFFUSE_TRANSMISSION_COLOR = 13,
    UV_SLOT_COUNT = 14,
};

static_assert(sizeof(MaterialDataCPU) == 656);
static_assert(offsetof(MaterialDataCPU, baseColorTextureIndex)       ==  16);
static_assert(offsetof(MaterialDataCPU, normalTextureIndex)          ==  32);
static_assert(offsetof(MaterialDataCPU, doubleSided)                 ==  40);
static_assert(offsetof(MaterialDataCPU, emissiveFactor)              ==  48);
static_assert(offsetof(MaterialDataCPU, emissiveTextureIndex)        ==  60);
static_assert(offsetof(MaterialDataCPU, alphaMode)                   ==  64);
static_assert(offsetof(MaterialDataCPU, spectralAlbedo)              ==  72);
static_assert(offsetof(MaterialDataCPU, spectralReflectanceCurveIndex) == 76);
static_assert(offsetof(MaterialDataCPU, irEmissivity)                ==  80);
static_assert(offsetof(MaterialDataCPU, irTransmittance)             ==  84);
static_assert(offsetof(MaterialDataCPU, irTemperature_K)             ==  88);
static_assert(offsetof(MaterialDataCPU, complexRefractiveIndexIndex) ==  92);
static_assert(offsetof(MaterialDataCPU, temperatureTextureIndex)     ==  96);
static_assert(offsetof(MaterialDataCPU, temperatureScale)            == 100);
static_assert(offsetof(MaterialDataCPU, temperatureOffset)           == 104);
static_assert(offsetof(MaterialDataCPU, irEmissivityCurveIndex)      == 108);
static_assert(offsetof(MaterialDataCPU, ior)                         == 112);
static_assert(offsetof(MaterialDataCPU, transmission)                == 116);
static_assert(offsetof(MaterialDataCPU, transmissionTextureIndex)    == 120);
static_assert(offsetof(MaterialDataCPU, irTransmittanceCurveIndex)   == 124);
static_assert(offsetof(MaterialDataCPU, attenuationColor)            == 128);
static_assert(offsetof(MaterialDataCPU, dispersion)                  == 152);
static_assert(offsetof(MaterialDataCPU, sheenReflectanceCurveIndex)  == 156);
static_assert(offsetof(MaterialDataCPU, endmemberCurveIndex1)        == 176);
static_assert(offsetof(MaterialDataCPU, endmemberCurveIndex2)        == 180);
static_assert(offsetof(MaterialDataCPU, endmemberCurveIndex3)        == 184);
static_assert(offsetof(MaterialDataCPU, weightTextureIndex)          == 188);
static_assert(offsetof(MaterialDataCPU, sheenColorFactor)            == 192);
static_assert(offsetof(MaterialDataCPU, sheenRoughnessFactor)        == 204);
static_assert(offsetof(MaterialDataCPU, sheenColorTextureIndex)      == 208);
static_assert(offsetof(MaterialDataCPU, sheenRoughnessTextureIndex)  == 212);
static_assert(offsetof(MaterialDataCPU, anisotropyStrength)          == 216);
static_assert(offsetof(MaterialDataCPU, anisotropyRotation)          == 220);
static_assert(offsetof(MaterialDataCPU, specularColorFactor)         == 224);
static_assert(offsetof(MaterialDataCPU, specularFactor)              == 236);
static_assert(offsetof(MaterialDataCPU, specularTextureIndex)        == 240);
static_assert(offsetof(MaterialDataCPU, specularColorTextureIndex)   == 244);
static_assert(offsetof(MaterialDataCPU, anisotropyTextureIndex)      == 248);
static_assert(offsetof(MaterialDataCPU, clearcoatFactor)             == 252);
static_assert(offsetof(MaterialDataCPU, clearcoatRoughnessFactor)    == 256);
static_assert(offsetof(MaterialDataCPU, clearcoatNormalScale)        == 260);
static_assert(offsetof(MaterialDataCPU, clearcoatTextureIndex)       == 264);
static_assert(offsetof(MaterialDataCPU, clearcoatRoughnessTextureIndex) == 268);
static_assert(offsetof(MaterialDataCPU, clearcoatNormalTextureIndex) == 272);
static_assert(offsetof(MaterialDataCPU, clearcoatReflectanceCurveIndex) == 276);
static_assert(offsetof(MaterialDataCPU, diffuseTransmissionFactor)   == 280);
static_assert(offsetof(MaterialDataCPU, diffuseTransmissionTextureIndex) == 284);
static_assert(offsetof(MaterialDataCPU, diffuseTransmissionColorFactor) == 288);
static_assert(offsetof(MaterialDataCPU, diffuseTransmissionColorTextureIndex) == 300);
static_assert(offsetof(MaterialDataCPU, diffuseTransmissionColorCurveIndex) == 304);
static_assert(offsetof(MaterialDataCPU, emissiveRadianceCurveIndex)  == 308);
static_assert(offsetof(MaterialDataCPU, uvTransformMat)              == 320);
static_assert(offsetof(MaterialDataCPU, uvTransformOffset)           == 544);

} // namespace quantiloom
