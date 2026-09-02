/**
 * @file test_material_conversion.cpp
 * @brief Cover for rendercore::ConvertMaterial
 *
 * The conversion is pure, so these need no device. The case that matters is the
 * wavelength one: ExternalRenderContext used to average each IR curve over its whole
 * range, because its copy of the conversion was a file-static function that could not
 * reach the current wavelength. A renderer that models spectral response cannot have
 * a thermal response that ignores which wavelength it is rendering.
 */

#include <gtest/gtest.h>

#include "renderer/RenderCore.hpp"

#include <glm/gtc/constants.hpp>

using namespace quantiloom;

namespace {

// Emissivity that falls across the band, so the value at a wavelength is nowhere
// near the mean except at the midpoint.
Material MakeIrMaterial() {
    Material mat;
    mat.name = "ir";
    mat.irEmissivityCurve = {{3000.0f, 0.9f}, {13000.0f, 0.1f}};
    mat.irTransmittanceCurve = {{3000.0f, 0.0f}, {13000.0f, 0.4f}};
    return mat;
}

}  // namespace

TEST(RenderCoreConvertMaterial, EvaluatesIrCurvesAtTheGivenWavelength) {
    const Material mat = MakeIrMaterial();

    const auto atShort = rendercore::ConvertMaterial(mat, 3000.0f);
    const auto atLong = rendercore::ConvertMaterial(mat, 13000.0f);

    EXPECT_NEAR(atShort.irEmissivity, 0.9f, 1e-4f);
    EXPECT_NEAR(atLong.irEmissivity, 0.1f, 1e-4f);
    EXPECT_NEAR(atShort.irTransmittance, 0.0f, 1e-4f);
    EXPECT_NEAR(atLong.irTransmittance, 0.4f, 1e-4f);
}

// The regression this replaced: averaging the curve gives 0.5 at both ends, which is
// what the GUI used to send the shader whatever band it was rendering.
TEST(RenderCoreConvertMaterial, DoesNotCollapseIrCurvesToTheirMean) {
    const Material mat = MakeIrMaterial();
    constexpr f32 curveMean = 0.5f;

    const auto atShort = rendercore::ConvertMaterial(mat, 3000.0f);

    EXPECT_GT(std::abs(atShort.irEmissivity - curveMean), 0.3f)
        << "emissivity must track the wavelength, not the curve average";
}

// -1 is the sentinel the shader reads as "no IR data, derive emissivity from
// metallic and roughness". An empty curve must not become 0, which would mean a
// perfect reflector.
TEST(RenderCoreConvertMaterial, SignalsAbsentEmissivityWithTheSentinel) {
    Material mat;
    const auto gpu = rendercore::ConvertMaterial(mat, 10000.0f);

    EXPECT_FLOAT_EQ(gpu.irEmissivity, -1.0f);
}

TEST(RenderCoreConvertMaterial, ClampsEmissivityToPhysicalRange) {
    Material mat;
    mat.irEmissivityCurve = {{3000.0f, 1.4f}, {13000.0f, 1.4f}};

    const auto gpu = rendercore::ConvertMaterial(mat, 10000.0f);

    EXPECT_LE(gpu.irEmissivity, 1.0f) << "emissivity above 1 is unphysical";
}

// The two front ends resolve these differently -- the context from the Material, the
// CLI from names matched against the config -- so the conversion takes them.
TEST(RenderCoreConvertMaterial, TakesCurveSlotsFromTheCaller) {
    Material mat;
    mat.spectralReflectanceCurveIndex = 7;
    mat.complexRefractiveIndexIndex = 9;

    const auto fromCaller = rendercore::ConvertMaterial(mat, 550.0f, {2, 3});
    EXPECT_EQ(fromCaller.spectralReflectanceCurveIndex, 2);
    EXPECT_EQ(fromCaller.complexRefractiveIndexIndex, 3);

    const auto unset = rendercore::ConvertMaterial(mat, 550.0f);
    EXPECT_EQ(unset.spectralReflectanceCurveIndex, -1) << "default is no curve";
    EXPECT_EQ(unset.complexRefractiveIndexIndex, -1);
}

TEST(RenderCoreConvertMaterial, TakesEndmemberSlotsFromTheCaller) {
    Material mat;

    const auto gpu = rendercore::ConvertMaterial(mat, 550.0f, {2, 3, 11, 12, 13, 4});
    EXPECT_EQ(gpu.endmemberCurveIndex1, 11);
    EXPECT_EQ(gpu.endmemberCurveIndex2, 12);
    EXPECT_EQ(gpu.endmemberCurveIndex3, 13);
    EXPECT_EQ(gpu.weightTextureIndex, 4);
}

// The default is not merely "unset", it is the pre-endmember behaviour: no
// extra curves and no weight texture means the shader reads w = (1, 0, 0, 0)
// and evaluates the single bound curve flat, as it always did.
TEST(RenderCoreConvertMaterial, NoEndmembersMeansTheOldFlatBehaviour) {
    Material mat;
    mat.spectralReflectanceCurveIndex = 5;

    const auto gpu = rendercore::ConvertMaterial(mat, 550.0f, {5, -1});

    EXPECT_EQ(gpu.endmemberCurveIndex1, -1);
    EXPECT_EQ(gpu.endmemberCurveIndex2, -1);
    EXPECT_EQ(gpu.endmemberCurveIndex3, -1);
    EXPECT_EQ(gpu.weightTextureIndex, -1);
}

// The interactive path resolves spectra into the Material and hands them back
// through this helper in two places; if it dropped a slot, one of those two
// uploads would be silently short.
TEST(RenderCoreConvertMaterial, IndicesFromMaterialCarriesEverySlot) {
    Material mat;
    mat.spectralReflectanceCurveIndex = 1;
    mat.complexRefractiveIndexIndex = 2;
    mat.endmemberCurveIndex1 = 3;
    mat.endmemberCurveIndex2 = 4;
    mat.endmemberCurveIndex3 = 5;
    mat.weightTextureIndex = 6;
    mat.sheenReflectanceCurveIndex = 7;
    mat.clearcoatReflectanceCurveIndex = 8;
    mat.diffuseTransmissionColorCurveIndex = 9;
    mat.emissiveRadianceCurveIndex = 10;
    mat.fluorescenceExcitationCurveIndex = 11;
    mat.fluorescenceEmissionCurveIndex = 12;

    const auto gpu = rendercore::ConvertMaterial(mat, 550.0f,
                                                 rendercore::IndicesFromMaterial(mat));

    EXPECT_EQ(gpu.spectralReflectanceCurveIndex, 1);
    EXPECT_EQ(gpu.complexRefractiveIndexIndex, 2);
    EXPECT_EQ(gpu.endmemberCurveIndex1, 3);
    EXPECT_EQ(gpu.endmemberCurveIndex2, 4);
    EXPECT_EQ(gpu.endmemberCurveIndex3, 5);
    EXPECT_EQ(gpu.weightTextureIndex, 6);
    EXPECT_EQ(gpu.sheenReflectanceCurveIndex, 7);
    EXPECT_EQ(gpu.clearcoatReflectanceCurveIndex, 8);
    EXPECT_EQ(gpu.diffuseTransmissionColorCurveIndex, 9);
    EXPECT_EQ(gpu.emissiveRadianceCurveIndex, 10);
    // The two that were added last, and the reason this test lists all of them:
    // MaterialGpuIndices is initialised positionally, so a slot appended to the
    // struct and not to the initialiser compiles and uploads the wrong index.
    EXPECT_EQ(gpu.fluorescenceExcitationCurveIndex, 11);
    EXPECT_EQ(gpu.fluorescenceEmissionCurveIndex, 12);
}

// The yield is a scalar the Material carries rather than an index the caller
// resolves, so it takes a different route to the GPU struct than the two curves
// beside it and needs saying separately.
TEST(RenderCoreConvertMaterial, TheFluorescenceYieldComesOffTheMaterial) {
    Material mat;
    mat.fluorescenceYield = 0.62f;
    const auto gpu = rendercore::ConvertMaterial(mat, 550.0f);
    EXPECT_FLOAT_EQ(gpu.fluorescenceYield, 0.62f);

    // And nothing fluoresces by default, in either half of the description.
    const auto plain = rendercore::ConvertMaterial(Material{}, 550.0f);
    EXPECT_FLOAT_EQ(plain.fluorescenceYield, 0.0f);
    EXPECT_EQ(plain.fluorescenceExcitationCurveIndex, -1);
    EXPECT_EQ(plain.fluorescenceEmissionCurveIndex, -1);
}

// A fluorescent surface must not enter the emitter-sampling table: it emits
// only what something else lit it with, so a triple here would have next-event
// estimation aim at a light that is dark on its own.
TEST(RenderCoreConvertMaterial, FluorescenceLeavesTheEmissiveFactorAlone) {
    Material mat;
    mat.fluorescenceExcitationCurveIndex = 3;
    mat.fluorescenceEmissionCurveIndex = 4;
    mat.fluorescenceYield = 1.0f;
    ASSERT_TRUE(mat.HasFluorescence());

    const auto gpu = rendercore::ConvertMaterial(mat, 550.0f,
                                                 rendercore::IndicesFromMaterial(mat));
    EXPECT_EQ(gpu.emissiveFactor, glm::vec3(0.0f));
}

// Any one of the three alone describes nothing: a shape with no strength, or a
// strength with no shape.
TEST(RenderCoreConvertMaterial, FluorescenceNeedsBothCurvesAndAYield) {
    Material mat;
    EXPECT_FALSE(mat.HasFluorescence());

    mat.fluorescenceExcitationCurveIndex = 3;
    EXPECT_FALSE(mat.HasFluorescence());
    mat.fluorescenceEmissionCurveIndex = 4;
    EXPECT_FALSE(mat.HasFluorescence()) << "no yield is no fluorescence";
    mat.fluorescenceYield = 0.5f;
    EXPECT_TRUE(mat.HasFluorescence());

    mat.fluorescenceEmissionCurveIndex = -1;
    EXPECT_FALSE(mat.HasFluorescence()) << "one curve is no fluorescence";
}

// ============================================================================
// Specular, anisotropy, clearcoat, diffuse transmission
// ============================================================================

// A default-constructed Material must produce the GPU struct the renderer
// behaved as if it had before these fields existed. Specular is the one that
// is easy to get wrong: its neutral element is 1, not 0.
TEST(RenderCoreConvertMaterial, DefaultsAreTheNeutralElementForEveryNewExtension) {
    const auto gpu = rendercore::ConvertMaterial(Material{}, 550.0f);

    EXPECT_EQ(gpu.specularFactor, 1.0f);
    EXPECT_EQ(gpu.specularColorFactor, glm::vec3(1.0f));
    EXPECT_EQ(gpu.specularTextureIndex, -1);
    EXPECT_EQ(gpu.specularColorTextureIndex, -1);

    EXPECT_EQ(gpu.anisotropyStrength, 0.0f);
    EXPECT_EQ(gpu.anisotropyRotation, 0.0f);
    EXPECT_EQ(gpu.anisotropyTextureIndex, -1);

    EXPECT_EQ(gpu.clearcoatFactor, 0.0f);
    EXPECT_EQ(gpu.clearcoatRoughnessFactor, 0.0f);
    EXPECT_EQ(gpu.clearcoatNormalScale, 1.0f);
    EXPECT_EQ(gpu.clearcoatTextureIndex, -1);
    EXPECT_EQ(gpu.clearcoatRoughnessTextureIndex, -1);
    EXPECT_EQ(gpu.clearcoatNormalTextureIndex, -1);
    EXPECT_EQ(gpu.clearcoatReflectanceCurveIndex, -1);

    EXPECT_EQ(gpu.diffuseTransmissionFactor, 0.0f);
    EXPECT_EQ(gpu.diffuseTransmissionColorFactor, glm::vec3(1.0f));
    EXPECT_EQ(gpu.diffuseTransmissionTextureIndex, -1);
    EXPECT_EQ(gpu.diffuseTransmissionColorTextureIndex, -1);
    EXPECT_EQ(gpu.diffuseTransmissionColorCurveIndex, -1);
}

TEST(RenderCoreConvertMaterial, CarriesTheFourMaterialExtensionsThrough) {
    Material mat;
    // AnisotropyBarnLamp's Lamp Metal carries anisotropy and clearcoat on one
    // material, with the coat's normal map sharing the base's texture.
    mat.specularFactor = 0.5f;
    mat.specularColorFactor = glm::vec3(10.0f, 0.6f, 0.0f);  // SpecularSilkPouf
    mat.specularTextureIndex = 1;
    mat.specularColorTextureIndex = 2;
    mat.anisotropyStrength = 1.0f;
    mat.anisotropyRotation = 0.5235988f;
    mat.anisotropyTextureIndex = 3;
    mat.clearcoatFactor = 0.25f;
    mat.clearcoatRoughnessFactor = 0.15f;
    mat.clearcoatNormalScale = 0.2f;
    mat.clearcoatTextureIndex = 4;
    mat.clearcoatRoughnessTextureIndex = 5;
    mat.clearcoatNormalTextureIndex = 0;
    mat.diffuseTransmissionFactor = 1.0f;
    mat.diffuseTransmissionColorFactor = glm::vec3(0.84f, 0.8f, 0.74f);  // the teacup
    mat.diffuseTransmissionTextureIndex = 6;
    mat.diffuseTransmissionColorTextureIndex = 7;

    const auto gpu = rendercore::ConvertMaterial(mat, 550.0f);

    EXPECT_FLOAT_EQ(gpu.specularFactor, 0.5f);
    EXPECT_FLOAT_EQ(gpu.specularColorFactor.r, 10.0f) << "an HDR specular colour is legal";
    EXPECT_FLOAT_EQ(gpu.specularColorFactor.g, 0.6f);
    EXPECT_FLOAT_EQ(gpu.specularColorFactor.b, 0.0f);
    EXPECT_EQ(gpu.specularTextureIndex, 1);
    EXPECT_EQ(gpu.specularColorTextureIndex, 2);

    EXPECT_FLOAT_EQ(gpu.anisotropyStrength, 1.0f);
    EXPECT_FLOAT_EQ(gpu.anisotropyRotation, 0.5235988f);
    EXPECT_EQ(gpu.anisotropyTextureIndex, 3);

    EXPECT_FLOAT_EQ(gpu.clearcoatFactor, 0.25f);
    EXPECT_FLOAT_EQ(gpu.clearcoatRoughnessFactor, 0.15f);
    EXPECT_FLOAT_EQ(gpu.clearcoatNormalScale, 0.2f);
    EXPECT_EQ(gpu.clearcoatTextureIndex, 4);
    EXPECT_EQ(gpu.clearcoatRoughnessTextureIndex, 5);
    EXPECT_EQ(gpu.clearcoatNormalTextureIndex, 0);

    EXPECT_FLOAT_EQ(gpu.diffuseTransmissionFactor, 1.0f);
    EXPECT_FLOAT_EQ(gpu.diffuseTransmissionColorFactor.r, 0.84f);
    EXPECT_FLOAT_EQ(gpu.diffuseTransmissionColorFactor.g, 0.8f);
    EXPECT_FLOAT_EQ(gpu.diffuseTransmissionColorFactor.b, 0.74f);
    EXPECT_EQ(gpu.diffuseTransmissionTextureIndex, 6);
    EXPECT_EQ(gpu.diffuseTransmissionColorTextureIndex, 7);
}

// The predicates gate whole blocks of shader work and drive the Studio panel's
// auto-expand, so what counts as "present" has to be exactly the deviation from
// glTF's default -- especially for specular, whose default is not zero.
TEST(MaterialExtensionPredicates, ReportAbsenceOnADefaultMaterial) {
    const Material mat;
    EXPECT_FALSE(mat.HasSpecular());
    EXPECT_FALSE(mat.HasAnisotropy());
    EXPECT_FALSE(mat.HasClearcoat());
    EXPECT_FALSE(mat.HasDiffuseTransmission());
}

TEST(MaterialExtensionPredicates, SpecularCountsAsAbsentAtItsNeutralElement) {
    Material mat;
    mat.specularFactor = 1.0f;
    mat.specularColorFactor = glm::vec3(1.0f);
    EXPECT_FALSE(mat.HasSpecular()) << "these are exactly the pre-extension F0 and F90";

    mat.specularColorFactor = glm::vec3(0.0f, 0.0f, 0.0f);  // GlamVelvetSofa champagne
    EXPECT_TRUE(mat.HasSpecular()) << "zero specular colour is a real, authored choice";

    mat.specularColorFactor = glm::vec3(1.0f);
    mat.specularFactor = 0.5f;
    EXPECT_TRUE(mat.HasSpecular());
}

TEST(MaterialExtensionPredicates, ReportPresenceFromFactorsAndCurves) {
    Material aniso;
    aniso.anisotropyStrength = 0.5f;
    EXPECT_TRUE(aniso.HasAnisotropy());

    // A rotation without strength rotates a lobe that is not stretched, and a
    // texture cannot rescue a zero factor -- the spec multiplies the two.
    Material rotationOnly;
    rotationOnly.anisotropyRotation = 1.0f;
    rotationOnly.anisotropyTextureIndex = 3;
    EXPECT_FALSE(rotationOnly.HasAnisotropy());

    Material coat;
    coat.clearcoatFactor = 0.25f;
    EXPECT_TRUE(coat.HasClearcoat());

    // The infrared bands read the curve instead of assuming a dielectric 0.04,
    // so a bound curve counts on its own.
    Material coatCurve;
    coatCurve.clearcoatReflectanceCurveIndex = 2;
    EXPECT_TRUE(coatCurve.HasClearcoat());

    Material dt;
    dt.diffuseTransmissionFactor = 0.1f;
    EXPECT_TRUE(dt.HasDiffuseTransmission());

    // The colour factor defaults to white and only tints what the factor lets
    // through, so on its own it is not presence.
    Material dtColorOnly;
    dtColorOnly.diffuseTransmissionColorFactor = glm::vec3(0.84f, 0.8f, 0.74f);
    EXPECT_FALSE(dtColorOnly.HasDiffuseTransmission());
}

// ============================================================================
// Sheen and UV transforms
// ============================================================================

TEST(RenderCoreConvertMaterial, CarriesSheenThrough) {
    Material mat;
    mat.sheenColorFactor = glm::vec3(0.9f, 0.7f, 0.6f);
    mat.sheenRoughnessFactor = 0.6f;
    mat.sheenColorTextureIndex = 3;
    mat.sheenRoughnessTextureIndex = 3;

    const auto gpu = rendercore::ConvertMaterial(mat, 550.0f);

    EXPECT_FLOAT_EQ(gpu.sheenColorFactor.r, 0.9f);
    EXPECT_FLOAT_EQ(gpu.sheenColorFactor.g, 0.7f);
    EXPECT_FLOAT_EQ(gpu.sheenColorFactor.b, 0.6f);
    EXPECT_FLOAT_EQ(gpu.sheenRoughnessFactor, 0.6f);
    EXPECT_EQ(gpu.sheenColorTextureIndex, 3);
    EXPECT_EQ(gpu.sheenRoughnessTextureIndex, 3);
}

// The default has to be the exact identity, not a rounded one: every scene
// without KHR_texture_transform goes through this path, and the guarantee that
// those render bit-identically is only worth as much as this assertion.
TEST(RenderCoreConvertMaterial, DefaultUvTransformIsExactlyIdentity) {
    const auto gpu = rendercore::ConvertMaterial(Material{}, 550.0f);

    for (int slot = 0; slot < UV_SLOT_COUNT; ++slot) {
        EXPECT_EQ(gpu.uvTransformMat[slot].x, 1.0f) << "slot " << slot;
        EXPECT_EQ(gpu.uvTransformMat[slot].y, 0.0f) << "slot " << slot;
        EXPECT_EQ(gpu.uvTransformMat[slot].z, 0.0f) << "slot " << slot;
        EXPECT_EQ(gpu.uvTransformMat[slot].w, 1.0f) << "slot " << slot;
        EXPECT_EQ(gpu.uvTransformOffset[slot].x, 0.0f) << "slot " << slot;
        EXPECT_EQ(gpu.uvTransformOffset[slot].y, 0.0f) << "slot " << slot;
    }
}

TEST(RenderCoreConvertMaterial, UvTransformScaleAndOffsetReachTheRightSlot) {
    Material mat;
    mat.baseColorUv.scale = glm::vec2(30.0f, 30.0f);   // SheenCloth
    mat.normalUv.offset = glm::vec2(0.5f, 0.25f);

    const auto gpu = rendercore::ConvertMaterial(mat, 550.0f);

    EXPECT_FLOAT_EQ(gpu.uvTransformMat[UV_SLOT_BASE_COLOR].x, 30.0f);
    EXPECT_FLOAT_EQ(gpu.uvTransformMat[UV_SLOT_BASE_COLOR].w, 30.0f);
    EXPECT_FLOAT_EQ(gpu.uvTransformOffset[UV_SLOT_NORMAL].x, 0.5f);
    EXPECT_FLOAT_EQ(gpu.uvTransformOffset[UV_SLOT_NORMAL].y, 0.25f);

    // A transform on one slot must not leak into another. SheenChair's fabric
    // is the case: base colour at scale 7, normal map at scale 2, one material.
    EXPECT_FLOAT_EQ(gpu.uvTransformMat[UV_SLOT_NORMAL].x, 1.0f);
    EXPECT_FLOAT_EQ(gpu.uvTransformOffset[UV_SLOT_BASE_COLOR].x, 0.0f);
}

// KHR_texture_transform composes translation * rotation * scale, and the
// rotation is clockwise in UV space because the second axis points down.
// Getting the sign backwards mirrors a rotated texture about the diagonal,
// which reads as a bad asset rather than as a bug here.
TEST(RenderCoreConvertMaterial, UvTransformRotationMatchesTheExtension) {
    Material mat;
    mat.baseColorUv.rotation = glm::half_pi<f32>();  // 90 degrees
    mat.baseColorUv.scale = glm::vec2(2.0f, 3.0f);

    const auto gpu = rendercore::ConvertMaterial(mat, 550.0f);
    const glm::vec4 m = gpu.uvTransformMat[UV_SLOT_BASE_COLOR];

    // cos = 0, sin = 1  =>  (c*sx, s*sy, -s*sx, c*sy) = (0, 3, -2, 0)
    EXPECT_NEAR(m.x, 0.0f, 1e-6f);
    EXPECT_NEAR(m.y, 3.0f, 1e-6f);
    EXPECT_NEAR(m.z, -2.0f, 1e-6f);
    EXPECT_NEAR(m.w, 0.0f, 1e-6f);

    // The point of the sign, stated as the mapping it produces: u turns into
    // -v's direction, not +v's.
    const glm::vec2 uv{1.0f, 0.0f};
    const glm::vec2 mapped{m.x * uv.x + m.y * uv.y, m.z * uv.x + m.w * uv.y};
    EXPECT_NEAR(mapped.x, 0.0f, 1e-6f);
    EXPECT_NEAR(mapped.y, -2.0f, 1e-6f);
}

TEST(RenderCoreConvertMaterial, CarriesThePbrFactorsThrough) {
    Material mat;
    mat.metallicFactor = 0.25f;
    mat.roughnessFactor = 0.75f;
    mat.doubleSided = true;
    mat.alphaCutoff = 0.3f;
    mat.alphaMode = Material::AlphaMode::Mask;

    const auto gpu = rendercore::ConvertMaterial(mat, 550.0f);

    EXPECT_FLOAT_EQ(gpu.metallicFactor, 0.25f);
    EXPECT_FLOAT_EQ(gpu.roughnessFactor, 0.75f);
    EXPECT_EQ(gpu.doubleSided, 1u);
    EXPECT_FLOAT_EQ(gpu.alphaCutoff, 0.3f);
    // The field the any-hit shader branches on. It was uploaded and never
    // checked until the shader started reading it.
    EXPECT_EQ(gpu.alphaMode, static_cast<u32>(Material::AlphaMode::Mask));
}

// ============================================================================
// SensorAdjustmentForMode
// ============================================================================
// A fused band renders per-nm average radiance and carries its own photon energy.
// The CLI applied both corrections inline; ExternalRenderContext applied neither, so
// the same scene through the same SensorParams was ~7e4 too dark in the GUI for LWIR.

TEST(RenderCoreSensorAdjustment, RecoversBandIntegratedRadianceForIrBands) {
    const auto lwir = rendercore::SensorAdjustmentForMode(SpectralMode::LWIR_Fused, false);
    EXPECT_FLOAT_EQ(lwir.radianceScale, 4000.0f) << "LWIR spans 8000-12000 nm";
    EXPECT_FLOAT_EQ(lwir.wavelengthNm, 10000.0f) << "photon energy at the band centre";

    const auto mwir = rendercore::SensorAdjustmentForMode(SpectralMode::MWIR_Fused, false);
    EXPECT_FLOAT_EQ(mwir.radianceScale, 2000.0f);
    EXPECT_FLOAT_EQ(mwir.wavelengthNm, 4000.0f);
}

// VIS_Fused outputs CIE-integrated RGB rather than scalar band radiance, so neither
// correction applies to it however fused it is.
TEST(RenderCoreSensorAdjustment, LeavesNonIrModesAlone) {
    for (const auto mode : {SpectralMode::RGB, SpectralMode::Single,
                            SpectralMode::VIS_Fused}) {
        const auto adjustment = rendercore::SensorAdjustmentForMode(mode, false);
        EXPECT_FLOAT_EQ(adjustment.radianceScale, 1.0f);
        EXPECT_FLOAT_EQ(adjustment.wavelengthNm, 0.0f) << "0 means leave it alone";
    }
}

// The radiance scale is a property of how the renderer writes the band; a host
// choosing its own photon wavelength does not change it.
TEST(RenderCoreSensorAdjustment, KeepsAHostChosenWavelength) {
    const auto adjustment =
        rendercore::SensorAdjustmentForMode(SpectralMode::LWIR_Fused, true);

    EXPECT_FLOAT_EQ(adjustment.wavelengthNm, 0.0f) << "0 leaves the caller's value";
    EXPECT_FLOAT_EQ(adjustment.radianceScale, 4000.0f) << "the scale still applies";
}
