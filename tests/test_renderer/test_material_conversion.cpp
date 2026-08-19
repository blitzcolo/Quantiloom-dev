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

    const auto gpu = rendercore::ConvertMaterial(mat, 550.0f,
                                                 rendercore::IndicesFromMaterial(mat));

    EXPECT_EQ(gpu.spectralReflectanceCurveIndex, 1);
    EXPECT_EQ(gpu.complexRefractiveIndexIndex, 2);
    EXPECT_EQ(gpu.endmemberCurveIndex1, 3);
    EXPECT_EQ(gpu.endmemberCurveIndex2, 4);
    EXPECT_EQ(gpu.endmemberCurveIndex3, 5);
    EXPECT_EQ(gpu.weightTextureIndex, 6);
    EXPECT_EQ(gpu.sheenReflectanceCurveIndex, 7);
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

    const auto gpu = rendercore::ConvertMaterial(mat, 550.0f);

    EXPECT_FLOAT_EQ(gpu.metallicFactor, 0.25f);
    EXPECT_FLOAT_EQ(gpu.roughnessFactor, 0.75f);
    EXPECT_EQ(gpu.doubleSided, 1u);
    EXPECT_FLOAT_EQ(gpu.alphaCutoff, 0.3f);
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
