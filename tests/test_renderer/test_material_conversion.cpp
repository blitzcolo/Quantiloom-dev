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

    const auto gpu = rendercore::ConvertMaterial(mat, 550.0f,
                                                 rendercore::IndicesFromMaterial(mat));

    EXPECT_EQ(gpu.spectralReflectanceCurveIndex, 1);
    EXPECT_EQ(gpu.complexRefractiveIndexIndex, 2);
    EXPECT_EQ(gpu.endmemberCurveIndex1, 3);
    EXPECT_EQ(gpu.endmemberCurveIndex2, 4);
    EXPECT_EQ(gpu.endmemberCurveIndex3, 5);
    EXPECT_EQ(gpu.weightTextureIndex, 6);
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
