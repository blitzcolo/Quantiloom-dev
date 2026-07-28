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
