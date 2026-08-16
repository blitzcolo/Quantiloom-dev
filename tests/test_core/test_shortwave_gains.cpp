// ============================================================================
// Quantiloom - Unit Tests for thermal/ShortwaveGains.hpp
// ============================================================================
// The bake is a matrix pass with no physics of its own beyond radiosity: an
// element's gain is the sum over its row of "what fraction of my hemisphere is
// that element" times "how much of what lands on it does it send on". So the
// way to check it is by hand, on a matrix small enough to write down, rather
// than by anything the solver produces.
//
// The scene: three elements, with element 0 seeing 1 and 2, element 1 seeing
// 0, and element 2 seeing nothing but sky.
// ============================================================================

#include <gtest/gtest.h>

#include "thermal/ShortwaveGains.hpp"

using namespace quantiloom;
using namespace quantiloom::thermal;

namespace {

Vector<ThermalElement> ThreeElements() {
    Vector<ThermalElement> elements(3);
    for (usize e = 0; e < 3; ++e) {
        elements[e].area_m2 = 1.0f;
        elements[e].normal = glm::vec3(0.0f, 1.0f, 0.0f);
        elements[e].materialId = static_cast<u32>(e);
    }
    elements[2].normal = glm::vec3(0.0f, -1.0f, 0.0f);  // faces away from the sun
    return elements;
}

/// Absorptivities 0.7, 0.25, 0.4 -- so reflectances 0.3, 0.75, 0.6.
Vector<ThermalMaterial> ThreeMaterials() {
    Vector<ThermalMaterial> materials(3);
    materials[0].shortwaveAbsorptivity = 0.7f;
    materials[1].shortwaveAbsorptivity = 0.25f;
    materials[2].shortwaveAbsorptivity = 0.4f;
    return materials;
}

ExchangeGeometry ThreeElementExchange() {
    ExchangeGeometry exchange;
    exchange.viewFactors.rowStart = {0, 2, 3, 3};
    exchange.viewFactors.column = {1, 2, 0};
    exchange.viewFactors.value = {0.3f, 0.2f, 0.4f};
    exchange.skyFraction = {0.5f, 0.6f, 1.0f};
    exchange.sunVisibility = {1.0f, 0.5f, 1.0f};
    return exchange;
}

SunVisibilityTable OneColumnAt(const glm::vec3& sunDirection) {
    SunVisibilityTable table;
    table.sampleTime_h = {12.0};
    table.visibility = {1.0f, 0.5f, 1.0f};
    table.sampleDirection = {sunDirection};
    return table;
}

}  // namespace

TEST(ShortwaveGainsTest, TheDiffuseGainIsTheSkyPlusOneBounceOfIt) {
    const auto elements = ThreeElements();
    const auto materials = ThreeMaterials();
    const auto exchange = ThreeElementExchange();

    SunVisibilityTable table = OneColumnAt(glm::vec3(0.0f, 1.0f, 0.0f));
    BakeShortwaveGains(exchange, elements, materials, table);

    ASSERT_EQ(table.diffuseGain.size(), 3u);
    // 0.5 + 0.3*0.75*0.6 + 0.2*0.6*1.0
    EXPECT_NEAR(table.diffuseGain[0], 0.755f, 1e-6f);
    // 0.6 + 0.4*0.3*0.5
    EXPECT_NEAR(table.diffuseGain[1], 0.66f, 1e-6f);
    // sees nothing but sky, so nothing is added to it
    EXPECT_NEAR(table.diffuseGain[2], 1.0f, 1e-6f);
}

TEST(ShortwaveGainsTest, TheReflectedGainGathersWhatTheLitElementsSendOn) {
    const auto elements = ThreeElements();
    const auto materials = ThreeMaterials();
    const auto exchange = ThreeElementExchange();

    SunVisibilityTable table = OneColumnAt(glm::vec3(0.0f, 1.0f, 0.0f));
    BakeShortwaveGains(exchange, elements, materials, table);

    // Radiosity per unit direct normal irradiance: rho * max(cos, 0) * v.
    // Element 2 faces away from the sun, so it sends nothing on however
    // reflective it is.
    ASSERT_EQ(table.reflectedGain.size(), 3u);
    EXPECT_NEAR(table.reflectedGain[0], 0.3f * (0.75f * 0.5f), 1e-6f);
    EXPECT_NEAR(table.reflectedGain[1], 0.4f * (0.3f * 1.0f), 1e-6f);
    EXPECT_NEAR(table.reflectedGain[2], 0.0f, 1e-6f);
}

TEST(ShortwaveGainsTest, EachSunColumnGetsItsOwnGain) {
    const auto elements = ThreeElements();
    const auto materials = ThreeMaterials();
    const auto exchange = ThreeElementExchange();

    // Two columns: the sun overhead, then below the horizon. The second is
    // night, and nothing bounces at night.
    SunVisibilityTable table;
    table.sampleTime_h = {12.0, 0.0};
    table.visibility = {1.0f, 0.5f, 1.0f, 0.0f, 0.0f, 0.0f};
    table.sampleDirection = {glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)};
    BakeShortwaveGains(exchange, elements, materials, table);

    ASSERT_EQ(table.reflectedGain.size(), 6u);
    EXPECT_NEAR(table.ReflectedColumn(0)[0], 0.3f * (0.75f * 0.5f), 1e-6f);
    for (usize e = 0; e < 3; ++e) {
        EXPECT_NEAR(table.ReflectedColumn(1)[e], 0.0f, 1e-6f) << "element " << e;
    }
    // The diffuse gain does not depend on the sun, so it stays one column.
    EXPECT_EQ(table.diffuseGain.size(), 3u);
}

TEST(ShortwaveGainsTest, WithoutSunDirectionsThereIsNoBounceToBake) {
    // The interactive path synthesises a table from the exchange and may not
    // know where the sun was. That has to degrade to "no bounce" rather than
    // to a bounce off an assumed direction.
    const auto elements = ThreeElements();
    const auto materials = ThreeMaterials();
    const auto exchange = ThreeElementExchange();

    SunVisibilityTable table = OneColumnAt(glm::vec3(0.0f, 1.0f, 0.0f));
    table.sampleDirection.clear();
    BakeShortwaveGains(exchange, elements, materials, table);

    EXPECT_TRUE(table.reflectedGain.empty());
    EXPECT_EQ(table.diffuseGain.size(), 3u) << "the sky does not need a sun direction";
}

TEST(ShortwaveGainsTest, AnEmptyExchangeLeavesBothGainsEmpty) {
    // A scene whose view factors never got built. The stepper falls back to
    // the bare sky fraction, so the bake must not invent a gain of its own.
    const auto elements = ThreeElements();
    const auto materials = ThreeMaterials();

    SunVisibilityTable table = OneColumnAt(glm::vec3(0.0f, 1.0f, 0.0f));
    BakeShortwaveGains(ExchangeGeometry{}, elements, materials, table);

    EXPECT_TRUE(table.reflectedGain.empty());
    EXPECT_TRUE(table.diffuseGain.empty());
}
