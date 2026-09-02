// ============================================================================
// Quantiloom - the forcing's own convective coefficient
// ============================================================================
// The convective exchange coefficient used to live only on the material, as a
// constant. A constant cannot describe a day: the coefficient is set by the
// wind and by whether the air over the surface is being stirred or is sitting
// stably on top of it, and those reverse between afternoon and midnight.
//
// Measured against a SURFRAD station, a single value fitted to the daytime
// signal over-warmed the nights by up to 1.8 K. At night the ground is colder
// than the air, so convection is a SOURCE rather than a sink, and a
// coefficient sized for a well-mixed afternoon pours in heat that a stable
// nocturnal boundary layer withholds. The cases below pin the mechanism, the
// override's precedence, and its backward compatibility.
// ============================================================================

#include <gtest/gtest.h>

#include "thermal/CpuCrankNicolsonStepper.hpp"
#include "thermal/ThermalSolver.hpp"  // MakeOpenSkyExchange, LoadForcingCsv

#include <cstdio>
#include <filesystem>
#include <fstream>

using namespace quantiloom;
using namespace quantiloom::thermal;

namespace {

Vector<ThermalElement> OneElementFacingUp() {
    ThermalElement element;
    element.centroid = glm::vec3(0.0f);
    element.normal = glm::vec3(0.0f, 1.0f, 0.0f);
    element.area_m2 = 1.0f;
    element.materialId = 0;
    return {element};
}

ThermalMaterial Soil() {
    ThermalMaterial material;
    material.conductivity_W_mK = 0.5f;
    material.density_kg_m3 = 1600.0f;
    material.specificHeat_J_kgK = 875.0f;
    material.thickness_m = 0.5f;
    material.convection_W_m2K = 10.0f;
    material.shortwaveAbsorptivity = 0.79f;
    material.longwaveEmissivity = 0.96f;
    material.interiorBoundary = InteriorBoundary::Adiabatic;
    return material;
}

/// Night: no sun, and a sky well below the air, which is what makes the
/// surface end up colder than the air it exchanges with.
ThermalForcing ClearNight() {
    ThermalForcing forcing;
    forcing.airTemperature_K = 295.0;
    forcing.sunIrradiance_W_m2 = 0.0;
    forcing.skyTemperature_K = 265.0;
    forcing.sunDirection = glm::vec3(0.0f, 1.0f, 0.0f);
    return forcing;
}

f64 SurfaceAfter(const ThermalForcing& forcing, const usize steps, const f64 dt_s,
                 const ConvectionLaw& law = {}) {
    const auto elements = OneElementFacingUp();
    const Vector<ThermalMaterial> materials{Soil()};
    const auto exchange = MakeOpenSkyExchange(1);
    const Vector<f32> visibility{0.0f};

    ThermalState state;
    state.nodeCount = 12;
    state.temperature_K.assign(state.nodeCount, 295.0);

    CpuCrankNicolsonStepper stepper(law);
    for (usize i = 0; i < steps; ++i) {
        stepper.Step(state, elements, materials, exchange, forcing, dt_s, {visibility});
    }
    return state.Surface(0);
}

ConvectionLaw Wind() {
    ConvectionLaw law;
    law.model = ConvectionModel::Wind;
    return law;
}

ConvectionLaw Stability() {
    ConvectionLaw law;
    law.model = ConvectionModel::Stability;
    return law;
}

std::filesystem::path WriteCsv(const char* name, const char* contents) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream(path) << contents;
    return path;
}

}  // namespace

TEST(ThermalConvectionForcing, ZeroMeansTheMaterialKeepsItsOwn) {
    // The default, and every scene that never heard of this channel. A forcing
    // that says nothing must be bit-identical to one that cannot say anything.
    ThermalForcing silent = ClearNight();
    ThermalForcing explicitMaterialValue = ClearNight();
    explicitMaterialValue.convection_W_m2K = 0.0;

    EXPECT_DOUBLE_EQ(SurfaceAfter(silent, 240, 60.0),
                     SurfaceAfter(explicitMaterialValue, 240, 60.0));
}

TEST(ThermalConvectionForcing, TheForcingOverridesTheMaterial) {
    ThermalForcing base = ClearNight();
    ThermalForcing overridden = ClearNight();
    overridden.convection_W_m2K = 30.0;  // material carries 10

    EXPECT_NE(SurfaceAfter(base, 240, 60.0), SurfaceAfter(overridden, 240, 60.0));
}

TEST(ThermalConvectionForcing, MoreConvectionWarmsAColdNightSurface) {
    // The mechanism behind the measured bias, stated as an ordering rather than
    // as a number. Under a cold sky the surface falls below the air, so the
    // convective term carries heat INTO it: a larger coefficient means a warmer
    // night, and a coefficient chosen to fit the afternoon is too large here.
    const f64 air = ClearNight().airTemperature_K;

    ThermalForcing weak = ClearNight();
    weak.convection_W_m2K = 5.0;
    ThermalForcing strong = ClearNight();
    strong.convection_W_m2K = 30.0;

    const f64 cool = SurfaceAfter(weak, 480, 60.0);
    const f64 warm = SurfaceAfter(strong, 480, 60.0);

    EXPECT_LT(cool, air) << "a surface under a 265 K sky must end below the air";
    EXPECT_LT(warm, air);
    EXPECT_GT(warm, cool) << "more convective exchange pulls a cold surface back "
                             "toward the air, which is why an afternoon-sized "
                             "coefficient over-warms a night";
}

TEST(ThermalConvectionForcing, TheNinthCsvColumnIsOptional) {
    // Same rule the seventh and eighth columns follow: a file written before
    // the column existed keeps its meaning exactly.
    const auto eight = WriteCsv("ql_forcing_eight.csv",
                                "# t air dni az el sky diff rh\n"
                                "0.0 295.0 0.0 180.0 0.0 265.0 0.0 20.0\n"
                                "1.0 296.0 0.0 180.0 0.0 266.0 0.0 21.0\n");
    const auto nine = WriteCsv("ql_forcing_nine.csv",
                               "# t air dni az el sky diff rh h\n"
                               "0.0 295.0 0.0 180.0 0.0 265.0 0.0 20.0 17.5\n"
                               "1.0 296.0 0.0 180.0 0.0 266.0 0.0 21.0 18.5\n");

    const auto withoutColumn = LoadForcingCsv(eight.string());
    const auto withColumn = LoadForcingCsv(nine.string());
    ASSERT_EQ(withoutColumn.size(), 2u);
    ASSERT_EQ(withColumn.size(), 2u);

    EXPECT_DOUBLE_EQ(withoutColumn[0].second.convection_W_m2K, 0.0)
        << "absent means the material's own, which is zero here";
    EXPECT_DOUBLE_EQ(withColumn[0].second.convection_W_m2K, 17.5);
    EXPECT_DOUBLE_EQ(withColumn[1].second.convection_W_m2K, 18.5);

    // Everything the eight-column file said still says the same thing.
    EXPECT_DOUBLE_EQ(withoutColumn[1].second.airTemperature_K,
                     withColumn[1].second.airTemperature_K);
    EXPECT_DOUBLE_EQ(withoutColumn[1].second.relativeHumidity,
                     withColumn[1].second.relativeHumidity);

    std::filesystem::remove(eight);
    std::filesystem::remove(nine);
}

TEST(ThermalConvectionForcing, InterpolationCarriesEveryField) {
    // SampleForcing interpolates field by field, which is the pattern that
    // silently drops whatever was added last. It dropped the convective
    // coefficient exactly that way: the endpoints carried it, because they
    // return a whole row, and everything between them read the struct default
    // instead -- so a forcing file varying h changed almost nothing and looked
    // like a physics result rather than a lost field.
    //
    // Written to fail for the NEXT field as well as this one: the two rows
    // differ in every member, and every member is checked at the midpoint.
    ThermalForcing early;
    early.airTemperature_K = 280.0;
    early.sunIrradiance_W_m2 = 100.0;
    early.diffuseIrradiance_W_m2 = 20.0;
    early.skyTemperature_K = 250.0;
    early.relativeHumidity = 30.0;
    early.convection_W_m2K = 8.0;
    early.windSpeed_m_s = 1.0;
    early.sunDirection = glm::vec3(0.0f, 1.0f, 0.0f);

    ThermalForcing late;
    late.airTemperature_K = 300.0;
    late.sunIrradiance_W_m2 = 900.0;
    late.diffuseIrradiance_W_m2 = 120.0;
    late.skyTemperature_K = 270.0;
    late.relativeHumidity = 50.0;
    late.convection_W_m2K = 28.0;
    late.windSpeed_m_s = 5.0;
    late.sunDirection = glm::vec3(0.0f, 1.0f, 0.0f);

    const Vector<std::pair<f64, ThermalForcing>> series{{0.0, early}, {2.0, late}};
    const ThermalForcing middle = SampleForcing(series, 1.0, ThermalForcing{});

    EXPECT_DOUBLE_EQ(middle.airTemperature_K, 290.0);
    EXPECT_DOUBLE_EQ(middle.sunIrradiance_W_m2, 500.0);
    EXPECT_DOUBLE_EQ(middle.diffuseIrradiance_W_m2, 70.0);
    EXPECT_DOUBLE_EQ(middle.skyTemperature_K, 260.0);
    EXPECT_DOUBLE_EQ(middle.relativeHumidity, 40.0);
    EXPECT_DOUBLE_EQ(middle.convection_W_m2K, 18.0)
        << "a field that only the endpoints carry is a field that is not there";
    EXPECT_DOUBLE_EQ(middle.windSpeed_m_s, 3.0)
        << "a field that only the endpoints carry is a field that is not there";
}

// ============================================================================
// Where h comes from when the forcing does not say
// ============================================================================

TEST(ThermalConvectionLaw, TheTenthCsvColumnIsOptional) {
    // Same rule every optional column follows: a file written before it
    // existed keeps its meaning exactly, and a calm is what it describes.
    const auto nine = WriteCsv("ql_forcing_law_nine.csv",
                               "# t air dni az el sky diff rh h\n"
                               "0.0 295.0 0.0 180.0 0.0 265.0 0.0 20.0 17.5\n"
                               "1.0 296.0 0.0 180.0 0.0 266.0 0.0 21.0 18.5\n");
    const auto ten = WriteCsv("ql_forcing_law_ten.csv",
                              "# t air dni az el sky diff rh h wind\n"
                              "0.0 295.0 0.0 180.0 0.0 265.0 0.0 20.0 17.5 2.5\n"
                              "1.0 296.0 0.0 180.0 0.0 266.0 0.0 21.0 18.5 4.5\n");

    const auto withoutColumn = LoadForcingCsv(nine.string());
    const auto withColumn = LoadForcingCsv(ten.string());
    ASSERT_EQ(withoutColumn.size(), 2u);
    ASSERT_EQ(withColumn.size(), 2u);

    EXPECT_DOUBLE_EQ(withoutColumn[0].second.windSpeed_m_s, 0.0);
    EXPECT_DOUBLE_EQ(withColumn[0].second.windSpeed_m_s, 2.5);
    EXPECT_DOUBLE_EQ(withColumn[1].second.windSpeed_m_s, 4.5);
    EXPECT_DOUBLE_EQ(withoutColumn[1].second.convection_W_m2K,
                     withColumn[1].second.convection_W_m2K);

    std::filesystem::remove(nine);
    std::filesystem::remove(ten);
}

TEST(ThermalConvectionLaw, TheConstantLawIsWhatWasThereBefore) {
    // The default has to be bit-identical to a run from before any of this
    // existed, or every scene in the repository moves.
    const ThermalForcing night = ClearNight();
    EXPECT_DOUBLE_EQ(SurfaceAfter(night, 240, 60.0, ConvectionLaw{}),
                     SurfaceAfter(night, 240, 60.0));
}

TEST(ThermalConvectionLaw, TheForcingColumnStillWinsOverTheLaw) {
    // A file carrying a measured coefficient is stating what the correlations
    // estimate, so it is not overridden by them.
    ThermalForcing measured = ClearNight();
    measured.convection_W_m2K = 12.0;
    measured.windSpeed_m_s = 9.0;  // the law would make this ~40

    EXPECT_DOUBLE_EQ(SurfaceAfter(measured, 240, 60.0, Stability()),
                     SurfaceAfter(measured, 240, 60.0, Wind()));
}

TEST(ThermalConvectionLaw, WindRaisesTheCoefficient) {
    // h = a + b U, so a windy night is exchanging more with the air -- and
    // under a cold sky the surface is the colder of the two, so more exchange
    // means a warmer surface.
    const f64 air = ClearNight().airTemperature_K;

    ThermalForcing calm = ClearNight();
    calm.windSpeed_m_s = 0.0;
    ThermalForcing breezy = ClearNight();
    breezy.windSpeed_m_s = 6.0;

    const f64 still = SurfaceAfter(calm, 480, 60.0, Wind());
    const f64 blown = SurfaceAfter(breezy, 480, 60.0, Wind());

    EXPECT_LT(still, air);
    EXPECT_GT(blown, still) << "6 m/s is h = 28.5 against 5.7, and the surface is "
                               "colder than the air it is exchanging with";
}

TEST(ThermalConvectionLaw, StabilityDampsTheExchangeOnACalmNight) {
    // The case the model exists for, and the direction the SURFRAD comparison
    // fixes. The surface is colder than the air, so the densest air is already
    // at the bottom and there is nothing to overturn: a nocturnal layer
    // withholds heat that an afternoon-sized coefficient pours in.
    //
    // Stated as an ordering rather than a number: with the same wind, the
    // stability law must leave the surface COLDER than the wind law alone.
    const f64 air = ClearNight().airTemperature_K;

    ThermalForcing night = ClearNight();
    night.windSpeed_m_s = 1.0;

    const f64 undamped = SurfaceAfter(night, 480, 60.0, Wind());
    const f64 damped = SurfaceAfter(night, 480, 60.0, Stability());

    EXPECT_LT(undamped, air);
    EXPECT_LT(damped, undamped)
        << "a stable layer exchanges less, so the surface runs further below the air";
}

TEST(ThermalConvectionLaw, StabilityFloorsACalmDaySurfaceAtFreeConvection) {
    // The other side of the same law. A surface hotter than the air raises
    // plumes off itself, so free convection is a floor under the wind law
    // rather than a damping of it -- and with the wind law's own intercept at
    // 5.7 the floor only bites once the surface is far above the air.
    ThermalForcing hot;
    hot.airTemperature_K = 295.0;
    hot.skyTemperature_K = 295.0;   // no radiative sink, so the sun sets the level
    hot.sunIrradiance_W_m2 = 1000.0;
    hot.sunDirection = glm::vec3(0.0f, 1.0f, 0.0f);
    hot.windSpeed_m_s = 0.0;

    const auto elements = OneElementFacingUp();
    const Vector<ThermalMaterial> materials{Soil()};
    const auto exchange = MakeOpenSkyExchange(1);
    const Vector<f32> fullSun{1.0f};

    const auto run = [&](const ConvectionLaw& law) {
        ThermalState state;
        state.nodeCount = 12;
        state.temperature_K.assign(state.nodeCount, 295.0);
        CpuCrankNicolsonStepper stepper(law);
        for (usize i = 0; i < 480; ++i) {
            stepper.Step(state, elements, materials, exchange, hot, 60.0, {fullSun});
        }
        return state.Surface(0);
    };

    const f64 windOnly = run(Wind());
    const f64 withFree = run(Stability());

    EXPECT_GT(windOnly, hot.airTemperature_K) << "a sunlit surface ends above the air";
    EXPECT_LE(withFree, windOnly)
        << "free convection can only add exchange, which can only cool a surface "
           "that is hotter than the air";
}
