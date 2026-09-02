// ============================================================================
// Quantiloom - Unit Tests for thermal/CpuCrankNicolsonStepper.hpp
// ============================================================================
// A surface temperature is not something a render can be checked against by
// eye: any smooth field looks plausible. What can be checked is the solver's
// agreement with cases that have closed-form answers -- a lumped body cooling
// exponentially, a slab settling into a linear profile, a step change
// diffusing as an error function -- and its convergence order, which says the
// discretisation is the one claimed rather than one that happens to be close.
//
// Every case here turns off what it is not testing. Radiation is quartic and
// couples elements together, so the conduction cases run with an emissivity of
// zero; the radiative case runs with no conduction to speak of.
// ============================================================================

#include <gtest/gtest.h>

#include "thermal/CpuCrankNicolsonStepper.hpp"

#include "thermal/ShortwaveGains.hpp"
#include "thermal/ThermalSolver.hpp"  // MakeOpenSkyExchange

#include <cmath>
#include <span>

using namespace quantiloom;
using namespace quantiloom::thermal;

namespace {

constexpr f64 kStefanBoltzmann = 5.670374419e-8;

/// One element, facing up, one square metre.
Vector<ThermalElement> OneElement() {
    ThermalElement element;
    element.centroid = glm::vec3(0.0f);
    element.normal = glm::vec3(0.0f, 1.0f, 0.0f);
    element.area_m2 = 1.0f;
    element.materialId = 0;
    return {element};
}

/// A material that conducts well enough to stay isothermal through its depth,
/// so the slab behaves as the lumped body the closed form describes.
ThermalMaterial LumpedMaterial() {
    ThermalMaterial material;
    material.conductivity_W_mK = 400.0f;   // copper, near enough
    material.density_kg_m3 = 8000.0f;
    material.specificHeat_J_kgK = 400.0f;
    material.thickness_m = 0.01f;
    material.convection_W_m2K = 10.0f;
    material.shortwaveAbsorptivity = 0.0f;
    material.longwaveEmissivity = 0.0f;    // convection only
    material.interiorBoundary = InteriorBoundary::Adiabatic;
    return material;
}

ThermalState MakeState(const usize elements, const u32 nodes, const f64 temperature_K) {
    ThermalState state;
    state.nodeCount = nodes;
    state.temperature_K.assign(elements * nodes, temperature_K);
    return state;
}

/// Saturation specific humidity, kg/kg. Written out again rather than shared
/// with the stepper: a test that calls the code it is checking checks nothing
/// about the formula, only about the plumbing.
f64 SaturationHumidity(const f64 temperature_K) {
    const f64 tC = temperature_K - 273.15;
    const f64 e = 610.94 * std::exp(17.625 * tC / (tC + 243.04));
    return 0.622 * e / (101325.0 - 0.378 * e);
}

/// The root of @p imbalance between @p low and @p high, to a millikelvin.
/// Bisection rather than Newton because the balances here are monotone and the
/// bracket is known, and a test should not need a derivative to be right.
template <typename F>
f64 BalanceTemperature(F&& imbalance, f64 low, f64 high) {
    for (i32 i = 0; i < 200; ++i) {
        const f64 mid = 0.5 * (low + high);
        if (imbalance(low) * imbalance(mid) <= 0.0) {
            high = mid;
        } else {
            low = mid;
        }
    }
    return 0.5 * (low + high);
}

}  // namespace

// ============================================================================
// Conduction, against closed forms
// ============================================================================

TEST(ThermalConductionTest, ALumpedBodyCoolsExponentially) {
    // T(t) = T_air + (T0 - T_air) exp(-t/tau), tau = rho c d / h. The one case
    // with an exact answer for the whole trajectory rather than for its limit.
    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{LumpedMaterial()};
    const auto exchange = MakeOpenSkyExchange(1);

    ThermalForcing forcing;
    forcing.airTemperature_K = 280.0;
    forcing.sunIrradiance_W_m2 = 0.0;

    const f64 T0 = 350.0;
    ThermalState state = MakeState(1, 12, T0);

    const ThermalMaterial& m = materials[0];
    const f64 tau = static_cast<f64>(m.density_kg_m3) * m.specificHeat_J_kgK *
                    m.thickness_m / m.convection_W_m2K;

    CpuCrankNicolsonStepper stepper;
    const f64 dt = tau / 200.0;
    for (i32 i = 0; i < 400; ++i) {  // two time constants
        stepper.Step(state, elements, materials, exchange, forcing, dt,
                     {exchange.sunVisibility});
    }

    const f64 elapsed = 400.0 * dt;
    const f64 expected = forcing.airTemperature_K +
                         (T0 - forcing.airTemperature_K) * std::exp(-elapsed / tau);
    EXPECT_NEAR(state.Surface(0), expected, 0.5)
        << "after " << elapsed << " s, tau = " << tau;
}

TEST(ThermalConductionTest, AHeldBackFaceProducesTheLinearSteadyProfile) {
    // With the back face held and the front exchanging with the air, the
    // steady profile through the slab is linear, and the surface sits where
    // the conductive and convective resistances divide the difference:
    //   (T_s - T_air) / (T_back - T_air) = R_conv / (R_conv + R_cond)
    ThermalMaterial material;
    material.conductivity_W_mK = 1.0f;
    material.density_kg_m3 = 2000.0f;
    material.specificHeat_J_kgK = 900.0f;
    material.thickness_m = 0.2f;
    material.convection_W_m2K = 10.0f;
    material.shortwaveAbsorptivity = 0.0f;
    material.longwaveEmissivity = 0.0f;
    material.interiorBoundary = InteriorBoundary::FixedTemperature;
    material.interiorTemperature_K = 293.15;

    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{material};
    const auto exchange = MakeOpenSkyExchange(1);

    ThermalForcing forcing;
    forcing.airTemperature_K = 273.15;

    ThermalState state = MakeState(1, 21, 283.15);
    CpuCrankNicolsonStepper stepper;
    for (i32 i = 0; i < 4000; ++i) {
        stepper.Step(state, elements, materials, exchange, forcing, 60.0,
                     {exchange.sunVisibility});
    }

    const f64 rCond = material.thickness_m / material.conductivity_W_mK;
    const f64 rConv = 1.0 / material.convection_W_m2K;
    const f64 expected = forcing.airTemperature_K +
                         (material.interiorTemperature_K - forcing.airTemperature_K) *
                             rConv / (rConv + rCond);
    EXPECT_NEAR(state.Surface(0), expected, 0.05);

    // And the interior really is linear: the midpoint sits halfway between the
    // surface and the back face.
    const f64 surface = state.temperature_K[0];
    const f64 middle = state.temperature_K[10];
    const f64 back = state.temperature_K[20];
    EXPECT_NEAR(middle, 0.5 * (surface + back), 0.05);
}

TEST(ThermalConductionTest, AnAdiabaticSlabReachesTheAirTemperature) {
    // Nothing else to exchange with, so it must end up where the air is --
    // the sanity check that catches a sign error in the convective term.
    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{LumpedMaterial()};
    const auto exchange = MakeOpenSkyExchange(1);

    ThermalForcing forcing;
    forcing.airTemperature_K = 300.0;

    ThermalState state = MakeState(1, 8, 250.0);
    CpuCrankNicolsonStepper stepper;
    // Ten time constants (tau = rho c d / h = 3200 s), by which the 50 K it
    // started away from the air is down to two thousandths of a kelvin.
    for (i32 i = 0; i < 3200; ++i) {
        stepper.Step(state, elements, materials, exchange, forcing, 10.0,
                     {exchange.sunVisibility});
    }
    EXPECT_NEAR(state.Surface(0), 300.0, 0.01);
}

TEST(ThermalConductionTest, TheSchemeIsSecondOrderInTime) {
    // Halving the step should quarter the error.
    //
    // Measured at steps of 800 s and 400 s, which is coarser than anything a
    // scene would use, and deliberately so: the slab has a finite Biot number
    // (hL/k = 2.5e-4 here) and sixteen nodes, and those contribute a fixed few
    // thousandths of a kelvin that no timestep removes. Below about 200 s that
    // fixed part is the same size as the time-discretisation error and the two
    // cancel -- the error passes through zero between 16 and 32 steps, which
    // makes a ratio taken there meaninglessly large. So the order is measured
    // where the time error still dominates, and the absolute accuracy is
    // checked separately in ALumpedBodyCoolsExponentially.
    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{LumpedMaterial()};
    const auto exchange = MakeOpenSkyExchange(1);

    ThermalForcing forcing;
    forcing.airTemperature_K = 280.0;

    const ThermalMaterial& m = materials[0];
    const f64 tau = static_cast<f64>(m.density_kg_m3) * m.specificHeat_J_kgK *
                    m.thickness_m / m.convection_W_m2K;
    const f64 T0 = 350.0;
    const f64 elapsed = tau;

    auto runWith = [&](const i32 steps) {
        ThermalState state = MakeState(1, 16, T0);
        CpuCrankNicolsonStepper stepper;
        const f64 dt = elapsed / steps;
        for (i32 i = 0; i < steps; ++i) {
            stepper.Step(state, elements, materials, exchange, forcing, dt,
                     {exchange.sunVisibility});
        }
        return state.Surface(0);
    };

    const f64 reference = runWith(8192);
    const f64 coarse = std::abs(runWith(4) - reference);
    const f64 fine = std::abs(runWith(8) - reference);

    EXPECT_GT(coarse, 0.0);
    // 4 is the asymptotic ratio; 4.65 is what these two steps actually give,
    // because at 800 s the next term in the expansion is still worth
    // something. The band is wide enough to hold that and narrow enough to
    // exclude first order (2) and third (8).
    EXPECT_GT(coarse / fine, 3.5) << "coarse " << coarse << ", fine " << fine;
    EXPECT_LT(coarse / fine, 5.5) << "coarse " << coarse << ", fine " << fine;
}

TEST(ThermalConductionTest, TheLumpedLimitIsApproachedFromAFiniteBiotNumber) {
    // The other half of the case above, stated so nobody later "fixes" the
    // order test by comparing against the closed form again. A slab with a
    // finite Biot number settles a little away from the lumped answer, and no
    // timestep makes that go away.
    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{LumpedMaterial()};
    const auto exchange = MakeOpenSkyExchange(1);

    ThermalForcing forcing;
    forcing.airTemperature_K = 280.0;

    const ThermalMaterial& m = materials[0];
    const f64 biot = static_cast<f64>(m.convection_W_m2K) * m.thickness_m /
                     m.conductivity_W_mK;
    EXPECT_LT(biot, 0.01) << "this material is meant to be nearly lumped";

    const f64 tau = static_cast<f64>(m.density_kg_m3) * m.specificHeat_J_kgK *
                    m.thickness_m / m.convection_W_m2K;
    const f64 T0 = 350.0;
    const f64 expected = forcing.airTemperature_K + (T0 - forcing.airTemperature_K) *
                                                        std::exp(-tau / tau);

    ThermalState state = MakeState(1, 16, T0);
    CpuCrankNicolsonStepper stepper;
    for (i32 i = 0; i < 8192; ++i) {
        stepper.Step(state, elements, materials, exchange, forcing, tau / 8192.0,
                     {exchange.sunVisibility});
    }
    // Close to the lumped answer, and not exactly it.
    EXPECT_NEAR(state.Surface(0), expected, 0.05);
}

// ============================================================================
// Radiation
// ============================================================================

TEST(ThermalConductionTest, ASurfaceUnderAColdSkySettlesBelowTheAir) {
    // Radiative cooling, which is the effect the whole clear-sky model exists
    // for. The steady state balances h(T_air - T) against eps sigma (T^4 -
    // T_sky^4), which has no closed form -- so the balance is solved here by
    // bisection and compared against what the stepper converged to.
    ThermalMaterial material = LumpedMaterial();
    material.longwaveEmissivity = 0.95f;
    material.convection_W_m2K = 10.0f;

    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{material};
    const auto exchange = MakeOpenSkyExchange(1);

    ThermalForcing forcing;
    forcing.airTemperature_K = 288.15;
    forcing.skyTemperature_K = 265.0;
    forcing.sunIrradiance_W_m2 = 0.0;

    ThermalState state = MakeState(1, 8, 288.15);
    CpuCrankNicolsonStepper stepper;
    for (i32 i = 0; i < 5000; ++i) {
        stepper.Step(state, elements, materials, exchange, forcing, 5.0,
                     {exchange.sunVisibility});
    }

    auto imbalance = [&](const f64 T) {
        const f64 sky = forcing.skyTemperature_K;
        return material.convection_W_m2K * (forcing.airTemperature_K - T) +
               material.longwaveEmissivity * kStefanBoltzmann *
                   (sky * sky * sky * sky - T * T * T * T);
    };
    f64 lo = 200.0, hi = 320.0;
    for (i32 i = 0; i < 200; ++i) {
        const f64 mid = 0.5 * (lo + hi);
        if (imbalance(mid) > 0.0) lo = mid; else hi = mid;
    }
    const f64 expected = 0.5 * (lo + hi);

    EXPECT_LT(state.Surface(0), forcing.airTemperature_K)
        << "a surface under a cold sky must sit below the air";
    EXPECT_NEAR(state.Surface(0), expected, 0.05);
}

TEST(ThermalConductionTest, AnElementSeeingOnlyItsNeighbourExchangesWithIt) {
    // Two elements, each filling the other's hemisphere entirely: no sky, so
    // the pair can only equalise. What this catches is a view factor applied
    // to the wrong element, which an open-sky test cannot see.
    Vector<ThermalElement> elements(2);
    for (auto& element : elements) {
        element.area_m2 = 1.0f;
        element.normal = glm::vec3(0.0f, 1.0f, 0.0f);
        element.materialId = 0;
    }

    ThermalMaterial material = LumpedMaterial();
    material.longwaveEmissivity = 0.95f;
    material.convection_W_m2K = 0.0f;  // radiation only
    const Vector<ThermalMaterial> materials{material};

    ExchangeGeometry exchange;
    exchange.viewFactors.rowStart = {0, 1, 2};
    exchange.viewFactors.column = {1, 0};
    exchange.viewFactors.value = {1.0f, 1.0f};
    exchange.skyFraction = {0.0f, 0.0f};
    exchange.sunVisibility = {0.0f, 0.0f};

    ThermalState state = MakeState(2, 8, 300.0);
    state.temperature_K[0] = 350.0;   // element 0, surface
    for (u32 n = 0; n < 8; ++n) state.temperature_K[n] = 350.0;

    CpuCrankNicolsonStepper stepper;
    ThermalForcing forcing;
    forcing.airTemperature_K = 300.0;

    const f64 startingSum = state.Surface(0) + state.Surface(1);
    // The pair's difference decays with tau/2, tau = rho c d / (4 eps sigma
    // T^3) = 5500 s here, so this is five of them: 50 K down to a hundredth.
    for (i32 i = 0; i < 28000; ++i) {
        stepper.Step(state, elements, materials, exchange, forcing, 1.0,
                     {exchange.sunVisibility});
    }

    EXPECT_NEAR(state.Surface(0), state.Surface(1), 0.5)
        << "an isolated pair must equalise";
    // Radiation moves heat between them without creating any: two identical
    // slabs, so the mean is conserved to within what the T^4 linearisation
    // costs over the transient.
    EXPECT_NEAR(state.Surface(0) + state.Surface(1), startingSum, 2.0);
}

TEST(ThermalConductionTest, MaterialsWithNoConductivityAreLeftAlone) {
    // Zero conductivity means "this material has no thermal properties", which
    // has to leave the temperature exactly as the config set it rather than
    // dividing by zero or drifting toward the air.
    ThermalMaterial inert;
    inert.conductivity_W_mK = 0.0f;

    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{inert};
    const auto exchange = MakeOpenSkyExchange(1);

    ThermalForcing forcing;
    forcing.airTemperature_K = 400.0;

    ThermalState state = MakeState(1, 8, 300.0);
    CpuCrankNicolsonStepper stepper;
    for (i32 i = 0; i < 100; ++i) {
        stepper.Step(state, elements, materials, exchange, forcing, 60.0,
                     {exchange.sunVisibility});
    }
    EXPECT_DOUBLE_EQ(state.Surface(0), 300.0);
}

// ============================================================================
// Short wave that did not come straight from the disc
// ============================================================================

TEST(ThermalConductionTest, DiffuseSkyWarmsAShadedSurfaceInProportionToItsSkyView) {
    // No direct sun at all -- the overcast case, which before this term had no
    // solar input whatsoever. With the radiation off, the balance is
    // alpha E_diff s = h (T - T_air) and the sky fraction is the whole of the
    // geometry: the element that sees half the sky warms by half as much.
    ThermalMaterial material = LumpedMaterial();
    material.shortwaveAbsorptivity = 0.7f;

    Vector<ThermalElement> elements = OneElement();
    elements.push_back(elements[0]);
    const Vector<ThermalMaterial> materials{material};

    ExchangeGeometry exchange = MakeOpenSkyExchange(2);
    exchange.skyFraction = {1.0f, 0.5f};

    ThermalForcing forcing;
    forcing.airTemperature_K = 290.0;
    forcing.sunIrradiance_W_m2 = 0.0;
    forcing.diffuseIrradiance_W_m2 = 400.0;

    ThermalState state = MakeState(2, 8, 290.0);
    CpuCrankNicolsonStepper stepper;
    for (i32 i = 0; i < 3000; ++i) {  // ~19 time constants
        stepper.Step(state, elements, materials, exchange, forcing, 20.0,
                     {exchange.sunVisibility});
    }

    const f64 open = forcing.airTemperature_K +
                     material.shortwaveAbsorptivity * forcing.diffuseIrradiance_W_m2 /
                         material.convection_W_m2K;
    EXPECT_NEAR(state.Surface(0), open, 0.05);
    EXPECT_NEAR(state.Surface(1), forcing.airTemperature_K + 0.5 * (open - forcing.airTemperature_K),
                0.05);
}

TEST(ThermalConductionTest, AReflectedBounceHeatsThePlateFacingAway) {
    // Two plates facing each other, half of each other's hemisphere. One is in
    // the sun; the other faces away from it and is never lit directly. Without
    // the bounce it sits at air temperature and the scene is wrong in the
    // obvious way -- a north wall that never warms.
    ThermalMaterial dark = LumpedMaterial();
    dark.shortwaveAbsorptivity = 0.7f;
    ThermalMaterial bright = LumpedMaterial();
    bright.shortwaveAbsorptivity = 0.2f;  // reflects 0.8
    const Vector<ThermalMaterial> materials{dark, bright};

    Vector<ThermalElement> elements = OneElement();
    elements.push_back(elements[0]);
    elements[0].normal = glm::vec3(0.0f, -1.0f, 0.0f);  // faces away from the sun
    elements[1].normal = glm::vec3(0.0f, 1.0f, 0.0f);
    elements[1].materialId = 1;

    ExchangeGeometry exchange;
    exchange.viewFactors.rowStart = {0, 1, 2};
    exchange.viewFactors.column = {1, 0};
    exchange.viewFactors.value = {0.5f, 0.5f};
    exchange.skyFraction = {0.5f, 0.5f};
    exchange.sunVisibility = {1.0f, 1.0f};

    SunVisibilityTable table;
    table.sampleTime_h = {0.0};
    table.visibility = {1.0f, 1.0f};
    table.sampleDirection = {glm::vec3(0.0f, 1.0f, 0.0f)};
    BakeShortwaveGains(exchange, elements, materials, table);

    ASSERT_EQ(table.reflectedGain.size(), 2u);
    EXPECT_NEAR(table.reflectedGain[0], 0.5f * 0.8f, 1e-6f);
    EXPECT_NEAR(table.reflectedGain[1], 0.0f, 1e-6f);

    ThermalForcing forcing;
    forcing.airTemperature_K = 290.0;
    forcing.sunIrradiance_W_m2 = 1000.0;
    forcing.sunDirection = glm::vec3(0.0f, 1.0f, 0.0f);

    auto runTo = [&](const std::span<const f32> reflected) {
        ThermalState state = MakeState(2, 8, 290.0);
        CpuCrankNicolsonStepper stepper;
        for (i32 i = 0; i < 3000; ++i) {
            stepper.Step(state, elements, materials, exchange, forcing, 20.0,
                         {exchange.sunVisibility, reflected, {}});
        }
        return state;
    };

    const ThermalState lit = runTo(table.reflectedGain);
    EXPECT_NEAR(lit.Surface(0),
                forcing.airTemperature_K + 0.7 * 1000.0 * 0.4 / dark.convection_W_m2K, 0.05);
    EXPECT_NEAR(lit.Surface(1),
                forcing.airTemperature_K + 0.2 * 1000.0 / bright.convection_W_m2K, 0.05);

    // And the control: with no gain the shaded plate has nothing at all.
    const ThermalState unlit = runTo({});
    EXPECT_NEAR(unlit.Surface(0), forcing.airTemperature_K, 0.05);
}

TEST(ThermalConductionTest, TheReflectedGainInterpolatesWithTheSunColumns) {
    // A batch step lands between two sun samples. The bounce has to be read on
    // the same indices as the visibility it was baked from -- a gain sampled
    // from one hour with a shadow mask from another is a surface lit by a sun
    // that is in two places.
    ThermalMaterial material = LumpedMaterial();
    material.shortwaveAbsorptivity = 0.7f;

    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{material};
    const auto exchange = MakeOpenSkyExchange(1);

    ThermalForcing forcing;
    forcing.airTemperature_K = 290.0;
    forcing.sunIrradiance_W_m2 = 1000.0;
    forcing.sunDirection = glm::vec3(0.0f, -1.0f, 0.0f);  // the disc misses it

    SunVisibilityTable table;
    table.sampleTime_h = {0.0, 2.0};
    table.visibility = {1.0f, 1.0f};
    table.reflectedGain = {0.0f, 0.4f};

    ThermalBatchStep step;
    step.forcing = forcing;
    step.dt_s = 30.0;
    step.sunSampleA = 0;
    step.sunSampleB = 1;
    step.sunBlend = 0.5;

    CpuCrankNicolsonStepper stepper;
    ThermalState blended = MakeState(1, 8, 290.0);
    const Vector<ThermalBatchStep> batch(400, step);
    stepper.StepMany(blended, elements, materials, exchange, table, batch);

    // Half of the second column's gain, applied directly, must land in the
    // same place.
    ThermalState direct = MakeState(1, 8, 290.0);
    const Vector<f32> halfGain{0.2f};
    for (i32 i = 0; i < 400; ++i) {
        stepper.Step(direct, elements, materials, exchange, forcing, 30.0,
                     {table.visibility, halfGain, {}});
    }
    EXPECT_DOUBLE_EQ(blended.Surface(0), direct.Surface(0));
}

// ============================================================================
// Evaporation
// ============================================================================

TEST(ThermalConductionTest, AWetSurfaceSettlesBelowTheAirItSitsIn) {
    // The reason a lawn is cooler than the pavement beside it. With the
    // radiation and the sun off, the balance is
    // h (T_air - T) = f_wet (h/c_p) L_v (q_sat(T) - RH q_sat(T_air)),
    // and it has no closed form -- so the expected value is bisected out of
    // the same balance, written independently.
    ThermalMaterial material = LumpedMaterial();
    material.wetnessFactor = 1.0f;

    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{material};
    const auto exchange = MakeOpenSkyExchange(1);

    ThermalForcing forcing;
    forcing.airTemperature_K = 300.0;
    forcing.relativeHumidity = 30.0;

    ThermalState state = MakeState(1, 8, 300.0);
    CpuCrankNicolsonStepper stepper;
    for (i32 i = 0; i < 4000; ++i) {
        stepper.Step(state, elements, materials, exchange, forcing, 10.0,
                     {exchange.sunVisibility});
    }

    const f64 h = material.convection_W_m2K;
    const f64 coefficient = (h / 1005.0) * 2.45e6;
    const f64 qAir = SaturationHumidity(forcing.airTemperature_K);
    const f64 expected = BalanceTemperature(
        [&](const f64 T) {
            return h * (forcing.airTemperature_K - T) -
                   coefficient * (SaturationHumidity(T) - 0.30 * qAir);
        },
        250.0, forcing.airTemperature_K);

    EXPECT_NEAR(state.Surface(0), expected, 0.1);
    EXPECT_LT(state.Surface(0), forcing.airTemperature_K - 5.0)
        << "evaporation into dry air has to cool the surface, not warm it";
}

TEST(ThermalConductionTest, ADrySurfaceIgnoresHumidityEntirely) {
    // A wetness of zero is not "a little evaporation": it is the balance as it
    // was before the term existed, to the last bit.
    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{LumpedMaterial()};
    const auto exchange = MakeOpenSkyExchange(1);

    auto runAtHumidity = [&](const f64 relativeHumidity) {
        ThermalForcing forcing;
        forcing.airTemperature_K = 300.0;
        forcing.relativeHumidity = relativeHumidity;

        ThermalState state = MakeState(1, 8, 320.0);
        CpuCrankNicolsonStepper stepper;
        for (i32 i = 0; i < 200; ++i) {
            stepper.Step(state, elements, materials, exchange, forcing, 30.0,
                         {exchange.sunVisibility});
        }
        return state.Surface(0);
    };

    EXPECT_DOUBLE_EQ(runAtHumidity(10.0), runAtHumidity(90.0));
}

TEST(ThermalConductionTest, AWetSurfaceRespondsFasterThanADryOne) {
    // Evaporation is a third way to shed a departure from equilibrium, so it
    // belongs in the time constant the solver warns against -- otherwise the
    // warning passes a timestep that the wettest surface in the scene cannot
    // resolve.
    const auto elements = OneElement();
    ThermalMaterial wet = LumpedMaterial();
    wet.wetnessFactor = 0.5f;

    const Vector<ThermalMaterial> dryMaterials{LumpedMaterial()};
    const Vector<ThermalMaterial> wetMaterials{wet};

    const f64 dry =
        CpuCrankNicolsonStepper::ShortestTimeConstantSeconds(elements, dryMaterials, 300.0);
    const f64 damp =
        CpuCrankNicolsonStepper::ShortestTimeConstantSeconds(elements, wetMaterials, 300.0);
    EXPECT_LT(damp, dry);
    EXPECT_GT(damp, 0.2 * dry) << "half-wet should not be an order of magnitude faster";
}

TEST(ThermalConductionTest, TheTimeConstantIsWhatTheSchemeIsJudgedAgainst) {
    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{LumpedMaterial()};

    const ThermalMaterial& m = materials[0];
    const f64 expected = static_cast<f64>(m.density_kg_m3) * m.specificHeat_J_kgK *
                         m.thickness_m / m.convection_W_m2K;  // emissivity is 0 here

    EXPECT_NEAR(CpuCrankNicolsonStepper::ShortestTimeConstantSeconds(elements, materials, 300.0),
                expected, expected * 1e-9);
}

// ============================================================================
// What is behind the surface
// ============================================================================

namespace {

/// A slab that conducts poorly enough to hold a gradient through its depth,
/// which is what the internal-source case is measuring.
ThermalMaterial GradientMaterial() {
    ThermalMaterial material;
    material.conductivity_W_mK = 1.0f;
    material.density_kg_m3 = 2000.0f;
    material.specificHeat_J_kgK = 900.0f;
    material.thickness_m = 0.10f;
    material.convection_W_m2K = 10.0f;
    material.shortwaveAbsorptivity = 0.0f;
    material.longwaveEmissivity = 0.0f;  // convection only, so the level is simple
    material.interiorBoundary = InteriorBoundary::Adiabatic;
    return material;
}

/// Step to steady state with a long timestep, the way the timeline's own
/// relaxation does.
void RelaxTo(ThermalState& state, const Vector<ThermalElement>& elements,
             const Vector<ThermalMaterial>& materials, const ExchangeGeometry& exchange,
             const ThermalForcing& forcing) {
    CpuCrankNicolsonStepper stepper;
    for (i32 i = 0; i < 400; ++i) {
        stepper.Step(state, elements, materials, exchange, forcing, 3600.0,
                     {exchange.sunVisibility});
    }
}

}  // namespace

TEST(ThermalInteriorTest, AnInternalSourceGivesTheLinearProfileFouriersLawAsksFor) {
    // In steady state every watt entering the back has to leave through the
    // front, so the flux through the slab is uniform and Fourier's law fixes
    // the gradient outright:
    //
    //     T_back - T_front = q d / k
    //
    // independently of what the front face is exchanging with -- which is what
    // makes it a closed form rather than a fit.
    const auto elements = OneElement();
    ThermalMaterial material = GradientMaterial();
    material.internalHeat_W_m2 = 100.0f;
    const Vector<ThermalMaterial> materials{material};
    const auto exchange = MakeOpenSkyExchange(1);

    ThermalForcing forcing;
    forcing.airTemperature_K = 290.0;
    forcing.sunIrradiance_W_m2 = 0.0;

    ThermalState state = MakeState(1, 21, 290.0);
    RelaxTo(state, elements, materials, exchange, forcing);

    const u32 last = state.nodeCount - 1;
    const f64 expected = static_cast<f64>(material.internalHeat_W_m2) *
                         material.thickness_m / material.conductivity_W_mK;
    EXPECT_NEAR(state.temperature_K[last] - state.Surface(0), expected, 1e-3)
        << "a uniform flux through a slab is a linear profile";

    // And the level: convection alone carries the same 100 W/m^2 away.
    EXPECT_NEAR(state.Surface(0),
                forcing.airTemperature_K + 100.0 / material.convection_W_m2K, 1e-3);
}

TEST(ThermalInteriorTest, NoInternalSourceIsTheSlabThatWasThereBefore) {
    // The default has to leave every existing scene where it was, to the bit.
    const auto elements = OneElement();
    const Vector<ThermalMaterial> silent{GradientMaterial()};
    ThermalMaterial zeroed = GradientMaterial();
    zeroed.internalHeat_W_m2 = 0.0f;
    const Vector<ThermalMaterial> explicitZero{zeroed};
    const auto exchange = MakeOpenSkyExchange(1);

    ThermalForcing forcing;
    forcing.airTemperature_K = 290.0;

    ThermalState a = MakeState(1, 21, 300.0);
    ThermalState b = MakeState(1, 21, 300.0);
    RelaxTo(a, elements, silent, exchange, forcing);
    RelaxTo(b, elements, explicitZero, exchange, forcing);

    for (u32 i = 0; i < a.nodeCount; ++i) {
        EXPECT_DOUBLE_EQ(a.temperature_K[i], b.temperature_K[i]) << "node " << i;
    }
}

TEST(ThermalInteriorTest, AnAmbientBackFaceSheddsHeatAnInsulatedOneCannot) {
    // A panel over a bay against a panel whose back is insulated. With a
    // source behind both, the one that can lose heat backwards runs cooler --
    // and the split is what a two-sided thin plate has that a wall does not.
    const auto elements = OneElement();

    ThermalMaterial insulated = GradientMaterial();
    insulated.internalHeat_W_m2 = 100.0f;

    ThermalMaterial open = insulated;
    open.interiorBoundary = InteriorBoundary::AmbientInterior;
    open.interiorTemperature_K = 290.0f;
    open.interiorConvection_W_m2K = 10.0f;

    const auto exchange = MakeOpenSkyExchange(1);
    ThermalForcing forcing;
    forcing.airTemperature_K = 290.0;

    ThermalState hot = MakeState(1, 21, 290.0);
    ThermalState split = MakeState(1, 21, 290.0);
    RelaxTo(hot, elements, Vector<ThermalMaterial>{insulated}, exchange, forcing);
    RelaxTo(split, elements, Vector<ThermalMaterial>{open}, exchange, forcing);

    EXPECT_LT(split.Surface(0), hot.Surface(0))
        << "a face that can lose heat backwards leaves less of it to the front";
    EXPECT_GT(split.Surface(0), forcing.airTemperature_K);

    // The source enters AT the back node, so the two paths out of it are not
    // symmetric: backwards it meets h_b alone, forwards it meets the slab and
    // the front film in series. With g = k/d,
    //
    //     T_back  = T_air + q / (h_b + g h_f / (g + h_f))
    //     T_front = T_air + (T_back - T_air) g / (g + h_f)
    //
    // which for 100 W/m^2 through k/d = 10 between two 10 W/(m^2 K) films is
    // 296.67 K at the back and 293.33 K at the front -- a third of the rise an
    // insulated back would have given.
    const f64 g = static_cast<f64>(open.conductivity_W_mK) / open.thickness_m;
    const f64 hFront = static_cast<f64>(open.convection_W_m2K);
    const f64 hBack = static_cast<f64>(open.interiorConvection_W_m2K);
    const f64 series = g * hFront / (g + hFront);
    const f64 backRise = static_cast<f64>(open.internalHeat_W_m2) / (hBack + series);
    EXPECT_NEAR(split.temperature_K[split.nodeCount - 1],
                forcing.airTemperature_K + backRise, 1e-3);
    EXPECT_NEAR(split.Surface(0),
                forcing.airTemperature_K + backRise * g / (g + hFront), 1e-3);
}

TEST(ThermalInteriorTest, AnAmbientBackFaceAtEquilibriumMovesNothing) {
    // Air, sky and interior all at one temperature: nothing anywhere has a
    // gradient to drive it, so a boundary that added a spurious flux would
    // show up as a slab that drifts off that temperature.
    const auto elements = OneElement();
    ThermalMaterial material = GradientMaterial();
    material.interiorBoundary = InteriorBoundary::AmbientInterior;
    material.interiorTemperature_K = 290.0f;
    material.longwaveEmissivity = 0.9f;
    const Vector<ThermalMaterial> materials{material};

    const auto exchange = MakeOpenSkyExchange(1);
    ThermalForcing forcing;
    forcing.airTemperature_K = 290.0;
    forcing.skyTemperature_K = 290.0;

    ThermalState state = MakeState(1, 21, 290.0);
    RelaxTo(state, elements, materials, exchange, forcing);

    for (u32 i = 0; i < state.nodeCount; ++i) {
        EXPECT_NEAR(state.temperature_K[i], 290.0, 1e-6) << "node " << i;
    }
}
