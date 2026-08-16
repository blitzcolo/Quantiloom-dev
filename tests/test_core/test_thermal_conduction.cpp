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

#include "thermal/ThermalSolver.hpp"  // MakeOpenSkyExchange

#include <cmath>

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
                     exchange.sunVisibility);
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
                     exchange.sunVisibility);
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
                     exchange.sunVisibility);
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
                     exchange.sunVisibility);
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
                     exchange.sunVisibility);
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
                     exchange.sunVisibility);
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
                     exchange.sunVisibility);
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
                     exchange.sunVisibility);
    }
    EXPECT_DOUBLE_EQ(state.Surface(0), 300.0);
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
