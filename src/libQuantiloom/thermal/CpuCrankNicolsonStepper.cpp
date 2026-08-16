/**
 * @file CpuCrankNicolsonStepper.cpp
 * @brief One-dimensional conduction through each element, on the CPU
 */

#include "thermal/CpuCrankNicolsonStepper.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace quantiloom::thermal {

namespace {

constexpr f64 kStefanBoltzmann = 5.670374419e-8;  // W m^-2 K^-4

// Air at the surface, for the latent term. One atmosphere and dry-air values:
// the balance is not sensitive to either at the precision the rest of it has.
constexpr f64 kAirPressure_Pa = 101325.0;
constexpr f64 kAirSpecificHeat_J_kgK = 1005.0;
constexpr f64 kLatentHeatVaporisation_J_kg = 2.45e6;

/// Saturation vapour pressure, Pa, by Magnus-Tetens. Good to a few tenths of a
/// percent between -40 and +50 C, which is the whole range a surface balance
/// visits outside a fire.
///
/// The clamp is not about accuracy, it is about not making things worse: the
/// state clamp lets a diverging element reach 5000 K, and Magnus at 5000 K
/// returns a vapour pressure of 1e10 Pa, which drives the mixing-ratio
/// denominator negative and turns the whole latent term into a NaN. A NaN
/// spreads through the radiative coupling to every element that can see the
/// one that produced it, so a single bad element takes the scene with it --
/// the divergence should stay visible as a wrong temperature, not become a
/// blank image.
f64 SaturationVapourPressure(const f64 temperature_K) {
    const f64 tC = std::clamp(temperature_K - 273.15, -80.0, 80.0);
    return 610.94 * std::exp(17.625 * tC / (tC + 243.04));
}

/// Saturation specific humidity, kg/kg.
f64 SaturationHumidity(const f64 temperature_K) {
    const f64 e = SaturationVapourPressure(temperature_K);
    return 0.622 * e / (kAirPressure_Pa - 0.378 * e);
}

/// The same, with its slope in temperature. The slope is what makes
/// evaporation stiff: saturation humidity roughly doubles every eleven
/// degrees, so a wet surface's latent admittance at 300 K is several times its
/// radiative one.
void SaturationHumidity(const f64 temperature_K, f64& q, f64& dq_dT) {
    // Same clamp as the vapour pressure, and for the same reason.
    const f64 tC = std::clamp(temperature_K - 273.15, -80.0, 80.0);
    const f64 denominator = tC + 243.04;
    const f64 e = SaturationVapourPressure(temperature_K);
    const f64 de_dT = e * (17.625 * 243.04) / (denominator * denominator);

    const f64 mixed = kAirPressure_Pa - 0.378 * e;
    q = 0.622 * e / mixed;
    dq_dT = 0.622 * kAirPressure_Pa * de_dT / (mixed * mixed);
}

/// The coefficient the latent flux and its slope in temperature share:
/// f_wet (h / c_p) L_v. Lewis analogy -- the same eddies that carry heat carry
/// vapour, so the mass transfer coefficient is the convective one over the
/// heat capacity of air, and nobody has to supply a second one.
f64 LatentCoefficient(const f64 wetnessFactor, const f64 convection_W_m2K) {
    return wetnessFactor * (convection_W_m2K / kAirSpecificHeat_J_kgK) *
           kLatentHeatVaporisation_J_kg;
}

/// Solve a tridiagonal system in place by Thomas elimination.
///
/// Exact rather than iterative, and O(n) rather than O(n^3): the matrix a
/// slab produces has three diagonals and nothing else, and a general solver
/// would spend its time proving that.
///
/// @param lower  sub-diagonal, lower[0] unused
/// @param diag   main diagonal, overwritten
/// @param upper  super-diagonal, upper[n-1] unused, overwritten
/// @param rhs    right-hand side, overwritten with the solution
void SolveTridiagonal(Vector<f64>& lower, Vector<f64>& diag, Vector<f64>& upper,
                      Vector<f64>& rhs) {
    const usize n = diag.size();
    if (n == 0) return;

    for (usize i = 1; i < n; ++i) {
        const f64 factor = lower[i] / diag[i - 1];
        diag[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }
    rhs[n - 1] /= diag[n - 1];
    for (usize i = n - 1; i-- > 0;) {
        rhs[i] = (rhs[i] - upper[i] * rhs[i + 1]) / diag[i];
    }
}

}  // namespace

void CpuCrankNicolsonStepper::Step(ThermalState& state, const Vector<ThermalElement>& elements,
                                   const Vector<ThermalMaterial>& materials,
                                   const ExchangeGeometry& exchange,
                                   const ThermalForcing& forcing, const f64 dt_s,
                                   const ShortwaveSample& shortwave) {
    const std::span<const f32> sunVisibility = shortwave.sunVisibility;
    const u32 nodes = state.nodeCount;
    if (nodes < 2 || elements.empty() || dt_s <= 0.0) {
        return;
    }

    // The radiative coupling reads every element's temperature, so it reads
    // the state as it was at the start of the step rather than as neighbours
    // update it. Surface values only: the interior nodes are not visible to
    // anything outside their own slab.
    Vector<f64> surfacePrevious(elements.size());
    for (usize i = 0; i < elements.size(); ++i) {
        surfacePrevious[i] = state.Surface(i);
    }

    Vector<f64> lower(nodes), diag(nodes), upper(nodes), rhs(nodes);

    for (usize e = 0; e < elements.size(); ++e) {
        const ThermalElement& element = elements[e];
        if (element.materialId >= materials.size()) continue;
        const ThermalMaterial& material = materials[element.materialId];
        if (!material.ParticipatesInSolve()) continue;

        const f64 k = material.conductivity_W_mK;
        const f64 rhoC = static_cast<f64>(material.density_kg_m3) *
                         material.specificHeat_J_kgK;
        const f64 dx = static_cast<f64>(material.thickness_m) / static_cast<f64>(nodes - 1);
        if (!(rhoC > 0.0) || !(dx > 0.0)) continue;

        // Fourier number: how far heat diffuses in one step, in node spacings.
        // Crank-Nicolson is stable at any value of it; this is only the
        // coefficient the scheme is written in terms of.
        const f64 r = (k * dt_s) / (rhoC * dx * dx);

        f64* T = &state.temperature_K[e * nodes];

        // ----------------------------------------------------------------
        // The exposed face
        // ----------------------------------------------------------------
        // Everything the outside does to this element, as one flux. The
        // radiative part is linearised against the previous temperatures, so
        // this is a number rather than a term in the unknown.
        const f64 emissivity = static_cast<f64>(material.longwaveEmissivity);
        f64 surfaceFlux_W_m2 = 0.0;

        const f64 absorptivity = material.shortwaveAbsorptivity;

        // Sun, direct. cos(theta) against the element's own normal, times the
        // precomputed visibility -- which is what carries the shadow.
        if (forcing.sunIrradiance_W_m2 > 0.0 && e < sunVisibility.size()) {
            const f64 cosTheta = static_cast<f64>(
                glm::dot(element.normal, glm::normalize(forcing.sunDirection)));
            if (cosTheta > 0.0) {
                surfaceFlux_W_m2 += absorptivity *
                                    forcing.sunIrradiance_W_m2 * cosTheta *
                                    static_cast<f64>(sunVisibility[e]);
            }
        }

        // Sun, off a neighbour. This is where a north wall gets its afternoon:
        // it is never in the sun, and the road in front of it is. The gain is
        // a gather over the whole hemisphere, so it carries no cos(theta) of
        // its own -- that was applied to the surfaces doing the reflecting,
        // when it was baked.
        if (forcing.sunIrradiance_W_m2 > 0.0 && e < shortwave.reflectedGain.size()) {
            surfaceFlux_W_m2 += absorptivity * forcing.sunIrradiance_W_m2 *
                                static_cast<f64>(shortwave.reflectedGain[e]);
        }

        // Sky, diffuse. For an isotropic dome the geometric factor is the
        // element's own sky fraction, plus whatever one bounce adds; that sum
        // is the baked gain, and the bare sky fraction is what it degrades to.
        // Under overcast this term is the entire solar input, which is why a
        // run without it has a cloudy day with no sunlight in it at all.
        if (forcing.diffuseIrradiance_W_m2 > 0.0) {
            f64 diffuseGain = 0.0;
            if (e < shortwave.diffuseGain.size()) {
                diffuseGain = static_cast<f64>(shortwave.diffuseGain[e]);
            } else if (e < exchange.skyFraction.size()) {
                diffuseGain = static_cast<f64>(exchange.skyFraction[e]);
            }
            surfaceFlux_W_m2 += absorptivity * forcing.diffuseIrradiance_W_m2 * diffuseGain;
        }

        // Long wave: what the hemisphere sends back, minus what this element
        // radiates. The sky fills whatever the other elements do not, which is
        // what makes a surface under an overhang cool more slowly than one
        // under open sky.
        f64 incoming = 0.0;
        if (e + 1 < exchange.viewFactors.rowStart.size()) {
            const u32 begin = exchange.viewFactors.rowStart[e];
            const u32 end = exchange.viewFactors.rowStart[e + 1];
            for (u32 n = begin; n < end; ++n) {
                const u32 j = exchange.viewFactors.column[n];
                if (j < surfacePrevious.size()) {
                    const f64 Tj = surfacePrevious[j];
                    incoming += exchange.viewFactors.value[n] * Tj * Tj * Tj * Tj;
                }
            }
        }
        if (e < exchange.skyFraction.size()) {
            const f64 Tsky = forcing.skyTemperature_K;
            incoming += static_cast<f64>(exchange.skyFraction[e]) * Tsky * Tsky * Tsky * Tsky;
        }
        const f64 Ti = surfacePrevious[e];
        surfaceFlux_W_m2 += emissivity * kStefanBoltzmann * (incoming - Ti * Ti * Ti * Ti);

        // Convection, which is linear in the unknown and therefore goes into
        // the matrix rather than into the flux.
        const f64 h = material.convection_W_m2K;

        // Evaporation. Non-linear in the unknown like the radiation, but with
        // several times its slope, so this one goes into the matrix as well:
        // linearised about the previous surface temperature and split
        // half-and-half the way Crank-Nicolson splits everything else. A wet
        // element under dry air can shed hundreds of watts per square metre,
        // and leaving that explicit oscillates at a minute per step.
        f64 latentAdmittance_W_m2K = 0.0;
        f64 latentFlux_W_m2 = 0.0;
        if (material.wetnessFactor > 0.0f) {
            f64 qSurface = 0.0;
            f64 dq_dT = 0.0;
            SaturationHumidity(Ti, qSurface, dq_dT);
            const f64 qAir = SaturationHumidity(forcing.airTemperature_K);

            const f64 humidity = std::clamp(forcing.relativeHumidity, 0.0, 100.0) / 100.0;
            const f64 coefficient =
                LatentCoefficient(static_cast<f64>(material.wetnessFactor), h);
            latentFlux_W_m2 = coefficient * (qSurface - humidity * qAir);
            latentAdmittance_W_m2K = coefficient * dq_dT;
        }

        // Crank-Nicolson on the half-cell at the face. The half cell has
        // capacity rho c dx/2 and exchanges with node 1 by conduction and with
        // the outside by h, by evaporation, and by the flux above.
        const f64 halfCell = rhoC * dx / (2.0 * dt_s);
        lower[0] = 0.0;
        diag[0] = halfCell + 0.5 * (k / dx + h + latentAdmittance_W_m2K);
        upper[0] = -0.5 * (k / dx);
        rhs[0] = halfCell * T[0] - 0.5 * (k / dx) * (T[0] - T[1]) +
                 0.5 * h * (2.0 * forcing.airTemperature_K - T[0]) + surfaceFlux_W_m2 -
                 latentFlux_W_m2 + 0.5 * latentAdmittance_W_m2K * T[0];

        // ----------------------------------------------------------------
        // The interior
        // ----------------------------------------------------------------
        for (u32 i = 1; i + 1 < nodes; ++i) {
            lower[i] = -0.5 * r;
            diag[i] = 1.0 + r;
            upper[i] = -0.5 * r;
            rhs[i] = T[i] + 0.5 * r * (T[i - 1] - 2.0 * T[i] + T[i + 1]);
        }

        // ----------------------------------------------------------------
        // The back face
        // ----------------------------------------------------------------
        const u32 last = nodes - 1;
        if (material.interiorBoundary == InteriorBoundary::FixedTemperature) {
            // A Dirichlet row: whatever is behind this surface holds it there.
            lower[last] = 0.0;
            diag[last] = 1.0;
            upper[last] = 0.0;
            rhs[last] = material.interiorTemperature_K;
        } else {
            // Adiabatic: the mirror condition, a half cell exchanging only
            // with the node in front of it.
            lower[last] = -0.5 * (k / dx);
            diag[last] = halfCell + 0.5 * (k / dx);
            upper[last] = 0.0;
            rhs[last] = halfCell * T[last] - 0.5 * (k / dx) * (T[last] - T[last - 1]);
        }

        SolveTridiagonal(lower, diag, upper, rhs);

        for (u32 i = 0; i < nodes; ++i) {
            // A temperature outside this range is a solver failure rather than
            // a cold night, and letting it through would put a NaN into the
            // render two steps later.
            T[i] = std::clamp(rhs[i], 1.0, 5000.0);
        }
    }
}

f64 CpuCrankNicolsonStepper::ShortestTimeConstantSeconds(
    const Vector<ThermalElement>& elements, const Vector<ThermalMaterial>& materials,
    const f64 referenceTemperature_K) {
    f64 shortest = std::numeric_limits<f64>::infinity();

    for (usize e = 0; e < elements.size(); ++e) {
        const u32 id = elements[e].materialId;
        if (id >= materials.size()) continue;
        const ThermalMaterial& material = materials[id];
        if (!material.ParticipatesInSolve()) continue;

        const f64 capacity = static_cast<f64>(material.density_kg_m3) *
                             material.specificHeat_J_kgK * material.thickness_m;
        const f64 eps = static_cast<f64>(material.longwaveEmissivity);
        const f64 radiative = 4.0 * eps * kStefanBoltzmann * referenceTemperature_K *
                              referenceTemperature_K * referenceTemperature_K;

        // A wet surface responds faster than a dry one of the same mass: the
        // latent path is another way for it to shed a departure from
        // equilibrium, and at 300 K it is the largest of the three.
        f64 latent = 0.0;
        if (material.wetnessFactor > 0.0f) {
            f64 q = 0.0;
            f64 dq_dT = 0.0;
            SaturationHumidity(referenceTemperature_K, q, dq_dT);
            latent = LatentCoefficient(static_cast<f64>(material.wetnessFactor),
                                       material.convection_W_m2K) *
                     dq_dT;
        }

        const f64 loss = material.convection_W_m2K + radiative + latent;
        if (loss > 0.0) {
            shortest = std::min(shortest, capacity / loss);
        }
    }
    return shortest;
}

}  // namespace quantiloom::thermal
