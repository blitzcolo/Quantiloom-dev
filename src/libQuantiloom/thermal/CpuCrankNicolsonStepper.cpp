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
                                   std::span<const f32> sunVisibility) {
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

        // Sun. cos(theta) against the element's own normal, times the
        // precomputed visibility -- which is what carries the shadow.
        if (forcing.sunIrradiance_W_m2 > 0.0 && e < sunVisibility.size()) {
            const f64 cosTheta = static_cast<f64>(
                glm::dot(element.normal, glm::normalize(forcing.sunDirection)));
            if (cosTheta > 0.0) {
                surfaceFlux_W_m2 += material.shortwaveAbsorptivity *
                                    forcing.sunIrradiance_W_m2 * cosTheta *
                                    static_cast<f64>(sunVisibility[e]);
            }
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

        // Crank-Nicolson on the half-cell at the face. The half cell has
        // capacity rho c dx/2 and exchanges with node 1 by conduction and with
        // the outside by h and the flux above.
        const f64 halfCell = rhoC * dx / (2.0 * dt_s);
        lower[0] = 0.0;
        diag[0] = halfCell + 0.5 * (k / dx + h);
        upper[0] = -0.5 * (k / dx);
        rhs[0] = halfCell * T[0] - 0.5 * (k / dx) * (T[0] - T[1]) +
                 0.5 * h * (2.0 * forcing.airTemperature_K - T[0]) + surfaceFlux_W_m2;

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
        const f64 loss = material.convection_W_m2K + radiative;
        if (loss > 0.0) {
            shortest = std::min(shortest, capacity / loss);
        }
    }
    return shortest;
}

}  // namespace quantiloom::thermal
