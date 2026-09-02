/**
 * @file CpuCrankNicolsonStepper.cpp
 * @brief One-dimensional conduction through each element, on the CPU
 */

#include "thermal/CpuCrankNicolsonStepper.hpp"

#include "thermal/ThermalMesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

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

/// Solve a tridiagonal system in place by Thomas elimination, for any number of
/// right-hand sides.
///
/// Exact rather than iterative, and O(n) rather than O(n^3): the matrix a
/// slab produces has three diagonals and nothing else, and a general solver
/// would spend its time proving that.
///
/// Several right-hand sides because every tangent this solver carries -- dT/dv
/// against a column of the sun's history, dT/dp against a material property --
/// obeys the same discrete operator as the temperature and differs only in what
/// drives it. Eliminating once and applying to all of them is the whole reason
/// a tangent costs a fraction of a second solve rather than a whole one, and it
/// is what keeps the cost of the Nth tangent a back-substitution rather than
/// another elimination.
///
/// @param lower  sub-diagonal, lower[0] unused
/// @param diag   main diagonal, overwritten
/// @param upper  super-diagonal, upper[n-1] unused, overwritten
/// @param rhs    the right-hand sides, each of length n, each overwritten with
///               its own solution. The first is the temperature's.
void FactorTridiagonal(const Vector<f64>& lower, Vector<f64>& diag,
                       const Vector<f64>& upper) {
    for (usize i = 1; i < diag.size(); ++i) {
        diag[i] -= (lower[i] / diag[i - 1]) * upper[i - 1];
    }
}

/// Apply a factorisation to right-hand sides. The elimination factor is
/// recomputed from the already-modified diagonal rather than stored, which is
/// the same number the factorisation used -- lower[i] / diag[i-1] with diag
/// as it stands after the sweep reached i-1.
///
/// Separate from the factorisation because a tangent in a MATERIAL parameter
/// needs the temperature this step produced, not the one it started from: the
/// matrix itself moves with k and rho c, and what that contributes to the
/// tangent is (dA/dp) T^{n+1}. So those rows can only be built after the
/// temperature has been back-substituted, and they then reuse the same
/// factorisation rather than paying for a second one.
void SolveFactored(const Vector<f64>& lower, const Vector<f64>& diag,
                   const Vector<f64>& upper, const std::span<const std::span<f64>> rhs) {
    const usize n = diag.size();
    if (n == 0 || rhs.empty()) return;

    for (usize i = 1; i < n; ++i) {
        const f64 factor = lower[i] / diag[i - 1];
        for (const std::span<f64>& b : rhs) b[i] -= factor * b[i - 1];
    }
    for (const std::span<f64>& b : rhs) b[n - 1] /= diag[n - 1];
    for (usize i = n - 1; i-- > 0;) {
        for (const std::span<f64>& b : rhs) {
            b[i] = (b[i] - upper[i] * b[i + 1]) / diag[i];
        }
    }
}

void SolveTridiagonal(Vector<f64>& lower, Vector<f64>& diag, Vector<f64>& upper,
                      const std::span<const std::span<f64>> rhs) {
    if (diag.empty()) return;
    FactorTridiagonal(lower, diag, upper);
    SolveFactored(lower, diag, upper, rhs);
}

/// Gravity, for the buoyancy in the Richardson number below.
constexpr f64 kGravity_m_s2 = 9.80665;

/// The wind speed the Richardson number is evaluated at. A bulk Richardson
/// number divides by U^2, so a dead calm would make it infinite; half a metre
/// per second is the light air a station reports as zero.
constexpr f64 kMinWind_m_s = 0.5;

/// What a strongly stable layer still exchanges. The damping below has no
/// bound of its own, and an h of zero would leave a surface radiating to the
/// sky with nothing at all drawing heat back into it -- which is colder than
/// any night.
constexpr f64 kMinStableConvection_W_m2K = 1.0;

/// The convective coefficient for one element at one instant, and the
/// admittance the matrix sees.
///
/// Priority: what the forcing states outright, then the law, then the
/// material's own constant. The forcing's column wins because a file that
/// carries a measured h is saying something the correlations are estimating.
///
/// The admittance is not h. The convective flux is q = h (T_air - T_s), and
/// under the stability law h itself depends on T_s, so
///
///     dq/dT_s = -h - (T_s - T_air) dh/dT_s
///
/// and the matrix wants that whole slope rather than its first term. Both
/// temperature-dependent branches close in one line:
///
///   free convection, h = C |dT|^(1/3):   the second term is h/3, so the
///                                        admittance is 4h/3
///   stable damping,  h = h_f/(1 + d Ri): the second term cancels down to
///                                        h/(1 + d Ri), which is below h
///
/// Under the constant and wind laws dh/dT_s is zero and the admittance is h,
/// which is what keeps those two bit-identical to a run from before this
/// existed.
///
/// The latent term derives its own coefficient from h, and its dependence on
/// T_s through h is NOT differentiated here -- the same order of
/// approximation as everything else linearised about the previous step.
void ConvectionAt(const ConvectionLaw& law, const ThermalForcing& forcing,
                  const ThermalMaterial& material, const f64 surfaceTemperature_K,
                  f64& h, f64& admittance_W_m2K) {
    if (forcing.convection_W_m2K > 0.0) {
        h = forcing.convection_W_m2K;
        admittance_W_m2K = h;
        return;
    }
    if (law.model == ConvectionModel::Constant) {
        h = material.convection_W_m2K;
        admittance_W_m2K = h;
        return;
    }

    h = law.windIntercept_W_m2K + law.windSlope_W_s_m3K * forcing.windSpeed_m_s;
    admittance_W_m2K = h;

    if (law.model != ConvectionModel::Stability) return;

    // Below a tenth of a degree neither branch has anything to say, and the
    // cube root's slope runs away.
    const f64 difference = surfaceTemperature_K - forcing.airTemperature_K;
    if (std::abs(difference) < 0.1) return;

    if (difference > 0.0) {
        // Unstable: the surface is hotter than the air above it, so plumes
        // rise off it. Free convection is a floor under the wind law rather
        // than a replacement for it.
        const f64 free_W_m2K = law.freeCoefficient * std::cbrt(difference);
        if (free_W_m2K > h) {
            h = free_W_m2K;
            admittance_W_m2K = h * (4.0 / 3.0);
        }
        return;
    }

    // Stable: the surface is colder than the air, the densest air is already
    // at the bottom, and there is nothing to overturn. This is the night the
    // whole model exists for -- convection is a source here, and a coefficient
    // sized for a well-mixed afternoon pours in heat a real nocturnal layer
    // withholds.
    const f64 wind = std::max(forcing.windSpeed_m_s, kMinWind_m_s);
    const f64 richardson = kGravity_m_s2 * law.referenceHeight_m * (-difference) /
                           (forcing.airTemperature_K * wind * wind);
    const f64 damping = 1.0 + law.stableDamping * richardson;
    if (!(damping > 1.0)) return;

    const f64 damped = h / damping;
    if (damped <= kMinStableConvection_W_m2K) {
        // At the floor the coefficient no longer moves with the surface, so
        // the admittance is the coefficient again.
        h = kMinStableConvection_W_m2K;
        admittance_W_m2K = h;
        return;
    }
    h = damped;
    admittance_W_m2K = h / damping;
}

/// Claim a slot for sun column @p column, evicting the oldest tracked one when
/// every slot is taken, and return which slot it is -- or the slot count when
/// this state tracks none, or when the column is older than everything already
/// tracked.
///
/// Evicting does not lose the evicted column's answer. What a slot holds is a
/// piece of sunSensitivity_K, and the shading pass applies the remainder --
/// the total less what the slots hold -- against the present sun exactly as it
/// did before slots existed. So a column leaving the window moves from "traced
/// at its own hour" back to "assumed to look like now", which is a loss of
/// resolution rather than a loss of energy.
u32 ClaimLagSlot(ThermalState& state, const u32 column) {
    const u32 slots = state.LagSlots();
    if (slots == 0) return 0;

    u32 oldest = 0;
    for (u32 s = 0; s < slots; ++s) {
        if (state.lagColumn[s] == column) return s;
        if (state.lagColumn[s] == ThermalState::kNoLagColumn) {
            oldest = s;
            break;
        }
        if (state.lagColumn[oldest] != ThermalState::kNoLagColumn &&
            state.lagColumn[s] < state.lagColumn[oldest]) {
            oldest = s;
        }
    }

    // A column older than every one tracked is history the window has already
    // moved past; leave the slots as they are rather than throwing away a
    // newer column for it.
    if (state.lagColumn[oldest] != ThermalState::kNoLagColumn &&
        column < state.lagColumn[oldest]) {
        return slots;
    }

    state.lagColumn[oldest] = column;
    const usize block = state.temperature_K.size();
    std::fill(state.lagSensitivity_K.begin() + static_cast<isize>(oldest * block),
              state.lagSensitivity_K.begin() + static_cast<isize>((oldest + 1) * block), 0.0);
    return oldest;
}

/// How far a tangent is allowed to travel. A sensitivity is a derivative, not
/// a temperature, so the state clamp does not apply to it; this one is here
/// for the same reason that one is -- a diverging element should stay visible
/// as a wrong number rather than turn into a NaN that spreads.
constexpr f64 kMaxSensitivity_K = 1000.0;

/// Everything the outside does to one exposed face, at the temperatures it had
/// at the start of a step.
///
/// Extracted from Step rather than written beside it, and that is the point:
/// the same numbers are wanted twice -- once to build the step's right-hand
/// side, once to answer "what is heating this element" for a probe -- and two
/// readings of one energy balance is exactly the kind of pair that drifts. The
/// accumulation order inside is the one Step had, so the flux it returns is
/// bit-for-bit the flux Step used to compute for itself.
struct SurfaceBalance {
    /// The whole of it, as the right-hand side wants it: short wave absorbed,
    /// plus the net long wave. Convection and evaporation are not here --
    /// they are linear in the unknown and belong in the matrix.
    f64 surfaceFlux_W_m2 = 0.0;
    /// The short-wave half with the absorptivity divided out, and the
    /// long-wave half with the emissivity divided out: each is its own
    /// derivative in that coefficient, which is what the parameter tangents
    /// read. Accumulated beside the flux so the two cannot disagree about
    /// which terms are in them.
    f64 shortwavePerAbsorptivity_W_m2 = 0.0;
    f64 longwavePerEmissivity_W_m2 = 0.0;
    /// d(flux)/dv: the direct beam with the sun visibility left out. The only
    /// term in the balance that has one.
    f64 directPerVisibility_W_m2 = 0.0;
    /// The convective coefficient and the admittance the matrix takes, which
    /// are the same number only under the constant law.
    f64 h = 0.0;
    f64 convectiveAdmittance_W_m2K = 0.0;
    /// Evaporation, linearised about the previous surface temperature.
    f64 latentFlux_W_m2 = 0.0;
    f64 latentAdmittance_W_m2K = 0.0;
};

SurfaceBalance EvaluateSurfaceBalance(const u32 e, const ThermalElement& element,
                                      const ThermalMaterial& material,
                                      const ExchangeGeometry& exchange,
                                      const ThermalForcing& forcing,
                                      const ShortwaveSample& shortwave,
                                      const std::span<const f32> sunVisibility,
                                      const Vector<f64>& surfacePrevious,
                                      const ConvectionLaw& convection) {
    SurfaceBalance out;
    const f64 emissivity = static_cast<f64>(material.longwaveEmissivity);

    const f64 absorptivity = material.shortwaveAbsorptivity;
    // The same flux with the coefficient divided out, which is its own
    // derivative in that coefficient. Accumulated beside the flux rather
    // than reconstructed later, so the two cannot come to disagree about
    // which terms are in it.

    // Sun, direct. cos(theta) against the element's own normal, times the
    // precomputed visibility -- which is what carries the shadow.
    //
    // directPerVisibility is the same product with the visibility left
    // out: d(flux)/dv, and the only term in the whole balance that has
    // one. The reflected gain below is driven by what OTHER elements see,
    // the diffuse by the sky, and neither moves when this element steps
    // into shade -- which is exactly why the tangent is local and costs a
    // right-hand side rather than a Jacobian.
    if (forcing.sunIrradiance_W_m2 > 0.0) {
        const f64 cosTheta = static_cast<f64>(
            glm::dot(element.normal, glm::normalize(forcing.sunDirection)));
        if (cosTheta > 0.0) {
            out.directPerVisibility_W_m2 =
                absorptivity * forcing.sunIrradiance_W_m2 * cosTheta;
            if (e < sunVisibility.size()) {
                out.surfaceFlux_W_m2 +=
                    out.directPerVisibility_W_m2 * static_cast<f64>(sunVisibility[e]);
                out.shortwavePerAbsorptivity_W_m2 += forcing.sunIrradiance_W_m2 * cosTheta *
                                                 static_cast<f64>(sunVisibility[e]);
            }
        }
    }

    // Sun, off a neighbour. This is where a north wall gets its afternoon:
    // it is never in the sun, and the road in front of it is. The gain is
    // a gather over the whole hemisphere, so it carries no cos(theta) of
    // its own -- that was applied to the surfaces doing the reflecting,
    // when it was baked.
    if (forcing.sunIrradiance_W_m2 > 0.0 && e < shortwave.reflectedGain.size()) {
        out.surfaceFlux_W_m2 += absorptivity * forcing.sunIrradiance_W_m2 *
                            static_cast<f64>(shortwave.reflectedGain[e]);
        out.shortwavePerAbsorptivity_W_m2 += forcing.sunIrradiance_W_m2 *
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
        out.surfaceFlux_W_m2 += absorptivity * forcing.diffuseIrradiance_W_m2 * diffuseGain;
        out.shortwavePerAbsorptivity_W_m2 += forcing.diffuseIrradiance_W_m2 * diffuseGain;
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
    out.longwavePerEmissivity_W_m2 = kStefanBoltzmann * (incoming - Ti * Ti * Ti * Ti);
    out.surfaceFlux_W_m2 += emissivity * out.longwavePerEmissivity_W_m2;

    // Convection, which is linear in the unknown and therefore goes into
    // the matrix rather than into the flux.
    //
    // The forcing's value wins when it has one, then the law, then the
    // material's constant. The latent term below derives from this same out.h,
    // so anything that varies it varies the evaporation with it -- which
    // is right, since both are the same turbulent exchange carrying
    // different quantities.
    ConvectionAt(convection, forcing, material, Ti, out.h, out.convectiveAdmittance_W_m2K);

    // Evaporation. Non-linear in the unknown like the radiation, but with
    // several times its slope, so this one goes into the matrix as well:
    // linearised about the previous surface temperature and split
    // half-and-half the way Crank-Nicolson splits everything else. A wet
    // element under dry air can shed hundreds of watts per square metre,
    // and leaving that explicit oscillates at a minute per step.
    if (material.wetnessFactor > 0.0f) {
        f64 qSurface = 0.0;
        f64 dq_dT = 0.0;
        SaturationHumidity(Ti, qSurface, dq_dT);
        const f64 qAir = SaturationHumidity(forcing.airTemperature_K);

        const f64 humidity = std::clamp(forcing.relativeHumidity, 0.0, 100.0) / 100.0;
        const f64 coefficient =
            LatentCoefficient(static_cast<f64>(material.wetnessFactor), out.h);
        out.latentFlux_W_m2 = coefficient * (qSurface - humidity * qAir);
        out.latentAdmittance_W_m2K = coefficient * dq_dT;
    }
    return out;
}

}  // namespace

bool CpuCrankNicolsonStepper::SurfaceFluxesAt(
    const ThermalState& state, const Vector<ThermalElement>& elements,
    const Vector<ThermalMaterial>& materials, const ExchangeGeometry& exchange,
    const ThermalForcing& forcing, const ShortwaveSample& shortwave, const u32 element,
    SurfaceFluxes& out) const {
    const u32 nodes = state.nodeCount;
    if (nodes < 2 || element >= elements.size()) return false;

    const ThermalElement& e = elements[element];
    if (e.materialId >= materials.size()) return false;
    const ThermalMaterial& material = materials[e.materialId];

    const f64 k = material.conductivity_W_mK;
    const f64 rhoC =
        static_cast<f64>(material.density_kg_m3) * material.specificHeat_J_kgK;
    const f64 dx = static_cast<f64>(material.thickness_m) / static_cast<f64>(nodes - 1);
    if (!(rhoC > 0.0) || !(dx > 0.0)) return false;

    // The same view of the field the step takes: surface values only, as they
    // were before anything moved.
    Vector<f64> surfacePrevious(elements.size());
    for (usize i = 0; i < elements.size(); ++i) surfacePrevious[i] = state.Surface(i);

    const SurfaceBalance balance =
        EvaluateSurfaceBalance(element, e, material, exchange, forcing, shortwave,
                               shortwave.sunVisibility, surfacePrevious, m_convection);

    const f64 emissivity = static_cast<f64>(material.longwaveEmissivity);
    const f64 Ts = surfacePrevious[element];
    const f64 T1 = state.temperature_K[static_cast<usize>(element) * nodes + 1];

    out = SurfaceFluxes{};
    out.longwave_W_m2 = emissivity * balance.longwavePerEmissivity_W_m2;
    // What is left of the flux once the long wave is taken out of it, rather
    // than the short wave recomputed: surfaceFlux was accumulated in the step's
    // own order and this keeps the two exactly consistent.
    out.shortwave_W_m2 = balance.surfaceFlux_W_m2 - out.longwave_W_m2;
    out.convection_W_m2 = balance.h * (forcing.airTemperature_K - Ts);
    out.latent_W_m2 = -balance.latentFlux_W_m2;
    out.conduction_W_m2 = (k / dx) * (T1 - Ts);

    // Lateral is a rate the step turns into a flux with the half-cell's
    // capacity, so the same product appears here.
    if (exchange.lateral.RowCount() == elements.size() && !exchange.lateral.value.empty() &&
        e.area_m2 > 0.0f) {
        f64 rate = 0.0;
        const usize base = static_cast<usize>(element) * nodes;
        for (u32 nz = exchange.lateral.rowStart[element];
             nz < exchange.lateral.rowStart[element + 1]; ++nz) {
            const u32 j = exchange.lateral.column[nz];
            rate += exchange.lateral.value[nz] *
                    (state.temperature_K[static_cast<usize>(j) * nodes] -
                     state.temperature_K[base]);
        }
        rate /= rhoC * static_cast<f64>(e.area_m2);
        out.lateral_W_m2 = rhoC * dx * 0.5 * rate;
    }
    return true;
}

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

    // Lateral conduction is explicit, and it reaches every node rather than
    // only the surface, so it needs the whole field as it was -- otherwise an
    // element would conduct against neighbours that have already moved, which
    // is a Gauss-Seidel sweep whose answer depends on the element order.
    // Allocated only for a scene that asked, since it doubles the state.
    const bool carryLateral = exchange.lateral.RowCount() == elements.size() &&
                              !exchange.lateral.value.empty();
    Vector<f64> lateralPrevious;
    Vector<f64> lateralPreviousSensitivity;
    Vector<f64> lateralPreviousLag;
    if (carryLateral) {
        lateralPrevious = state.temperature_K;
        if (state.HasSensitivity()) lateralPreviousSensitivity = state.sunSensitivity_K;
        if (state.HasLagSensitivity()) lateralPreviousLag = state.lagSensitivity_K;
    }

    Vector<f64> lower(nodes), diag(nodes), upper(nodes), rhs(nodes);

    // The tangent rides in the same matrix, so it needs a second right-hand
    // side and nothing else. No ping-pong copy beside surfacePrevious: the
    // tangent this carries is the derivative with respect to an element's OWN
    // sun visibility, so it reads only its own previous value, which is still
    // in the state when its turn comes.
    const bool carryTangent = state.HasSensitivity();
    Vector<f64> rhsTangent(carryTangent ? nodes : 0);

    // One more right-hand side per tracked sun column, and one source weight
    // each: what fraction of this step's short wave came from that column.
    // Only the two columns the step is interpolating between are non-zero, so
    // the rest ride along as pure decay -- which is the whole of what "the
    // ground is still warm from an hour ago" means.
    // Stepped whenever they exist, sourced only when the step knows which
    // columns it came from: a tangent that decayed on some steps and not
    // others would not be the derivative of anything.
    const bool carryLags = carryTangent && state.HasLagSensitivity();
    const u32 lagSlots = carryLags ? state.LagSlots() : 0u;
    Vector<f64> rhsLag(static_cast<usize>(lagSlots) * nodes);
    Vector<f64> lagWeight(lagSlots, 0.0);
    if (carryLags && shortwave.columnsKnown) {
        const f64 blend = std::clamp(shortwave.columnBlend, 0.0, 1.0);
        const f64 weights[2] = {1.0 - blend, blend};
        const u32 columns[2] = {static_cast<u32>(shortwave.columnA),
                                static_cast<u32>(shortwave.columnB)};

        for (i32 i = 0; i < 2; ++i) {
            if (weights[i] > 0.0) ClaimLagSlot(state, columns[i]);
        }
        // Read the weights back off the slots rather than off the claims: with
        // a one-slot window the second claim evicts the first, and a source
        // written to a slot that no longer tracks its column would be
        // attributed to the wrong hour. What is dropped here stays in the
        // total, which is where the shading pass looks for it.
        for (u32 s = 0; s < lagSlots; ++s) {
            for (i32 i = 0; i < 2; ++i) {
                if (weights[i] > 0.0 && state.lagColumn[s] == columns[i]) {
                    lagWeight[s] += weights[i];
                }
            }
        }
    }

    // What the elimination is applied to, built once: the temperature first,
    // then whichever tangents this state carries. Every element has the same
    // node count, so the list does not change inside the loop.
    Vector<std::span<f64>> rightHandSides;
    rightHandSides.emplace_back(rhs);
    if (carryTangent) rightHandSides.emplace_back(rhsTangent);
    for (u32 s = 0; s < lagSlots; ++s) {
        rightHandSides.emplace_back(rhsLag.data() + static_cast<usize>(s) * nodes, nodes);
    }

    // The lateral gain of each node of the element being stepped, in K/s. One
    // rate per node rather than one per element because the two faces of a
    // slab can be at very different temperatures and conduct sideways at
    // different rates; the node's own cell height cancels out of it, so the
    // form is the same for the half cells at the faces and the whole cells
    // between them.
    Vector<f64> lateralRate(carryLateral ? nodes : 0, 0.0);
    Vector<f64> lateralRateTangent(carryLateral && carryTangent ? nodes : 0, 0.0);
    Vector<f64> lateralRateLag(carryLateral ? static_cast<usize>(lagSlots) * nodes : 0, 0.0);

    // One more right-hand side per material parameter being differentiated.
    // These are solved after the temperature rather than beside it: the matrix
    // itself moves with k and rho c, and what that contributes to the tangent
    // is (dA/dp) T^{n+1}, which does not exist until the temperature has been
    // back-substituted. They reuse the factorisation.
    const bool carryParameters = state.HasParameterSensitivity();
    const usize parameterCount = carryParameters ? state.parameters.size() : 0;
    Vector<f64> rhsParameter(parameterCount * nodes);
    Vector<std::span<f64>> parameterSides;
    for (usize p = 0; p < parameterCount; ++p) {
        parameterSides.emplace_back(rhsParameter.data() + p * nodes, nodes);
    }
    Vector<f64> lateralRateParameter(
        carryLateral ? parameterCount * nodes : 0, 0.0);
    Vector<f64> lateralPreviousParameter;
    if (carryLateral && carryParameters) {
        lateralPreviousParameter = state.parameterSensitivity;
    }

    // Which elements are the far side of a shell, and therefore not stepped:
    // their temperature is the owner's back node and is written there at the
    // end. The lower index owns, so a pair is decided without a tie-break.
    const auto shellPartnerOf = [this, &elements](const usize e) -> u32 {
        if (m_shellPartner.size() != elements.size()) return ThermalMesh::kNoShellPartner;
        return m_shellPartner[e];
    };

    for (usize e = 0; e < elements.size(); ++e) {
        const ThermalElement& element = elements[e];
        if (element.materialId >= materials.size()) continue;
        const ThermalMaterial& material = materials[element.materialId];
        if (!material.ParticipatesInSolve()) continue;

        // The far side of a shell has no column of its own. Skipped here and
        // filled in after the loop, from the column it shares.
        const u32 shellPartner = shellPartnerOf(e);
        const bool isShellOwner =
            shellPartner != ThermalMesh::kNoShellPartner && shellPartner > e;
        if (shellPartner != ThermalMesh::kNoShellPartner && !isShellOwner) continue;

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
        // Sideways, to whatever this element shares an edge with
        // ----------------------------------------------------------------
        // Explicit, from the snapshot, and applied to the tangents on the same
        // terms as to the temperature -- a neighbour that steps into shade
        // cools this element too, and leaving that out of the tangent would
        // make dT/dv describe a column the solver is no longer stepping.
        //
        // g_ij is a conductance per metre of slab depth, so dividing by the
        // element's heat capacity per unit area leaves a rate that does not
        // depend on which node it is applied to.
        if (carryLateral) {
            std::fill(lateralRate.begin(), lateralRate.end(), 0.0);
            std::fill(lateralRateTangent.begin(), lateralRateTangent.end(), 0.0);
            std::fill(lateralRateLag.begin(), lateralRateLag.end(), 0.0);
            std::fill(lateralRateParameter.begin(), lateralRateParameter.end(), 0.0);
        }
        if (carryLateral && element.area_m2 > 0.0f) {
            const f64* Ti = &lateralPrevious[e * nodes];
            const f64* Si = lateralPreviousSensitivity.empty()
                                ? nullptr
                                : &lateralPreviousSensitivity[e * nodes];

            for (u32 nz = exchange.lateral.rowStart[e]; nz < exchange.lateral.rowStart[e + 1];
                 ++nz) {
                const u32 j = exchange.lateral.column[nz];
                const f64 g = exchange.lateral.value[nz];
                const f64* Tj = &lateralPrevious[j * nodes];
                for (u32 i = 0; i < nodes; ++i) {
                    lateralRate[i] += g * (Tj[i] - Ti[i]);
                }
                if (Si != nullptr && !lateralRateTangent.empty()) {
                    const f64* Sj = &lateralPreviousSensitivity[j * nodes];
                    for (u32 i = 0; i < nodes; ++i) {
                        lateralRateTangent[i] += g * (Sj[i] - Si[i]);
                    }
                }
                for (u32 s = 0; s < lagSlots && !lateralPreviousLag.empty(); ++s) {
                    const usize block = s * elements.size() * nodes;
                    const f64* Li = &lateralPreviousLag[block + e * nodes];
                    const f64* Lj = &lateralPreviousLag[block + j * nodes];
                    f64* rate = &lateralRateLag[static_cast<usize>(s) * nodes];
                    for (u32 i = 0; i < nodes; ++i) {
                        rate[i] += g * (Lj[i] - Li[i]);
                    }
                }
                for (usize p = 0; p < parameterCount && !lateralPreviousParameter.empty();
                     ++p) {
                    const usize block = p * elements.size() * nodes;
                    const f64* Pi = &lateralPreviousParameter[block + e * nodes];
                    const f64* Pj = &lateralPreviousParameter[block + j * nodes];
                    f64* rate = &lateralRateParameter[p * nodes];
                    for (u32 i = 0; i < nodes; ++i) {
                        rate[i] += g * (Pj[i] - Pi[i]);
                    }
                }
            }

            const f64 perCapacity = 1.0 / (rhoC * static_cast<f64>(element.area_m2));
            for (u32 i = 0; i < nodes; ++i) {
                lateralRate[i] *= perCapacity;
                if (!lateralRateTangent.empty()) lateralRateTangent[i] *= perCapacity;
            }
            for (f64& rate : lateralRateLag) rate *= perCapacity;
            for (f64& rate : lateralRateParameter) rate *= perCapacity;
        }

        // ----------------------------------------------------------------
        // The exposed face
        // ----------------------------------------------------------------
        // Everything the outside does to this element, as one flux. The
        // radiative part is linearised against the previous temperatures, so
        // this is a number rather than a term in the unknown.
        // Everything the outside does to this element, as one flux, plus the
        // pieces of it the tangents and a probe read. One evaluation, so a
        // decomposition shown in a panel is the decomposition the step used.
        const SurfaceBalance balance = EvaluateSurfaceBalance(
            e, element, material, exchange, forcing, shortwave, sunVisibility,
            surfacePrevious, m_convection);
        const f64 emissivity = static_cast<f64>(material.longwaveEmissivity);
        const f64 surfaceFlux_W_m2 = balance.surfaceFlux_W_m2;
        const f64 shortwavePerAbsorptivity_W_m2 = balance.shortwavePerAbsorptivity_W_m2;
        const f64 longwavePerEmissivity_W_m2 = balance.longwavePerEmissivity_W_m2;
        const f64 directPerVisibility_W_m2 = balance.directPerVisibility_W_m2;
        const f64 h = balance.h;
        const f64 convectiveAdmittance_W_m2K = balance.convectiveAdmittance_W_m2K;
        const f64 latentFlux_W_m2 = balance.latentFlux_W_m2;
        const f64 latentAdmittance_W_m2K = balance.latentAdmittance_W_m2K;
        // The surface temperature the balance was linearised about, which the
        // tangent rows below linearise about too.
        const f64 Ti = surfacePrevious[e];

        // Crank-Nicolson on the half-cell at the face. The half cell has
        // capacity rho c dx/2 and exchanges with node 1 by conduction and with
        // the outside by h, by evaporation, and by the flux above.
        const f64 halfCell = rhoC * dx / (2.0 * dt_s);
        lower[0] = 0.0;
        diag[0] = halfCell + 0.5 * (k / dx + convectiveAdmittance_W_m2K + latentAdmittance_W_m2K);
        upper[0] = -0.5 * (k / dx);
        // The trailing term is what a temperature-dependent h adds, and it is
        // exactly zero -- and therefore exactly nothing, in floating point --
        // whenever the admittance is h.
        rhs[0] = halfCell * T[0] - 0.5 * (k / dx) * (T[0] - T[1]) +
                 0.5 * h * (2.0 * forcing.airTemperature_K - T[0]) + surfaceFlux_W_m2 -
                 latentFlux_W_m2 + 0.5 * latentAdmittance_W_m2K * T[0] +
                 0.5 * (convectiveAdmittance_W_m2K - h) * T[0];
        if (carryLateral) rhs[0] += halfCell * dt_s * lateralRate[0];

        // The tangent's face row: the same equation differentiated in v.
        // Term by term against the line above -- the air temperature and the
        // latent flux are constants of v and drop, the explicit long-wave
        // loss contributes its own slope, and the short wave contributes the
        // only source. What is left of the convection is -Y_conv/2 sigma0
        // rather than the +h*Tair the temperature gets, and of the
        // evaporation -Y_lat/2 sigma0, because the flux and the half-implicit
        // correction cancel to half. Y_conv rather than h: under a
        // temperature-dependent law the coefficient moves with the surface,
        // and the tangent is differentiating the same equation the matrix
        // solves.
        //
        // The neighbours' share of `incoming` is deliberately NOT
        // differentiated. Its derivative is the off-diagonal of a Jacobian
        // over every element that sees this one, and what it would add is the
        // second-order fact that a colder patch of ground makes its
        // neighbours very slightly colder too. Keeping it out is what makes
        // this a per-element quantity a shader can apply per pixel.
        // The one term of the face row that only a tangent uses. Hoisted so the
        // rows below can be written once and applied to every tangent this
        // state carries.
        const f64 radiativeSlope = 4.0 * emissivity * kStefanBoltzmann * Ti * Ti * Ti;

        // ----------------------------------------------------------------
        // The interior
        // ----------------------------------------------------------------
        for (u32 i = 1; i + 1 < nodes; ++i) {
            lower[i] = -0.5 * r;
            diag[i] = 1.0 + r;
            upper[i] = -0.5 * r;
            rhs[i] = T[i] + 0.5 * r * (T[i - 1] - 2.0 * T[i] + T[i + 1]);
            if (carryLateral) rhs[i] += dt_s * lateralRate[i];
        }

        // ----------------------------------------------------------------
        // The back face
        // ----------------------------------------------------------------
        const u32 last = nodes - 1;
        f64 backAdmittance_W_m2K = 0.0;
        f64 backRadiativeSlope_W_m2K = 0.0;
        if (isShellOwner) {
            // The other face of the same slab, exposed. Not a boundary
            // condition standing in for what is behind the surface -- there is
            // nothing behind it, and the same balance the front face gets is
            // what the back one gets, evaluated with the PARTNER's geometry:
            // its own normal against the sun, its own sky fraction, its own
            // view factors.
            //
            // EvaluateSurfaceBalance linearises about surfacePrevious[partner],
            // and surfacePrevious carries the owner's back node there -- which
            // is what makes this the same function rather than a second reading
            // of the balance written for one side.
            const SurfaceBalance back = EvaluateSurfaceBalance(
                shellPartner, elements[shellPartner], material, exchange, forcing,
                shortwave, sunVisibility, surfacePrevious, m_convection);

            // Mirror of the front row. The internal source is deliberately not
            // added: a shell has two exposed faces and no interior for a source
            // to enter from, and a config that asks for both is asking for a
            // panel that is also a wall.
            lower[last] = -0.5 * (k / dx);
            diag[last] = halfCell + 0.5 * (k / dx + back.convectiveAdmittance_W_m2K +
                                           back.latentAdmittance_W_m2K);
            upper[last] = 0.0;
            rhs[last] = halfCell * T[last] - 0.5 * (k / dx) * (T[last] - T[last - 1]) +
                        0.5 * back.h * (2.0 * forcing.airTemperature_K - T[last]) +
                        back.surfaceFlux_W_m2 - back.latentFlux_W_m2 +
                        0.5 * back.latentAdmittance_W_m2K * T[last] +
                        0.5 * (back.convectiveAdmittance_W_m2K - back.h) * T[last];
            if (carryLateral) rhs[last] += halfCell * dt_s * lateralRate[last];

            // The tangent rows below read these two, and for a shell the back
            // face's own convection and radiation are what they see.
            backAdmittance_W_m2K = back.convectiveAdmittance_W_m2K;
            backRadiativeSlope_W_m2K =
                4.0 * emissivity * kStefanBoltzmann * T[last] * T[last] * T[last];
        } else if (material.interiorBoundary == InteriorBoundary::FixedTemperature) {
            // A Dirichlet row: whatever is behind this surface holds it there.
            lower[last] = 0.0;
            diag[last] = 1.0;
            upper[last] = 0.0;
            rhs[last] = material.interiorTemperature_K;
        } else {
            // A half cell at the back, exchanging with the node in front of it
            // and with whatever the boundary says is behind it. Adiabatic is
            // the mirror condition -- nothing behind -- and the two flux terms
            // below are then both zero, which is exactly the row this was
            // before either existed.
            //
            // The internal source is a flux and enters the right-hand side
            // whole. The back-face exchange is linear in the unknown, so it
            // splits between the two sides the way the front face's convection
            // does.
            f64 backFlux_W_m2 = material.internalHeat_W_m2;
            if (material.interiorBoundary == InteriorBoundary::AmbientInterior) {
                const f64 interior = static_cast<f64>(material.interiorTemperature_K);
                const f64 hBack = static_cast<f64>(material.interiorConvection_W_m2K);
                // Radiation to a background at the interior temperature,
                // linearised about the previous back-face temperature exactly
                // as the exposed face linearises its own T^4.
                const f64 Tb = T[last];
                const f64 radiative = emissivity * kStefanBoltzmann *
                                      (interior * interior * interior * interior -
                                       Tb * Tb * Tb * Tb);
                backAdmittance_W_m2K = hBack;
                backRadiativeSlope_W_m2K = 4.0 * emissivity * kStefanBoltzmann * Tb * Tb * Tb;
                backFlux_W_m2 += hBack * (interior - Tb) + radiative;
            }

            lower[last] = -0.5 * (k / dx);
            diag[last] = halfCell + 0.5 * (k / dx + backAdmittance_W_m2K);
            upper[last] = 0.0;
            rhs[last] = halfCell * T[last] - 0.5 * (k / dx) * (T[last] - T[last - 1]) +
                        backFlux_W_m2 + 0.5 * backAdmittance_W_m2K * T[last];
            if (carryLateral) rhs[last] += halfCell * dt_s * lateralRate[last];
        }

        // ----------------------------------------------------------------
        // Every tangent, through the same rows
        // ----------------------------------------------------------------
        // The same equation differentiated in v. Term by term against the
        // temperature's rows: the air temperature, the latent flux and
        // everything behind the back face are constants of v and drop; the
        // explicit long-wave loss contributes its own slope; the short wave
        // contributes the only source. What is left of the convection is
        // -Y_conv/2 sigma rather than the +h*Tair the temperature gets, and of
        // the evaporation -Y_lat/2 sigma, because the flux and the
        // half-implicit correction cancel to half. Y_conv rather than h: under
        // a temperature-dependent law the coefficient moves with the surface.
        //
        // The neighbours' share of `incoming` is deliberately NOT
        // differentiated. Its derivative is the off-diagonal of a Jacobian
        // over every element that sees this one, and what it would add is the
        // second-order fact that a colder patch of ground makes its
        // neighbours very slightly colder too. Keeping it out is what makes
        // this a per-element quantity a shader can apply per pixel.
        //
        // Written once and applied to each: the tangent against the whole
        // day's visibility, and one per tracked sun column, which differ only
        // in how much of this step's short wave is theirs.
        const auto buildTangentRows = [&](const f64* sigma, f64* out, const f64 source,
                                          const f64* lateral) {
            out[0] = halfCell * sigma[0] - 0.5 * (k / dx) * (sigma[0] - sigma[1]) -
                     0.5 * convectiveAdmittance_W_m2K * sigma[0] -
                     radiativeSlope * sigma[0] -
                     0.5 * latentAdmittance_W_m2K * sigma[0] + source;
            if (lateral != nullptr) out[0] += halfCell * dt_s * lateral[0];

            for (u32 i = 1; i + 1 < nodes; ++i) {
                out[i] = sigma[i] + 0.5 * r * (sigma[i - 1] - 2.0 * sigma[i] + sigma[i + 1]);
                if (lateral != nullptr) out[i] += dt_s * lateral[i];
            }

            if (material.interiorBoundary == InteriorBoundary::FixedTemperature) {
                // A room held at its own temperature does not care about the
                // sun, so the tangent's Dirichlet value is zero rather than
                // the temperature's.
                out[last] = 0.0;
                return;
            }
            out[last] = halfCell * sigma[last] -
                        0.5 * (k / dx) * (sigma[last] - sigma[last - 1]) -
                        0.5 * backAdmittance_W_m2K * sigma[last] -
                        backRadiativeSlope_W_m2K * sigma[last];
            if (lateral != nullptr) out[last] += halfCell * dt_s * lateral[last];
        };

        f64* sigma = nullptr;
        if (carryTangent) {
            sigma = &state.sunSensitivity_K[e * nodes];
            buildTangentRows(sigma, rhsTangent.data(), directPerVisibility_W_m2,
                             carryLateral ? lateralRateTangent.data() : nullptr);
        }
        for (u32 s = 0; s < lagSlots; ++s) {
            const usize block = s * elements.size() * nodes + e * nodes;
            buildTangentRows(&state.lagSensitivity_K[block],
                             rhsLag.data() + static_cast<usize>(s) * nodes,
                             directPerVisibility_W_m2 * lagWeight[s],
                             carryLateral ? &lateralRateLag[static_cast<usize>(s) * nodes]
                                          : nullptr);
        }

        FactorTridiagonal(lower, diag, upper);
        SolveFactored(lower, diag, upper, rightHandSides);

        // ----------------------------------------------------------------
        // The material parameters, on the temperature this step just produced
        // ----------------------------------------------------------------
        // Same operator again, so the same factorisation, and a source that
        // is this row's own derivative in the parameter. Three of them touch
        // only the flux; two move the matrix, and what that contributes is
        // -(dA/dp) T^{n+1} -- which is why these are built here rather than
        // beside the tangents above.
        for (usize p = 0; p < parameterCount; ++p) {
            const usize block = p * elements.size() * nodes;
            const f64* sigma = &state.parameterSensitivity[block + e * nodes];
            f64* out = rhsParameter.data() + p * nodes;
            buildTangentRows(sigma, out, 0.0,
                             carryLateral ? &lateralRateParameter[p * nodes] : nullptr);

            const f64 conduction = k / dx;
            switch (state.parameters[p]) {
                case ThermalParameter::Absorptivity:
                    out[0] += shortwavePerAbsorptivity_W_m2;
                    break;

                case ThermalParameter::Emissivity:
                    out[0] += longwavePerEmissivity_W_m2;
                    if (material.interiorBoundary == InteriorBoundary::AmbientInterior) {
                        const f64 interior = static_cast<f64>(material.interiorTemperature_K);
                        const f64 Tb = T[last];
                        out[last] += kStefanBoltzmann *
                                     (interior * interior * interior * interior -
                                      Tb * Tb * Tb * Tb);
                    }
                    break;

                case ThermalParameter::Convection: {
                    // Only a free parameter where nothing else decides it: a
                    // forcing column or a wind law is what h is then, and the
                    // material's own number does not reach the answer.
                    if (forcing.convection_W_m2K > 0.0 ||
                        m_convection.model != ConvectionModel::Constant) {
                        break;
                    }
                    const f64 latentPerH = h > 0.0 ? latentFlux_W_m2 / h : 0.0;
                    const f64 latentAdmittancePerH =
                        h > 0.0 ? latentAdmittance_W_m2K / h : 0.0;
                    out[0] += (forcing.airTemperature_K - T[0]) - latentPerH +
                              0.5 * (1.0 + latentAdmittancePerH) * (T[0] - rhs[0]);
                    break;
                }

                case ThermalParameter::Conductivity: {
                    // The face and back rows exchange k/dx with their
                    // neighbour, half explicit and half implicit; the interior
                    // rows carry it inside r.
                    out[0] += -0.5 / dx * ((T[0] - T[1]) + (rhs[0] - rhs[1]));
                    for (u32 i = 1; i + 1 < nodes; ++i) {
                        const f64 rPerK = r / k;
                        out[i] += 0.5 * rPerK *
                                  ((T[i - 1] - 2.0 * T[i] + T[i + 1]) +
                                   (rhs[i - 1] - 2.0 * rhs[i] + rhs[i + 1]));
                    }
                    if (material.interiorBoundary != InteriorBoundary::FixedTemperature) {
                        out[last] += -0.5 / dx *
                                     ((T[last] - T[last - 1]) + (rhs[last] - rhs[last - 1]));
                    }
                    break;
                }

                case ThermalParameter::HeatCapacity: {
                    // rho c scales the capacity up and the Fourier number
                    // down, so the two rows differ in sign.
                    out[0] += halfCell / rhoC * (T[0] - rhs[0]);
                    for (u32 i = 1; i + 1 < nodes; ++i) {
                        const f64 rPerRhoC = r / rhoC;
                        out[i] += -0.5 * rPerRhoC *
                                  ((T[i - 1] - 2.0 * T[i] + T[i + 1]) +
                                   (rhs[i - 1] - 2.0 * rhs[i] + rhs[i + 1]));
                    }
                    if (material.interiorBoundary != InteriorBoundary::FixedTemperature) {
                        out[last] += halfCell / rhoC * (T[last] - rhs[last]);
                    }
                    break;
                }

                case ThermalParameter::Count:
                    break;
            }
            (void)conduction;
        }
        SolveFactored(lower, diag, upper, parameterSides);

        for (u32 i = 0; i < nodes; ++i) {
            // A temperature outside this range is a solver failure rather than
            // a cold night, and letting it through would put a NaN into the
            // render two steps later.
            T[i] = std::clamp(rhs[i], 1.0, 5000.0);
            if (carryTangent) {
                sigma[i] = std::clamp(rhsTangent[i], -kMaxSensitivity_K, kMaxSensitivity_K);
            }
        }
        for (u32 s = 0; s < lagSlots; ++s) {
            f64* slot = &state.lagSensitivity_K[s * elements.size() * nodes + e * nodes];
            for (u32 i = 0; i < nodes; ++i) {
                slot[i] = std::clamp(rhsLag[static_cast<usize>(s) * nodes + i],
                                     -kMaxSensitivity_K, kMaxSensitivity_K);
            }
        }
        for (usize p = 0; p < parameterCount; ++p) {
            f64* slot = &state.parameterSensitivity[p * elements.size() * nodes + e * nodes];
            for (u32 i = 0; i < nodes; ++i) {
                // The clamp is in kelvin per unit of the parameter, and the
                // parameters differ by orders of magnitude in scale -- rho c is
                // millions, emissivity is one. So this is only the NaN guard the
                // others are, not a statement about a plausible sensitivity.
                slot[i] = std::clamp(rhsParameter[p * nodes + i], -1e12, 1e12);
            }
        }
    }

    // The far side of each shell, from the column it shares. Written into the
    // partner's own surface slot rather than mapped at every reader: the
    // radiative exchange, the render's temperature buffer, a dump and a probe
    // all reach for state.Surface(e), and one write per pair per step is what
    // lets none of them know a shell is involved.
    //
    // Its tangent is left at zero, which is a claim rather than an omission.
    // What the column carries is dT/dv for the OWNER's sun visibility, and the
    // shading pass would pair that derivative with the partner's own
    // visibility -- a different quantity. Zero means the back face gets the
    // triangle's own temperature with no sub-triangle shadow correction, which
    // is what every surface got before the correction existed.
    if (m_shellPartner.size() == elements.size()) {
        for (usize e = 0; e < elements.size(); ++e) {
            const u32 partner = m_shellPartner[e];
            if (partner == ThermalMesh::kNoShellPartner || partner < e) continue;
            if (elements[e].materialId >= materials.size()) continue;
            if (!materials[elements[e].materialId].ParticipatesInSolve()) continue;

            const usize ownerBack = e * nodes + (nodes - 1);
            const usize partnerFront = static_cast<usize>(partner) * nodes;
            if (ownerBack >= state.temperature_K.size() ||
                partnerFront >= state.temperature_K.size()) {
                continue;
            }
            state.temperature_K[partnerFront] = state.temperature_K[ownerBack];
            if (state.HasSensitivity()) {
                state.sunSensitivity_K[partnerFront] = 0.0;
            }
        }
    }
}

f64 CpuCrankNicolsonStepper::ShortestTimeConstantSeconds(
    const Vector<ThermalElement>& elements, const Vector<ThermalMaterial>& materials,
    const f64 referenceTemperature_K, const ConvectionLaw& law, const f64 windSpeed_m_s) {
    f64 shortest = std::numeric_limits<f64>::infinity();

    // What the law would give at its windiest. The free branch is evaluated at
    // a ten-degree surface-to-air difference, which is a warm afternoon or a
    // clear night rather than a worst case -- this is an advisory, and a
    // number chosen to make it fire on every scene would say nothing.
    f64 lawCoefficient = 0.0;
    if (law.model != ConvectionModel::Constant) {
        lawCoefficient = law.windIntercept_W_m2K + law.windSlope_W_s_m3K * windSpeed_m_s;
        if (law.model == ConvectionModel::Stability) {
            lawCoefficient =
                std::max(lawCoefficient, law.freeCoefficient * std::cbrt(10.0));
        }
    }

    for (usize e = 0; e < elements.size(); ++e) {
        const u32 id = elements[e].materialId;
        if (id >= materials.size()) continue;
        const ThermalMaterial& material = materials[id];
        if (!material.ParticipatesInSolve()) continue;

        const f64 capacity = static_cast<f64>(material.density_kg_m3) *
                             material.specificHeat_J_kgK * material.thickness_m;
        const f64 convection = std::max(static_cast<f64>(material.convection_W_m2K),
                                        lawCoefficient);
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
                                       convection) *
                     dq_dT;
        }

        const f64 loss = convection + radiative + latent;
        if (loss > 0.0) {
            shortest = std::min(shortest, capacity / loss);
        }
    }
    return shortest;
}

}  // namespace quantiloom::thermal
