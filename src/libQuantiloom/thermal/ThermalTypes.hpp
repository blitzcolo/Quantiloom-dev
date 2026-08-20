/**
 * @file ThermalTypes.hpp
 * @brief What the surface energy balance is made of
 *
 * A rendered thermogram is only as good as the temperatures it is given, and
 * until now those were typed into a config. What actually sets them is an
 * energy balance running through a day: sun absorbed, heat conducted into the
 * wall behind the surface, long-wave traded with the sky and with whatever
 * else is in view, convection with the air. Aguerre et al. 2020 solve that
 * with FEM over a volume mesh; this solves the same balance one dimension at a
 * time, per surface element, which is the part that carries thermal inertia --
 * the reason a wall is still warm at 19:00 and a thin blind is not.
 *
 * The pieces:
 *
 *   ThermalElement       one triangle, in world space, with the area and
 *                        material it exchanges through
 *   ThermalMaterial      the properties that decide how fast it responds
 *   CsrMatrix            who sees whom, and how much: F_ij is the fraction of
 *                        element i's hemisphere that element j fills
 *   ThermalState         node temperatures, one column per element
 *   ThermalForcing       what the outside world is doing at one instant
 */

#pragma once

#include "core/Types.hpp"

#include <glm/glm.hpp>

#include <span>

namespace quantiloom::thermal {

/// How a surface's back face is held. A wall has room behind it at a known
/// temperature; a free-standing plate has nothing.
enum class InteriorBoundary : u8 {
    Adiabatic = 0,  ///< no heat crosses the back face
    FixedTemperature  ///< held at interiorTemperature_K, as a room would
};

/**
 * @brief Thermal properties of one material
 *
 * Kept out of Material: that header is a layout contract Quantiloom-Qt reads
 * by offset, and these are solver inputs rather than anything the shader
 * samples. Matched to materials by name, the way the spectral bindings are.
 */
struct ThermalMaterial {
    /// Thermal conductivity, W/(m K). Zero means this material does not take
    /// part: its temperature stays whatever the config gave it.
    f32 conductivity_W_mK = 0.0f;
    f32 density_kg_m3 = 2000.0f;
    f32 specificHeat_J_kgK = 900.0f;
    /// How deep the slab is. With the conductivity and the heat capacity this
    /// is what sets the thermal inertia -- the time constant on which the
    /// surface follows the air rather than the sun.
    f32 thickness_m = 0.2f;
    /// Convective exchange with the air, W/(m^2 K). 5 is still air, 25 is a
    /// brisk wind; the correlation that would derive it from a wind speed is
    /// left to whoever has a wind speed.
    f32 convection_W_m2K = 5.0f;
    /// Fraction of short-wave sunlight the surface absorbs. Distinct from the
    /// infrared emissivity, which is what it radiates with: fresh snow absorbs
    /// almost no sunlight and radiates nearly as a blackbody.
    f32 shortwaveAbsorptivity = 0.7f;
    /// Long-wave emissivity, what the surface radiates with. Taken from the
    /// material's own IR curve rather than typed twice -- the balance and the
    /// render have to agree about it, or a surface radiates one amount into
    /// the solver and another into the camera.
    f32 longwaveEmissivity = 0.9f;
    /// How much of the surface evaporates, 0 for dry and 1 for open water.
    /// Latent heat is what makes a lawn ten degrees cooler than the pavement
    /// beside it under the same sun, and no combination of the properties
    /// above can say so -- they all describe a surface that only conducts,
    /// convects and radiates. Zero by default, which is exactly the old
    /// balance.
    f32 wetnessFactor = 0.0f;

    InteriorBoundary interiorBoundary = InteriorBoundary::Adiabatic;
    f32 interiorTemperature_K = 293.15f;

    [[nodiscard]] bool ParticipatesInSolve() const { return conductivity_W_mK > 0.0f; }
};

/**
 * @brief One surface element: a triangle that exchanges heat
 *
 * World space, because everything it exchanges with is. Per triangle rather
 * than per material because the whole point is that one wall is not one
 * temperature -- the part under the roof overhang runs warmer at night, and a
 * per-material temperature cannot say so.
 */
struct ThermalElement {
    glm::vec3 centroid{0.0f};
    f32 area_m2 = 0.0f;
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    u32 materialId = 0;
};

/**
 * @brief Sparse rows of view factors, compressed by row
 *
 * F_ij is the fraction of element i's hemisphere that element j fills, and the
 * remainder -- whatever the hemisphere sees that is not another element -- is
 * skyFraction. Rows therefore satisfy sum_j F_ij + s_i = 1 by construction,
 * and the builder restores that after truncating the small entries: a row that
 * does not sum to one is a surface exchanging with nothing, which cools to
 * absolute zero given long enough.
 *
 * Sparse because the dense form is n^2 and n is the triangle count: 15k
 * triangles would be 2.25e8 entries, nearly a gigabyte, almost all of it
 * numerically zero.
 */
struct CsrMatrix {
    Vector<u32> rowStart;   ///< size n+1, rowStart[i]..rowStart[i+1] is row i
    Vector<u32> column;     ///< which element
    Vector<f32> value;      ///< what fraction of the hemisphere it fills

    [[nodiscard]] usize RowCount() const {
        return rowStart.empty() ? 0 : rowStart.size() - 1;
    }
    [[nodiscard]] usize NonZeros() const { return value.size(); }
};

/**
 * @brief What every element sees that is not another element
 *
 * Held beside the matrix rather than as a column in it because the sky is not
 * an element: it has a temperature nothing computes from an energy balance,
 * and it is the only thing in the model that can be colder than everything
 * else at once.
 */
struct ExchangeGeometry {
    CsrMatrix viewFactors;
    Vector<f32> skyFraction;      ///< s_i, the unoccluded part of the hemisphere
    Vector<f32> sunVisibility;    ///< 0 = shadowed, 1 = full sun, fractional at an edge
};

/**
 * @brief Node temperatures through the slab, for every element
 *
 * Layout is element-major: element i owns [i*nodeCount, (i+1)*nodeCount),
 * node 0 at the exposed surface and node nodeCount-1 at the back face. That
 * order is what makes the tridiagonal solve contiguous.
 */
struct ThermalState {
    Vector<f64> temperature_K;

    /// dT/dv: how far each node moves per unit change in this element's own
    /// sun visibility, held for the whole trajectory rather than derived at
    /// the end. Same element-major layout as temperature_K.
    ///
    /// The reason it is a state and not a formula. A shadow's contrast is not
    /// the steady response to losing the sun -- that would be
    /// alpha E cos(theta) / (h + 4 eps sigma T^3), 31 K for dry sand at noon,
    /// and the ground never gets there because its time constant is hours.
    /// Nor is it the one-step response, which is 5 K and ignores every step
    /// before it. It is the integral of the same slab equation the
    /// temperature obeys, driven by the short-wave term alone, and that is
    /// exactly what this carries: the tangent of the trajectory, stepped by
    /// the same operator, so it inherits the slab's thickness, its node
    /// count, its boundary condition and its history for free.
    ///
    /// Empty means nobody asked for it, and every stepper leaves it alone --
    /// which is what keeps a caller that only wants temperatures paying
    /// nothing. Size it like temperature_K to turn it on.
    Vector<f64> sunSensitivity_K;

    u32 nodeCount = 0;

    [[nodiscard]] usize ElementCount() const {
        return nodeCount == 0 ? 0 : temperature_K.size() / nodeCount;
    }
    [[nodiscard]] f64 Surface(const usize element) const {
        return temperature_K[element * nodeCount];
    }
    /// Whether the tangent is being carried, and therefore whether a stepper
    /// should advance it.
    [[nodiscard]] bool HasSensitivity() const {
        return sunSensitivity_K.size() == temperature_K.size();
    }
    [[nodiscard]] f64 SurfaceSensitivity(const usize element) const {
        return sunSensitivity_K[element * nodeCount];
    }
};

/**
 * @brief The outside world at one instant
 *
 * Interpolated from a forcing file when there is one, held constant when there
 * is not.
 */
struct ThermalForcing {
    f64 airTemperature_K = 288.15;
    /// Direct normal irradiance, W/m^2. The element's own cos(theta) and its
    /// precomputed sun visibility are what turn it into an absorbed flux.
    f64 sunIrradiance_W_m2 = 0.0;
    /// Diffuse horizontal irradiance, W/m^2 -- the part of the sunlight that
    /// arrives from the sky dome rather than from the disc. Under overcast it
    /// is the whole of it, which is why a run without this term has no solar
    /// input at all on a cloudy day.
    f64 diffuseIrradiance_W_m2 = 0.0;
    glm::vec3 sunDirection{0.0f, 1.0f, 0.0f};  ///< from surface toward the sun
    /// Effective sky temperature: the blackbody radiating what the sky does.
    /// SkyThermal derives it from the air temperature and humidity, or the NN
    /// atmosphere's own downwelling gives it directly.
    f64 skyTemperature_K = 268.0;
    /// Relative humidity of the air, percent. Read only by the latent term:
    /// what a wet surface can evaporate is set by how far the air is from
    /// saturated. Same quantity the clear-sky model reads.
    f64 relativeHumidity = 50.0;
};

/**
 * @brief What the short wave does, sampled at several times of day
 *
 * Each column is one precomputed sun direction, one visibility fraction per
 * element; layout is sample-major (column k starts at k * elementCount).
 * A constant-forcing run has K=1 and a single column equal to the exchange's
 * own sunVisibility.
 *
 * The two gain fields are baked beside it by BakeShortwaveGains. They carry
 * the light that reached an element off something else, which the direct term
 * cannot: a north wall in a street sees no sun at any hour and is still warm,
 * because the road in front of it is bright.
 */
struct SunVisibilityTable {
    Vector<f64> sampleTime_h;  ///< sorted, K entries (K >= 1)
    Vector<f32> visibility;    ///< K * elementCount, sample-major
    /// Where the sun was for each column, from surface toward it. K entries
    /// when the builder recorded them; only the bounce bake reads it, and it
    /// leaves the reflected gain empty when it is missing.
    Vector<glm::vec3> sampleDirection;
    /// R_ik: direct sunlight reaching element i after one bounce off the other
    /// elements, per unit direct normal irradiance, for sun column k. Same
    /// K * elementCount sample-major layout as the visibility. Empty means no
    /// bounce was baked, which is the same answer as all zeros.
    Vector<f32> reflectedGain;
    /// G_i: the sky's own diffuse light reaching element i, per unit diffuse
    /// horizontal irradiance -- its sky fraction plus what one bounce off the
    /// other elements adds. Time-invariant, so one column rather than K.
    /// Empty means fall back to the exchange's bare sky fraction.
    Vector<f32> diffuseGain;

    [[nodiscard]] usize SampleCount() const { return sampleTime_h.size(); }
    [[nodiscard]] usize ElementCount() const {
        return sampleTime_h.empty() ? 0 : visibility.size() / sampleTime_h.size();
    }
    [[nodiscard]] const f32* Column(usize k) const {
        return visibility.data() + k * ElementCount();
    }
    /// The reflected-gain column matching Column(k), or nullptr when none was
    /// baked.
    [[nodiscard]] const f32* ReflectedColumn(usize k) const {
        return reflectedGain.empty() ? nullptr : reflectedGain.data() + k * ElementCount();
    }

    /// Interpolation indices and blend for time @p t: result is
    /// (1-blend)*Column(a) + blend*Column(b).
    void SampleIndices(f64 t, usize& a, usize& b, f64& blend) const;
};

/**
 * @brief The short-wave geometry one step needs, already interpolated
 *
 * Three per-element spans rather than three parameters, because they are one
 * thing: what fraction of each of the sun, its bounce, and the sky dome this
 * element receives right now. An empty span means that path contributes
 * nothing -- except diffuseGain, where empty falls back to the exchange's sky
 * fraction, the answer with no bounce.
 */
struct ShortwaveSample {
    std::span<const f32> sunVisibility;
    std::span<const f32> reflectedGain;
    std::span<const f32> diffuseGain;
};

/// One step in a batch, carrying the forcing and where in the sun table it is.
struct ThermalBatchStep {
    ThermalForcing forcing;
    f64 dt_s = 60.0;
    usize sunSampleA = 0;
    usize sunSampleB = 0;
    f64 sunBlend = 0.0;
};

}  // namespace quantiloom::thermal
