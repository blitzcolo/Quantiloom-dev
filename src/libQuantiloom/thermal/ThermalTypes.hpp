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
    u32 nodeCount = 0;

    [[nodiscard]] usize ElementCount() const {
        return nodeCount == 0 ? 0 : temperature_K.size() / nodeCount;
    }
    [[nodiscard]] f64 Surface(const usize element) const {
        return temperature_K[element * nodeCount];
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
    glm::vec3 sunDirection{0.0f, 1.0f, 0.0f};  ///< from surface toward the sun
    /// Effective sky temperature: the blackbody radiating what the sky does.
    /// SkyThermal derives it from the air temperature and humidity, or the NN
    /// atmosphere's own downwelling gives it directly.
    f64 skyTemperature_K = 268.0;
};

/**
 * @brief Sun visibility at several times of day, for diurnal interpolation
 *
 * Each column is one precomputed sun direction, one visibility fraction per
 * element; layout is sample-major (column k starts at k * elementCount).
 * A constant-forcing run has K=1 and a single column equal to the exchange's
 * own sunVisibility.
 */
struct SunVisibilityTable {
    Vector<f64> sampleTime_h;  ///< sorted, K entries (K >= 1)
    Vector<f32> visibility;    ///< K * elementCount, sample-major

    [[nodiscard]] usize SampleCount() const { return sampleTime_h.size(); }
    [[nodiscard]] usize ElementCount() const {
        return sampleTime_h.empty() ? 0 : visibility.size() / sampleTime_h.size();
    }
    [[nodiscard]] const f32* Column(usize k) const {
        return visibility.data() + k * ElementCount();
    }

    /// Interpolation indices and blend for time @p t: result is
    /// (1-blend)*Column(a) + blend*Column(b).
    void SampleIndices(f64 t, usize& a, usize& b, f64& blend) const;
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
