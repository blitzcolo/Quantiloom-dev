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
/// temperature; a free-standing plate has nothing; a panel over a bay has air.
enum class InteriorBoundary : u8 {
    Adiabatic = 0,  ///< no heat crosses the back face
    FixedTemperature,  ///< held at interiorTemperature_K, as a room would
    /// Convecting to air at interiorTemperature_K, and radiating to a
    /// background at the same temperature. What a thin panel with its back
    /// open to a shaded interior has: a fuselage skin over a bay, a sign, a
    /// fence. Distinct from Adiabatic, which is a panel whose back is
    /// perfectly insulated, and from FixedTemperature, which pins the back
    /// node itself and lets a slab of any thickness dump heat into it.
    AmbientInterior
};

/// Where the convective coefficient comes from when the forcing does not state
/// one outright.
enum class ConvectionModel : u8 {
    /// The material's own number, unchanged all day. What every scene written
    /// before the others existed gets.
    Constant = 0,
    /// Forced convection from the wind: h = a + b U.
    Wind,
    /// The wind law, corrected for how the air is layered over the surface:
    /// damped when the surface is colder than the air, floored by free
    /// convection when it is warmer.
    Stability
};

/**
 * @brief A material property the trajectory can be differentiated with respect to
 *
 * Which ones are worth having is decided by what a fit or an uncertainty
 * budget needs: the three that set the surface's exchange with the outside,
 * and the two that set how fast the slab behind it responds.
 *
 * The derivative is taken holding the NEIGHBOURS' temperatures fixed, exactly
 * as dT/dv is -- what it would add is the second-order fact that a warmer
 * patch warms what can see it. With lateral conduction on, elements that
 * conduct into each other do share the derivative, which is the right reading
 * for a fit: one h is fitted for a material, not one per triangle.
 */
enum class ThermalParameter : u8 {
    Convection = 0,   ///< h, W/(m^2 K). Only under the constant law: the others derive it
    Emissivity,       ///< eps_lw, what the surface radiates with
    Absorptivity,     ///< alpha_s, the short-wave fraction it absorbs
    Conductivity,     ///< k, W/(m K)
    HeatCapacity,     ///< rho c, J/(m^3 K), the two as the balance uses them
    Count
};

/**
 * @brief The surface energy balance for one element, term by term
 *
 * Every number is a flux density in W/m^2, signed POSITIVE INTO the exposed
 * face. They sum to the rate the surface half-cell is storing heat, which is
 * what makes the set readable as an explanation rather than as six unrelated
 * numbers: a surface that is warming has a positive sum, and which term made
 * it positive is the answer to why.
 *
 * The signs are worth stating because two of them are usually negative in
 * daylight. Long wave is a NET: what the hemisphere sends back minus what this
 * element radiates, so a surface warmer than its sky is losing by it.
 * Evaporation only ever leaves.
 */
struct SurfaceFluxes {
    /// Sunlight absorbed: the direct beam through whatever shadow the element
    /// is in, plus what neighbours reflected onto it, plus the sky's diffuse.
    f64 shortwave_W_m2 = 0.0;
    /// Net long wave against the hemisphere and the sky.
    f64 longwave_W_m2 = 0.0;
    /// h (T_air - T_surface). Positive when the air is the warmer.
    f64 convection_W_m2 = 0.0;
    /// Evaporation, which is never positive.
    f64 latent_W_m2 = 0.0;
    /// Conduction from the node below into the surface node. Negative through
    /// a sunlit afternoon, when the slab is where the heat is going.
    f64 conduction_W_m2 = 0.0;
    /// What the neighbouring elements conduct in across the shared edges.
    /// Zero unless the mesh was built with contacts and the stepper carries
    /// them.
    f64 lateral_W_m2 = 0.0;

    [[nodiscard]] f64 Sum() const {
        return shortwave_W_m2 + longwave_W_m2 + convection_W_m2 + latent_W_m2 +
               conduction_W_m2 + lateral_W_m2;
    }
};

/// The name a config writes for each, and the name a dump column carries.
[[nodiscard]] const char* ThermalParameterName(ThermalParameter parameter);
/// The reverse; Count for a name this build does not know.
[[nodiscard]] ThermalParameter ThermalParameterFromName(StringView name);

/**
 * @brief The correlation that turns wind and a temperature difference into h
 *
 * A convective coefficient is not a material property, and a constant one
 * cannot describe a day: it is set by the wind and by whether the air over the
 * surface is being stirred or is lying stably on top of it, and those reverse
 * between afternoon and midnight. Measured against a SURFRAD station, one
 * value fitted to the daytime signal over-warmed the nights by up to 1.8 K.
 *
 * That bias is the whole reason for the stability model, and it fixes the sign
 * of the correction. At night the ground is colder than the air, so convection
 * is a SOURCE; too large a coefficient pours in heat, and the model has to
 * make h SMALLER there rather than larger. A cold surface under still air is
 * stably stratified -- the densest air is already at the bottom, so there is
 * nothing to overturn and the exchange is suppressed. Free convection is the
 * opposite case: a surface hotter than the air raises plumes, and that is a
 * floor under h rather than a cap.
 *
 * So the stability law is the wind law damped on the stable side and floored
 * by free convection on the unstable side:
 *
 *     Ri  = g z (T_air - T_s) / (T_air max(U, 0.5)^2)     bulk Richardson
 *     h   = (a + b U) / (1 + d Ri)                        stable, Ri > 0
 *     h   = max(a + b U, C |T_s - T_air|^(1/3))           unstable, Ri <= 0
 *
 * The stable branch is the Louis 1979 form with its usual d = 10, and it is
 * floored at 1 W/(m^2 K) -- a real stable layer still exchanges something, and
 * an h of zero would let a surface radiate to the sky with nothing at all
 * drawing heat back.
 *
 * Held by the stepper rather than by the forcing, because it says how the
 * balance is modelled rather than what the weather is doing. The wind speed
 * itself is in ThermalForcing, where the rest of the weather is.
 */
struct ConvectionLaw {
    ConvectionModel model = ConvectionModel::Constant;

    /// McAdams for a flat plate in parallel flow, h = 5.7 + 3.8 U in
    /// W/(m^2 K) for U in m/s. The same correlation the SURFRAD comparison
    /// applied outside the renderer before this could do it inside.
    f64 windIntercept_W_m2K = 5.7;
    f64 windSlope_W_s_m3K = 3.8;

    /// Free convection over a horizontal plate, h = C |T_s - T_air|^(1/3),
    /// C in W/(m^2 K^(4/3)). 1.52 is the usual turbulent value. It is a floor
    /// on the unstable side only: on the stable side there is no free
    /// convection to have.
    f64 freeCoefficient = 1.52;

    /// Where the air temperature and the wind are measured, in metres. Two is
    /// the screen height a weather station reports at, and the Richardson
    /// number is only meaningful against the height its gradient spans.
    f64 referenceHeight_m = 2.0;
    /// The Louis stable-side damping constant, h = h_forced / (1 + d Ri).
    /// Zero turns the damping off and leaves the wind law with a free-
    /// convection floor.
    f64 stableDamping = 10.0;
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

    /// A flux entering the back face, W/m^2 of surface. What is behind the
    /// surface rather than what falls on it: an engine, a battery, a compartment
    /// with people in it. Positive heats the slab from behind, which is the
    /// only way a shaded surface can be warmer than everything around it --
    /// and in an infrared scene that is the whole signature.
    ///
    /// Read by the Adiabatic and AmbientInterior boundaries. Under
    /// FixedTemperature the back node is pinned, so whatever flux is applied
    /// there is absorbed by the thing doing the pinning and changes nothing;
    /// the config warns rather than pretending.
    f32 internalHeat_W_m2 = 0.0f;

    InteriorBoundary interiorBoundary = InteriorBoundary::Adiabatic;
    /// The temperature behind the surface: what FixedTemperature pins the back
    /// node to, and what AmbientInterior convects and radiates against.
    f32 interiorTemperature_K = 293.15f;
    /// Convective coefficient at the back face, W/(m^2 K), for
    /// AmbientInterior. Still air inside a bay rather than the wind outside it.
    f32 interiorConvection_W_m2K = 3.0f;

    /// Two sides of one thin slab rather than a surface with something behind
    /// it: a car panel, a road sign, a tent, an aircraft skin.
    ///
    /// An asset models such a thing as two sheets of triangles, and solved
    /// naively that is two independent slabs each insulating against nothing --
    /// so a panel in the sun comes out as hot as if its back were against a
    /// wall, and the back face itself sits wherever the initial condition left
    /// it. With this the two faces share one column and the back row is a full
    /// surface balance rather than a boundary condition.
    ///
    /// The pairing is a heuristic over geometry nobody authored for it, so the
    /// mesh reports how many triangles of a shell material found no partner.
    /// Those are solved one-sided, which is the old behaviour rather than a
    /// failure.
    bool isShell = false;

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

    /// Who touches whom, and how well heat crosses the join: g_ij in W/(m K),
    /// the conductance per metre of slab depth between two elements sharing an
    /// edge. Empty is the model this solver had until now, where every element
    /// is an independent column and a shadow edge is therefore as sharp as the
    /// mesh -- which is right for dry sand at an hour's timescale and wrong for
    /// a metal panel at any.
    ///
    /// Beside the view factors rather than in them because it is the same
    /// question asked of contact rather than of sight, and because putting it
    /// here is what lets every stepper reach it: they all already take an
    /// ExchangeGeometry.
    CsrMatrix lateral;
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

    /// dT/dv_k for a sliding window of the most recent sun columns: how far
    /// each node would move per unit of this element's visibility IN COLUMN k
    /// alone, rather than across the whole day at once.
    ///
    /// Why both. sunSensitivity_K above answers "what if this element had seen
    /// more sun, all day", and a shading pass applies it with the visibility
    /// difference it traces NOW -- which assumes the pixel's shadow history
    /// matches its present shadow. Under a moving sun it does not: a pixel
    /// that is shaded now may have been lit an hour ago, and the ground under
    /// it is still warm. Each of these answers "what if this element had seen
    /// more sun AT THAT HOUR", and a shading pass can trace the pixel's own
    /// visibility toward each of those sun positions.
    ///
    /// The window is the point rather than a limitation. What is not in it is
    /// not lost: the shading pass applies (total - sum of the tracked columns)
    /// with the present sun, which is exactly what it did before any of this
    /// existed. So a slot count of zero is the old behaviour, and each slot
    /// moves one column's worth of the answer from "assume the history looks
    /// like now" to "trace it".
    ///
    /// Layout: slot-major over the element-major blocks, so slot s owns
    /// [s * ElementCount() * nodeCount, (s+1) * ...).
    Vector<f64> lagSensitivity_K;
    /// Which sun column each slot tracks, kNoLagColumn for a slot that has not
    /// been claimed yet. Size is the slot count.
    Vector<u32> lagColumn;

    /// dT/dp for each material parameter this state was asked to
    /// differentiate, in the order `parameters` lists them. Same
    /// parameter-major layout over the element-major blocks as the lag
    /// tangents, and the same reason for existing: a number a fit or an
    /// uncertainty budget needs is the derivative of the trajectory, and the
    /// trajectory is the only thing that has it.
    Vector<f64> parameterSensitivity;
    Vector<ThermalParameter> parameters;

    static constexpr u32 kNoLagColumn = 0xFFFFFFFFu;

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

    /// How many sun columns this state tracks separately. Zero is the whole of
    /// the old behaviour.
    [[nodiscard]] u32 LagSlots() const { return static_cast<u32>(lagColumn.size()); }
    [[nodiscard]] bool HasLagSensitivity() const {
        return !lagColumn.empty() &&
               lagSensitivity_K.size() == lagColumn.size() * temperature_K.size();
    }
    [[nodiscard]] f64 SurfaceLagSensitivity(const usize slot, const usize element) const {
        return lagSensitivity_K[(slot * ElementCount() + element) * nodeCount];
    }

    [[nodiscard]] bool HasParameterSensitivity() const {
        return !parameters.empty() &&
               parameterSensitivity.size() == parameters.size() * temperature_K.size();
    }
    [[nodiscard]] f64 SurfaceParameterSensitivity(const usize index,
                                                  const usize element) const {
        return parameterSensitivity[(index * ElementCount() + element) * nodeCount];
    }

    /// What one snapshot of this state costs. The timeline stores whole copies
    /// of it as checkpoints, so this is what a scrub backwards is paid for in
    /// memory, and it is here rather than at the caller so that a state vector
    /// added later cannot be left out of the total by being forgotten in one
    /// file. Every vector this struct owns belongs in the sum.
    [[nodiscard]] usize ByteSize() const {
        return (temperature_K.size() + sunSensitivity_K.size() + lagSensitivity_K.size() +
                parameterSensitivity.size()) *
                   sizeof(f64) +
               lagColumn.size() * sizeof(u32) +
               parameters.size() * sizeof(ThermalParameter);
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
    /// Convective exchange coefficient for this instant, W/(m^2 K). Zero or
    /// negative means "the material's own", which is what every scene that
    /// does not supply one gets.
    ///
    /// It is here rather than only on the material because a constant cannot
    /// describe a day. The coefficient is set by the wind and by whether the
    /// air over the surface is being stirred or is sitting stably on top of
    /// it, and those reverse between afternoon and midnight. Measured against
    /// a SURFRAD station, one constant fitted to the daytime signal
    /// over-warmed the nights by up to 1.8 K -- at night the ground is colder
    /// than the air, so convection is a source, and too large a coefficient
    /// pours in heat that the real stable boundary layer withholds.
    f64 convection_W_m2K = 0.0;
    /// Wind speed at the reference height, m/s. What ConvectionLaw reads when
    /// the column above is silent. A file that carries no wind describes a
    /// calm, which under the stability law is free convection rather than no
    /// convection at all.
    f64 windSpeed_m_s = 0.0;
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

    /// Which two sun columns the visibility above was interpolated from, and
    /// how far between them. A step that knows this can attribute its
    /// short-wave source to the columns that produced it, which is what the
    /// per-column tangents need; one that does not -- the steady-state
    /// relaxation, a caller that passes only a visibility -- leaves
    /// `columnsKnown` false and the per-column tangents alone.
    usize columnA = 0;
    usize columnB = 0;
    f64 columnBlend = 0.0;
    bool columnsKnown = false;
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
