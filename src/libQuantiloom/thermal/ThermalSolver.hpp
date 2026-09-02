/**
 * @file ThermalSolver.hpp
 * @brief From a scene and a time of day to a temperature per triangle
 *
 * The orchestration: build the elements, get the view factors, pick a starting
 * state, and step to the hour asked for. What comes back is one temperature per
 * triangle, which the renderer uploads and the closest-hit shader reads in
 * place of the material's own.
 *
 * The physics is in CpuCrankNicolsonStepper; the geometry is in
 * ThermalExchangePrecompute, or in the analytic fallback here when there is no
 * device to run it on.
 */

#pragma once

#include "scene/Scene.hpp"
#include "thermal/ThermalStepper.hpp"
#include "thermal/ThermalTypes.hpp"

#include <unordered_map>

namespace quantiloom::thermal {

/// How to start a run. A day's simulation forgets its initial condition within
/// a few hours for a thin surface and never quite does for a thick one, which
/// is what makes the choice matter.
enum class InitialCondition : u8 {
    /// Every element at initialTemperature_K.
    Uniform = 0,
    /// The temperature each element would settle at under the starting
    /// forcing, found by stepping with a long timestep until nothing moves.
    /// What a scene wants when the hour being rendered is the first one.
    Steady
};

/**
 * @brief Everything the [thermal] section says
 */
struct ThermalConfig {
    bool enabled = false;

    /// Hour of the simulated day to render, and where the run starts. A
    /// sequence renders one job per time, each starting from startTime_h --
    /// so a frame at 18:00 costs eighteen hours of stepping, not six.
    f64 time_h = 12.0;
    f64 startTime_h = 0.0;
    f64 timestep_s = 60.0;

    u32 nodeCount = 10;
    InitialCondition initial = InitialCondition::Steady;
    f64 initialTemperature_K = 288.15;
    /// How often the trajectory stores a full state, in simulated hours.
    /// Only the interactive path scrubs, but the offline one pays for the
    /// setting too -- a shorter stride is more memory and a cheaper replay.
    f64 checkpointStride_h = 1.0;

    /// Rays per element for the view-factor precompute, and how many of the
    /// resulting entries to keep. A row past the top few is mostly noise, and
    /// keeping it would make the matrix dense for no accuracy.
    u32 exchangeRays = 256;
    u32 exchangeTopK = 32;

    /// Constant forcing, used when no forcing file is given.
    f64 airTemperature_K = 288.15;
    f64 sunIrradiance_W_m2 = 0.0;
    f64 diffuseIrradiance_W_m2 = 0.0;
    glm::vec3 sunDirection{0.0f, 1.0f, 0.0f};
    f64 skyTemperature_K = 268.0;
    f64 relativeHumidity = 50.0;

    /// CSV of time-varying forcing: time_h, air_temperature_k,
    /// sun_irradiance_w_m2, sun_azimuth_deg, sun_elevation_deg,
    /// sky_temperature_k, and optionally diffuse_irradiance_w_m2,
    /// relative_humidity, convection_w_m2k and wind_speed_m_s. Linearly
    /// interpolated, held flat outside its range; a row that stops early keeps
    /// the defaults for what it did not say.
    ///
    /// The last two are two ways of saying the same thing, and the ninth wins:
    /// a file that carries a measured coefficient is stating what the
    /// correlations below are estimating.
    String forcingFile;

    /// Where h comes from when the forcing does not state one. Constant by
    /// default, which is the material's own number and the behaviour of every
    /// scene written before the others existed.
    ConvectionLaw convection;

    /// Carry dT/dv through the trajectory, so the shading pass can resolve a
    /// shadow edge inside a triangle rather than at its border. On by default;
    /// off exists so the two renders can be compared, which is the only way to
    /// show what the correction is worth. Turning it off does not merely
    /// suppress the correction at the shader -- it sizes the tangent out of
    /// ThermalState, so the solve neither carries nor pays for it.
    bool sunCorrection = true;

    /// Where to write the per-element solve, if anywhere. One row per triangle
    /// in BuildThermalMesh order: the centroid and normal the solve used, the
    /// temperature it reached, and the (dT/dv, v) pair the shader would apply.
    /// Empty writes nothing. The renderer's own outputs are images, and an
    /// image is the temperature after the per-pixel correction and the
    /// radiance inversion -- this is what the solver itself produced, which is
    /// what a mesh-resolution study has to measure against.
    String dumpElementsFile;

    /// Thermal properties by material name, from [[materials]].
    std::unordered_map<String, ThermalMaterial> materials;
};

/// What a run produced, for the log line and for the renderer.
struct ThermalResult {
    /// One temperature per triangle, in the order BuildThermalMesh produced
    /// them. Empty when the solve did not run.
    Vector<f32> surfaceTemperature_K;
    /// Element base per instance, for the shader's PrimitiveIndex() lookup.
    Vector<u32> instanceElementBase;

    /// dT/dv at the exposed face, one per triangle: how far this element's
    /// temperature would move if it saw a unit more of the sun, all day.
    /// Read from the trajectory's tangent, not from a steady-state formula --
    /// see ThermalState::sunSensitivity_K.
    Vector<f32> sunSensitivity_K;
    /// v: the sun visibility the solve used for each element at time_h. The
    /// pair (sensitivity, visibility) is what lets a shading pass replace a
    /// triangle-average shadow with the one it traced for its own pixel:
    ///     T(x) = T_element + (v(x) - v_element) * dT/dv
    /// A shadow boundary then lands where the geometry puts it instead of on
    /// the nearest triangle edge.
    Vector<f32> sunVisibility;
    /// Where the sun was at time_h, from surface toward it. The forcing file
    /// owns this and it need not agree with [lighting] sun_direction, so the
    /// shading pass has to be told rather than assume.
    glm::vec3 sunDirection{0.0f, 1.0f, 0.0f};

    u32 elementCount = 0;
    u32 participatingElements = 0;
    u32 exchangeNonZeros = 0;
    u32 stepsTaken = 0;
    f64 minTemperature_K = 0.0;
    f64 maxTemperature_K = 0.0;
    f64 meanTemperature_K = 0.0;
    /// Empty when the run succeeded.
    String error;
};

/**
 * @brief Run the balance from startTime_h to time_h
 *
 * @param scene     the geometry, and the materials the config's thermal
 *                  properties are matched to by name
 * @param config    the [thermal] section
 * @param exchange  view factors and sun visibility. Pass an empty one to have
 *                  the analytic fallback used instead: every element sees only
 *                  the sky, which is right for a scene with nothing in it to
 *                  shade anything else and wrong -- conservatively, too cold at
 *                  night -- for a street.
 * @param sunTable  per-sample sun visibility (empty → synthesised from
 *                  exchange.sunVisibility as a single column)
 * @param stepper   who advances the trajectory. Null, the default, builds a
 *                  CpuCrankNicolsonStepper here -- which keeps this callable
 *                  without a Vulkan device, as the tests need. A caller with a
 *                  device may pass the GPU stepper instead; it mirrors the
 *                  same maths in f32, so the two do not agree to the last bit
 *                  and their results must not be mixed.
 */
[[nodiscard]] ThermalResult RunThermalSolve(const Scene& scene, const ThermalConfig& config,
                                            const ExchangeGeometry& exchange,
                                            const SunVisibilityTable& sunTable = {},
                                            IThermalStepper* stepper = nullptr);

/**
 * @brief The per-material properties the solve will actually use
 *
 * One entry per scene material, in scene order: the config's [[materials]]
 * thermal block matched by name, over the long-wave emissivity read from the
 * material's own IR curve. The emissivity wins over anything the config typed
 * -- a material bound to a measured spectrum is solved at the Planck-weighted
 * band average of that curve, which differs from the typed number by about
 * 0.4 K in the trajectory.
 *
 * Hoisted out of RunThermalSolve because the solve cache has to hash exactly
 * these numbers. Two implementations of the merge would let the key and the
 * solve disagree, and a cache that keys on the wrong emissivity serves a
 * trajectory from the wrong surface without saying so.
 *
 * @param namedCount  optional out: how many scene materials matched a config
 *                    entry. Zero means the solve would fail for want of any
 *                    material with thermal properties.
 */
[[nodiscard]] Vector<ThermalMaterial> BuildSolvedMaterials(const Scene& scene,
                                                           const ThermalConfig& config,
                                                           u32* namedCount = nullptr);

/// The one line the downstream gates parse out of a render's log. It lives here
/// rather than at the end of RunThermalSolve so that a solve served from cache
/// prints it identically -- a gate that reads nothing reports every render
/// clean, which is the failure it exists to catch.
void LogThermalSolveSummary(const ThermalResult& result);

/// The sky-only exchange the fallback uses: every row empty, every sky
/// fraction 1, every element in full sun. Exposed so a caller can say
/// explicitly that this is what it wants.
[[nodiscard]] ExchangeGeometry MakeOpenSkyExchange(usize elementCount);

/// Write the solve out one row per element, for the studies that have to
/// measure the temperature field rather than look at it. Deliberately the
/// solver's own numbers: the renderer's images carry the per-pixel sun
/// correction and a radiance inversion on top, and a mesh-resolution study
/// needs the field underneath both. The material block above the table is what
/// the SOLVE saw -- a material bound to a measured spectrum is solved at the
/// Planck-weighted band average of that curve, not at the emissivity its config
/// typed, and reproducing a trajectory from the config instead is a 0.4 K error
/// that looks exactly like a result.
///
/// Takes the two fields rather than a ThermalResult so the interactive solve,
/// which produces its own result type, writes byte-identical files to the
/// offline one instead of a second format that drifts.
///
/// @param temperature_K  per element; an entry of 0 marks an element the solve
///                       did not participate in
/// @param sunSensitivity_K  per element, or empty when the tangent was not
///                       carried -- which the file distinguishes from zero,
///                       because a temperature that does not move is a
///                       different claim from not having asked
/// @param visibility     v per element at the dumped instant. It rides along
///                       because a pointwise reference integration assumes an
///                       element whose hemisphere is mostly sky, and a file
///                       where sky_fraction is far from one says the reference
///                       does not apply.
void DumpThermalElements(const String& path, const Vector<ThermalElement>& elements,
                         const Vector<ThermalMaterial>& materials,
                         const ExchangeGeometry& geometry,
                         const Vector<f32>& temperature_K,
                         const Vector<f32>& sunSensitivity_K,
                         const Vector<f32>& visibility);

/// Read a forcing CSV. Returns an empty vector and logs when it cannot be
/// read, which the caller treats as constant forcing.
[[nodiscard]] Vector<std::pair<f64, ThermalForcing>> LoadForcingCsv(const String& path);

/// The sun visibility every element had at one time, on the same two-column
/// interpolation the solver's own steps use. Falls back to the exchange's
/// single column, and with neither to full sun -- which is what a scene with
/// no shadowing precompute is.
[[nodiscard]] Vector<f32> SampleSunVisibilityAt(const SunVisibilityTable& table,
                                                const ExchangeGeometry& exchange,
                                                f64 time_h, usize elementCount);

/// Interpolate a forcing series at one time, holding the ends flat.
[[nodiscard]] ThermalForcing SampleForcing(
    const Vector<std::pair<f64, ThermalForcing>>& series, f64 time_h,
    const ThermalForcing& fallback);

}  // namespace quantiloom::thermal
