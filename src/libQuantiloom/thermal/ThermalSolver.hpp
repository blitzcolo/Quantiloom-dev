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
    /// sky_temperature_k, and optionally diffuse_irradiance_w_m2 and
    /// relative_humidity. Linearly interpolated, held flat outside its range;
    /// a row that stops early keeps the defaults for what it did not say.
    String forcingFile;

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
 */
[[nodiscard]] ThermalResult RunThermalSolve(const Scene& scene, const ThermalConfig& config,
                                            const ExchangeGeometry& exchange,
                                            const SunVisibilityTable& sunTable = {});

/// The sky-only exchange the fallback uses: every row empty, every sky
/// fraction 1, every element in full sun. Exposed so a caller can say
/// explicitly that this is what it wants.
[[nodiscard]] ExchangeGeometry MakeOpenSkyExchange(usize elementCount);

/// Read a forcing CSV. Returns an empty vector and logs when it cannot be
/// read, which the caller treats as constant forcing.
[[nodiscard]] Vector<std::pair<f64, ThermalForcing>> LoadForcingCsv(const String& path);

/// Interpolate a forcing series at one time, holding the ends flat.
[[nodiscard]] ThermalForcing SampleForcing(
    const Vector<std::pair<f64, ThermalForcing>>& series, f64 time_h,
    const ThermalForcing& fallback);

}  // namespace quantiloom::thermal
