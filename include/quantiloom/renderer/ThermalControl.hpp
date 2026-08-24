/**
 * @file ThermalControl.hpp
 * @brief Parameter and status structs for the interactive thermal solve
 *
 * Pure POD, no QL_API: the facade methods that pass them are exported, and
 * their layout is stable enough that the SDK installs them alongside the
 * existing public headers, but adding every member to the export list would
 * be churn for a struct whose fields the host reads directly.
 */

#pragma once

#include "core/Types.hpp"

namespace quantiloom {

enum class ThermalInitialCondition : u8 {
    Steady = 0,
    Uniform
};

struct ThermalSolveParams {
    f64 startTime_h = 0.0;
    f64 timestep_s = 60.0;
    u32 layerCount = 10;
    ThermalInitialCondition initial = ThermalInitialCondition::Steady;
    f64 initialTemperature_K = 288.15;
    u32 exchangeRays = 256;
    u32 exchangeTopK = 32;
    f64 airTemperature_K = 288.15;
    f64 sunIrradiance_W_m2 = 0.0;
    /// Diffuse horizontal irradiance, W/m^2 -- the sky dome rather than the
    /// disc, and the whole of the solar input under overcast.
    f64 diffuseIrradiance_W_m2 = 0.0;
    f64 skyTemperature_K = 268.0;
    /// Percent. Only the latent term reads it, and only for materials with a
    /// wetness factor. Same quantity the clear-sky model uses.
    f64 relativeHumidity = 50.0;
    String forcingFile;
    f64 checkpointStride_h = 1.0;

    /// Carry dT/dv through the trajectory, so the shading pass can resolve a
    /// shadow edge inside a triangle rather than at its border. On by default;
    /// off exists so the two renders can be compared, which is the only way to
    /// show what the correction is worth.
    ///
    /// Off does not merely suppress the correction at the shader -- it sizes
    /// the tangent out of the solve state, so the trajectory neither carries
    /// nor pays for it. Changing it therefore rebuilds the timeline.
    bool sunCorrection = true;

    /// Where DumpThermalElements() writes, when it is called with no path of
    /// its own. Empty for the scenes that never want one.
    ///
    /// Naming a file here does NOT make the solve write it. The offline path
    /// runs once and a dump per run is exactly right; a viewport re-solves on
    /// every scrub of the hour slider, and a parameter that wrote a file each
    /// time would turn dragging a slider into hundreds of writes. So the path
    /// travels with the parameters -- a config carries it, a host round-trips
    /// it -- and the write is an explicit call.
    String dumpElementsFile;
};

struct ThermalMaterialParams {
    f32 conductivity_W_mK = 0.0f;
    f32 density_kg_m3 = 2000.0f;
    f32 specificHeat_J_kgK = 900.0f;
    f32 thickness_m = 0.2f;
    f32 convection_W_m2K = 5.0f;
    f32 shortwaveAbsorptivity = 0.7f;
    /// 0 for dry, 1 for open water. Evaporation is what puts a lawn ten
    /// degrees below the pavement beside it under the same sun.
    f32 wetnessFactor = 0.0f;
    bool interiorFixedTemperature = false;
    f32 interiorTemperature_K = 293.15f;
};

struct ThermalSolveStatus {
    bool enabled = false;
    bool solveValid = false;
    bool exchangeValid = false;
    u32 elementCount = 0;
    u32 participatingElements = 0;
    u32 exchangeNonZeros = 0;
    u32 exchangeRunCount = 0;
    /// Sun columns the trajectory interpolates between: one for constant
    /// forcing, one per row of a forcing file. More than one is what makes a
    /// shadow move across the day rather than sit where it was at hour zero.
    u32 sunSampleCount = 0;
    u32 lastStepCount = 0;
    u32 checkpointCount = 0;
    f64 currentTime_h = 0.0;
    f64 minTemperature_K = 0.0;
    f64 maxTemperature_K = 0.0;
    f64 meanTemperature_K = 0.0;
    f64 shortestTimeConstant_s = 0.0;
    f64 sliderStartTime_h = 0.0;
    f64 sliderEndTime_h = 24.0;
    String stepperName;
    String error;
};

}  // namespace quantiloom
