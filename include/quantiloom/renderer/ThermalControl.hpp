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

/// Where the convective coefficient comes from when the forcing file does not
/// carry one. Constant is the material's own number all day; Wind is
/// h = a + b U from the forcing's wind speed; Stability adds free convection,
/// which is what carries the exchange on a calm night.
enum class ThermalConvectionModel : u8 {
    Constant = 0,
    Wind,
    Stability
};

/// A material parameter the solve can carry a derivative with respect to,
/// beside the sun-visibility one it always carries. Each costs a state vector
/// and an elimination pass per step, so they are asked for by name rather than
/// all carried: a fit wants one or two, a viewport usually wants none.
///
/// Mirrors thermal::ThermalParameter, which is where the derivatives are
/// actually taken; this is the spelling that crosses the SDK boundary.
enum class ThermalSensitivityParameter : u8 {
    Convection = 0,  ///< h, W/(m^2 K). Only under the constant law
    Emissivity,      ///< eps_lw, what the surface radiates with
    Absorptivity,    ///< alpha_s, the short-wave fraction it absorbs
    Conductivity,    ///< k, W/(m K)
    HeatCapacity     ///< rho c, J/(m^3 K)
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

    /// The convection correlation, and its constants: McAdams h = 5.7 + 3.8 U
    /// for the wind, C |T_s - T_air|^(1/3) with C = 1.52 as the free-convection
    /// floor when the surface is the warmer, and a Louis damping
    /// h / (1 + d Ri) over a reference height when it is the colder.
    /// Anything but Constant is solved on the CPU stepper -- the GPU one does
    /// not evaluate them, and a step that quietly used a different h would be a
    /// wrong trajectory rather than a slower one.
    ThermalConvectionModel convectionModel = ThermalConvectionModel::Constant;
    f64 convectionWindA_W_m2K = 5.7;
    f64 convectionWindB_W_s_m3K = 3.8;
    f64 convectionFreeC = 1.52;
    f64 convectionReferenceHeight_m = 2.0;
    f64 convectionStableDamping = 10.0;

    /// Let heat cross the edge between two triangles of one object. Off by
    /// default, and CPU-only for the same reason the convection laws are.
    bool lateralConduction = false;

    /// How many of the sun's recent columns carry a tangent of their own, so a
    /// shading pass can trace the pixel's shadow at the hour it was cast
    /// instead of assuming it looked like now. Zero is the old behaviour; each
    /// slot costs a state vector, an elimination pass and a ray per shaded
    /// pixel, and only a forcing file with several sun columns has anything
    /// for them to track.
    u32 sunMemoryLags = 0;

    /// Carry dT/dv through the trajectory, so the shading pass can resolve a
    /// shadow edge inside a triangle rather than at its border. On by default;
    /// off exists so the two renders can be compared, which is the only way to
    /// show what the correction is worth.
    ///
    /// Off does not merely suppress the correction at the shader -- it sizes
    /// the tangent out of the solve state, so the trajectory neither carries
    /// nor pays for it. Changing it therefore rebuilds the timeline.
    bool sunCorrection = true;

    /// Material parameters to differentiate the trajectory with respect to.
    /// Empty is the default and costs nothing. What they are for is a fit --
    /// dT/dh against a measured series is what turns a guessed convection
    /// coefficient into a measured one -- and a viewport that wants to show
    /// what a slider would do before the re-solve finishes.
    ///
    /// Like sunCorrection, this sizes the state rather than being read later,
    /// so changing it rebuilds the timeline.
    Vector<ThermalSensitivityParameter> parameterSensitivities;

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
    /// A flux entering the back face, W/m^2: an engine, a battery, a
    /// compartment. The only way a shaded surface can be the warmest thing in
    /// an infrared scene.
    f32 internalHeat_W_m2 = 0.0f;
    /// Two sides of one thin slab rather than a surface with something behind
    /// it. The pair shares a column with a full surface balance at each end,
    /// and the three interior fields below then have nothing to act on.
    ///
    /// Turning it on repairs the mesh, not just a flag: the pairing is found
    /// while the geometry is being walked.
    bool isShell = false;
    bool interiorFixedTemperature = false;
    /// The back face convects and radiates to interiorTemperature_K instead of
    /// being insulated. A panel over a bay rather than a wall. Ignored when
    /// interiorFixedTemperature pins the node outright.
    bool interiorAmbient = false;
    f32 interiorTemperature_K = 293.15f;
    f32 interiorConvection_W_m2K = 3.0f;
};

/**
 * @brief One element's surface energy balance, term by term
 *
 * Every number is W/m^2, signed POSITIVE INTO the exposed face, and the six
 * sum to the rate the surface is storing heat. That is what makes them an
 * explanation rather than six unrelated readings: a surface that is warming
 * has a positive sum, and which term made it positive is the answer to why.
 *
 * Two are usually negative in daylight, and the reason is in their
 * definitions. Long wave is a NET -- what the hemisphere sends back minus what
 * this element radiates -- so a surface warmer than its sky loses by it.
 * Evaporation only ever leaves.
 */
struct ThermalSurfaceFluxes {
    f64 shortwave_W_m2 = 0.0;   ///< absorbed sun: direct through its shadow, reflected, diffuse
    f64 longwave_W_m2 = 0.0;    ///< net against the hemisphere and the sky
    f64 convection_W_m2 = 0.0;  ///< h (T_air - T_surface)
    f64 latent_W_m2 = 0.0;      ///< evaporation, never positive
    f64 conduction_W_m2 = 0.0;  ///< from the node below into the surface node
    f64 lateral_W_m2 = 0.0;     ///< across shared edges; zero unless the mesh carries contacts
};

/**
 * @brief What one element did over a stretch of the day
 *
 * The probe behind a time-series panel. Sampled by replaying the trajectory,
 * which is what makes it cheap: the checkpoints are already there, so asking
 * about an element costs stepping between them rather than solving again.
 *
 * Every vector has the same length, or `fluxes` is empty -- which is what a
 * stepper that does not decompose its own balance reports, and the difference
 * between "no heat moved" and "nobody asked" is worth keeping.
 */
struct ThermalElementTrajectory {
    /// Hours, ascending, the samples the caller asked for.
    Vector<f64> time_h;
    /// The exposed face.
    Vector<f64> surfaceTemperature_K;
    /// The back face: the other end of the slab, which is what says whether a
    /// wall has finished responding to yesterday.
    Vector<f64> backTemperature_K;
    /// Empty when the stepper declines to decompose its balance.
    Vector<ThermalSurfaceFluxes> fluxes;
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
