// ============================================================================
// Quantiloom - Thermal Column Tool
// ============================================================================
// One surface element, one 1-D column, one trajectory -- the surface energy
// balance with the scene taken away.
//
// Two experiments need exactly this and neither wants a renderer:
//
//   * Validating the solver against a measured site.  A flat, open field
//     station is one element under an unoccluded sky; putting geometry around
//     it would add view factors the station does not have and call the
//     disagreement physics.
//
//   * Building a pointwise reference for a mesh-resolution study.  The model
//     gives every element an independent column driven by its own visibility
//     history and no lateral conduction, so the exact temperature at a point
//     is that point's own visibility series integrated through the same column.
//     That reference costs O(sample count) and is independent of how finely
//     the scene happens to be tessellated -- which is the whole point, since
//     the tessellation is the variable under study.
//
// It is deliberately the same code the renderer runs: ThermalTimeline over
// CpuCrankNicolsonStepper, so the steady-state initial condition, the fixed
// time grid, the midpoint forcing sample and the tangent all behave here
// exactly as they do in a scene.  A reference implemented twice is a reference
// that disagrees for its own reasons.
//
// Usage:
//   thermal_column_tool <spec.toml> [--visibility vis.csv] [--out traj.csv]
//
// See the header comment on ReadSpec below for the TOML the spec file takes.
// ============================================================================

#include "core/Config.hpp"
#include "core/Log.hpp"
#include "thermal/CpuCrankNicolsonStepper.hpp"
#include "thermal/ShortwaveGains.hpp"
#include "thermal/ThermalSolver.hpp"
#include "thermal/ThermalTimeline.hpp"
#include "thermal/ThermalTypes.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

using namespace quantiloom;
using namespace quantiloom::thermal;

namespace {

struct Spec {
    ThermalMaterial material;

    f64 startTime_h = 0.0;
    f64 endTime_h = 24.0;
    f64 timestep_s = 60.0;
    u32 nodeCount = 10;
    InitialCondition initial = InitialCondition::Steady;
    f64 initialTemperature_K = 288.15;
    f64 checkpointStride_h = 1.0;
    f64 outputStep_h = 0.25;
    bool sunCorrection = true;

    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    f32 area_m2 = 1.0f;
    f32 skyFraction = 1.0f;

    String forcingFile;
    ThermalForcing constant;
};

/// The sun direction the forcing CSV's own reader builds, so a constant
/// forcing and a one-row file mean the same thing by an azimuth.
glm::vec3 SunFrom(const f64 azimuth_deg, const f64 elevation_deg) {
    const f64 az = azimuth_deg * std::numbers::pi / 180.0;
    const f64 el = elevation_deg * std::numbers::pi / 180.0;
    return glm::normalize(glm::vec3(static_cast<f32>(std::cos(el) * std::sin(az)),
                                    static_cast<f32>(std::sin(el)),
                                    static_cast<f32>(-std::cos(el) * std::cos(az))));
}

/**
 * @brief Read the spec TOML
 *
 * ```toml
 * [column]                          # the slab, keys named as [[materials]] names them
 * thermal_conductivity_w_mk = 0.3
 * density_kg_m3             = 1600.0
 * specific_heat_j_kgk       = 800.0
 * thickness_m               = 0.15
 * convection_h_w_m2k        = 14.0
 * shortwave_absorptivity    = 0.72
 * ir_emissivity             = 0.90
 * wetness_factor            = 0.0
 * interior_bc               = "adiabatic"   # or "fixed"
 * interior_temperature_k    = 293.15
 *
 * [solve]                           # keys named as [thermal] names them
 * start_time_h        = 0.0
 * end_time_h          = 24.0
 * timestep_s          = 60.0
 * layers              = 10
 * initial             = "steady"            # or "uniform"
 * initial_temperature_k = 288.15
 * checkpoint_stride_h = 1.0
 * output_step_h       = 0.25                # how often a row is written
 * sun_correction      = true                # carry dT/dv
 *
 * [geometry]                        # what the element is, absent a scene
 * normal       = [0.0, 1.0, 0.0]
 * area_m2      = 1.0
 * sky_fraction = 1.0                # 1 is an unoccluded sky; lower it to
 *                                   # stand in for a horizon that is not open
 *
 * [forcing]
 * file = "site.csv"                 # the eight-column CSV of [thermal]
 * # ...or, with no file, the constant forcing:
 * air_temperature_k        = 293.15
 * sun_irradiance_w_m2      = 900.0
 * diffuse_irradiance_w_m2  = 100.0
 * sky_temperature_k        = 268.0
 * relative_humidity        = 50.0
 * sun_azimuth_deg          = 180.0
 * sun_elevation_deg        = 60.0
 * ```
 */
Spec ReadSpec(const Config& config, const std::filesystem::path& specDir) {
    Spec spec;

    ThermalMaterial& m = spec.material;
    m.conductivity_W_mK = config.GetFloat("column.thermal_conductivity_w_mk", 0.3f);
    m.density_kg_m3 = config.GetFloat("column.density_kg_m3", 1600.0f);
    m.specificHeat_J_kgK = config.GetFloat("column.specific_heat_j_kgk", 800.0f);
    m.thickness_m = config.GetFloat("column.thickness_m", 0.15f);
    m.convection_W_m2K = config.GetFloat("column.convection_h_w_m2k", 14.0f);
    m.shortwaveAbsorptivity = config.GetFloat("column.shortwave_absorptivity", 0.72f);
    m.longwaveEmissivity = config.GetFloat("column.ir_emissivity", 0.90f);
    m.wetnessFactor = config.GetFloat("column.wetness_factor", 0.0f);
    m.interiorTemperature_K = config.GetFloat("column.interior_temperature_k", 293.15f);
    m.interiorBoundary = config.GetString("column.interior_bc", "adiabatic") == "fixed"
                             ? InteriorBoundary::FixedTemperature
                             : InteriorBoundary::Adiabatic;

    spec.startTime_h = config.GetDouble("solve.start_time_h", 0.0);
    spec.endTime_h = config.GetDouble("solve.end_time_h", 24.0);
    spec.timestep_s = config.GetDouble("solve.timestep_s", 60.0);
    spec.nodeCount = config.GetUInt("solve.layers", 10);
    spec.initialTemperature_K = config.GetDouble("solve.initial_temperature_k", 288.15);
    spec.checkpointStride_h = config.GetDouble("solve.checkpoint_stride_h", 1.0);
    spec.outputStep_h = config.GetDouble("solve.output_step_h", 0.25);
    spec.sunCorrection = config.GetBool("solve.sun_correction", true);
    spec.initial = config.GetString("solve.initial", "steady") == "uniform"
                       ? InitialCondition::Uniform
                       : InitialCondition::Steady;

    const auto normal = config.GetArray<f32>("geometry.normal");
    if (normal.size() == 3) {
        spec.normal = glm::normalize(glm::vec3(normal[0], normal[1], normal[2]));
    }
    spec.area_m2 = config.GetFloat("geometry.area_m2", 1.0f);
    spec.skyFraction = config.GetFloat("geometry.sky_fraction", 1.0f);

    const String forcing = config.GetString("forcing.file", "");
    if (!forcing.empty()) {
        const std::filesystem::path path(forcing);
        spec.forcingFile =
            path.is_absolute() ? forcing : (specDir / path).lexically_normal().string();
    }

    spec.constant.airTemperature_K = config.GetDouble("forcing.air_temperature_k", 293.15);
    spec.constant.sunIrradiance_W_m2 = config.GetDouble("forcing.sun_irradiance_w_m2", 0.0);
    spec.constant.diffuseIrradiance_W_m2 =
        config.GetDouble("forcing.diffuse_irradiance_w_m2", 0.0);
    spec.constant.skyTemperature_K = config.GetDouble("forcing.sky_temperature_k", 268.0);
    spec.constant.relativeHumidity = config.GetDouble("forcing.relative_humidity", 50.0);
    spec.constant.sunDirection = SunFrom(config.GetDouble("forcing.sun_azimuth_deg", 180.0),
                                         config.GetDouble("forcing.sun_elevation_deg", 60.0));
    return spec;
}

/// A two-column `time_h, visibility` CSV.  Commas or whitespace, `#` comments,
/// and a non-numeric first line skipped as a header -- the same tolerances
/// LoadForcingCsv has, so one hand-written file works with either.
Vector<std::pair<f64, f32>> ReadVisibilityCsv(const String& path) {
    Vector<std::pair<f64, f32>> series;
    std::ifstream file(path);
    if (!file) {
        QL_LOG_ERROR("cannot read visibility CSV '{}'", path);
        return series;
    }

    String line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream fields(line);
        f64 time_h = 0.0;
        f64 visibility = 0.0;
        if (!(fields >> time_h >> visibility)) continue;  // header or short row
        series.emplace_back(time_h, static_cast<f32>(visibility));
    }
    std::sort(series.begin(), series.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return series;
}

/// Linear interpolation with the ends held flat -- SampleForcing's rule, so
/// the visibility a step sees is sampled the way its forcing is.
f32 SampleAt(const Vector<std::pair<f64, f32>>& series, const f64 t) {
    if (series.empty()) return 1.0f;
    if (t <= series.front().first) return series.front().second;
    if (t >= series.back().first) return series.back().second;
    for (usize i = 1; i < series.size(); ++i) {
        if (t <= series[i].first) {
            const f64 span = series[i].first - series[i - 1].first;
            const f64 blend = span > 0.0 ? (t - series[i - 1].first) / span : 0.0;
            return static_cast<f32>(series[i - 1].second +
                                    blend * (series[i].second - series[i - 1].second));
        }
    }
    return series.back().second;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "Usage: thermal_column_tool <spec.toml> [--visibility vis.csv] [--out traj.csv]\n"
               "\n"
               "  Integrates one 1-D column through the same surface energy balance a\n"
               "  scene element gets, and writes the trajectory as CSV.\n"
               "\n"
               "  --visibility  two columns, time_h and sun visibility in [0,1].  The\n"
               "                column's own shadow history; without it the sun is\n"
               "                unobstructed for the whole run.  Sample it at least as\n"
               "                finely as solve.timestep_s -- the solver interpolates\n"
               "                linearly between the rows it is given.\n"
               "  --out         where the trajectory goes (default: stdout)\n";
        return 1;
    }

    const std::filesystem::path specPath(argv[1]);
    String visibilityPath;
    String outputPath;
    for (int i = 2; i < argc; ++i) {
        const String arg = argv[i];
        if (arg == "--visibility" && i + 1 < argc) {
            visibilityPath = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            outputPath = argv[++i];
        } else {
            std::cerr << "unrecognised argument: " << arg << "\n";
            return 1;
        }
    }

    Log::Init();

    auto loaded = Config::Load(specPath.string());
    if (!loaded.has_value()) {
        QL_LOG_ERROR("cannot read spec '{}': {}", specPath.string(), loaded.error());
        return 1;
    }
    const Spec spec = ReadSpec(loaded.value(), specPath.parent_path());

    if (spec.material.conductivity_W_mK <= 0.0f) {
        QL_LOG_ERROR("column.thermal_conductivity_w_mk must be positive; "
                     "zero is how a material opts out of the solve entirely");
        return 1;
    }

    // ------------------------------------------------------------------
    // One element, one material, an open sky
    // ------------------------------------------------------------------
    Vector<ThermalElement> elements(1);
    elements[0].centroid = glm::vec3(0.0f);
    elements[0].normal = spec.normal;
    elements[0].area_m2 = spec.area_m2;
    elements[0].materialId = 0;

    const Vector<ThermalMaterial> materials{spec.material};

    ExchangeGeometry exchange = MakeOpenSkyExchange(1);
    exchange.skyFraction[0] = spec.skyFraction;

    const auto forcingSeries = LoadForcingCsv(spec.forcingFile);
    if (!spec.forcingFile.empty() && forcingSeries.empty()) {
        QL_LOG_ERROR("forcing.file '{}' produced no rows", spec.forcingFile);
        return 1;
    }

    // ------------------------------------------------------------------
    // The sun visibility table
    // ------------------------------------------------------------------
    // Columns at the visibility file's own times when there is one, so that a
    // shadow crossing can be described as finely as the study needs without
    // the forcing having to be resampled to match.  Otherwise one column per
    // forcing row, which is what the scene path's shadow dispatch produces.
    const auto visibilitySeries =
        visibilityPath.empty() ? Vector<std::pair<f64, f32>>{}
                               : ReadVisibilityCsv(visibilityPath);
    if (!visibilityPath.empty() && visibilitySeries.empty()) {
        QL_LOG_ERROR("--visibility '{}' produced no rows", visibilityPath);
        return 1;
    }

    SunVisibilityTable table;
    if (!visibilitySeries.empty()) {
        for (const auto& [time_h, v] : visibilitySeries) {
            table.sampleTime_h.push_back(time_h);
            table.visibility.push_back(std::clamp(v, 0.0f, 1.0f));
            table.sampleDirection.push_back(
                SampleForcing(forcingSeries, time_h, spec.constant).sunDirection);
        }
    } else if (!forcingSeries.empty()) {
        for (const auto& [time_h, forcing] : forcingSeries) {
            table.sampleTime_h.push_back(time_h);
            table.visibility.push_back(1.0f);
            table.sampleDirection.push_back(forcing.sunDirection);
        }
    } else {
        table.sampleTime_h.push_back(spec.startTime_h);
        table.visibility.push_back(1.0f);
        table.sampleDirection.push_back(spec.constant.sunDirection);
    }

    // A lone element reflects onto nothing, so this only ever writes the
    // diffuse gain -- but it is the same call the scene path makes, and
    // letting it write that gain is what keeps the diffuse term identical.
    BakeShortwaveGains(exchange, elements, materials, table);

    // ------------------------------------------------------------------
    // Run it
    // ------------------------------------------------------------------
    CpuCrankNicolsonStepper stepper;

    const f64 shortest = CpuCrankNicolsonStepper::ShortestTimeConstantSeconds(
        elements, materials, spec.constant.airTemperature_K);
    if (std::isfinite(shortest) && spec.timestep_s > shortest) {
        QL_LOG_WARN("timestep {:.0f} s is longer than the surface time constant ({:.0f} s); "
                    "the trajectory is smoothed rather than unstable",
                    spec.timestep_s, shortest);
    }

    ThermalTimeline::Desc desc;
    desc.startTime_h = spec.startTime_h;
    desc.timestep_s = spec.timestep_s;
    desc.checkpointStride_h = spec.checkpointStride_h;
    desc.nodeCount = spec.nodeCount;
    desc.initial = spec.initial;
    desc.initialTemperature_K = spec.initialTemperature_K;
    desc.carrySunSensitivity = spec.sunCorrection;

    ThermalTimeline timeline(desc, elements, materials, exchange, table, forcingSeries,
                             spec.constant, stepper);

    std::ofstream fileOut;
    if (!outputPath.empty()) {
        fileOut.open(outputPath);
        if (!fileOut) {
            QL_LOG_ERROR("cannot write '{}'", outputPath);
            return 1;
        }
    }
    std::ostream& out = outputPath.empty() ? std::cout : fileOut;

    out << "time_h,T_surface_K,T_back_K,dTdv_K,v,air_K,sky_K,dni_W_m2\n";
    out << std::setprecision(9);

    const f64 step = spec.outputStep_h > 0.0 ? spec.outputStep_h : 0.25;
    const int rows =
        static_cast<int>(std::floor((spec.endTime_h - spec.startTime_h) / step + 1e-9)) + 1;
    for (int i = 0; i < std::max(rows, 1); ++i) {
        const f64 t = spec.startTime_h + static_cast<f64>(i) * step;
        const ThermalState& state = timeline.StateAt(t);
        const ThermalForcing forcing = SampleForcing(forcingSeries, t, spec.constant);

        // The back node from the state's own node count, not the spec's: the
        // timeline floors it at two, and a column indexed by what was asked
        // for rather than by what was built reports the surface twice.
        out << t << ',' << state.Surface(0) << ','
            << state.temperature_K[state.nodeCount - 1] << ',';
        if (state.HasSensitivity()) out << state.SurfaceSensitivity(0);
        out << ',' << SampleAt(visibilitySeries, t) << ',' << forcing.airTemperature_K << ','
            << forcing.skyTemperature_K << ',' << forcing.sunIrradiance_W_m2 << '\n';
    }

    QL_LOG_INFO("thermal_column_tool: {} nodes through {:.3f} m, {:.1f} h to {:.1f} h at "
                "{:.0f} s -> {}",
                spec.nodeCount, spec.material.thickness_m, spec.startTime_h, spec.endTime_h,
                spec.timestep_s, outputPath.empty() ? "stdout" : outputPath);
    return 0;
}
