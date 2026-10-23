#pragma once

#include "core/Platform.hpp"

#include <cstdint>
#include <string>

namespace quantiloom {

// ============================================================================
// AtmosphereNNConfig - configuration for the NN atmosphere ([atmosphere] TOML)
// ============================================================================
// Replaces the analytic AtmosphericConfig. The nine weather features map
// directly onto the MODTRAN surrogate network inputs; out-of-domain values
// are clamped by the network input spec (with a one-time warning).
//
// Presets replicate scripts/make_anchor_configs.py::preset_value applied to
// the atmospheres.json scenarios, with discrete features snapped to the
// training domain.

struct QL_API AtmosphereNNConfig {
    bool enabled = false;
    std::string modelPackDir;          // Directory of *.safetensors; empty = disabled
    std::string preset = "clear";

    // MODTRAN sampled weather features (training domain in comments)
    double atmosModel = 2.0;           // {2, 3}
    double ihaze = 1.0;                // {1, 4, 5, 9, 10}
    double icld = 0.0;                 // {0, 6, 18}
    double visKm = 23.0;               // [0.5, 50] log-uniform
    double rainrtMmH = 0.0;            // [0, 50]
    double tGroundK = 293.15;          // [253, 328]
    double rh = 0.5;                   // [0.05, 1]
    double pHPa = 1013.25;             // [950, 1040]
    double h2oScale = 1.0;             // [0.5, 2]

    // Solar geometry. When sunFromLighting is true (default), zenith/azimuth
    // are derived from the renderer's sun direction at bake time; an explicit
    // [atmosphere] sun_zenith_deg sets it false and these values win.
    bool sunFromLighting = true;
    double sunZenithDeg = 45.0;
    double sunAzimuthDeg = 180.0;

    // Observer altitude in km. When h1FromCamera is true (default), the
    // renderer resolves h1 from camera.y * world_units_to_meters at bake time
    // and h1Km is ignored; an explicit [atmosphere] h1_km sets it false.
    bool h1FromCamera = true;
    double h1Km = 0.1;

    // Scale applied to the solar-scatter content of path radiance at bake
    // time. The networks were trained for the unattenuated sun; low-light
    // scenarios (e.g. moonlight approximated by a scaled solar LUT) need the
    // scatter term scaled by the same factor or it adds an unphysical
    // twilight-bright fog. Single scattering is linear in source irradiance,
    // so this is a first-order-correct scaling. MWIR's lpath product is
    // decomposed, so only its SOL_SCAT term is scaled and the thermal
    // path-emission term survives; VIS/NIR/SWIR emit a single combined
    // product that is essentially all scatter, so the whole lpath is scaled
    // there (and vanishes entirely for sun below the 85 deg training domain).
    // LWIR lpath is pure thermal emission and is never scaled.
    // 1.0 = unscaled sun.
    double solarScatterScale = 1.0;

    // LUT resolution
    int lutASamples = 64;              // Range / zenith axis
    int lutAzSamples = 16;             // Relative azimuth axis (lpath only)

    // Applies one of the 8 named presets (or "disabled"). A named preset also
    // sets enabled = true and "disabled" sets it false, so the last preset
    // applied decides; a caller that cannot run the atmosphere (no model pack)
    // clears enabled itself afterwards. Returns false and leaves the config
    // untouched for unknown names.
    bool ApplyPreset(const std::string& name);

    // Stable hash over everything that affects baked LUT contents, with
    // h1 quantized to 50 m and the sun zenith to 0.5 deg so per-frame jitter
    // does not trigger rebakes. The absolute sun azimuth is not hashed: a bake
    // is indexed by relative azimuth and never reads it. Band and lambda grid
    // are hashed separately by the baker.
    uint64_t BakeHash() const;
};

}  // namespace quantiloom
