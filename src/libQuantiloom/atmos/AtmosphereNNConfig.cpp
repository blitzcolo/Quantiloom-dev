#include "atmos/AtmosphereNNConfig.hpp"

#include <cmath>
#include <cstring>

namespace quantiloom {

namespace {

struct Preset {
    const char* name;
    double atmosModel, ihaze, icld, visKm, rainrtMmH, tGroundK, rh, pHPa;
    double sunZenithDeg, sunAzimuthDeg;
};

// From atmospheres.json via make_anchor_configs.py::preset_value:
//   modtran_model -> atmos_model (1 snapped to 2, outside {2,3})
//   aerosol Rural->1 Maritime->4 Urban->5 Industrial->5; weather Fog -> 9
//   icld: rain > 0 -> 6, else 0; h2o_scale = 1 everywhere
//   sun azimuth folded to [0, 180] (mirror symmetry)
constexpr Preset kPresets[] = {
    {"clear",           2, 1, 0, 23.0, 0.0,  293.15, 0.50, 1013.25, 45, 180},
    {"turbulent_clear", 2, 1, 0, 20.0, 0.0,  308.15, 0.30, 1010.00, 30, 170},
    {"urban_haze",      2, 5, 0,  8.0, 0.0,  298.15, 0.75, 1010.00, 60, 160},
    {"fog",             2, 9, 0,  0.5, 0.0,  288.15, 1.00, 1015.00, 70, 160},
    {"light_rain",      2, 4, 6, 10.0, 2.5,  290.15, 0.85, 1008.00, 50, 150},
    {"heavy_rain",      2, 4, 6,  2.0, 25.0, 288.15, 0.95, 1005.00, 80, 150},
    {"snow",            3, 1, 6,  2.5, 5.0,  273.15, 0.90, 1020.00, 65, 170},
    {"haze",            2, 5, 0,  3.0, 0.0,  303.15, 0.40, 1000.00, 40, 120},
};

uint64_t Fnv1a(const void* data, size_t n, uint64_t h) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

uint64_t HashDouble(double v, uint64_t h) {
    return Fnv1a(&v, sizeof v, h);
}

}  // namespace

bool AtmosphereNNConfig::ApplyPreset(const std::string& name) {
    if (name == "disabled") {
        enabled = false;
        preset = name;
        return true;
    }
    for (const Preset& p : kPresets) {
        if (name == p.name) {
            preset = name;
            atmosModel = p.atmosModel;
            ihaze = p.ihaze;
            icld = p.icld;
            visKm = p.visKm;
            rainrtMmH = p.rainrtMmH;
            tGroundK = p.tGroundK;
            rh = p.rh;
            pHPa = p.pHPa;
            h2oScale = 1.0;
            sunZenithDeg = p.sunZenithDeg;
            sunAzimuthDeg = p.sunAzimuthDeg;
            return true;
        }
    }
    return false;
}

uint64_t AtmosphereNNConfig::BakeHash() const {
    uint64_t h = 14695981039346656037ULL;
    h = Fnv1a(modelPackDir.data(), modelPackDir.size(), h);
    h = HashDouble(atmosModel, h);
    h = HashDouble(ihaze, h);
    h = HashDouble(icld, h);
    h = HashDouble(visKm, h);
    h = HashDouble(rainrtMmH, h);
    h = HashDouble(tGroundK, h);
    h = HashDouble(rh, h);
    h = HashDouble(pHPa, h);
    h = HashDouble(h2oScale, h);
    // Quantized geometry: 0.5 deg sun, 50 m altitude
    h = HashDouble(std::round(sunZenithDeg * 2.0), h);
    h = HashDouble(std::round(sunAzimuthDeg * 2.0), h);
    h = HashDouble(std::round(h1Km * 20.0), h);
    h = HashDouble(static_cast<double>(lutASamples), h);
    h = HashDouble(static_cast<double>(lutAzSamples), h);
    return h;
}

}  // namespace quantiloom
