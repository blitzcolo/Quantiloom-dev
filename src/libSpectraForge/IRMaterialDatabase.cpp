#include "IRMaterialDatabase.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <array>
#include <limits>

namespace spectraforge {

// ============================================================================
// Built-in IR Material Database
// ============================================================================
// 11 common surface types covering the majority of outdoor/indoor scenes.
// Emissivity values are typical gray-body approximations from literature.
//
// Matching priority: metallic filter -> L* range -> chroma/hue scoring.
// generic_dielectric is the fallback (last entry).
// ============================================================================

static constexpr std::array<IRMaterialEntry, 11> g_database = {{
    // Metals (requiresMetallic = true)
    {
        "polished_metal", "Polished metal surface (low emissivity)",
        0.02f, 0.03f, 0.04f, 0.05f,
        70.0f, 100.0f,   // high L*
        200.0f,           // any chroma
        0.0f, 360.0f,    // any hue
        true
    },
    {
        "oxidized_metal", "Oxidized/weathered metal",
        0.15f, 0.20f, 0.28f, 0.35f,
        30.0f, 70.0f,    // medium L*
        200.0f,
        0.0f, 360.0f,
        true
    },
    {
        "dark_metal", "Dark/corroded metal (high emissivity for metal)",
        0.25f, 0.35f, 0.42f, 0.50f,
        0.0f, 30.0f,     // low L*
        200.0f,
        0.0f, 360.0f,
        true
    },

    // Dielectrics (requiresMetallic = false)
    {
        "vegetation", "Green vegetation (very high LWIR emissivity)",
        0.70f, 0.75f, 0.94f, 0.97f,
        15.0f, 80.0f,    // broad L* range
        200.0f,           // any chroma
        90.0f, 160.0f,   // green hue range (a*<0, b*>0)
        false
    },
    {
        "water", "Water surface",
        0.93f, 0.95f, 0.96f, 0.97f,
        10.0f, 60.0f,
        200.0f,
        200.0f, 280.0f,  // blue hue range (b*<0)
        false
    },
    {
        "asphalt", "Asphalt/dark road surface",
        0.88f, 0.90f, 0.93f, 0.95f,
        0.0f, 25.0f,     // very low L*
        15.0f,            // achromatic
        0.0f, 360.0f,
        false
    },
    {
        "dry_soil", "Dry soil/earth",
        0.85f, 0.88f, 0.91f, 0.93f,
        20.0f, 55.0f,    // low-medium L*
        200.0f,
        30.0f, 80.0f,    // warm/brown hue
        false
    },
    {
        "concrete", "Concrete/stone surface",
        0.85f, 0.87f, 0.89f, 0.91f,
        35.0f, 75.0f,    // medium L*
        12.0f,            // achromatic
        0.0f, 360.0f,
        false
    },
    {
        "painted_surface", "Painted surface (generic, high emissivity)",
        0.88f, 0.90f, 0.91f, 0.92f,
        60.0f, 100.0f,   // high L*
        15.0f,            // low chroma (neutral paint)
        0.0f, 360.0f,
        false
    },
    {
        "glass", "Glass/window (partial transmittance in NIR/SWIR)",
        0.15f, 0.20f, 0.78f, 0.88f,
        40.0f, 95.0f,
        10.0f,            // achromatic
        0.0f, 360.0f,
        false
    },

    // Fallback — must be last
    {
        "generic_dielectric", "Generic dielectric (fallback)",
        0.85f, 0.87f, 0.88f, 0.90f,
        0.0f, 100.0f,    // any L*
        200.0f,           // any chroma
        0.0f, 360.0f,    // any hue
        false
    },
}};

Span<const IRMaterialEntry> GetBuiltinDatabase() {
    return Span<const IRMaterialEntry>(g_database.data(), g_database.size());
}

// ============================================================================
// LAB chroma and hue helpers
// ============================================================================

static f32 LABChroma(const LABColor& c) {
    return std::sqrt(c.a * c.a + c.b * c.b);
}

static f32 LABHue(const LABColor& c) {
    f32 h = std::atan2(c.b, c.a) * (180.0f / 3.14159265f);
    if (h < 0.0f) h += 360.0f;
    return h;
}

// ============================================================================
// Matching Algorithm
// ============================================================================
// For each database entry, compute a score. Lower = better match.
// Hard filters (metallic, L* range) eliminate entries.
// Soft scoring: L* distance + chroma/hue penalty.

const IRMaterialEntry& MatchMaterial(const LABColor& color, f32 metallicFactor) {
    bool isMetal = metallicFactor > 0.5f;
    f32 chroma = LABChroma(color);
    f32 hue = LABHue(color);

    f32 bestScore = std::numeric_limits<f32>::max();
    // Fallback: last metallic entry for metals, last entry overall for dielectrics
    u32 metalFallback = 2; // dark_metal (broadest emissivity for unknown metal)
    u32 bestIdx = isMetal ? metalFallback : static_cast<u32>(g_database.size() - 1);

    for (u32 i = 0; i < g_database.size(); ++i) {
        const auto& entry = g_database[i];

        // Hard filter: metallic mismatch
        if (entry.requiresMetallic && !isMetal) continue;
        if (!entry.requiresMetallic && isMetal) continue;

        // Hard filter: L* out of range (inclusive boundaries)
        if (color.L < entry.L_min - 0.5f || color.L > entry.L_max + 0.5f) continue;

        // Score: distance from L* range center
        f32 Lcenter = (entry.L_min + entry.L_max) * 0.5f;
        f32 score = std::abs(color.L - Lcenter) * 0.5f;

        // Chroma check
        if (entry.chromaMax < 50.0f) {
            // Entry wants achromatic surface
            if (chroma > entry.chromaMax)
                score += (chroma - entry.chromaMax) * 2.0f;
        }

        // Hue check (for chromatic materials)
        if (entry.hueMax - entry.hueMin < 359.0f) {
            // Wrap-aware hue distance
            f32 hMin = entry.hueMin;
            f32 hMax = entry.hueMax;
            bool inRange = (hMin <= hMax)
                ? (hue >= hMin && hue <= hMax)
                : (hue >= hMin || hue <= hMax);
            if (!inRange) score += 50.0f; // heavy penalty
        }

        if (score < bestScore) {
            bestScore = score;
            bestIdx = i;
        }
    }

    return g_database[bestIdx];
}

// ============================================================================
// IR Curve Generation
// ============================================================================
// For each IR band, place 2 sample points at band start/end wavelengths.
// Kirchhoff: reflectance = 1 - emissivity for opaque materials (τ=0).
// Exception: glass has non-zero transmittance in NIR/SWIR bands.

void GenerateIRCurves(
    const IRMaterialEntry& entry,
    Vector<std::pair<f32, f32>>& outEmissivity,
    Vector<std::pair<f32, f32>>& outReflectance,
    Vector<std::pair<f32, f32>>& outTransmittance)
{
    outEmissivity.clear();
    outReflectance.clear();
    outTransmittance.clear();

    // Band definitions: {startNm, endNm, emissivity}
    struct Band { f32 start; f32 end; f32 eps; };
    const Band bands[] = {
        {  780.0f,  1400.0f, entry.emissivityNIR  },
        { 1000.0f,  2500.0f, entry.emissivitySWIR },
        { 3000.0f,  5000.0f, entry.emissivityMWIR },
        { 8000.0f, 12000.0f, entry.emissivityLWIR },
    };

    bool isGlass = (std::strcmp(entry.name, "glass") == 0);

    for (const auto& band : bands) {
        f32 eps = band.eps;
        f32 tau = 0.0f;

        // Glass: non-zero transmittance in NIR/SWIR, reduced in thermal bands
        if (isGlass) {
            if (band.start < 2600.0f)
                tau = 0.70f; // NIR/SWIR glass transmittance ~70%
            else if (band.start < 6000.0f)
                tau = 0.10f; // MWIR: glass becomes opaque
            // LWIR: tau stays 0 (glass is opaque beyond ~5μm)
        }

        // Kirchhoff: rho = 1 - eps - tau
        f32 rho = 1.0f - eps - tau;
        if (rho < 0.0f) rho = 0.0f;

        outEmissivity.push_back({band.start, eps});
        outEmissivity.push_back({band.end,   eps});

        outReflectance.push_back({band.start, rho});
        outReflectance.push_back({band.end,   rho});

        outTransmittance.push_back({band.start, tau});
        outTransmittance.push_back({band.end,   tau});
    }

    // Sort by wavelength (bands may overlap SWIR/NIR)
    auto cmp = [](const auto& a, const auto& b) { return a.first < b.first; };
    std::sort(outEmissivity.begin(), outEmissivity.end(), cmp);
    std::sort(outReflectance.begin(), outReflectance.end(), cmp);
    std::sort(outTransmittance.begin(), outTransmittance.end(), cmp);

    // Deduplicate wavelengths (keep first occurrence)
    auto dedup = [](Vector<std::pair<f32, f32>>& v) {
        if (v.size() < 2) return;
        auto it = std::unique(v.begin(), v.end(),
            [](const auto& a, const auto& b) { return a.first == b.first; });
        v.erase(it, v.end());
    };
    dedup(outEmissivity);
    dedup(outReflectance);
    dedup(outTransmittance);
}

} // namespace spectraforge
