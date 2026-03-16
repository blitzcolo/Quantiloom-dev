/**
 * @file IRMaterialDatabase.hpp
 * @brief Built-in IR material lookup table and LAB-space matching
 *
 * Provides a hardcoded database of common surface types with per-band
 * emissivity values. Match algorithm scores LAB centroids against entries
 * to find the best IR material class for each texture cluster.
 */

#pragma once

#include "Platform.hpp"
#include "ColorCluster.hpp"
#include "core/Types.hpp"
#include <utility>

namespace spectraforge {

using quantiloom::f32;
using quantiloom::u32;
using quantiloom::Span;
using quantiloom::Vector;

struct IRMaterialEntry {
    const char* name;
    const char* description;

    // Per-band average emissivity (gray body approximation)
    f32 emissivityNIR;    // 780-1400nm
    f32 emissivitySWIR;   // 1000-2500nm
    f32 emissivityMWIR;   // 3000-5000nm
    f32 emissivityLWIR;   // 8000-12000nm

    // LAB matching criteria
    f32 L_min, L_max;
    f32 chromaMax;         // for achromatic materials (chroma < this)
    f32 hueMin, hueMax;   // for chromatic materials (degrees)
    bool requiresMetallic; // only match if metallicFactor > 0.5
};

// Built-in database (11 entries)
SF_API Span<const IRMaterialEntry> GetBuiltinDatabase();

// Match a LAB color + metallic factor to the best IR material
SF_API const IRMaterialEntry& MatchMaterial(const LABColor& color, f32 metallicFactor);

// Generate wavelength-dependent IR curves from a matched entry
// Fills emissivity, reflectance, transmittance as (wavelength_nm, value) pairs
SF_API void GenerateIRCurves(
    const IRMaterialEntry& entry,
    Vector<std::pair<f32, f32>>& outEmissivity,
    Vector<std::pair<f32, f32>>& outReflectance,
    Vector<std::pair<f32, f32>>& outTransmittance
);

} // namespace spectraforge
