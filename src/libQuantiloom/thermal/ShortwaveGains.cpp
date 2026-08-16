/**
 * @file ShortwaveGains.cpp
 * @brief Sunlight that arrived off something else
 */

#include "thermal/ShortwaveGains.hpp"

#include <algorithm>

namespace quantiloom::thermal {

namespace {

/// What an element sends back on: one minus what it absorbs, since a surface
/// the solver models is opaque. Out-of-range materials reflect nothing rather
/// than reflecting the default, which would invent light from an index bug.
f32 ReflectanceOf(const ThermalElement& element, const Vector<ThermalMaterial>& materials) {
    if (element.materialId >= materials.size()) return 0.0f;
    return std::clamp(1.0f - materials[element.materialId].shortwaveAbsorptivity, 0.0f, 1.0f);
}

}  // namespace

void BakeShortwaveGains(const ExchangeGeometry& exchange,
                        const Vector<ThermalElement>& elements,
                        const Vector<ThermalMaterial>& materials,
                        SunVisibilityTable& table) {
    table.reflectedGain.clear();
    table.diffuseGain.clear();

    const usize n = elements.size();
    const CsrMatrix& F = exchange.viewFactors;
    if (n == 0 || F.RowCount() < n) return;

    // Reflectance is read once per element rather than once per entry: the
    // matrix has tens of entries per row and the lookup is the same each time.
    Vector<f32> reflectance(n);
    for (usize e = 0; e < n; ++e) {
        reflectance[e] = ReflectanceOf(elements[e], materials);
    }

    // ------------------------------------------------------------------
    // Diffuse: the sky directly, plus the sky off one other surface
    // ------------------------------------------------------------------
    // s_i is already the cosine-weighted fraction of element i's hemisphere
    // that is sky, so for an isotropic dome the direct part is E_diff * s_i --
    // one for a flat roof, a half for an unobstructed wall.
    if (exchange.skyFraction.size() >= n) {
        table.diffuseGain.resize(n);
        for (usize i = 0; i < n; ++i) {
            f64 gain = static_cast<f64>(exchange.skyFraction[i]);
            for (u32 k = F.rowStart[i]; k < F.rowStart[i + 1]; ++k) {
                const u32 j = F.column[k];
                if (j >= n) continue;
                gain += static_cast<f64>(F.value[k]) * reflectance[j] *
                        static_cast<f64>(exchange.skyFraction[j]);
            }
            table.diffuseGain[i] = static_cast<f32>(gain);
        }
    }

    // ------------------------------------------------------------------
    // Reflected: the disc off one other surface, per sun column
    // ------------------------------------------------------------------
    const usize samples = table.SampleCount();
    if (samples == 0 || table.sampleDirection.size() != samples ||
        table.ElementCount() != n) {
        return;
    }

    table.reflectedGain.assign(samples * n, 0.0f);
    for (usize s = 0; s < samples; ++s) {
        const glm::vec3 sun = glm::normalize(table.sampleDirection[s]);
        const f32* visible = table.Column(s);

        // What each element is lit by, per unit direct normal irradiance.
        // Computed for the whole column first because a row gathers from many
        // elements and most of them appear in several rows.
        Vector<f32> radiosity(n, 0.0f);
        for (usize j = 0; j < n; ++j) {
            const f32 cosTheta = glm::dot(elements[j].normal, sun);
            if (cosTheta <= 0.0f) continue;
            radiosity[j] = reflectance[j] * cosTheta * visible[j];
        }

        f32* out = table.reflectedGain.data() + s * n;
        for (usize i = 0; i < n; ++i) {
            f64 gain = 0.0;
            for (u32 k = F.rowStart[i]; k < F.rowStart[i + 1]; ++k) {
                const u32 j = F.column[k];
                if (j >= n) continue;
                gain += static_cast<f64>(F.value[k]) * radiosity[j];
            }
            out[i] = static_cast<f32>(gain);
        }
    }
}

}  // namespace quantiloom::thermal
