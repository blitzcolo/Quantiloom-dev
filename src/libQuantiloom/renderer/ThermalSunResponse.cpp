/**
 * @file ThermalSunResponse.cpp
 * @brief Packing the solver's sun response for binding 26
 */

#include "renderer/ThermalSunResponse.hpp"

#include <cmath>

namespace quantiloom::rendercore {

Vector<ThermalSunResponseGpu> MakeThermalSunResponse(const Vector<f32>& sunSensitivity_K,
                                                     const Vector<f32>& sunVisibility,
                                                     const glm::vec3& sunDirection) {
    const usize n = sunSensitivity_K.size();
    const f32 length = std::sqrt(glm::dot(sunDirection, sunDirection));
    if (n == 0 || sunVisibility.size() != n || !(length > 0.0f)) {
        return Vector<ThermalSunResponseGpu>(1);
    }

    Vector<ThermalSunResponseGpu> records(n + 1);
    records[0].a = sunDirection / length;
    records[0].b = 1.0f;
    for (usize e = 0; e < n; ++e) {
        records[e + 1].a = glm::vec3(sunSensitivity_K[e], sunVisibility[e], 0.0f);
    }
    return records;
}

}  // namespace quantiloom::rendercore
