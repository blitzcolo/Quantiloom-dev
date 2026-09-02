/**
 * @file ThermalSunResponse.cpp
 * @brief Packing the solver's sun response for binding 26
 */

#include "renderer/ThermalSunResponse.hpp"

#include <cmath>

namespace quantiloom::rendercore {

Vector<ThermalSunResponseGpu> MakeThermalSunResponse(const Vector<f32>& sunSensitivity_K,
                                                     const Vector<f32>& sunVisibility,
                                                     const glm::vec3& sunDirection,
                                                     const Vector<f32>& lagSensitivity_K,
                                                     const Vector<f32>& lagVisibility,
                                                     const Vector<glm::vec3>& lagDirection) {
    const usize n = sunSensitivity_K.size();
    const f32 length = std::sqrt(glm::dot(sunDirection, sunDirection));
    if (n == 0 || sunVisibility.size() != n || !(length > 0.0f)) {
        return Vector<ThermalSunResponseGpu>(1);
    }

    // Which slots are worth carrying: one whose sun direction is unknown has
    // nothing for the shader to trace toward, and its share stays in the
    // present-sun record where it always was.
    Vector<usize> slots;
    const usize offered = lagDirection.size();
    if (lagSensitivity_K.size() == offered * n && lagVisibility.size() == offered * n) {
        for (usize s = 0; s < offered; ++s) {
            if (glm::dot(lagDirection[s], lagDirection[s]) > 0.0f) slots.push_back(s);
        }
    }

    const usize stride = 1 + slots.size();
    Vector<ThermalSunResponseGpu> records(stride * (n + 1));
    records[0].a = sunDirection / length;
    records[0].b = static_cast<f32>(stride);

    for (usize j = 0; j < slots.size(); ++j) {
        const glm::vec3& direction = lagDirection[slots[j]];
        records[1 + j].a = glm::normalize(direction);
    }

    for (usize e = 0; e < n; ++e) {
        ThermalSunResponseGpu* element = &records[stride * (e + 1)];
        f32 attributed = 0.0f;
        for (usize j = 0; j < slots.size(); ++j) {
            const usize at = slots[j] * n + e;
            element[1 + j].a = glm::vec3(lagSensitivity_K[at], lagVisibility[at], 0.0f);
            attributed += lagSensitivity_K[at];
        }
        // The present-sun record is the remainder, so the terms the shader
        // adds sum to the whole day's sensitivity whatever the split.
        element[0].a = glm::vec3(sunSensitivity_K[e] - attributed, sunVisibility[e], 0.0f);
    }
    return records;
}

}  // namespace quantiloom::rendercore
