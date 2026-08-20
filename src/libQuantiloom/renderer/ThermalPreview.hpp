/**
 * @file ThermalPreview.hpp
 * @brief Interactive thermal solve state, isolated from the render pipeline
 *
 * Owns the thermal mesh, material table, exchange geometry, sun visibility
 * table, timeline with checkpoints, and the stepper (GPU when available, CPU
 * fallback). Never touches the pipeline or descriptors — the ERC facade
 * reads the results and updates binding 24 itself.
 *
 * Dirty flags defer expensive work: only the first SetThermalTime after a
 * geometry change pays for the exchange precompute, and only the first after
 * a material or parameter change rebuilds the timeline. Scrubbing time alone
 * never invalidates either.
 */

#pragma once

#include "renderer/ThermalControl.hpp"
#include "renderer/VulkanContext.hpp"
#include "thermal/ThermalTypes.hpp"

#include <memory>
#include <unordered_map>
#include <vulkan/vulkan.h>

namespace quantiloom {

class Scene;

namespace thermal {
class ThermalTimeline;
class IThermalStepper;
}  // namespace thermal

namespace rendercore {

class ThermalPreview {
public:
    explicit ThermalPreview(VulkanContext& context);
    ~ThermalPreview();

    ThermalPreview(const ThermalPreview&) = delete;
    ThermalPreview& operator=(const ThermalPreview&) = delete;

    void SetParams(const ThermalSolveParams& params);
    void SetMaterial(const String& name, const ThermalMaterialParams& params);
    void ClearMaterials();
    void SetEnabled(bool enabled);
    void SetFallbackSunDirection(const glm::vec3& dir);

    void InvalidateGeometry();
    void InvalidateMaterialEmissivity();

    struct SolveResult {
        Vector<f32> surfaceTemperature_K;
        Vector<u32> instanceElementBase;
        /// dT/dv and the visibility it was taken about, per element, plus the
        /// sun direction the solve used at this instant. Together they let the
        /// shading pass trade a triangle-average shadow for the one it traces
        /// per pixel -- see thermal::ThermalResult for the arithmetic.
        Vector<f32> sunSensitivity_K;
        Vector<f32> sunVisibility;
        glm::vec3 sunDirection{0.0f, 1.0f, 0.0f};
        u32 elementCount = 0;
        bool elementCountChanged = false;
        String error;
    };

    SolveResult SolveAt(f64 time_h, const Scene& scene, VkAccelerationStructureKHR tlas);

    [[nodiscard]] ThermalSolveStatus Status() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace rendercore
}  // namespace quantiloom
