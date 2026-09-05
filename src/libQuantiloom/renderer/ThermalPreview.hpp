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

class EpochGeometryHost;

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

    /**
     * @brief Who moves the scene when the geometry has to be measured again
     *
     * Optional. Without it the preview measures the world once, wherever the
     * scene happens to be -- which is right for a scene that does not move and
     * is what every scene did before there was a timeline.
     *
     * The host is borrowed, not owned, and must outlive this.
     */
    void SetEpochHost(EpochGeometryHost* host);

    /**
     * @brief When to measure it
     *
     * @param times_s  timeline seconds, from thermal::PlanEpochTimes
     * @param from_h   the same instants as solve hours; entry 0 is ignored,
     *                 since epoch zero reaches back forever
     *
     * One entry or none is the static case and costs exactly what it did
     * before. Setting this marks the exchange dirty, because it is a statement
     * about geometry.
     */
    void SetEpochPlan(Vector<f64> times_s, Vector<f64> from_h);

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
        /// The same, per tracked sun column, when the parameters asked for
        /// any: slot-major sensitivity and visibility, and where the sun was.
        Vector<f32> lagSensitivity_K;
        Vector<f32> lagVisibility;
        Vector<glm::vec3> lagDirection;
        u32 elementCount = 0;
        bool elementCountChanged = false;
        String error;
    };

    SolveResult SolveAt(f64 time_h, const Scene& scene, VkAccelerationStructureKHR tlas);

    /// Write the solve at the instant last shown, one row per element. Empty
    /// takes the path from the parameters. Returns the file written.
    [[nodiscard]] Result<String, String> DumpElements(const String& pathOrEmpty = "");

    /// One element's history, by replaying the trajectory rather than solving
    /// again. The hour the viewport is showing is restored before this returns.
    [[nodiscard]] Result<ThermalElementTrajectory, String> ElementTrajectory(
        u32 element, f64 fromHour, f64 toHour, u32 samples);

    /// The surface field of dT/dp for one parameter, at the hour on screen,
    /// one float per element and empty when the solve does not carry that
    /// parameter. Empty is what a caller must be able to tell from all-zero:
    /// a tangent nobody asked for is not a tangent that came out flat.
    [[nodiscard]] Vector<f32> ParameterSensitivityField(
        ThermalSensitivityParameter parameter) const;

    /// The flat element index for an instance and one of its triangles.
    /// False when that instance is not one the solve carries, which is a real
    /// answer about the geometry rather than a lookup failure.
    [[nodiscard]] bool ElementFor(u32 instanceIndex, u32 primitiveIndex, u32& out) const;

    [[nodiscard]] ThermalSolveStatus Status() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace rendercore
}  // namespace quantiloom
