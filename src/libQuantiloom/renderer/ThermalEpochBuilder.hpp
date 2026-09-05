/**
 * @file ThermalEpochBuilder.hpp
 * @brief Measuring the world once per epoch
 *
 * The planner said when; this does the expensive part. For each epoch it puts
 * the scene where that epoch says, refits the acceleration structure, walks the
 * geometry into elements, traces the hemisphere for view factors and the sun
 * for visibility, and bakes the short-wave gains -- then puts the scene back.
 *
 * Shared by the two hosts that own a device and a scene, because they need the
 * same schedule out of the same steps and a second implementation would be a
 * second set of view factors: ExternalRenderContext (which has to do this in
 * the middle of an interactive session) and OfflineRenderer (which does it once
 * before a render or a whole sequence).
 *
 * ## What it costs
 *
 * K epochs is K exchange precomputes, and one of those is the single most
 * expensive thing a thermal scene does. The knobs are the stride (default: one
 * thermal timestep, since a finer division is invisible to the stepper) and the
 * minimum move. A truck that drives for ten seconds and parks costs two or
 * three epochs; a config that asks for a boundary every second over a week
 * costs what it asked for.
 *
 * ## The sun table
 *
 * Each epoch keeps only the forcing rows its own span needs, plus one on each
 * side so the interpolation at the boundary has something to reach for. A
 * month-long timeline with a row every ten minutes is four thousand columns;
 * keeping all of them in every epoch would be the memory that stops this
 * working at all. The exception is `sun_memory_lags`, where a state carries
 * column INDICES across boundaries and a subset would renumber them -- there
 * the full table is kept and the cost is reported.
 */

#pragma once

#include "renderer/ThermalExchangePrecompute.hpp"
#include "scene/Scene.hpp"
#include "thermal/ThermalMesh.hpp"
#include "thermal/ThermalTypes.hpp"

#include <vulkan/vulkan.h>

namespace quantiloom::rendercore {

/**
 * @brief How the builder borrows a host's scene for a moment
 *
 * Implemented by whoever owns the scene and its acceleration structure. The
 * builder moves the scene through every epoch and hands it back at the end;
 * what "hands it back" means is the host's business, because only the host
 * knows where its clock stands.
 */
class EpochGeometryHost {
public:
    virtual ~EpochGeometryHost() = default;

    /// Put the scene at @p t_s and return the acceleration structure to trace
    /// against. VK_NULL_HANDLE means the host could not, and the builder falls
    /// back to an open-sky exchange for that epoch.
    virtual VkAccelerationStructureKHR ApplyEpoch(f64 t_s) = 0;

    /// Put the scene back where the host had it.
    virtual void Restore() = 0;
};

struct EpochBuildInput {
    /// One per epoch, in time order, from PlanEpochTimes.
    Vector<f64> epochTimes_s;
    /// The same epochs in solve hours. Entry 0 is ignored -- epoch zero always
    /// reaches back forever.
    Vector<f64> epochFrom_h;

    thermal::ThermalMeshOptions meshOptions;
    /// Indexed by ThermalElement::materialId.
    const Vector<thermal::ThermalMaterial>* materials = nullptr;
    /// Per scene material, the fraction of its area that is really there.
    Vector<f32> materialCoverage;

    ThermalExchangePrecompute::Params precompute;

    /// Rows of the forcing file, which decide the sun columns.
    const Vector<std::pair<f64, thermal::ThermalForcing>>* forcingSeries = nullptr;
    /// Used when the forcing file has one row or none.
    glm::vec3 fallbackSunDirection{0.0f, 1.0f, 0.0f};

    /// When non-zero the sun table is kept whole in every epoch; see the file
    /// comment.
    u32 sunMemoryLags = 0;
};

/**
 * @brief Build the schedule, leaving the scene where the host wants it
 *
 * @param scene    the host's scene, mutated through the epochs and restored
 * @param epoch0MeshOut  optional: the mesh built for epoch zero, which is
 *                       where the instance bases, the shell partners and the
 *                       contacts come from. All of them are topological, so
 *                       one epoch's answer is every epoch's.
 */
thermal::ThermalGeometrySchedule BuildThermalGeometrySchedule(
    VulkanContext& context, EpochGeometryHost& host, const Scene& scene,
    const EpochBuildInput& input, thermal::ThermalMesh* epoch0MeshOut = nullptr);

}  // namespace quantiloom::rendercore
