/**
 * @file ThermalExchangePrecompute.hpp
 * @brief Who sees whom, on the GPU, once per scene
 *
 * The view factors the energy balance needs are a geometry question -- what
 * fraction of each surface's hemisphere every other surface fills, and whether
 * the sun reaches it -- and they do not change while the geometry does not.
 * So they are computed once, before any stepping, by casting a hemisphere of
 * rays per element against the acceleration structure the renderer already
 * built.
 *
 * Aguerre et al. compute the same matrix with Embree on the CPU and report
 * eleven minutes for a hundred thousand elements. This is the same algorithm
 * against hardware ray tracing.
 *
 * Self-contained: its own pipeline, descriptor set and buffers, in the manner
 * of GpuSpectralReconstructor rather than as another method on
 * ExternalRenderContext. The offline renderer has no compute infrastructure of
 * its own, and this is the first thing to need any.
 */

#pragma once

#include "renderer/VulkanContext.hpp"
#include "thermal/ThermalTypes.hpp"

#include <vulkan/vulkan.h>

#include <memory>
#include <span>

namespace quantiloom::rendercore {

/**
 * @brief One run of the exchange precompute
 */
class ThermalExchangePrecompute {
public:
    struct Params {
        /// Rays per element. The estimator's error goes as 1/sqrt(N), and a
        /// view factor wrong by a few percent moves a temperature by a
        /// fraction of a kelvin, so a few hundred is enough.
        u32 hemisphereRays = 256;
        /// Rays across the sun's disc. Several rather than one so an element at
        /// the edge of a shadow gets a fraction, which is what makes the
        /// shadow boundary in the temperature field as soft as the geometry.
        u32 sunRays = 8;
        /// Entries kept per row. Past the largest few a row is mostly the
        /// noise floor of the estimator, and keeping it would make the matrix
        /// dense for no accuracy.
        u32 topK = 32;
        /// From surface toward the sun, normalised.
        glm::vec3 sunDirection{0.0f, 1.0f, 0.0f};
        /// Angular radius of the sun, radians. The real one is 0.00465.
        f32 sunAngularRadius = 0.00465f;
    };

    ThermalExchangePrecompute(VulkanContext& context);
    ~ThermalExchangePrecompute();

    ThermalExchangePrecompute(const ThermalExchangePrecompute&) = delete;
    ThermalExchangePrecompute& operator=(const ThermalExchangePrecompute&) = delete;

    /**
     * @brief Per material, the fraction of its area that is actually there
     *
     * 1 for an opaque surface; for alphaMode MASK or BLEND, the mean of its
     * base colour alpha. An occluder is committed with that probability, so a
     * leaf with holes lets radiation through them as it lets light through.
     *
     * A mean rather than a per-texel test, deliberately. A view factor is an
     * area integral estimated by a histogram of a few hundred rays; over that
     * many the hit points spread across the occluder, so their expectation is
     * exactly this mean, and resolving each ray against its own texel would
     * make this pass carry the render pipeline's whole bindless texture set to
     * answer a question finer than the estimator can hear.
     *
     * Empty -- the default -- means every material is opaque, which is what a
     * scene with no alpha coverage wants. Set before either Run; it feeds both.
     */
    void SetMaterialCoverage(Vector<f32> coverage);

    /// True when the pipeline came up. False means the shader could not be
    /// found or the device refused it, and the caller should fall back to the
    /// analytic open-sky exchange rather than failing the render.
    [[nodiscard]] bool IsValid() const;

    /**
     * @brief Cast the rays and reduce them into sparse rows
     *
     * @param tlas      the scene's top-level acceleration structure
     * @param elements  surface elements, in BuildThermalMesh order
     * @param instanceElementBase  first element of each TLAS instance
     * @param params    ray counts and the sun
     * @return the exchange geometry, or an empty one on failure
     */
    [[nodiscard]] thermal::ExchangeGeometry Run(
        VkAccelerationStructureKHR tlas, const Vector<thermal::ThermalElement>& elements,
        const Vector<u32>& instanceElementBase, const Params& params);

    /**
     * @brief Sun visibility for several directions in one submit
     *
     * Each direction gets one dispatch; the output is direction-major:
     * result[k * elementCount .. (k+1) * elementCount). The hemisphere is
     * skipped (rayCount=0), so this is much cheaper than a full Run.
     *
     * @return K * elementCount floats, or empty on failure
     */
    [[nodiscard]] Vector<f32> RunSunVisibility(
        VkAccelerationStructureKHR tlas, const Vector<thermal::ThermalElement>& elements,
        const Vector<u32>& instanceElementBase,
        std::span<const glm::vec3> directions, u32 sunRays = 8,
        f32 sunAngularRadius = 0.00465f);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace quantiloom::rendercore
