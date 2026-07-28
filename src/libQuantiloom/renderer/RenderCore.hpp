/**
 * @file RenderCore.hpp
 * @brief Render orchestration shared by the offline CLI and the GUI's ExternalRenderContext
 *
 * Internal on purpose: this header is not in include/quantiloom/ and nothing here
 * carries QL_API. Consumers reach it through a facade -- ExternalRenderContext for
 * the GUI today, an offline context for the CLI once the migration below finishes.
 * The unit tests link quantiloom_core and see it directly.
 *
 * ## Why this exists
 *
 * `src/app/main.cpp` drives the Vulkan layer itself -- GpuBuffer, CommandHelper,
 * BLAS/TLAS, RayTracingPipeline -- as a ~25-stage pipeline that runs in parallel with
 * the equivalent private steps inside ExternalRenderContext::Impl. Two orchestrators
 * for one renderer means a fix lands in one and not the other, and they have already
 * drifted: the GUI runs the sensor chain on the GPU while the CLI runs GenericSensor
 * on the CPU, and the two equirectangular-to-cubemap conversions disagreed on which
 * way is up (see EquirectToCubemap below).
 *
 * Stages move here one at a time, each verified by a render before the next one
 * starts. When the set is complete, ExternalRenderContext becomes a thin adapter over
 * it and the CLI stops needing library internals -- which is what still keeps ~274
 * symbols exported for its sake alone.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Config.hpp"
#include "core/Image.hpp"
#include "core/Types.hpp"
#include "scene/Scene.hpp"

namespace quantiloom::rendercore {

/**
 * @brief Build a Scene from a parsed TOML scene configuration
 *
 * Resolves, in order: `scene.usd`, `scene.gltf`, `scene.preset`, then a Cornell box
 * with a warning. Loading files is all this does -- no GPU resources are touched, so
 * both the offline path and the interactive context can call it before they have a
 * device.
 *
 * @note Merged from two implementations that had drifted. The CLI's supported
 *       presets and fell back to a Cornell box; ExternalRenderContext::LoadScene
 *       supported neither and rejected any config without scene.gltf or scene.usd,
 *       which is why a procedural TOML renders from the CLI but would not open in
 *       the GUI. The CLI's superset is what survived.
 * @note The two also disagreed on precedence when a config names both a USD and a
 *       glTF file: the CLI took the USD, the context took the glTF. No config in
 *       assets/configs does both, so the case was unreachable either way; the CLI's
 *       order is kept because it is the one covered by a render test.
 */
Result<Scene, String> LoadSceneFromConfig(const Config& config);

/**
 * @brief Convert an equirectangular (latitude-longitude) environment map to cubemap faces
 *
 * @param equirect Source map. Row 0 is the zenith, which is what OpenEXR's scanline
 *                 order gives and what ImageIO::ReadEXR passes through unchanged.
 * @param faceSize Edge length of each square face
 * @return Six RGB faces in Vulkan order: +X, -X, +Y, -Y, +Z, -Z
 *
 * @note This replaced two implementations that were vertically mirrored relative to
 *       each other. The CLI mapped v = acos(y)/pi, so +Y sampled row 0; the GUI mapped
 *       v = (asin(y) + pi/2)/pi, so +Y sampled the last row -- the ground. Since
 *       acos(y)/pi == 1 - (asin(y) + pi/2)/pi exactly, the GUI rendered every
 *       environment map upside down. The CLI's mapping is the one kept.
 */
Vector<Image> EquirectToCubemap(const Image& equirect, u32 faceSize);

} // namespace quantiloom::rendercore
