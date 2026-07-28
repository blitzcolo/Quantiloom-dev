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
 * Resolves `scene.usd`, then `scene.gltf`. A config naming neither is an error:
 * quietly rendering something else turns a misspelt key into a wrong picture rather
 * than a message. Loading files is all this does -- no GPU resources are touched, so
 * both the offline path and the interactive context can call it before they have a
 * device.
 *
 * @note Merged from two implementations that had drifted. The CLI also accepted a
 *       `scene.preset` naming one of three meshes built in code, and fell back to a
 *       procedural Cornell box when a config named no scene at all;
 *       ExternalRenderContext::LoadScene did neither. Both went away rather than
 *       being adopted: assets/models/cornell_box/ holds a committed Cornell box
 *       built to the original specification with ECOSTRESS spectral reflectances,
 *       which the cornell_box_{vis,swir,mwir,lwir} configs use, and a scene whose
 *       materials carry no spectral data is of little use to a spectral renderer.
 * @note The two disagreed on precedence when a config names both a USD and a glTF:
 *       the CLI took the USD, the context the glTF. No config in assets/configs does
 *       both, so the case was unreachable either way; the CLI's order is kept
 *       because it is the one covered by a render test.
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
