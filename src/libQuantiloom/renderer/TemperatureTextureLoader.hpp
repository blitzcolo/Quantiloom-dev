/**
 * @file TemperatureTextureLoader.hpp
 * @brief Authored temperature maps into scene textures
 *
 * Material::temperatureTextureIndex has reached the GPU since the field
 * existed, but only glTF extras and USD attributes could fill it. This mounts
 * the TOML side: a [[materials]] entry names an image, and the R channel
 * becomes the normalised temperature the shader already decodes as
 * T = r * temperatureScale + temperatureOffset.
 *
 * TIMING. Must run before TextureManager::UploadTextures, which fixes the
 * texture indices. Same window BuildUnmixWeightTextures uses; see the call
 * sites in OfflineRenderer and ExternalRenderContext.
 */

#pragma once

#include "core/Types.hpp"
#include "renderer/ConfigApply.hpp"
#include "scene/Scene.hpp"

namespace quantiloom::rendercore {

/**
 * @brief Load every material's authored temperature map into scene.textures
 *
 * A material with a non-empty temperatureTexturePath gets the image loaded,
 * its R channel re-encoded to UNORM8, and temperatureTextureIndex pointed at
 * the new texture -- overwriting an index a scene file may have provided,
 * because the config is the override channel. A path that fails to load warns
 * and leaves the material exactly as the scene file set it.
 *
 * @param baseDir  directory a temperature_texture path is relative to
 */
void MountTemperatureTextures(Scene& scene, const String& baseDir, ConfigApplyReport& report);

}  // namespace quantiloom::rendercore
