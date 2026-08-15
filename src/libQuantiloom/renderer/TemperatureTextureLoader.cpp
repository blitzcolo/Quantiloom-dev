/**
 * @file TemperatureTextureLoader.cpp
 * @brief Authored temperature maps into scene textures
 */

#include "renderer/TemperatureTextureLoader.hpp"

#include "core/Log.hpp"
#include "io/ImageIO.hpp"
#include "renderer/ConfigResolve.hpp"

#include <algorithm>

namespace quantiloom::rendercore {

namespace {

/// Load an authored temperature map and re-encode its R channel to UNORM8.
/// Authors normalise the map to [0, 1] against the material's scale/offset;
/// values outside that range are clamped, because a map carrying absolute
/// kelvin would otherwise silently saturate at temperatureOffset + scale.
std::optional<Texture> LoadTemperatureTexture(const String& path, const String& materialName) {
    auto imageResult = ImageIO::ReadImage(path);
    if (!imageResult.has_value()) {
        QL_LOG_WARN("  Material '{}': cannot read temperature texture '{}'", materialName, path);
        return std::nullopt;
    }

    const Image& img = imageResult.value();
    const usize texelCount = static_cast<usize>(img.width) * img.height;
    if (texelCount == 0 || img.channels < 1) {
        QL_LOG_WARN("  Material '{}': temperature texture '{}' is empty", materialName, path);
        return std::nullopt;
    }

    Texture tex;
    tex.name = "__temperature_" + materialName;
    tex.width = img.width;
    tex.height = img.height;
    tex.channels = 4;
    tex.sourceUri = path;
    tex.isSRGB = false;              // temperatures are numbers, not colour
    tex.skipBlockCompression = true; // and BC7 would smear them across blocks
    tex.pixels.resize(texelCount * 4);

    usize clipped = 0;
    for (usize i = 0; i < texelCount; ++i) {
        const f32 v = img.data[i * img.channels];
        if (v < 0.0f || v > 1.0f) {
            ++clipped;
        }
        tex.pixels[i * 4] = static_cast<u8>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
        tex.pixels[i * 4 + 1] = 0;
        tex.pixels[i * 4 + 2] = 0;
        tex.pixels[i * 4 + 3] = 255;
    }
    if (clipped > 0) {
        QL_LOG_WARN("  Material '{}': {} texel(s) outside [0, 1] in '{}' were clamped -- "
                    "the map should be normalised, with kelvin carried by "
                    "temperature_scale and temperature_offset",
                    materialName, clipped, path);
    }
    return tex;
}

}  // namespace

void MountTemperatureTextures(Scene& scene, const String& baseDir, ConfigApplyReport& report) {
    u32 mounted = 0;
    for (auto& material : scene.materials) {
        if (material.temperatureTexturePath.empty()) {
            continue;
        }

        auto tex = LoadTemperatureTexture(
            ResolveConfigPath(material.temperatureTexturePath, baseDir), material.name);
        if (!tex.has_value()) {
            // The scene file's own map, if it had one, stays in effect; a
            // config typo costs the override and never the render.
            continue;
        }

        material.temperatureTextureIndex = static_cast<i32>(scene.textures.size());
        scene.textures.push_back(std::move(tex.value()));
        ++mounted;

        QL_LOG_INFO("  Material '{}': temperature texture index {} ({}x{}), "
                    "T = r * {:.1f} + {:.1f} K, resolution {:.2f} K/step",
                    material.name, material.temperatureTextureIndex,
                    scene.textures.back().width, scene.textures.back().height,
                    material.temperatureScale, material.temperatureOffset,
                    material.temperatureScale / 255.0f);
    }

    if (mounted > 0) {
        report.messages.push_back(
            {ConfigApplyMessage::Severity::Info, "materials.temperature_texture",
             "  Mounted " + std::to_string(mounted) + " temperature texture(s)"});
    }
}

}  // namespace quantiloom::rendercore
