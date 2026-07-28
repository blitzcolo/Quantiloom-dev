#include "renderer/RenderCore.hpp"

#include "core/Log.hpp"
#include "io/GltfLoader.hpp"
#include "io/UsdLoader.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>

namespace quantiloom::rendercore {

Result<Scene, String> LoadSceneFromConfig(const Config& config) {
    if (config.Has("scene.usd")) {
        const auto usdPath = config.Get<String>("scene.usd");
        QL_LOG_INFO("Loading USD scene: {}", usdPath);

        auto result = UsdLoader::LoadFromFile(usdPath);
        if (!result.has_value()) {
            return Result<Scene, String>::Err("Failed to load USD: " + result.error());
        }
        return Result<Scene, String>(std::move(result.value()));
    }

    if (config.Has("scene.gltf")) {
        const auto gltfPath = config.Get<String>("scene.gltf");
        QL_LOG_INFO("Loading glTF model: {}", gltfPath);

        auto result = GltfLoader::LoadFromFile(gltfPath);
        if (!result.has_value()) {
            return Result<Scene, String>::Err("Failed to load glTF: " + result.error());
        }
        return Result<Scene, String>(std::move(result.value()));
    }

    return Result<Scene, String>::Err(
        "No scene.usd or scene.gltf in config -- nothing to render");
}

namespace {

// Direction through the centre of a cubemap texel. Face order is Vulkan's:
// 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
glm::vec3 CubemapFaceDirection(u32 face, f32 u, f32 v) {
    const f32 uc = 2.0f * u - 1.0f;
    const f32 vc = 2.0f * v - 1.0f;

    switch (face) {
        case 0:  return glm::normalize(glm::vec3( 1.0f,   -vc,   -uc));
        case 1:  return glm::normalize(glm::vec3(-1.0f,   -vc,    uc));
        case 2:  return glm::normalize(glm::vec3(   uc,  1.0f,    vc));
        case 3:  return glm::normalize(glm::vec3(   uc, -1.0f,   -vc));
        case 4:  return glm::normalize(glm::vec3(   uc,   -vc,  1.0f));
        case 5:  return glm::normalize(glm::vec3(  -uc,   -vc, -1.0f));
        default: return glm::vec3(0.0f);
    }
}

// Bilinear lookup into the equirectangular source.
//
// v = acos(y)/pi, so +Y lands on row 0. That is the zenith: OpenEXR stores scanlines
// top-down and ImageIO::ReadEXR does not flip them, which a 4K noon-sky HDRI confirms
// -- its top quarter is ~47x brighter than its bottom quarter.
glm::vec3 SampleEquirect(const Image& equirect, const glm::vec3& dir) {
    const f32 theta = std::acos(glm::clamp(dir.y, -1.0f, 1.0f));  // [0, pi] from +Y
    const f32 phi   = std::atan2(dir.z, dir.x);                   // [-pi, pi]

    const f32 u = (phi + glm::pi<f32>()) / (2.0f * glm::pi<f32>());
    const f32 v = theta / glm::pi<f32>();

    const u32 width  = equirect.width;
    const u32 height = equirect.height;

    const f32 fx = u * static_cast<f32>(width - 1);
    const f32 fy = v * static_cast<f32>(height - 1);

    const u32 x0 = static_cast<u32>(fx) % width;
    const u32 y0 = static_cast<u32>(fy) % height;
    const u32 x1 = (x0 + 1) % width;
    const u32 y1 = std::min(y0 + 1, height - 1);

    const f32 wx = fx - std::floor(fx);
    const f32 wy = fy - std::floor(fy);

    const glm::vec3 c00(equirect(x0, y0, 0), equirect(x0, y0, 1), equirect(x0, y0, 2));
    const glm::vec3 c10(equirect(x1, y0, 0), equirect(x1, y0, 1), equirect(x1, y0, 2));
    const glm::vec3 c01(equirect(x0, y1, 0), equirect(x0, y1, 1), equirect(x0, y1, 2));
    const glm::vec3 c11(equirect(x1, y1, 0), equirect(x1, y1, 1), equirect(x1, y1, 2));

    const glm::vec3 c0 = c00 * (1.0f - wx) + c10 * wx;
    const glm::vec3 c1 = c01 * (1.0f - wx) + c11 * wx;

    return c0 * (1.0f - wy) + c1 * wy;
}

}  // namespace

Vector<Image> EquirectToCubemap(const Image& equirect, const u32 faceSize) {
    QL_LOG_INFO("  Converting equirectangular to cubemap ({}x{} per face)...",
                faceSize, faceSize);

    Vector<Image> faces(6);
    for (u32 face = 0; face < 6; ++face) {
        faces[face] = Image(faceSize, faceSize, 3);

        for (u32 y = 0; y < faceSize; ++y) {
            for (u32 x = 0; x < faceSize; ++x) {
                const f32 u = (static_cast<f32>(x) + 0.5f) / static_cast<f32>(faceSize);
                const f32 v = (static_cast<f32>(y) + 0.5f) / static_cast<f32>(faceSize);

                const glm::vec3 color =
                    SampleEquirect(equirect, CubemapFaceDirection(face, u, v));

                faces[face](x, y, 0) = color.r;
                faces[face](x, y, 1) = color.g;
                faces[face](x, y, 2) = color.b;
            }
        }
    }

    return faces;
}

}  // namespace quantiloom::rendercore
