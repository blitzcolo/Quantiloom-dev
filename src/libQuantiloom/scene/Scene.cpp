#include "Scene.hpp"
#include "core/Log.hpp"
#include "io/LUTLoader.hpp"
#include <filesystem>

namespace quantiloom {

// ============================================================================
// Scene::FromConfig - Load scene from TOML configuration
// ============================================================================

Result<Scene, String> Scene::FromConfig(const Config& config) {
    Scene scene;

    // ========================================================================
    // Camera Configuration
    // ========================================================================

    if (config.Has("camera")) {
        auto cameraTable = config.GetTable("camera");
        if (cameraTable) {
            const Config& cam = *cameraTable;

            // Position
            glm::vec3 position = scene.camera.GetPosition();
            if (cam.Has("position")) {
                auto pos = cam.GetArray<f32>("position");
                if (pos.size() == 3) {
                    position = glm::vec3(pos[0], pos[1], pos[2]);
                }
            }

            // Look-at
            glm::vec3 lookAt = scene.camera.GetLookAt();
            if (cam.Has("look_at")) {
                auto lookAtArr = cam.GetArray<f32>("look_at");
                if (lookAtArr.size() == 3) {
                    lookAt = glm::vec3(lookAtArr[0], lookAtArr[1], lookAtArr[2]);
                }
            }

            // Up vector
            glm::vec3 up = scene.camera.GetUp();
            if (cam.Has("up")) {
                auto upArr = cam.GetArray<f32>("up");
                if (upArr.size() == 3) {
                    up = glm::vec3(upArr[0], upArr[1], upArr[2]);
                }
            }

            // Field of view
            f32 fov = cam.Get<f32>("fov", 45.0f);

            // Update camera with new values
            scene.camera.SetPosition(position);
            scene.camera.SetLookAt(lookAt);
            scene.camera.SetUp(up);
            scene.camera.SetFovY(fov);
        }
    }

    // ========================================================================
    // Renderer Configuration (resolution)
    // ========================================================================

    if (config.Has("renderer.resolution")) {
        auto res = config.GetArray<i32>("renderer.resolution");
        if (res.size() == 2) {
            scene.width = static_cast<u32>(res[0]);
            scene.height = static_cast<u32>(res[1]);
            // Update camera aspect ratio
            f32 aspectRatio = static_cast<f32>(scene.width) / static_cast<f32>(scene.height);
            scene.camera.SetAspectRatio(aspectRatio);
        }
    }

    // ========================================================================
    // Spectral Configuration
    // ========================================================================

    // MS-RT mode: band definitions
    if (config.Has("spectral.bands")) {
        auto bandsTable = config.GetTable("spectral");
        if (bandsTable) {
            // Try to parse bands array (TOML array of tables)
            // Note: toml++ doesn't have direct array-of-tables getter,
            // need to access via root table
            const auto& root = config.GetRoot();
            if (root["spectral"]["bands"].is_array()) {
                const auto& bandsArray = *root["spectral"]["bands"].as_array();

                for (const auto& bandNode : bandsArray) {
                    if (bandNode.is_table()) {
                        const auto& bandTable = *bandNode.as_table();

                        SpectralBand band;
                        if (bandTable["name"].is_string()) {
                            band.name = *bandTable["name"].value<std::string>();
                        }
                        if (bandTable["center_nm"].is_number()) {
                            band.center_nm = static_cast<f32>(*bandTable["center_nm"].value<double>());
                        }
                        if (bandTable["fwhm_nm"].is_number()) {
                            band.fwhm_nm = static_cast<f32>(*bandTable["fwhm_nm"].value<double>());
                        }

                        if (band.IsValid()) {
                            scene.bands.push_back(band);
                        } else {
                            QL_LOG_WARN("Invalid spectral band: {}", band.name);
                        }
                    }
                }

                QL_LOG_INFO("Loaded {} spectral bands", scene.bands.size());
            }
        }
    }

    // HS-OFF mode: wavelength range
    if (config.Has("spectral.range_nm")) {
        auto range = config.GetArray<f32>("spectral.range_nm");
        if (range.size() == 2) {
            scene.lambda_min = range[0];
            scene.lambda_max = range[1];
        }
    }

    if (config.Has("spectral.step_nm")) {
        scene.delta_lambda = config.Get<f32>("spectral.step_nm", 5.0f);
    }

    // ========================================================================
    // Atmosphere LUT (optional)
    // ========================================================================

    if (config.Has("atmosphere.lut")) {
        String lutPath = config.Get<String>("atmosphere.lut", "");
        if (!lutPath.empty() && std::filesystem::exists(lutPath)) {
            auto lut = LUTLoader::LoadHDF5(lutPath);
            if (lut && lut->IsValid()) {
                scene.atmosphereLUT = std::move(*lut);
                QL_LOG_INFO("Loaded atmosphere LUT: {} wavelength samples", scene.atmosphereLUT->Size());
            } else {
                QL_LOG_WARN("Failed to load atmosphere LUT: {}", lutPath);
            }
        }
    }

    // ========================================================================
    // Scene Metadata
    // ========================================================================

    scene.name = config.Get<String>("scene.name", "Untitled Scene");
    scene.description = config.Get<String>("scene.description", "");

    // ========================================================================
    // Validation
    // ========================================================================

    if (!scene.IsValid()) {
        return Result<Scene, String>::Err("Scene validation failed after loading");
    }

    QL_LOG_INFO("Scene loaded successfully: {}", scene.name);
    scene.PrintSummary();

    return scene;
}

// ============================================================================
// Utilities
// ============================================================================

bool Scene::IsValid() const {
    // Resolution must be non-zero
    if (width == 0 || height == 0) {
        return false;
    }

    // Must have either bands (MS-RT) or wavelength range (HS-OFF)
    if (bands.empty() && (lambda_min >= lambda_max || delta_lambda <= 0.0f)) {
        return false;
    }

    // All materials must be valid
    for (const auto& mat : materials) {
        if (!mat.IsValid()) {
            return false;
        }
    }

    // All meshes must be valid
    for (const auto& mesh : meshes) {
        if (!mesh.IsValid()) {
            return false;
        }
    }

    return true;
}

u32 Scene::GetTotalTriangleCount() const {
    u32 total = 0;
    for (const auto& mesh : meshes) {
        total += mesh.GetTotalTriangleCount();
    }
    return total;
}

u32 Scene::GetTotalVertexCount() const {
    u32 total = 0;
    for (const auto& mesh : meshes) {
        total += mesh.GetTotalVertexCount();
    }
    return total;
}

void Scene::PrintSummary() const {
    QL_LOG_INFO("========================================");
    QL_LOG_INFO("  Scene: {}", name);
    QL_LOG_INFO("========================================");
    QL_LOG_INFO("Camera:");
    QL_LOG_INFO("  Position: ({:.2f}, {:.2f}, {:.2f})",
                camera.GetPosition().x, camera.GetPosition().y, camera.GetPosition().z);
    QL_LOG_INFO("  Look-at:  ({:.2f}, {:.2f}, {:.2f})",
                camera.GetLookAt().x, camera.GetLookAt().y, camera.GetLookAt().z);
    QL_LOG_INFO("  FOV: {:.1f} deg", camera.GetFovY());
    QL_LOG_INFO("  Resolution: {}x{}", width, height);

    QL_LOG_INFO("Geometry:");
    QL_LOG_INFO("  Meshes: {}", meshes.size());
    QL_LOG_INFO("  Materials: {}", materials.size());
    QL_LOG_INFO("  Triangles: {}", GetTotalTriangleCount());
    QL_LOG_INFO("  Vertices: {}", GetTotalVertexCount());

    QL_LOG_INFO("Spectral:");
    if (!bands.empty()) {
        QL_LOG_INFO("  Mode: MS-RT");
        QL_LOG_INFO("  Bands: {}", bands.size());
        for (const auto& band : bands) {
            QL_LOG_INFO("    - {}: {:.1f} nm (FWHM: {:.1f} nm)",
                        band.name, band.center_nm, band.fwhm_nm);
        }
    } else {
        QL_LOG_INFO("  Mode: HS-OFF");
        QL_LOG_INFO("  Range: {:.1f} - {:.1f} nm", lambda_min, lambda_max);
        QL_LOG_INFO("  Step: {:.1f} nm", delta_lambda);
        u32 numBands = static_cast<u32>((lambda_max - lambda_min) / delta_lambda) + 1;
        QL_LOG_INFO("  Total bands: {}", numBands);
    }

    if (atmosphereLUT.has_value()) {
        QL_LOG_INFO("Atmosphere:");
        QL_LOG_INFO("  LUT loaded: {} wavelength samples", atmosphereLUT->Size());
    }

    QL_LOG_INFO("========================================");
}

} // namespace quantiloom
