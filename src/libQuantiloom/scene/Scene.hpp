/**
 * @file Scene.hpp
 * @brief Top-level scene container for geometry, materials, camera, and spectral configuration
 *
 * Provides Scene class as the central container for all rendering data:
 * - Geometry: Meshes (geometry primitives) and SceneNodes (instances with transforms)
 * - Materials: PBR material definitions with spectral/IR properties
 * - Textures: CPU-side texture images (uploaded to GPU by TextureManager)
 * - Camera: Viewpoint configuration
 * - Spectral config: Wavelength bands, ranges, atmosphere LUTs
 *
 * Scene lifetime:
 * - Created at application startup (Scene::FromConfig or manually)
 * - Must outlive Renderer (renderer holds references, not ownership)
 * - Destroyed at shutdown after renderer cleanup
 *
 * Loading sources:
 * - TOML configuration files (Scene::FromConfig)
 * - glTF 2.0 models (GltfLoader::LoadFromFile)
 * - Procedural generation (SceneBuilder test scenes)
 *
 * @author wtflmao
 */

#pragma once

#include "Camera.hpp"
#include "Mesh.hpp"
#include "Material.hpp"
#include "Texture.hpp"
#include "core/Types.hpp"
#include "core/Config.hpp"
#include "core/LUT.hpp"
#include <vector>
#include <string>

// ============================================================================
// SpectralBand - Band-pass configuration for multi-spectral rendering
// ============================================================================

namespace quantiloom {

/**
 * @struct SpectralBand
 * @brief Defines a single spectral band with center wavelength and bandwidth
 *
 * Used in multi-spectral rendering mode to define output channels.
 * Each band represents a filter response function (typically Gaussian).
 *
 * Example:
 * @code
 * SpectralBand visGreen = {"VIS_Green", 550.0f, 40.0f};  // 530-570nm (FWHM=40nm)
 * SpectralBand nirBand = {"NIR_850", 850.0f, 30.0f};     // 835-865nm
 * @endcode
 *
 * @see Scene::bands for multi-spectral channel configuration
 */
struct SpectralBand {
    String name;
    f32 center_nm = 550.0f;
    f32 fwhm_nm = 40.0f;

    bool IsValid() const {
        return center_nm > 0.0f && fwhm_nm > 0.0f;
    }
};

// ============================================================================
// Scene - Top-level scene container
// ============================================================================
/**
 * @class Scene
 * @brief Top-level container for all scene data (geometry, materials, camera, spectral config)
 *
 * Central data structure holding all rendering resources:
 * - Geometry: meshes (GeometryPrimitive collections) + nodes (instances with transforms)
 * - Materials: PBR definitions with spectral/IR properties
 * - Textures: CPU image data (RGB/RGBA, uploaded to GPU separately)
 * - Camera: Viewpoint configuration (position, FOV, look-at)
 * - Spectral: Wavelength bands, ranges, atmosphere LUTs
 *
 * Scene graph structure:
 * @code
 * Scene
 * ├── meshes[] (geometry definitions)
 * │   └── primitives[] (vertex/index data + material ID)
 * ├── nodes[] (mesh instances with transforms)
 * │   ├── meshIndex (reference to meshes[])
 * │   └── transform (4x4 matrix, world space)
 * ├── materials[] (PBR + spectral properties)
 * └── textures[] (CPU images, referenced by materials)
 * @endcode
 *
 * Usage example:
 * @code
 * // Load from config
 * auto config = Config::Load("scene.toml");
 * auto sceneResult = Scene::FromConfig(*config);
 * if (!sceneResult.has_value()) {
 *     QL_LOG_ERROR("Failed to load scene: {}", sceneResult.error());
 *     return;
 * }
 * Scene scene = std::move(sceneResult.value());
 *
 * // Or load glTF
 * auto gltfResult = GltfLoader::LoadFromFile("model.gltf");
 * Scene scene = std::move(gltfResult.value());
 *
 * // Access scene data
 * QL_LOG_INFO("Scene: {} meshes, {} materials, {} triangles",
 *             scene.meshes.size(), scene.materials.size(),
 *             scene.GetTotalTriangleCount());
 * @endcode
 *
 * @note Scene must outlive Renderer (renderer holds references)
 * @note Meshes are indexed by nodes (one mesh can have multiple instances)
 * @note Materials are indexed by primitives (triangle groups)
 * @note Textures are indexed by materials (bindless descriptor array on GPU)
 *
 * @see GltfLoader for loading glTF 2.0 models
 * @see SceneBuilder for procedural test scenes
 * @see main.cpp for complete integration example
 */
class QL_API Scene {
public:
    // ========================================================================
    // Construction
    // ========================================================================

    Scene() = default;

    // Load scene from TOML configuration
    static Result<Scene, String> FromConfig(const Config& config);

    // ========================================================================
    // Scene Data (public for direct access)
    // ========================================================================

    // Rendering configuration
    Camera camera;
    u32 width = 1280;   // Render resolution width
    u32 height = 720;   // Render resolution height

    // Scene graph and resources
    std::vector<Mesh> meshes;            // Mesh definitions (geometry primitives)
    std::vector<SceneNode> nodes;        // Scene instances (mesh + transform)
    std::vector<Material> materials;     // Material definitions
    std::vector<Texture> textures;       // Texture images (CPU-side data)

    // Spectral configuration
    std::vector<SpectralBand> bands;  // For MS-RT mode
    f32 lambda_min = 380.0f;          // For HS-OFF mode (nm)
    f32 lambda_max = 760.0f;
    f32 delta_lambda = 5.0f;

    // Atmosphere LUT (optional, for LUT-fast mode)
    Optional<AtmosphereLUT> atmosphereLUT;

    // Metadata
    String name = "Untitled Scene";
    String description;

    // ========================================================================
    // Utilities
    // ========================================================================

    // Check if scene is valid
    bool IsValid() const;

    // Get total triangle count
    u32 GetTotalTriangleCount() const;

    // Get total vertex count
    u32 GetTotalVertexCount() const;

    // Get scene bounding box diagonal length (meters)
    // Used for automatic atmospheric scattering enable/disable
    f32 GetBoundingBoxSize() const;

    // Print scene summary (for debugging)
    void PrintSummary() const;
};

} // namespace quantiloom
