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
// SpectralBand - Band-pass configuration for MS-RT mode
// ============================================================================
// Defines a single spectral band with:
// - center_nm: Center wavelength (nanometers)
// - fwhm_nm: Full-width at half-maximum (bandwidth)
// - name: Human-readable identifier
//
// Used in MS-RT mode to define output channels
// ============================================================================

namespace quantiloom {

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
// Responsibilities:
// - Hold all scene data (geometry, materials, camera, spectral config)
// - Load from TOML configuration (Scene::FromConfig)
// - Provide lifetime management (owns all data)
//
// Usage:
//   auto config = Config::Load("scene.toml");
//   auto scene = Scene::FromConfig(*config);
//   renderer.SetScene(scene);  // Renderer references Scene
//
// Lifetime:
// - Scene must outlive Renderer (Renderer holds reference, not ownership)
// - Typically created at application startup, destroyed at shutdown
// ============================================================================

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

    // Print scene summary (for debugging)
    void PrintSummary() const;
};

} // namespace quantiloom
