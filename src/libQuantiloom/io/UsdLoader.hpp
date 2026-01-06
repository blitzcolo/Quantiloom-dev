/**
 * @file UsdLoader.hpp
 * @brief OpenUSD scene loader for 3D scenes with PBR materials
 *
 * Provides UsdLoader class for loading OpenUSD files:
 * - Parses .usd, .usda (ASCII), .usdc (Crate binary), and .usdz (zipped) formats
 * - Converts UsdGeomMesh to Quantiloom GeometryPrimitive format
 * - Converts UsdPreviewSurface materials to Quantiloom Material format
 * - Loads texture assets (PNG/JPEG/EXR) via ImageIO
 * - Flattens Xform hierarchy to world-space SceneNode transforms
 * - Supports basic USD References for external file composition
 *
 * Supported OpenUSD features:
 * - UsdGeomMesh with triangle/polygon primitives (auto-triangulated)
 * - UsdShadeMaterial with UsdPreviewSurface (PBR metallic-roughness)
 * - Textures: diffuseColor, metallic, roughness, normal, emissive
 * - Xform hierarchy (flattened to world-space transforms)
 * - Basic References (external USD file composition)
 *
 * Quantiloom spectral extensions via custom primvars:
 * - primvars:quantiloom:materialType - Spectral database type
 * - primvars:quantiloom:materialRef - Material name in database
 * - primvars:quantiloom:emissivityCurve - Path to emissivity CSV
 * - primvars:quantiloom:reflectanceCurve - Path to reflectance CSV
 * - primvars:quantiloom:transmittanceCurve - Path to transmittance CSV
 * - primvars:quantiloom:temperature_K - Surface temperature (K)
 *
 * NOT supported:
 * - Animation/skeletal deformation (UsdSkel)
 * - Complex USD composition (Variants, Payloads, Inherits)
 * - MaterialX materials
 * - UsdGeomBasisCurves/Points (only meshes)
 * - UsdLux lights (use Quantiloom config instead)
 * - UsdGeomCamera (use Quantiloom config instead)
 *
 * Uses TinyUSDZ library for USD parsing.
 *
 * @note Returns Result<Scene, String> for explicit error handling
 * @note Scene graph flattened to world space (no hierarchy preserved)
 * @note Polygons are triangulated during loading
 *
 * @author wtflmao
 */

#pragma once

#include "core/Types.hpp"
#include "scene/Scene.hpp"
#include "scene/Mesh.hpp"
#include "scene/Material.hpp"
#include "scene/Texture.hpp"
#include <string>
#include <vector>

// ============================================================================
// UsdLoader - Loads OpenUSD scenes (.usd, .usda, .usdc, .usdz)
// ============================================================================

namespace quantiloom {

/**
 * @class UsdLoader
 * @brief Static utility class for loading OpenUSD 3D scenes into Quantiloom scenes
 *
 * Converts OpenUSD files into Quantiloom Scene format.
 * Handles all USD data extraction, format conversion, and resource loading.
 *
 * Loading workflow:
 * 1. Parse USD file using TinyUSDZ library
 * 2. Resolve References (external USD files)
 * 3. Extract and convert meshes -> Quantiloom Mesh/GeometryPrimitive
 * 4. Extract and convert materials -> Quantiloom Material (PBR)
 * 5. Load textures -> Quantiloom Texture (via ImageIO)
 * 6. Flatten Xform hierarchy -> SceneNode list with world transforms
 * 7. Parse Quantiloom spectral extensions (custom primvars)
 * 8. Return complete Scene object
 *
 * Usage example:
 * @code
 * // Load USD scene
 * auto result = UsdLoader::LoadFromFile("scenes/my_scene.usda");
 * if (!result.has_value()) {
 *     QL_LOG_ERROR("Failed to load USD: {}", result.error());
 *     return;
 * }
 *
 * Scene scene = std::move(result.value());
 * QL_LOG_INFO("Loaded: {} meshes, {} materials, {} textures",
 *             scene.meshes.size(),
 *             scene.materials.size(),
 *             scene.textures.size());
 *
 * // Scene ready for rendering
 * // Build BLAS for each mesh primitive...
 * @endcode
 *
 * Error handling:
 * @code
 * auto result = UsdLoader::LoadFromFile("scene.usd");
 * if (!result.has_value()) {
 *     // Error message in result.error()
 *     QL_LOG_ERROR("Load failed: {}", result.error());
 * }
 * @endcode
 *
 * @note Supports .usd, .usda (ASCII), .usdc (Crate binary), .usdz (zipped)
 * @note Scene graph flattened: parent transforms accumulated into world space
 * @note Polygon meshes are automatically triangulated
 * @note Texture data loaded via ImageIO (PNG/JPEG/EXR)
 *
 * @see Scene for output scene structure
 * @see Material for PBR material mapping
 * @see Texture for texture data format
 * @see GltfLoader for similar glTF 2.0 loading functionality
 */
class QL_API UsdLoader {
public:
    // ========================================================================
    // Loading Functions
    // ========================================================================

    /**
     * @brief Load USD file (.usd, .usda, .usdc, .usdz)
     * @param path Path to USD file
     * @return Scene with meshes, materials, textures, and nodes on success
     * @return Error message string on failure
     */
    static Result<Scene, String> LoadFromFile(const String& path);

private:
    // ========================================================================
    // Internal Parsing Functions
    // ========================================================================

    /**
     * @brief Parse UsdGeomMesh to Quantiloom Mesh
     * @note Polygons are triangulated during parsing
     * @note Each mesh maps to one Quantiloom Mesh with one or more GeometryPrimitives
     */
    static Mesh ParseMesh(const void* stage, const void* geomMesh,
                          const std::vector<Material>& materials);

    /**
     * @brief Parse UsdShadeMaterial with UsdPreviewSurface to Quantiloom Material
     * @note Converts PBR metallic-roughness workflow
     */
    static Material ParseMaterial(const void* stage, const void* shadeMaterial,
                                   const std::vector<Texture>& textures,
                                   const String& usdFilePath);

    /**
     * @brief Parse texture from USD shader node
     * @note Loads image via ImageIO (PNG/JPEG/EXR)
     */
    static Texture ParseTexture(const void* stage, const String& assetPath,
                                 const String& usdFilePath);

    /**
     * @brief Flatten USD Xform hierarchy to world-space nodes
     * @note Resolves References to external USD files
     * @note Computes accumulated transforms for each node
     */
    static std::vector<SceneNode> FlattenXformHierarchy(const void* stage);

    /**
     * @brief Parse Quantiloom spectral extensions from USD custom primvars
     * @note Mirrors glTF QUANTILOOM_material_ir extension behavior
     */
    static void ParseSpectralExtensions(Material& mat, const void* prim,
                                         const String& usdFilePath);

    // ========================================================================
    // Geometry Utilities
    // ========================================================================

    /**
     * @brief Triangulate polygon mesh
     * @param faceVertexCounts Number of vertices per face
     * @param faceVertexIndices Vertex indices for all faces
     * @return Triangulated index buffer
     */
    static std::vector<u32> TriangulatePolygons(
        const std::vector<i32>& faceVertexCounts,
        const std::vector<i32>& faceVertexIndices);
};

} // namespace quantiloom
