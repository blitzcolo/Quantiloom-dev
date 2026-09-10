/**
 * @file UsdLoader.hpp
 * @brief OpenUSD scene loader for 3D scenes with PBR materials
 *
 * Provides UsdLoader class for loading OpenUSD files:
 * - Parses .usd, .usda (ASCII), .usdc (Crate binary), and .usdz (zipped) formats
 * - Converts UsdGeomMesh to Quantiloom GeometryPrimitive format
 * - Converts UsdPreviewSurface and MaterialX materials to Quantiloom Material format
 * - Loads texture assets (PNG/JPEG/EXR) via ImageIO
 * - Flattens Xform hierarchy to world-space SceneNode transforms
 * - Full USD composition support (sublayers, references, payloads, variants, inherits)
 *
 * Supported OpenUSD features:
 * - UsdGeomMesh with triangle/polygon primitives (auto-triangulated)
 * - UsdShadeMaterial with UsdPreviewSurface (PBR metallic-roughness)
 * - MaterialX standard_surface shader (converted to PBR)
 * - Textures: diffuseColor, metallic, roughness, normal, emissive (with connections)
 * - Xform hierarchy (flattened to world-space transforms)
 * - Full USD composition (sublayers, references, payloads, variants, inherits)
 * - Variant selection via UsdLoadOptions
 * - GeomSubsets for multi-material meshes
 * - PointInstancer for efficient instancing
 *
 * Quantiloom spectral extensions via custom attributes:
 * - quantiloom:materialType - Spectral database type
 * - quantiloom:materialRef - Material name in database
 * - quantiloom:emissivityCurve - Path to emissivity CSV
 * - quantiloom:reflectanceCurve - Path to reflectance CSV
 * - quantiloom:transmittanceCurve - Path to transmittance CSV
 * - quantiloom:temperature_K - Surface temperature (K)
 *
 * NOT supported:
 * - Animation/skeletal deformation (UsdSkel)
 * - UsdGeomBasisCurves/Points (only meshes)
 * - UsdLux lights (use Quantiloom config instead)
 * - UsdGeomCamera (use Quantiloom config instead)
 *
 * Uses Pixar OpenUSD library for USD parsing (conditional compilation).
 * When OpenUSD is not available, returns an error message.
 *
 * @note Returns Result<Scene, String> for explicit error handling
 * @note Scene graph flattened to world space (no hierarchy preserved)
 * @note Polygons are triangulated during loading
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include "scene/Scene.hpp"
#include "scene/Mesh.hpp"
#include "scene/Material.hpp"
#include "scene/Texture.hpp"
#include <string>
#include <vector>
#include <unordered_map>

// ============================================================================
// UsdLoader - Loads OpenUSD scenes (.usd, .usda, .usdc, .usdz)
// ============================================================================

namespace quantiloom {

/**
 * @struct UsdLoadOptions
 * @brief Configuration options for USD scene loading
 *
 * Allows customization of USD loading behavior including variant selection,
 * payload loading, and feature toggles.
 */
struct UsdLoadOptions {
    // ========================================================================
    // Variant Selection
    // ========================================================================

    /**
     * @brief Variant selections to apply before loading
     *
     * Map of prim path -> (variant set name -> variant name)
     * Example: { "/Root/Model": { "LOD": "high", "material": "metal" } }
     *
     * The empty prim path is a wildcard: that set is selected on every prim
     * that owns it, which is how a config can say `lod=low` without knowing
     * where in the hierarchy the sets are.
     *
     * If empty, default variants are used.
     */
    using VariantSelections = std::unordered_map<String, std::unordered_map<String, String>>;
    VariantSelections variantSelections;

    // ========================================================================
    // Payload Control
    // ========================================================================

    /**
     * @brief Initial payload loading policy
     * - LoadAll: Load all payloads immediately (default)
     * - LoadNone: Don't load any payloads (faster initial load)
     */
    enum class PayloadPolicy { LoadAll, LoadNone };
    PayloadPolicy payloadPolicy = PayloadPolicy::LoadAll;

    // ========================================================================
    // Material Options
    // ========================================================================

    /**
     * @brief Enable MaterialX material parsing
     * When true, parses MaterialX standard_surface and other MaterialX shaders.
     * When false, only UsdPreviewSurface is parsed.
     */
    bool enableMaterialX = true;

    /**
     * @brief Load textures from shader connections
     * When false, only scalar material values are loaded (faster).
     */
    bool loadTextures = true;

    // ========================================================================
    // Geometry Options
    // ========================================================================

    /**
     * @brief Enable GeomSubsets for multi-material meshes
     * When true, meshes with GeomSubsets are split into multiple primitives.
     */
    bool enableGeomSubsets = true;

    /**
     * @brief Enable PointInstancer expansion
     * When true, PointInstancer prims are expanded to individual instances.
     * When false, PointInstancer prims are skipped.
     */
    bool enablePointInstancer = true;

    /**
     * @brief Fold the stage's own upAxis and metersPerUnit into the node transforms
     *
     * A Z-up stage is rotated to Y-up and an authored metersPerUnit is applied as
     * a uniform scale, so a scene assembled from files in different conventions
     * lines up. On by default; `scene.world_units_to_meters` is not an
     * alternative, as that scales lighting, camera and thermal inputs rather than
     * geometry and is one scalar for a whole config.
     */
    bool applyStageMetrics = true;

    /**
     * @brief Time code for attribute queries
     * Default is UsdTimeCode::Default() (no animation).
     */
    double timeCode = 0.0;  // 0.0 = default time
    bool useDefaultTime = true;  // When true, ignores timeCode and uses Default()

    // ========================================================================
    // Factory Methods
    // ========================================================================

    /**
     * @brief Create default options (load everything)
     */
    static UsdLoadOptions Default() { return UsdLoadOptions{}; }

    /**
     * @brief Create fast loading options (minimal features)
     */
    static UsdLoadOptions Fast() {
        UsdLoadOptions opts;
        opts.payloadPolicy = PayloadPolicy::LoadNone;
        opts.loadTextures = false;
        opts.enableGeomSubsets = false;
        opts.enablePointInstancer = false;
        return opts;
    }
};

/**
 * @brief Parse a variant selection spec into UsdLoadOptions::variantSelections
 *
 * The syntax a config's `variant` key uses for a USD file:
 *
 *     /Root/Car{color=red}          one set on one prim
 *     lod=low                       that set on every prim that owns it
 *     /A{x=1},/B{y=2},lod=low       comma-separated, any mix
 *
 * A malformed entry is an error rather than a warning. An unknown *variant
 * name* is a typo in the data and glTF warns about it, but the syntax is the
 * contract between the config and the loader: if it does not parse, nobody
 * knows what was asked for.
 *
 * pxr-free, so it is available and testable in a build without OpenUSD.
 */
Result<UsdLoadOptions::VariantSelections, String> ParseUsdVariantSpec(StringView spec);

/**
 * @class UsdLoader
 * @brief Static utility class for loading OpenUSD 3D scenes into Quantiloom scenes
 *
 * Converts OpenUSD files into Quantiloom Scene format.
 * Handles all USD data extraction, format conversion, and resource loading.
 *
 * Loading workflow:
 * 1. Open USD stage using OpenUSD (automatic composition)
 * 2. Apply variant selections from options
 * 3. Traverse stage to collect materials (UsdPreviewSurface + MaterialX)
 * 4. Extract and convert meshes -> Quantiloom Mesh/GeometryPrimitive
 * 5. Handle GeomSubsets for multi-material meshes
 * 6. Load textures from shader connections -> Quantiloom Texture
 * 7. Expand PointInstancer to scene nodes
 * 8. Compute world transforms for each node
 * 9. Parse Quantiloom spectral extensions (custom attributes)
 * 10. Return complete Scene object
 *
 * Usage example:
 * @code
 * // Load USD scene with default options
 * auto result = UsdLoader::LoadFromFile("scenes/my_scene.usda");
 *
 * // Load with custom variant selection
 * UsdLoadOptions opts;
 * opts.variantSelections["/Root/Car"] = { {"color", "red"}, {"LOD", "high"} };
 * auto result = UsdLoader::LoadFromFile("scenes/car.usd", opts);
 *
 * if (!result.has_value()) {
 *     QL_LOG_ERROR("Failed to load USD: {}", result.error());
 *     return;
 * }
 *
 * Scene scene = std::move(result.value());
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
class UsdLoader {
public:
    // ========================================================================
    // Loading Functions
    // ========================================================================

    /**
     * @brief Check if OpenUSD support is available
     * @return true if OpenUSD was compiled in, false otherwise
     */
    static bool IsAvailable();

    /**
     * @brief Load USD file with default options
     * @param path Path to USD file
     * @return Scene with meshes, materials, textures, and nodes on success
     * @return Error message string on failure
     */
    static Result<Scene, String> LoadFromFile(const String& path);

    /**
     * @brief Load USD file with custom options
     * @param path Path to USD file
     * @param options Loading configuration (variants, payloads, features)
     * @return Scene with meshes, materials, textures, and nodes on success
     * @return Error message string on failure
     */
    static Result<Scene, String> LoadFromFile(const String& path, const UsdLoadOptions& options);

    // ========================================================================
    // Utility Functions
    // ========================================================================

    /**
     * @brief List available variant sets and their options for a prim
     * @param path Path to USD file
     * @param primPath Path to prim (e.g., "/Root/Model")
     * @return Map of variant set name -> available variant names
     */
    static Result<std::unordered_map<String, std::vector<String>>, String>
        ListVariants(const String& path, const String& primPath);

    /**
     * @brief List all prims with variant sets in a USD file
     * @param path Path to USD file
     * @return List of prim paths that have variant sets
     */
    static Result<std::vector<String>, String> ListPrimsWithVariants(const String& path);

    // ========================================================================
    // Texture Utility Functions
    // ========================================================================

    /**
     * @brief Resolve an asset path against the USD file and decode it to RGBA8
     * @note The decode itself is usd::DecodeTextureFile, which UsdTextureBank
     *       also uses; this adds the path resolution a loose asset path needs.
     */
    static Texture ParseTexture(const void* stage, const String& assetPath,
                                 const String& usdFilePath);

private:
    // ========================================================================
    // Internal Parsing Functions
    // ========================================================================

    /**
     * @brief Parse UsdGeomMesh to Quantiloom Mesh
     * @note Polygons are triangulated during parsing
     * @note Each mesh maps to one Quantiloom Mesh with one or more GeometryPrimitives
     * @note GeomSubsets are parsed to create separate primitives per material
     */
    static Mesh ParseMesh(const void* stage, const void* geomMesh,
                          const std::unordered_map<String, int>& materialPathMap,
                          const String& usdFilePath,
                          const UsdLoadOptions& options);

    /**
     * @brief Flatten USD Xform hierarchy to world-space nodes
     * @note Resolves References to external USD files
     * @note Computes accumulated transforms for each node
     */
    static std::vector<SceneNode> FlattenXformHierarchy(const void* stage);

    /**
     * @brief Parse PointInstancer to scene nodes
     * @note Expands instances with per-instance transforms
     */
    static void ParsePointInstancer(const void* stage, const void* instancer,
                                     Scene& scene,
                                     const std::unordered_map<String, int>& materialPathMap,
                                     const String& usdFilePath,
                                     const UsdLoadOptions& options);


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

    /**
     * @brief Triangulate polygon mesh with face mapping
     * @param faceVertexCounts Number of vertices per face
     * @param faceVertexIndices Vertex indices for all faces
     * @param outTriangleToFace Output: maps each output triangle to source face index
     * @return Triangulated index buffer
     */
    static std::vector<u32> TriangulatePolygonsWithFaceMap(
        const std::vector<i32>& faceVertexCounts,
        const std::vector<i32>& faceVertexIndices,
        std::vector<u32>& outTriangleToFace);
};

} // namespace quantiloom
