/**
 * @file GltfLoader.hpp
 * @brief glTF 2.0 model loader for 3D scenes with PBR materials
 *
 * Provides GltfLoader class for loading glTF 2.0 files:
 * - Parses .gltf (JSON + external resources) and .glb (binary) formats
 * - Converts glTF meshes to Quantiloom GeometryPrimitive format
 * - Converts glTF PBR materials to Quantiloom Material format
 * - Loads embedded/external textures (PNG/JPEG) to Quantiloom Texture format
 * - Flattens scene graph to world-space SceneNode transforms
 *
 * Supported glTF 2.0 features:
 * - Meshes with multiple primitives (triangle lists)
 * - PBR metallic-roughness materials
 * - Textures: base color, metallic-roughness, normal maps, emissive
 * - Embedded textures (base64 DataURI in .gltf files)
 * - External textures (PNG/JPEG files via URI references)
 * - Binary embedded textures (.glb binary chunks)
 * - Scene graph hierarchies (flattened to world-space transforms)
 * - Transform nodes (translation, rotation, scale, matrix)
 *
 * Supported extensions:
 * - KHR_materials_transmission, KHR_materials_ior, KHR_materials_volume
 * - KHR_materials_dispersion (and the QUANTILOOM_materials_dispersion original)
 * - KHR_materials_sheen (factors and textures)
 * - KHR_texture_transform (per texture slot)
 * - QUANTILOOM_material_ir (measured IR curves, temperature field)
 *
 * NOT supported:
 * - Animations/skinning/morph targets
 * - Cameras/lights (use Quantiloom config instead)
 * - A second UV set: only TEXCOORD_0 is read, so a textureInfo naming
 *   texCoord 1 is warned about and sampled against set 0
 * - Other KHR_materials_* extensions (specular, clearcoat, variants, ...),
 *   which are ignored silently
 *
 * Uses tinygltf library for glTF parsing.
 *
 * @note Returns Result<Scene, String> for explicit error handling
 * @note Scene graph flattened to world space (no hierarchy preserved)
 * @note All mesh primitives become separate GeometryPrimitive instances
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

// ============================================================================
// GltfLoader - Loads glTF 2.0 models (.gltf, .glb)
// ============================================================================

namespace quantiloom {

/**
 * @class GltfLoader
 * @brief Static utility class for loading glTF 2.0 3D models into Quantiloom scenes
 *
 * Converts glTF 2.0 files (JSON or binary) into Quantiloom Scene format.
 * Handles all glTF data extraction, format conversion, and resource loading.
 *
 * Loading workflow:
 * 1. Parse glTF file using tinygltf library
 * 2. Extract and convert meshes → Quantiloom Mesh/GeometryPrimitive
 * 3. Extract and convert materials → Quantiloom Material (PBR)
 * 4. Load textures → Quantiloom Texture (decode PNG/JPEG)
 * 5. Flatten scene graph → SceneNode list with world transforms
 * 6. Return complete Scene object
 *
 * Usage example:
 * @code
 * // Load glTF model
 * auto result = GltfLoader::LoadFromFile("models/DamagedHelmet.gltf");
 * if (!result.has_value()) {
 *     QL_LOG_ERROR("Failed to load glTF: {}", result.error());
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
 * auto result = GltfLoader::LoadFromFile("model.gltf");
 * if (!result.has_value()) {
 *     // Error message in result.error()
 *     QL_LOG_ERROR("Load failed: {}", result.error());
 * }
 * @endcode
 *
 * @note Supports both .gltf (JSON + external files) and .glb (binary single-file)
 * @note Scene graph flattened: parent transforms accumulated into world space
 * @note Each glTF primitive becomes separate GeometryPrimitive (enables per-primitive materials)
 * @note Texture data decoded to f32 in CPU memory (not uploaded to GPU yet)
 *
 * @see Scene for output scene structure
 * @see Material for PBR material mapping
 * @see Texture for texture data format
 */
class GltfLoader {
public:
    // ========================================================================
    // Loading Functions
    // ========================================================================

    // Load glTF file (.gltf or .glb)
    // Returns Scene with meshes, materials, textures, and nodes
    static Result<Scene, String> LoadFromFile(const String& path);

private:
    // ========================================================================
    // Internal Parsing Functions
    // ========================================================================

    // Parse glTF mesh to Quantiloom Mesh
    // Each glTF primitive becomes a GeometryPrimitive
    static Mesh ParseMesh(const void* gltfModel, int meshIndex,
                          const std::vector<Material>& materials);

    // Parse glTF material to Quantiloom Material
    // Converts PBR metallic-roughness to our format
    static Material ParseMaterial(const void* gltfModel, int materialIndex,
                                   const std::vector<Texture>& textures,
                                   const String& gltfFilePath);

    // Parse glTF texture to Quantiloom Texture
    // Loads embedded image data (PNG/JPEG)
    static Texture ParseTexture(const void* gltfModel, int textureIndex);

    // Flatten glTF scene graph to world-space nodes
    // Computes accumulated transforms for each node
    static std::vector<SceneNode> FlattenSceneGraph(const void* gltfModel);

    // ========================================================================
    // Accessor Utilities
    // ========================================================================

    // Read vertex attribute from glTF accessor
    // Returns std::vector<T> where T is glm::vec2, glm::vec3, etc.
    template<typename T>
    static std::vector<T> ReadAccessor(const void* gltfModel, int accessorIndex);

    // Read index buffer from glTF accessor
    // Handles u8, u16, u32 indices
    static std::vector<u32> ReadIndices(const void* gltfModel, int accessorIndex);
};

} // namespace quantiloom
