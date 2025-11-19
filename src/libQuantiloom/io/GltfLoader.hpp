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
// Responsibilities:
// - Parse glTF 2.0 files using tinygltf library
// - Convert glTF meshes to Quantiloom Mesh/GeometryPrimitive format
// - Convert glTF PBR materials to Quantiloom Material format
// - Load embedded textures (PNG/JPEG) to Quantiloom Texture format
// - Build Scene hierarchy with nodes and transforms
//
// Supported Features:
// - Meshes with multiple primitives
// - PBR metallic-roughness materials
// - Embedded textures (PNG/JPEG via base64 or binary glb)
// - Scene graph transforms (flattened to world space)
// - Normal maps, emissive maps
//
// Not Supported (M2):
// - Animations
// - Skinning/morphing
// - Cameras (use config file instead)
// - Lights (use config file instead)
// - Extensions (KHR_materials_*, etc.)
//
// Usage:
//   auto result = GltfLoader::LoadFromFile("model.gltf");
//   if (!result.has_value()) {
//       QL_LOG_ERROR("Failed to load glTF: {}", result.error());
//   }
//   Scene scene = result.value();
// ============================================================================

namespace quantiloom {

class QL_API GltfLoader {
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
                                   const std::vector<Texture>& textures);

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
