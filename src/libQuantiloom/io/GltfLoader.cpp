#include "GltfLoader.hpp"
#include "core/Log.hpp"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE  // We don't need write functionality
#include <tiny_gltf.h>

#define GLM_ENABLE_EXPERIMENTAL  // Required for GLM experimental extensions
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <filesystem>
#include <stdexcept>

namespace quantiloom {

// ============================================================================
// Helper: Convert tinygltf types to GLM types
// ============================================================================

static glm::mat4 MatrixFromGltf(const std::vector<double>& mat) {
    // glTF matrices are column-major (same as GLM)
    glm::mat4 result;
    if (mat.size() == 16) {
        for (int i = 0; i < 16; ++i) {
            result[i / 4][i % 4] = static_cast<float>(mat[i]);
        }
    } else {
        result = glm::mat4(1.0f);  // Identity fallback
    }
    return result;
}

static glm::mat4 TRSToMatrix(const std::vector<double>& translation,
                              const std::vector<double>& rotation,
                              const std::vector<double>& scale) {
    // Translation
    glm::vec3 t = (translation.size() == 3)
        ? glm::vec3(translation[0], translation[1], translation[2])
        : glm::vec3(0.0f);

    // Rotation (quaternion: x, y, z, w)
    glm::quat r = (rotation.size() == 4)
        ? glm::quat(static_cast<float>(rotation[3]),  // w
                    static_cast<float>(rotation[0]),  // x
                    static_cast<float>(rotation[1]),  // y
                    static_cast<float>(rotation[2]))  // z
        : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);  // Identity

    // Scale
    glm::vec3 s = (scale.size() == 3)
        ? glm::vec3(scale[0], scale[1], scale[2])
        : glm::vec3(1.0f);

    // Compose: T * R * S
    glm::mat4 matT = glm::translate(glm::mat4(1.0f), t);
    glm::mat4 matR = glm::mat4_cast(r);
    glm::mat4 matS = glm::scale(glm::mat4(1.0f), s);

    return matT * matR * matS;
}

// ============================================================================
// ParseNodeTransform (helper function for scene graph flattening)
// ============================================================================

static glm::mat4 ParseNodeTransform(const tinygltf::Node& node) {
    // glTF supports two forms: matrix or TRS (translation/rotation/scale)
    if (!node.matrix.empty()) {
        return MatrixFromGltf(node.matrix);
    } else {
        return TRSToMatrix(node.translation, node.rotation, node.scale);
    }
}

// ============================================================================
// ReadAccessor (Template Specializations)
// ============================================================================

template<>
std::vector<glm::vec3> GltfLoader::ReadAccessor<glm::vec3>(const void* gltfModelPtr, int accessorIndex) {
    const auto& model = *static_cast<const tinygltf::Model*>(gltfModelPtr);

    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size())) {
        return {};
    }

    const auto& accessor = model.accessors[accessorIndex];
    const auto& bufferView = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[bufferView.buffer];

    const u8* dataPtr = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
    size_t stride = bufferView.byteStride ? bufferView.byteStride : sizeof(float) * 3;

    std::vector<glm::vec3> result;
    result.reserve(accessor.count);

    for (size_t i = 0; i < accessor.count; ++i) {
        const float* floatPtr = reinterpret_cast<const float*>(dataPtr + i * stride);
        result.emplace_back(floatPtr[0], floatPtr[1], floatPtr[2]);
    }

    return result;
}

template<>
std::vector<glm::vec2> GltfLoader::ReadAccessor<glm::vec2>(const void* gltfModelPtr, int accessorIndex) {
    const auto& model = *static_cast<const tinygltf::Model*>(gltfModelPtr);

    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size())) {
        return {};
    }

    const auto& accessor = model.accessors[accessorIndex];
    const auto& bufferView = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[bufferView.buffer];

    const u8* dataPtr = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
    size_t stride = bufferView.byteStride ? bufferView.byteStride : sizeof(float) * 2;

    std::vector<glm::vec2> result;
    result.reserve(accessor.count);

    for (size_t i = 0; i < accessor.count; ++i) {
        const float* floatPtr = reinterpret_cast<const float*>(dataPtr + i * stride);
        result.emplace_back(floatPtr[0], floatPtr[1]);
    }

    return result;
}

// ============================================================================
// ReadIndices
// ============================================================================

std::vector<u32> GltfLoader::ReadIndices(const void* gltfModelPtr, int accessorIndex) {
    const auto& model = *static_cast<const tinygltf::Model*>(gltfModelPtr);

    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size())) {
        return {};
    }

    const auto& accessor = model.accessors[accessorIndex];
    const auto& bufferView = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[bufferView.buffer];

    const u8* dataPtr = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;

    std::vector<u32> result;
    result.reserve(accessor.count);

    // glTF indices can be u8, u16, or u32
    if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
        for (size_t i = 0; i < accessor.count; ++i) {
            result.push_back(static_cast<u32>(dataPtr[i]));
        }
    } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
        const u16* indices = reinterpret_cast<const u16*>(dataPtr);
        for (size_t i = 0; i < accessor.count; ++i) {
            result.push_back(static_cast<u32>(indices[i]));
        }
    } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
        const u32* indices = reinterpret_cast<const u32*>(dataPtr);
        for (size_t i = 0; i < accessor.count; ++i) {
            result.push_back(indices[i]);
        }
    }

    return result;
}

// ============================================================================
// ParseTexture
// ============================================================================

Texture GltfLoader::ParseTexture(const void* gltfModelPtr, int textureIndex) {
    const auto& model = *static_cast<const tinygltf::Model*>(gltfModelPtr);

    if (textureIndex < 0 || textureIndex >= static_cast<int>(model.textures.size())) {
        QL_LOG_ERROR("Invalid texture index: {}", textureIndex);
        return Texture{};
    }

    const auto& gltfTexture = model.textures[textureIndex];
    const auto& gltfImage = model.images[gltfTexture.source];

    Texture tex;
    tex.name = gltfImage.name.empty() ? ("Texture_" + std::to_string(textureIndex)) : gltfImage.name;
    tex.width = static_cast<u32>(gltfImage.width);
    tex.height = static_cast<u32>(gltfImage.height);
    tex.channels = static_cast<u32>(gltfImage.component);

    // Copy pixel data (tinygltf already decoded PNG/JPEG)
    tex.pixels = gltfImage.image;

    // Parse sampler parameters
    if (gltfTexture.sampler >= 0 && gltfTexture.sampler < static_cast<int>(model.samplers.size())) {
        const auto& gltfSampler = model.samplers[gltfTexture.sampler];

        // Min/Mag filter
        if (gltfSampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST ||
            gltfSampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST ||
            gltfSampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR) {
            tex.sampler.minFilter = TextureSampler::Filter::Nearest;
        } else {
            tex.sampler.minFilter = TextureSampler::Filter::Linear;
        }

        if (gltfSampler.magFilter == TINYGLTF_TEXTURE_FILTER_NEAREST) {
            tex.sampler.magFilter = TextureSampler::Filter::Nearest;
        } else {
            tex.sampler.magFilter = TextureSampler::Filter::Linear;
        }

        // Wrap mode S
        switch (gltfSampler.wrapS) {
            case TINYGLTF_TEXTURE_WRAP_REPEAT:
                tex.sampler.wrapS = TextureSampler::WrapMode::Repeat;
                break;
            case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
                tex.sampler.wrapS = TextureSampler::WrapMode::ClampToEdge;
                break;
            case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
                tex.sampler.wrapS = TextureSampler::WrapMode::MirroredRepeat;
                break;
            default:
                tex.sampler.wrapS = TextureSampler::WrapMode::Repeat;
        }

        // Wrap mode T
        switch (gltfSampler.wrapT) {
            case TINYGLTF_TEXTURE_WRAP_REPEAT:
                tex.sampler.wrapT = TextureSampler::WrapMode::Repeat;
                break;
            case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
                tex.sampler.wrapT = TextureSampler::WrapMode::ClampToEdge;
                break;
            case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
                tex.sampler.wrapT = TextureSampler::WrapMode::MirroredRepeat;
                break;
            default:
                tex.sampler.wrapT = TextureSampler::WrapMode::Repeat;
        }
    }

    QL_LOG_INFO("  Loaded texture '{}' ({}x{}, {} channels)",
                tex.name, tex.width, tex.height, tex.channels);

    return tex;
}

// ============================================================================
// ParseMaterial
// ============================================================================

Material GltfLoader::ParseMaterial(const void* gltfModelPtr, int materialIndex,
                                    const std::vector<Texture>& textures) {
    const auto& model = *static_cast<const tinygltf::Model*>(gltfModelPtr);

    Material mat;

    if (materialIndex < 0 || materialIndex >= static_cast<int>(model.materials.size())) {
        // Default material
        mat.name = "DefaultMaterial";
        mat.baseColorFactor = glm::vec4(1.0f);
        mat.metallicFactor = 0.0f;
        mat.roughnessFactor = 1.0f;
        mat.ComputeSpectralAlbedo();
        return mat;
    }

    const auto& gltfMaterial = model.materials[materialIndex];
    mat.name = gltfMaterial.name.empty() ? ("Material_" + std::to_string(materialIndex)) : gltfMaterial.name;

    // PBR metallic-roughness
    const auto& pbr = gltfMaterial.pbrMetallicRoughness;

    // Base color
    mat.baseColorFactor = glm::vec4(
        static_cast<float>(pbr.baseColorFactor[0]),
        static_cast<float>(pbr.baseColorFactor[1]),
        static_cast<float>(pbr.baseColorFactor[2]),
        static_cast<float>(pbr.baseColorFactor[3])
    );

    if (pbr.baseColorTexture.index >= 0) {
        mat.baseColorTextureIndex = pbr.baseColorTexture.index;
    }

    // Metallic-roughness
    mat.metallicFactor = static_cast<float>(pbr.metallicFactor);
    mat.roughnessFactor = static_cast<float>(pbr.roughnessFactor);

    if (pbr.metallicRoughnessTexture.index >= 0) {
        mat.metallicRoughnessTextureIndex = pbr.metallicRoughnessTexture.index;
    }

    // Normal map
    if (gltfMaterial.normalTexture.index >= 0) {
        mat.normalTextureIndex = gltfMaterial.normalTexture.index;
        mat.normalScale = static_cast<float>(gltfMaterial.normalTexture.scale);
    }

    // Emissive
    mat.emissiveFactor = glm::vec3(
        static_cast<float>(gltfMaterial.emissiveFactor[0]),
        static_cast<float>(gltfMaterial.emissiveFactor[1]),
        static_cast<float>(gltfMaterial.emissiveFactor[2])
    );

    if (gltfMaterial.emissiveTexture.index >= 0) {
        mat.emissiveTextureIndex = gltfMaterial.emissiveTexture.index;
    }

    // Alpha mode
    if (gltfMaterial.alphaMode == "OPAQUE") {
        mat.alphaMode = Material::AlphaMode::Opaque;
    } else if (gltfMaterial.alphaMode == "MASK") {
        mat.alphaMode = Material::AlphaMode::Mask;
        mat.alphaCutoff = static_cast<float>(gltfMaterial.alphaCutoff);
    } else if (gltfMaterial.alphaMode == "BLEND") {
        mat.alphaMode = Material::AlphaMode::Blend;
    }

    // Compute spectral albedo for M1 compatibility
    mat.ComputeSpectralAlbedo();

    QL_LOG_INFO("  Loaded material '{}' (metallic={:.2f}, roughness={:.2f})",
                mat.name, mat.metallicFactor, mat.roughnessFactor);

    return mat;
}

// ============================================================================
// ParseMesh
// ============================================================================

Mesh GltfLoader::ParseMesh(const void* gltfModelPtr, int meshIndex,
                            const std::vector<Material>& materials) {
    const auto& model = *static_cast<const tinygltf::Model*>(gltfModelPtr);

    if (meshIndex < 0 || meshIndex >= static_cast<int>(model.meshes.size())) {
        QL_LOG_ERROR("Invalid mesh index: {}", meshIndex);
        return Mesh{};
    }

    const auto& gltfMesh = model.meshes[meshIndex];

    Mesh mesh;
    mesh.name = gltfMesh.name.empty() ? ("Mesh_" + std::to_string(meshIndex)) : gltfMesh.name;

    // Parse each primitive
    for (size_t primIdx = 0; primIdx < gltfMesh.primitives.size(); ++primIdx) {
        const auto& gltfPrimitive = gltfMesh.primitives[primIdx];

        GeometryPrimitive primitive;

        // Material ID
        primitive.materialId = (gltfPrimitive.material >= 0) ? gltfPrimitive.material : 0;

        // Positions (required)
        auto posIt = gltfPrimitive.attributes.find("POSITION");
        if (posIt != gltfPrimitive.attributes.end()) {
            primitive.positions = ReadAccessor<glm::vec3>(gltfModelPtr, posIt->second);
        } else {
            QL_LOG_ERROR("Primitive {} in mesh '{}' has no POSITION attribute", primIdx, mesh.name);
            continue;
        }

        // Normals (optional)
        auto normIt = gltfPrimitive.attributes.find("NORMAL");
        if (normIt != gltfPrimitive.attributes.end()) {
            primitive.normals = ReadAccessor<glm::vec3>(gltfModelPtr, normIt->second);
        }

        // UVs (optional, use TEXCOORD_0)
        auto uvIt = gltfPrimitive.attributes.find("TEXCOORD_0");
        if (uvIt != gltfPrimitive.attributes.end()) {
            primitive.uvs = ReadAccessor<glm::vec2>(gltfModelPtr, uvIt->second);
        }

        // Indices (required for indexed geometry)
        if (gltfPrimitive.indices >= 0) {
            primitive.indices = ReadIndices(gltfModelPtr, gltfPrimitive.indices);
        } else {
            // Non-indexed geometry: generate sequential indices
            primitive.indices.resize(primitive.positions.size());
            for (size_t i = 0; i < primitive.positions.size(); ++i) {
                primitive.indices[i] = static_cast<u32>(i);
            }
        }

        QL_LOG_INFO("    Primitive {}: {} vertices, {} triangles, material {}",
                    primIdx, primitive.GetVertexCount(), primitive.GetTriangleCount(), primitive.materialId);

        mesh.primitives.push_back(std::move(primitive));
    }

    QL_LOG_INFO("  Loaded mesh '{}' with {} primitive(s)", mesh.name, mesh.primitives.size());

    return mesh;
}

// ============================================================================
// FlattenSceneGraph
// ============================================================================

static void TraverseNode(const tinygltf::Model& model, int nodeIndex,
                          const glm::mat4& parentTransform,
                          std::vector<SceneNode>& outNodes,
                          const std::vector<Mesh>& meshes) {
    if (nodeIndex < 0 || nodeIndex >= static_cast<int>(model.nodes.size())) {
        return;
    }

    const auto& gltfNode = model.nodes[nodeIndex];

    // Compute local transform
    glm::mat4 localTransform = ParseNodeTransform(gltfNode);

    // Compute world transform
    glm::mat4 worldTransform = parentTransform * localTransform;

    // If this node has a mesh, add a SceneNode
    if (gltfNode.mesh >= 0) {
        SceneNode sceneNode;
        sceneNode.meshIndex = static_cast<u32>(gltfNode.mesh);
        sceneNode.transform = worldTransform;
        sceneNode.name = gltfNode.name.empty() ? ("Node_" + std::to_string(nodeIndex)) : gltfNode.name;
        outNodes.push_back(sceneNode);
    }

    // Recursively traverse children
    for (int childIndex : gltfNode.children) {
        TraverseNode(model, childIndex, worldTransform, outNodes, meshes);
    }
}

std::vector<SceneNode> GltfLoader::FlattenSceneGraph(const void* gltfModelPtr) {
    const auto& model = *static_cast<const tinygltf::Model*>(gltfModelPtr);

    std::vector<SceneNode> nodes;

    // glTF can have multiple scenes, use the default scene
    int sceneIndex = (model.defaultScene >= 0) ? model.defaultScene : 0;
    if (sceneIndex < 0 || sceneIndex >= static_cast<int>(model.scenes.size())) {
        QL_LOG_WARN("No valid scene found in glTF file");
        return nodes;
    }

    const auto& scene = model.scenes[sceneIndex];

    // Empty meshes vector (not used in traversal, kept for signature)
    std::vector<Mesh> meshes;

    // Traverse all root nodes
    for (int rootNodeIndex : scene.nodes) {
        TraverseNode(model, rootNodeIndex, glm::mat4(1.0f), nodes, meshes);
    }

    QL_LOG_INFO("  Flattened scene graph: {} node(s)", nodes.size());

    return nodes;
}

// ============================================================================
// LoadFromFile
// ============================================================================

Result<Scene, String> GltfLoader::LoadFromFile(const String& path) {
    QL_LOG_INFO("Loading glTF model from: {}", path);

    if (!std::filesystem::exists(path)) {
        return Result<Scene, String>::Err("File not found: " + path);
    }

    // Determine file type (.gltf or .glb)
    std::filesystem::path filePath(path);
    String ext = filePath.extension().string();
    bool isBinary = (ext == ".glb" || ext == ".GLB");

    // Load glTF using tinygltf
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    String error, warning;

    bool success = isBinary
        ? loader.LoadBinaryFromFile(&model, &error, &warning, path)
        : loader.LoadASCIIFromFile(&model, &error, &warning, path);

    if (!warning.empty()) {
        QL_LOG_WARN("glTF warning: {}", warning);
    }

    if (!success || !error.empty()) {
        return Result<Scene, String>::Err("Failed to load glTF: " + error);
    }

    QL_LOG_INFO("  glTF loaded: {} meshes, {} materials, {} textures",
                model.meshes.size(), model.materials.size(), model.textures.size());

    // Build Scene
    Scene scene;
    scene.name = filePath.stem().string();

    // Load textures
    scene.textures.reserve(model.textures.size());
    for (size_t i = 0; i < model.textures.size(); ++i) {
        scene.textures.push_back(ParseTexture(&model, static_cast<int>(i)));
    }

    // Load materials
    scene.materials.reserve(model.materials.size());
    for (size_t i = 0; i < model.materials.size(); ++i) {
        scene.materials.push_back(ParseMaterial(&model, static_cast<int>(i), scene.textures));
    }

    // Ensure at least one default material exists
    if (scene.materials.empty()) {
        scene.materials.push_back(Material::CreateLambertian(glm::vec3(0.8f), "DefaultMaterial"));
    }

    // Load meshes
    scene.meshes.reserve(model.meshes.size());
    for (size_t i = 0; i < model.meshes.size(); ++i) {
        scene.meshes.push_back(ParseMesh(&model, static_cast<int>(i), scene.materials));
    }

    // Flatten scene graph to nodes
    scene.nodes = FlattenSceneGraph(&model);

    QL_LOG_INFO("  Scene '{}' loaded: {} meshes, {} nodes, {} materials, {} textures",
                scene.name, scene.meshes.size(), scene.nodes.size(),
                scene.materials.size(), scene.textures.size());

    return std::move(scene);
}

} // namespace quantiloom
