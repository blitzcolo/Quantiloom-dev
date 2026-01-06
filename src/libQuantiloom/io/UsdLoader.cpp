/**
 * @file UsdLoader.cpp
 * @brief OpenUSD scene loader implementation using TinyUSDZ
 *
 * Implementation of UsdLoader class for loading OpenUSD files.
 * Uses TinyUSDZ library for USD parsing.
 *
 * @author wtflmao
 */

#include "UsdLoader.hpp"
#include "SpectralIO.hpp"
#include "core/Log.hpp"

// TinyUSDZ headers (linking against tinyusdz_static library)
#include <tinyusdz.hh>
#include <io-util.hh>
#include <pprinter.hh>
#include <prim-types.hh>
#include <usdGeom.hh>
#include <usdShade.hh>
#include <tydra/scene-access.hh>
#include <tydra/render-data.hh>

// Use ImageIO for texture loading (avoids stb_image symbol conflicts with tinygltf)
#include "ImageIO.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <filesystem>
#include <unordered_map>

namespace quantiloom {

// ============================================================================
// Helper: Convert TinyUSDZ types to GLM types
// ============================================================================

[[maybe_unused]]
static glm::mat4 MatrixFromUsd(const tinyusdz::value::matrix4d& mat) {
    glm::mat4 result;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result[i][j] = static_cast<float>(mat.m[i][j]);
        }
    }
    return result;
}

// ============================================================================
// Helper: Extract vec3 array from VertexAttribute
// ============================================================================

static std::vector<glm::vec3> ExtractVec3FromVertexAttribute(
    const tinyusdz::tydra::VertexAttribute& attr) {

    std::vector<glm::vec3> result;

    if (attr.empty()) {
        return result;
    }

    // Check format is Vec3
    if (attr.format != tinyusdz::tydra::VertexAttributeFormat::Vec3) {
        return result;
    }

    size_t count = attr.vertex_count();
    result.reserve(count);

    const float* floatData = reinterpret_cast<const float*>(attr.data.data());

    for (size_t i = 0; i < count; ++i) {
        result.emplace_back(floatData[i * 3 + 0],
                           floatData[i * 3 + 1],
                           floatData[i * 3 + 2]);
    }

    return result;
}

// ============================================================================
// Helper: Extract vec2 array from VertexAttribute
// ============================================================================

static std::vector<glm::vec2> ExtractVec2FromVertexAttribute(
    const tinyusdz::tydra::VertexAttribute& attr) {

    std::vector<glm::vec2> result;

    if (attr.empty()) {
        return result;
    }

    // Check format is Vec2
    if (attr.format != tinyusdz::tydra::VertexAttributeFormat::Vec2) {
        return result;
    }

    size_t count = attr.vertex_count();
    result.reserve(count);

    const float* floatData = reinterpret_cast<const float*>(attr.data.data());

    for (size_t i = 0; i < count; ++i) {
        result.emplace_back(floatData[i * 2 + 0],
                           floatData[i * 2 + 1]);
    }

    return result;
}

// ============================================================================
// TriangulatePolygons - Convert polygon faces to triangles
// ============================================================================

std::vector<u32> UsdLoader::TriangulatePolygons(
    const std::vector<i32>& faceVertexCounts,
    const std::vector<i32>& faceVertexIndices) {

    std::vector<u32> triangleIndices;

    size_t indexOffset = 0;
    for (const i32 vertexCount : faceVertexCounts) {
        if (vertexCount < 3) {
            indexOffset += vertexCount;
            continue;
        }

        // Fan triangulation for convex polygons
        for (i32 i = 1; i < vertexCount - 1; ++i) {
            triangleIndices.push_back(static_cast<u32>(faceVertexIndices[indexOffset]));
            triangleIndices.push_back(static_cast<u32>(faceVertexIndices[indexOffset + i]));
            triangleIndices.push_back(static_cast<u32>(faceVertexIndices[indexOffset + i + 1]));
        }

        indexOffset += vertexCount;
    }

    return triangleIndices;
}

// ============================================================================
// ParseTexture - Load texture from USD asset path using ImageIO
// ============================================================================

Texture UsdLoader::ParseTexture(const void* /* stagePtr */, const String& assetPath,
                                  const String& usdFilePath) {
    Texture tex;

    // Resolve asset path relative to USD file
    std::filesystem::path usdDir = std::filesystem::path(usdFilePath).parent_path();
    std::filesystem::path fullPath = usdDir / assetPath;

    if (!std::filesystem::exists(fullPath)) {
        QL_LOG_ERROR("Texture file not found: {}", fullPath.string());
        return tex;
    }

    // Determine file type
    String ext = fullPath.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".exr") {
        // Load EXR via ImageIO
        auto imageResult = ImageIO::ReadEXR(fullPath.string());
        if (!imageResult.has_value()) {
            QL_LOG_ERROR("Failed to load EXR texture '{}'", assetPath);
            return tex;
        }

        const Image& img = imageResult.value();

        tex.name = fullPath.filename().string();
        tex.width = img.width;
        tex.height = img.height;
        tex.channels = img.channels;
        tex.sourceUri = assetPath;

        // Convert float image data to u8 pixels
        size_t pixelCount = static_cast<size_t>(img.width) * img.height * img.channels;
        tex.pixels.resize(pixelCount);
        for (size_t i = 0; i < pixelCount; ++i) {
            float val = std::clamp(img.data[i], 0.0f, 1.0f);
            tex.pixels[i] = static_cast<u8>(val * 255.0f + 0.5f);
        }
    } else {
        // PNG/JPEG textures are loaded by tydra's texture loader
        // This function is currently unused - tydra handles textures internally
        // TODO: Add PNG/JPEG support if needed for manual texture loading
        QL_LOG_WARN("Texture format '{}' not directly supported, use tydra for PNG/JPEG", ext);
        tex.name = fullPath.filename().string();
        tex.sourceUri = assetPath;
        return tex;
    }

    QL_LOG_INFO("  Loaded texture '{}' ({}x{}, {} channels)",
                tex.name, tex.width, tex.height, tex.channels);

    return tex;
}

// ============================================================================
// ParseSpectralExtensions - Parse Quantiloom custom primvars
// ============================================================================

void UsdLoader::ParseSpectralExtensions(Material& mat, const void* /* primPtr */,
                                          const String& /* usdFilePath */) {
    // Spectral extensions via custom primvars would be parsed here
    // For now, mark as RGB-upsampled; full primvar support requires
    // additional USD traversal for custom attributes

    if (mat.HasQuantiloomRef() || mat.HasIRData()) {
        mat.spectralSource = Material::SpectralSource::Measured;

        if (mat.HasIRData() && !mat.ValidateIRKirchhoffLaw()) {
            QL_LOG_WARN("  Material '{}' violates Kirchhoff's law (epsilon+rho+tau > 1)", mat.name);
        }
    }
}

// ============================================================================
// ParseMaterial - Convert UsdPreviewSurface to Quantiloom Material
// ============================================================================

Material UsdLoader::ParseMaterial(const void* /* stagePtr */, const void* /* matPtr */,
                                    const std::vector<Texture>& /* textures */,
                                    const String& /* usdFilePath */) {
    Material mat;

    // Default material values
    mat.baseColorFactor = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
    mat.metallicFactor = 0.0f;
    mat.roughnessFactor = 0.5f;
    mat.emissiveFactor = glm::vec3(0.0f);
    mat.alphaMode = Material::AlphaMode::Opaque;
    mat.alphaCutoff = 0.5f;

    mat.spectralSource = Material::SpectralSource::RGBUpsampled;
    mat.ComputeSpectralAlbedo();

    return mat;
}

// ============================================================================
// ParseMesh - Convert UsdGeomMesh to Quantiloom Mesh
// ============================================================================

Mesh UsdLoader::ParseMesh(const void* /* stagePtr */, const void* /* meshPtr */,
                            const std::vector<Material>& /* materials */) {
    Mesh mesh;
    return mesh;
}

// ============================================================================
// FlattenXformHierarchy - Flatten USD scene graph to world-space nodes
// ============================================================================

std::vector<SceneNode> UsdLoader::FlattenXformHierarchy(const void* /* stagePtr */) {
    std::vector<SceneNode> nodes;
    return nodes;
}

// ============================================================================
// LoadFromFile - Main entry point using tydra RenderSceneConverter
// ============================================================================

Result<Scene, String> UsdLoader::LoadFromFile(const String& path) {
    QL_LOG_INFO("Loading USD scene from: {}", path);

    if (!std::filesystem::exists(path)) {
        return Result<Scene>(Result<Scene>::Err("File not found: " + path));
    }

    // Determine file type and load appropriately
    std::filesystem::path filePath(path);
    String ext = filePath.extension().string();

    // Convert to lowercase for comparison
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    tinyusdz::Stage stage;
    std::string warn, err;
    bool success = false;

    if (ext == ".usda" || ext == ".usd") {
        success = tinyusdz::LoadUSDFromFile(path, &stage, &warn, &err);
    } else if (ext == ".usdc") {
        success = tinyusdz::LoadUSDCFromFile(path, &stage, &warn, &err);
    } else if (ext == ".usdz") {
        success = tinyusdz::LoadUSDZFromFile(path, &stage, &warn, &err);
    } else {
        return Result<Scene>(Result<Scene>::Err(
            "Unsupported USD file extension: " + ext +
            " (supported: .usd, .usda, .usdc, .usdz)"));
    }

    if (!warn.empty()) {
        QL_LOG_WARN("USD warning: {}", warn);
    }

    if (!success || !err.empty()) {
        return Result<Scene>(Result<Scene>::Err("Failed to load USD: " + err));
    }

    QL_LOG_INFO("  USD file loaded successfully");

    // Use tydra RenderSceneConverter to convert USD stage to render-ready data
    tinyusdz::tydra::RenderSceneConverter converter;
    tinyusdz::tydra::RenderSceneConverterEnv env(stage);

    // Configure converter
    env.timecode = tinyusdz::value::TimeCode::Default();
    env.mesh_config.triangulate = true;

    tinyusdz::tydra::RenderScene renderScene;

    // ConvertToRenderScene takes only 2 arguments: (env, scene*)
    bool converged = converter.ConvertToRenderScene(env, &renderScene);

    if (!converged) {
        std::string convErr = converter.GetError();
        if (!convErr.empty()) {
            QL_LOG_WARN("USD conversion warning: {}", convErr);
        }
    }

    QL_LOG_INFO("  Converted to render scene: {} meshes, {} materials",
                renderScene.meshes.size(),
                renderScene.materials.size());

    // Build Quantiloom Scene
    Scene scene;
    scene.name = filePath.stem().string();

    // ========================================================================
    // Load Materials from RenderScene
    // ========================================================================
    std::unordered_map<int, int> usdMatToSceneMat;

    for (size_t i = 0; i < renderScene.materials.size(); ++i) {
        const auto& usdMat = renderScene.materials[i];

        Material mat;
        mat.name = usdMat.name.empty() ? ("Material_" + std::to_string(i)) : usdMat.name;

        // UsdPreviewSurface PBR parameters from tydra surfaceShader
        const auto& shader = usdMat.surfaceShader;

        // Diffuse color
        if (shader.diffuseColor.is_texture()) {
            // Texture binding - would need texture ID lookup
            mat.baseColorFactor = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
        } else {
            const auto& color = shader.diffuseColor.value;
            mat.baseColorFactor = glm::vec4(color[0], color[1], color[2], 1.0f);
        }

        // Metallic
        if (!shader.metallic.is_texture()) {
            mat.metallicFactor = shader.metallic.value;
        }

        // Roughness
        if (!shader.roughness.is_texture()) {
            mat.roughnessFactor = shader.roughness.value;
        }

        // Emissive
        if (!shader.emissiveColor.is_texture()) {
            const auto& emiss = shader.emissiveColor.value;
            mat.emissiveFactor = glm::vec3(emiss[0], emiss[1], emiss[2]);
        }

        // Opacity
        if (!shader.opacity.is_texture()) {
            float opacity = shader.opacity.value;
            if (opacity < 1.0f) {
                mat.alphaMode = Material::AlphaMode::Blend;
                mat.baseColorFactor.a = opacity;
            }
        }

        mat.ComputeSpectralAlbedo();
        mat.spectralSource = Material::SpectralSource::RGBUpsampled;

        usdMatToSceneMat[static_cast<int>(i)] = static_cast<int>(scene.materials.size());
        scene.materials.push_back(std::move(mat));

        QL_LOG_INFO("  Loaded material '{}' (metallic={:.2f}, roughness={:.2f})",
                    scene.materials.back().name,
                    scene.materials.back().metallicFactor,
                    scene.materials.back().roughnessFactor);
    }

    // Ensure at least one default material exists
    if (scene.materials.empty()) {
        scene.materials.push_back(Material::CreateLambertian(glm::vec3(0.8f), "DefaultMaterial"));
    }

    // ========================================================================
    // Load Meshes from RenderScene
    // ========================================================================
    for (size_t i = 0; i < renderScene.meshes.size(); ++i) {
        const auto& usdMesh = renderScene.meshes[i];

        Mesh mesh;
        mesh.name = usdMesh.prim_name.empty() ? ("Mesh_" + std::to_string(i)) : usdMesh.prim_name;

        GeometryPrimitive primitive;

        // Convert positions (points is std::vector<vec3>)
        primitive.positions.reserve(usdMesh.points.size());
        for (const auto& p : usdMesh.points) {
            primitive.positions.emplace_back(p[0], p[1], p[2]);
        }

        // Convert normals from VertexAttribute
        if (!usdMesh.normals.empty()) {
            std::vector<glm::vec3> normals = ExtractVec3FromVertexAttribute(usdMesh.normals);
            primitive.normals = std::move(normals);
        }

        // Convert UVs (texcoords is unordered_map<uint32_t, VertexAttribute>)
        // Use slot 0 as primary UV
        auto uvIt = usdMesh.texcoords.find(0);
        if (uvIt != usdMesh.texcoords.end()) {
            std::vector<glm::vec2> uvs = ExtractVec2FromVertexAttribute(uvIt->second);
            primitive.uvs = std::move(uvs);
        }

        // Get triangle indices (tydra already triangulated if configured)
        const auto& indices = usdMesh.faceVertexIndices();
        primitive.indices.reserve(indices.size());
        for (const auto idx : indices) {
            primitive.indices.push_back(static_cast<u32>(idx));
        }

        // Material assignment
        int materialId = 0;
        if (usdMesh.material_id >= 0) {
            if (auto it = usdMatToSceneMat.find(usdMesh.material_id); it != usdMatToSceneMat.end()) {
                materialId = it->second;
            }
        }
        primitive.materialId = materialId;

        // Handle face-varying data expansion if needed
        if (!primitive.normals.empty() &&
            primitive.normals.size() != primitive.positions.size() &&
            primitive.normals.size() == primitive.indices.size()) {

            QL_LOG_DEBUG("  Expanding face-varying data for mesh '{}'", mesh.name);

            std::vector<glm::vec3> newPositions;
            std::vector<glm::vec3> newNormals;
            std::vector<glm::vec2> newUVs;
            std::vector<u32> newIndices;

            newPositions.reserve(primitive.indices.size());
            newNormals.reserve(primitive.indices.size());
            if (!primitive.uvs.empty()) {
                newUVs.reserve(primitive.indices.size());
            }

            for (size_t idx = 0; idx < primitive.indices.size(); ++idx) {
                u32 vertexIdx = primitive.indices[idx];
                newPositions.push_back(primitive.positions[vertexIdx]);
                newNormals.push_back(primitive.normals[idx]);

                if (!primitive.uvs.empty()) {
                    if (idx < primitive.uvs.size()) {
                        newUVs.push_back(primitive.uvs[idx]);
                    } else if (vertexIdx < primitive.uvs.size()) {
                        newUVs.push_back(primitive.uvs[vertexIdx]);
                    }
                }

                newIndices.push_back(static_cast<u32>(idx));
            }

            primitive.positions = std::move(newPositions);
            primitive.normals = std::move(newNormals);
            primitive.uvs = std::move(newUVs);
            primitive.indices = std::move(newIndices);
        }

        mesh.primitives.push_back(std::move(primitive));

        QL_LOG_INFO("  Loaded mesh '{}': {} vertices, {} triangles, material {}",
                    mesh.name,
                    mesh.primitives[0].GetVertexCount(),
                    mesh.primitives[0].GetTriangleCount(),
                    mesh.primitives[0].materialId);

        scene.meshes.push_back(std::move(mesh));
    }

    // ========================================================================
    // Build Scene Nodes (one per mesh with identity or world transform)
    // ========================================================================
    for (size_t i = 0; i < scene.meshes.size(); ++i) {
        SceneNode node;
        node.meshIndex = static_cast<u32>(i);
        node.name = scene.meshes[i].name;

        // tydra RenderMesh has world transform baked in
        // For now, use identity transform since tydra handles flattening
        node.transform = glm::mat4(1.0f);

        scene.nodes.push_back(node);
    }

    QL_LOG_INFO("  Scene '{}' loaded: {} meshes, {} nodes, {} materials, {} textures",
                scene.name, scene.meshes.size(), scene.nodes.size(),
                scene.materials.size(), scene.textures.size());

    return Result(std::move(scene));
}

} // namespace quantiloom
