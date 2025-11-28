// ============================================================================
// Quantiloom - Spectral Path Tracer
// ============================================================================
// Main entry point for Quantiloom spectral rendering system
// Supports single-wavelength and multi-wavelength rendering modes
// ============================================================================

#include "core/Log.hpp"
#include "core/Config.hpp"
#include "core/Image.hpp"
#include "io/ImageIO.hpp"
#include "io/GltfLoader.hpp"
#include "renderer/VulkanContext.hpp"
#include "renderer/RayTracingPipeline.hpp"
#include "renderer/AccelerationStructure.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/TextureManager.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/BRDFLutGenerator.hpp"
#include "renderer/PerformanceLogger.hpp"
#include "scene/Mesh.hpp"
#include "scene/Material.hpp"
#include "scene/Camera.hpp"
#include "SceneBuilder.hpp"

#include <glm/glm.hpp>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <cstddef>  // For offsetof
#include <cstdlib>  // For std::rand (random seed generation)

using namespace quantiloom;

// ============================================================================
// LUT Data Structure (matches shader LUTData structure)
// ============================================================================
// Dual mode support: RGB and spectral
// ============================================================================

struct LUTData {
    glm::vec3 sunDirection;         // FROM surface TO sun (normalized), offset 0
    f32 sunRadiance_spectral;       // Spectral radiance at current λ, offset 12

    glm::vec3 sunRadiance_rgb;      // RGB radiance for RGB mode, offset 16
    f32 skyRadiance_spectral;       // Spectral radiance at current λ, offset 28

    glm::vec3 skyRadiance_rgb;      // RGB radiance for RGB mode, offset 32
    f32 transmittance;              // Atmospheric transmittance τ(λ) [0, 1], offset 44
};  // Total: 48 bytes

// ============================================================================
// Material Data Structure (matches shader MaterialData structure)
// ============================================================================
// Must match the layout in common.hlsli exactly for GPU upload
// ============================================================================

struct MaterialDataCPU {
    glm::vec4 baseColorFactor;           // offset 0, size 16
    i32 baseColorTextureIndex;           // offset 16, size 4
    f32 metallicFactor;                  // offset 20, size 4
    f32 roughnessFactor;                 // offset 24, size 4
    i32 metallicRoughnessTextureIndex;   // offset 28, size 4

    i32 normalTextureIndex;              // offset 32, size 4
    f32 normalScale;                     // offset 36, size 4

    glm::vec3 emissiveFactor;            // offset 40, size 12
    i32 emissiveTextureIndex;            // offset 52, size 4

    u32 alphaMode;                       // offset 56, size 4
    f32 alphaCutoff;                     // offset 60, size 4

    f32 spectralAlbedo;                  // offset 64, size 4

    f32 irEmissivity;                    // offset 68, size 4
    f32 irReflectance;                   // offset 72, size 4
    f32 irTransmittance;                 // offset 76, size 4
    f32 irTemperature_K;                 // offset 80, size 4
};  // Total: 84 bytes (must match GPU MaterialData in common.hlsli)

// Verify struct layout matches shader expectations
// If this fails, the CPU/GPU struct layouts are mismatched, which WILL cause GPU crashes
static_assert(sizeof(MaterialDataCPU) == 84, "MaterialDataCPU size mismatch! Expected 84 bytes to match GPU MaterialData struct");
static_assert(offsetof(MaterialDataCPU, baseColorTextureIndex) == 16, "baseColorTextureIndex offset mismatch");
static_assert(offsetof(MaterialDataCPU, normalTextureIndex) == 32, "normalTextureIndex offset mismatch");
static_assert(offsetof(MaterialDataCPU, emissiveFactor) == 40, "emissiveFactor offset mismatch");
static_assert(offsetof(MaterialDataCPU, emissiveTextureIndex) == 52, "emissiveTextureIndex offset mismatch");
static_assert(offsetof(MaterialDataCPU, spectralAlbedo) == 64, "spectralAlbedo offset mismatch");
static_assert(offsetof(MaterialDataCPU, irEmissivity) == 68, "irEmissivity offset mismatch");
static_assert(offsetof(MaterialDataCPU, irReflectance) == 72, "irReflectance offset mismatch");
static_assert(offsetof(MaterialDataCPU, irTransmittance) == 76, "irTransmittance offset mismatch");
static_assert(offsetof(MaterialDataCPU, irTemperature_K) == 80, "irTemperature_K offset mismatch");

// ============================================================================
// Environment Map Helpers
// ============================================================================

// Convert equirectangular (latitude-longitude) to cubemap face direction
// face: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
// u, v: normalized coordinates [0, 1] within the face
// Returns: 3D direction vector (unnormalized)
inline glm::vec3 CubemapFaceDirection(u32 face, f32 u, f32 v) {
    // Convert UV to [-1, +1] range
    f32 x = 2.0f * u - 1.0f;
    f32 y = 2.0f * v - 1.0f;

    glm::vec3 dir;
    switch (face) {
        case 0: dir = glm::vec3( 1.0f,    -y,    -x); break;  // +X
        case 1: dir = glm::vec3(-1.0f,    -y,     x); break;  // -X
        case 2: dir = glm::vec3(    x,  1.0f,     y); break;  // +Y
        case 3: dir = glm::vec3(    x, -1.0f,    -y); break;  // -Y
        case 4: dir = glm::vec3(    x,    -y,  1.0f); break;  // +Z
        case 5: dir = glm::vec3(   -x,    -y, -1.0f); break;  // -Z
        default: dir = glm::vec3(0.0f, 0.0f, 0.0f); break;
    }
    return dir;
}

// Sample equirectangular map using direction vector
// Returns RGB color from the equirect map
inline glm::vec3 SampleEquirect(const Image& equirect, const glm::vec3& dir) {
    glm::vec3 normalized = glm::normalize(dir);

    // Convert Cartesian direction to spherical coordinates (θ, φ)
    // θ (theta): polar angle [0, π], φ (phi): azimuthal angle [0, 2π]
    f32 theta = std::acos(normalized.y);         // [0, π]
    f32 phi = std::atan2(normalized.z, normalized.x);  // [-π, π]

    // Convert to UV coordinates [0, 1]
    f32 u = (phi + glm::pi<f32>()) / (2.0f * glm::pi<f32>());  // [0, 1]
    f32 v = theta / glm::pi<f32>();                             // [0, 1]

    // Sample equirect with bilinear filtering
    u32 width = equirect.width;
    u32 height = equirect.height;

    f32 fx = u * static_cast<f32>(width - 1);
    f32 fy = v * static_cast<f32>(height - 1);

    u32 x0 = static_cast<u32>(fx) % width;
    u32 y0 = static_cast<u32>(fy) % height;
    u32 x1 = (x0 + 1) % width;
    u32 y1 = std::min(y0 + 1, height - 1);

    f32 wx = fx - std::floor(fx);
    f32 wy = fy - std::floor(fy);

    // Bilinear interpolation (assume RGB channels = 3)
    auto lerp = [](f32 a, f32 b, f32 t) { return a * (1.0f - t) + b * t; };

    glm::vec3 c00(equirect(x0, y0, 0), equirect(x0, y0, 1), equirect(x0, y0, 2));
    glm::vec3 c10(equirect(x1, y0, 0), equirect(x1, y0, 1), equirect(x1, y0, 2));
    glm::vec3 c01(equirect(x0, y1, 0), equirect(x0, y1, 1), equirect(x0, y1, 2));
    glm::vec3 c11(equirect(x1, y1, 0), equirect(x1, y1, 1), equirect(x1, y1, 2));

    glm::vec3 c0 = c00 * (1.0f - wx) + c10 * wx;
    glm::vec3 c1 = c01 * (1.0f - wx) + c11 * wx;

    return c0 * (1.0f - wy) + c1 * wy;
}

// Convert equirectangular image to cubemap faces
// Returns vector of 6 images (one per face), each with faceSize×faceSize resolution
std::vector<Image> EquirectToCubemap(const Image& equirect, u32 faceSize) {
    QL_LOG_INFO("Converting equirectangular map to cubemap ({}x{} per face)...", faceSize, faceSize);

    std::vector<Image> faces(6);

    for (u32 face = 0; face < 6; ++face) {
        faces[face] = Image(faceSize, faceSize, 3);  // RGB

        for (u32 y = 0; y < faceSize; ++y) {
            for (u32 x = 0; x < faceSize; ++x) {
                // Convert pixel to UV [0, 1]
                f32 u = (static_cast<f32>(x) + 0.5f) / static_cast<f32>(faceSize);
                f32 v = (static_cast<f32>(y) + 0.5f) / static_cast<f32>(faceSize);

                // Get 3D direction for this pixel
                glm::vec3 dir = CubemapFaceDirection(face, u, v);

                // Sample equirect map
                glm::vec3 color = SampleEquirect(equirect, dir);

                // Store in face
                faces[face](x, y, 0) = color.r;
                faces[face](x, y, 1) = color.g;
                faces[face](x, y, 2) = color.b;
            }
        }
    }

    QL_LOG_INFO("  Cubemap conversion complete");
    return faces;
}

// ============================================================================
// Scene Loading Helper
// ============================================================================

// Load scene from config file
// Returns either procedural scene or glTF-loaded scene
// For glTF scenes, also populates materials and textures
Result<Scene, String> LoadSceneFromConfig(const Config& config) {
    Scene scene;

    // Check for glTF file
    if (config.Has("scene.gltf")) {
        String gltfPath = config.Get<String>("scene.gltf");
        QL_LOG_INFO("Loading glTF model: {}", gltfPath);

        auto result = GltfLoader::LoadFromFile(gltfPath);
        if (!result.has_value()) {
            return Result<Scene, String>::Err("Failed to load glTF: " + result.error());
        }

        return std::move(result.value());
    }

    // Check for procedural preset
    if (config.Has("scene.preset")) {
        String preset = config.Get<String>("scene.preset", "cornell_box");
        QL_LOG_INFO("Loading built-in scene preset: {}", preset);

        Mesh mesh;
        if (preset == "cornell_box") {
            mesh = TestScenes::CreateCornellBoxScene();
        } else if (preset == "multi_object") {
            mesh = TestScenes::CreateMultiObjectScene();
        } else if (preset == "lighting_test") {
            mesh = TestScenes::CreateLightingTestScene();
        } else {
            QL_LOG_WARN("Unknown scene preset '{}', defaulting to cornell_box", preset);
            mesh = TestScenes::CreateCornellBoxScene();
        }

        // Wrap in Scene
        scene.name = preset;
        scene.meshes.push_back(std::move(mesh));

        // Create single node with identity transform
        SceneNode node;
        node.meshIndex = 0;
        node.transform = glm::mat4(1.0f);
        node.name = "SceneRoot";
        scene.nodes.push_back(node);

        return std::move(scene);
    }

    // Default: Cornell box
    QL_LOG_WARN("No scene specified in config, using cornell_box preset");
    Mesh mesh = TestScenes::CreateCornellBoxScene();
    scene.name = "cornell_box";
    scene.meshes.push_back(std::move(mesh));

    SceneNode node;
    node.meshIndex = 0;
    node.transform = glm::mat4(1.0f);
    node.name = "SceneRoot";
    scene.nodes.push_back(node);

    return std::move(scene);
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char* argv[]) {
    // ========================================================================
    // Initialize Logging
    // ========================================================================
    Log::Init("quantiloom.log", Log::Level::Info);

    QL_LOG_INFO("========================================");
    QL_LOG_INFO("  Quantiloom Spectral Path Tracer");
    QL_LOG_INFO("========================================");

    // ========================================================================
    // Load Configuration
    // ========================================================================
    if (argc < 2) {
        QL_LOG_ERROR("No configuration file provided");
        QL_LOG_INFO("Usage: {} <config.toml>", argv[0]);
        QL_LOG_INFO("Example: {} assets/configs/spectral_single.toml", argv[0]);
        Log::Shutdown();
        return 1;
    }

    std::filesystem::path configPath(argv[1]);
    QL_LOG_INFO("Loading configuration: {}", configPath.string());

    auto configResult = Config::Load(configPath);
    if (!configResult.has_value()) {
        QL_LOG_ERROR("Failed to load configuration: {}", configResult.error());
        Log::Shutdown();
        return 1;
    }

    Config config = configResult.value();
    QL_LOG_INFO("Configuration loaded successfully");

    try {
        // ====================================================================
        // Parse Configuration
        // ====================================================================
        QL_LOG_INFO("Parsing configuration...");

        // Renderer settings
        auto resArray = config.GetArray<u32>("renderer.resolution");
        if (resArray.size() != 2) {
            QL_LOG_ERROR("renderer.resolution must be array of 2 integers, but got {} and they are {} {}", resArray.size(), resArray[0], resArray[1]);
            return 1;
        }
        u32 width = resArray[0];
        u32 height = resArray[1];
        u32 spp = config.Get<u32>("renderer.spp", 1);
        String outputPath = config.Get<String>("renderer.output", "spectral_output.exr");

        QL_LOG_INFO("  Resolution: {}x{}", width, height);
        QL_LOG_INFO("  Samples per pixel: {}", spp);
        QL_LOG_INFO("  Output: {}", outputPath);

        // Spectral settings
        String spectralModeStr = config.Get<String>("spectral.mode", "rgb");  // Default to RGB

        // Parse spectral mode
        auto spectralModeResult = ParseSpectralMode(spectralModeStr);
        if (!spectralModeResult.has_value()) {
            QL_LOG_ERROR("Invalid spectral mode: {}", spectralModeStr);
            QL_LOG_ERROR("Supported modes: single, rgb, mwir_fused, lwir_fused");
            return 1;
        }
        SpectralMode spectral_mode = *spectralModeResult;

        QL_LOG_INFO("  Spectral mode: {}", spectralModeStr);

        // Only read wavelength_nm for modes that require it (single, MWIR, LWIR)
        f32 wavelength_nm = 550.0f;  // Default value (unused in RGB mode)
        if (spectral_mode == SpectralMode::Single ||
            spectral_mode == SpectralMode::MWIR_Fused ||
            spectral_mode == SpectralMode::LWIR_Fused) {
            wavelength_nm = config.Get<f32>("spectral.wavelength_nm", 550.0f);
            QL_LOG_INFO("  Wavelength: {:.1f} nm", wavelength_nm);
        }

        // Camera settings
        f32 aspectRatio = static_cast<f32>(width) / static_cast<f32>(height);
        auto cameraResult = Camera::FromConfig(config, aspectRatio);
        if (!cameraResult.has_value()) {
            QL_LOG_ERROR("Failed to load camera: {}", cameraResult.error());
            return 1;
        }
        Camera camera = cameraResult.value();

        // Lighting settings
        auto sunDirArray = config.GetArray<f32>("lighting.sun_direction");
        if (sunDirArray.size() != 3) {
            QL_LOG_ERROR("lighting.sun_direction must be array of 3 floats, but got {}", sunDirArray.size());
            return 1;
        }
        glm::vec3 sunDirection = glm::normalize(glm::vec3(sunDirArray[0], sunDirArray[1], sunDirArray[2]));

        auto sunRadArray = config.GetArray<f32>("lighting.sun_radiance");
        if (sunRadArray.size() != 3) {
            QL_LOG_ERROR("lighting.sun_radiance must be array of 3 floats, but got {}", sunRadArray.size());
            return 1;
        }
        glm::vec3 sunRadiance(sunRadArray[0], sunRadArray[1], sunRadArray[2]);

        auto skyRadArray = config.GetArray<f32>("lighting.sky_radiance");
        if (skyRadArray.size() != 3) {
            QL_LOG_ERROR("lighting.sky_radiance must be array of 3 floats, but got {}", skyRadArray.size());
            return 1;
        }
        glm::vec3 skyRadiance(skyRadArray[0], skyRadArray[1], skyRadArray[2]);

        // Atmospheric transmittance (Beer-Lambert law)
        // Default: 0.9 (relatively clear atmosphere, ~10% attenuation)
        f32 transmittance = config.Get<f32>("lighting.transmittance", 0.9f);
        transmittance = std::clamp(transmittance, 0.0f, 1.0f);

        QL_LOG_INFO("  Sun direction: [{:.2f}, {:.2f}, {:.2f}]",
                    sunDirection.x, sunDirection.y, sunDirection.z);
        QL_LOG_INFO("  Sun radiance: [{:.2f}, {:.2f}, {:.2f}]",
                    sunRadiance.x, sunRadiance.y, sunRadiance.z);
        QL_LOG_INFO("  Sky radiance: [{:.2f}, {:.2f}, {:.2f}]",
                    skyRadiance.x, skyRadiance.y, skyRadiance.z);
        QL_LOG_INFO("  Atmospheric transmittance: {:.3f}", transmittance);

        // Material settings
        auto albedoArray = config.GetArray<f32>("material.albedo");
        if (albedoArray.size() != 3) {
            QL_LOG_ERROR("material.albedo must be array of 3 floats, but got {}", albedoArray.size());
            return 1;
        }
        glm::vec3 albedo(albedoArray[0], albedoArray[1], albedoArray[2]);
        QL_LOG_INFO("  Material albedo: [{:.2f}, {:.2f}, {:.2f}]", albedo.x, albedo.y, albedo.z);

        // ====================================================================
        // Initialize Vulkan Context
        // ====================================================================
        QL_LOG_INFO("Initializing Vulkan context...");
        VulkanContext context;

        if (!context.IsRayTracingSupported()) {
            QL_LOG_ERROR("Ray tracing not supported on this device");
            return 1;
        }

        // ====================================================================
        // Load Scene Geometry
        // ====================================================================
        QL_LOG_INFO("Loading scene...");

        auto sceneResult = LoadSceneFromConfig(config);
        if (!sceneResult.has_value()) {
            QL_LOG_ERROR("Failed to load scene: {}", sceneResult.error());
            return 1;
        }

        Scene loadedScene = sceneResult.value();

        // If scene has no materials (procedural), create default from config
        if (loadedScene.materials.empty()) {
            Material defaultMaterial = Material::CreateLambertian(albedo, "DefaultMaterial");
            loadedScene.materials.push_back(defaultMaterial);
            QL_LOG_INFO("  Created default material (spectral albedo: {:.3f})",
                        defaultMaterial.spectralAlbedo);
        }

        QL_LOG_INFO("  Scene loaded: {} meshes, {} nodes, {} materials",
                    loadedScene.meshes.size(), loadedScene.nodes.size(),
                    loadedScene.materials.size());

        // ====================================================================
        // Validate Material Spectral Sources (sRGB upsampling gate - R5)
        // ====================================================================
        bool requireQuantitative = (spectral_mode == SpectralMode::Multispectral ||
                                     spectral_mode == SpectralMode::MWIR_Fused ||
                                     spectral_mode == SpectralMode::LWIR_Fused);

        bool failOnSRGB = config.Get<bool>("quality.fail_on_srgb_upsample", false);
        bool logSources = config.Get<bool>("quality.log_material_sources", false);

        if (requireQuantitative && (failOnSRGB || logSources)) {
            QL_LOG_INFO("Validating material spectral sources for quantitative mode...");

            bool hasInvalidMaterials = false;
            for (const auto& mat : loadedScene.materials) {
                const char* sourceStr = "Unknown";
                switch (mat.spectralSource) {
                    case Material::SpectralSource::Measured:
                        sourceStr = "Measured (quantitative)";
                        break;
                    case Material::SpectralSource::RGBUpsampled:
                        sourceStr = "RGB-upsampled (NOT quantitative)";
                        hasInvalidMaterials = true;
                        break;
                    case Material::SpectralSource::Procedural:
                        sourceStr = "Procedural";
                        break;
                    default:
                        sourceStr = "Unknown";
                        break;
                }

                if (logSources) {
                    QL_LOG_INFO("  Material '{}': source = {}", mat.name, sourceStr);
                }

                if (mat.spectralSource == Material::SpectralSource::RGBUpsampled) {
                    QL_LOG_WARN("  ⚠️  Material '{}' uses RGB-upsampled spectra (not quantitative)",
                                mat.name);
                }
            }

            if (hasInvalidMaterials && failOnSRGB) {
                QL_LOG_ERROR("========================================");
                QL_LOG_ERROR("  ABORTED: RGB-upsampled materials detected");
                QL_LOG_ERROR("========================================");
                QL_LOG_ERROR("RGB-upsampled materials are NOT suitable for quantitative analysis.");
                QL_LOG_ERROR("To proceed (non-quantitative preview), set 'quality.fail_on_srgb_upsample = false'.");
                QL_LOG_ERROR("For quantitative results, provide measured spectral material data.");
                QL_LOG_ERROR("========================================");
                Log::Shutdown();
                return 1;
            }

            if (hasInvalidMaterials && !failOnSRGB) {
                QL_LOG_WARN("⚠️  WARNING: Proceeding with RGB-upsampled materials (non-quantitative preview).");
            }
        }

        // M2: Build BLAS for each primitive in each mesh
        // This allows per-primitive materials and proper glTF support
        std::vector<BLAS> blasList;
        std::vector<u32> primitiveMaterialIds;  // Track material ID for each BLAS

        for (const auto& mesh : loadedScene.meshes) {
            for (const auto& primitive : mesh.primitives) {
                blasList.emplace_back(context, primitive);
                primitiveMaterialIds.push_back(primitive.materialId);
            }
        }

        u32 totalTriangles = 0;
        for (const auto& mesh : loadedScene.meshes) {
            totalTriangles += mesh.GetTotalTriangleCount();
        }

        QL_LOG_INFO("  Created {} BLAS(es) for {} total triangles",
                    blasList.size(), totalTriangles);

        // ====================================================================
        // Build Acceleration Structures
        // ====================================================================
        QL_LOG_INFO("Building acceleration structures...");

        // Build TLAS with all instances
        TLAS tlas(context);

        CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
            // Build all BLAS
            for (auto& blas : blasList) {
                blas.Build(cmd);
            }

            // Add instances to TLAS
            size_t blasIndex = 0;
            for (const auto& node : loadedScene.nodes) {
                const Mesh& mesh = loadedScene.meshes[node.meshIndex];

                for (size_t primIdx = 0; primIdx < mesh.primitives.size(); ++primIdx) {
                    const auto& primitive = mesh.primitives[primIdx];
                    tlas.AddInstance(
                        blasList[blasIndex],
                        primitive.materialId,
                        node.transform
                    );
                    ++blasIndex;
                }
            }

            tlas.Build(cmd);
        });

        QL_LOG_INFO("  TLAS built with {} instance(s)", loadedScene.nodes.size());

        // ====================================================================
        // Create Output Image
        // ====================================================================
        QL_LOG_INFO("Creating output image ({}x{})...", width, height);

        GpuImage outputImage(
            context.GetAllocator(),
            context.GetDevice(),
            width, height,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY
        );

        CommandHelper::TransitionImageLayoutImmediate(
            context,
            outputImage.GetImage(),
            outputImage.GetFormat(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL
        );

        // ====================================================================
        // Create LUT Buffer (Dual Mode: RGB + Spectral)
        // ====================================================================
        QL_LOG_INFO("Creating LUT buffer...");

        // For spectral mode: Convert RGB to scalar (average of RGB channels)
        f32 sunRadiance_spectral = (sunRadiance.r + sunRadiance.g + sunRadiance.b) / 3.0f;
        f32 skyRadiance_spectral = (skyRadiance.r + skyRadiance.g + skyRadiance.b) / 3.0f;

        if (spectral_mode == SpectralMode::RGB) {
            QL_LOG_INFO("  Sun RGB radiance: [{:.2f}, {:.2f}, {:.2f}] W·sr⁻¹·m⁻²",
                        sunRadiance.r, sunRadiance.g, sunRadiance.b);
            QL_LOG_INFO("  Sky RGB radiance: [{:.2f}, {:.2f}, {:.2f}] W·sr⁻¹·m⁻²",
                        skyRadiance.r, skyRadiance.g, skyRadiance.b);
        } else {
            QL_LOG_INFO("  Sun spectral radiance: {:.3f} W·sr⁻¹·m⁻²·nm⁻¹", sunRadiance_spectral);
            QL_LOG_INFO("  Sky spectral radiance: {:.3f} W·sr⁻¹·m⁻²·nm⁻¹", skyRadiance_spectral);
        }

        LUTData lutData;
        lutData.sunDirection = sunDirection;

        // Fill both spectral and RGB fields for flexibility
        lutData.sunRadiance_spectral = sunRadiance_spectral;
        lutData.skyRadiance_spectral = skyRadiance_spectral;
        lutData.sunRadiance_rgb = sunRadiance;
        lutData.skyRadiance_rgb = skyRadiance;
        lutData.transmittance = transmittance;

        GpuBuffer lutBuffer(
            context.GetAllocator(),
            sizeof(LUTData),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );

        lutBuffer.Upload(&lutData, sizeof(LUTData));

        // ====================================================================
        // Upload Textures to GPU
        // ====================================================================
        QL_LOG_INFO("Uploading textures to GPU...");

        TextureManager textureManager(context);
        textureManager.UploadTextures(loadedScene.textures);

        QL_LOG_INFO("  {} textures uploaded", textureManager.GetTextureCount());

        // ====================================================================
        // Create Material Buffer (PBR)
        // ====================================================================
        QL_LOG_INFO("Creating PBR material buffer...");

        // Upload all materials with full PBR parameters
        std::vector<MaterialDataCPU> materialData;
        materialData.reserve(loadedScene.materials.size());

        for (const auto& mat : loadedScene.materials) {
            MaterialDataCPU cpuMat;

            // Base color
            cpuMat.baseColorFactor = mat.baseColorFactor;
            cpuMat.baseColorTextureIndex = mat.baseColorTextureIndex;

            // Metallic-Roughness
            cpuMat.metallicFactor = mat.metallicFactor;
            cpuMat.roughnessFactor = mat.roughnessFactor;
            cpuMat.metallicRoughnessTextureIndex = mat.metallicRoughnessTextureIndex;

            // Normal mapping
            cpuMat.normalTextureIndex = mat.normalTextureIndex;
            cpuMat.normalScale = mat.normalScale;

            // Emissive
            cpuMat.emissiveFactor = mat.emissiveFactor;
            cpuMat.emissiveTextureIndex = mat.emissiveTextureIndex;

            // Alpha mode
            cpuMat.alphaMode = static_cast<u32>(mat.alphaMode);
            cpuMat.alphaCutoff = mat.alphaCutoff;

            // Spectral (M1 compatibility)
            cpuMat.spectralAlbedo = mat.spectralAlbedo;

            // Infrared material properties (evaluate curves at current wavelength)
            cpuMat.irEmissivity = mat.GetIREmissivity(wavelength_nm);
            cpuMat.irReflectance = mat.GetIRReflectance(wavelength_nm);
            cpuMat.irTransmittance = mat.GetIRTransmittance(wavelength_nm);
            cpuMat.irTemperature_K = mat.irTemperature_K;

            materialData.push_back(cpuMat);

            QL_LOG_INFO("  Material '{}': base=[{:.2f},{:.2f},{:.2f},{:.2f}] metal={:.2f} rough={:.2f}",
                        mat.name,
                        mat.baseColorFactor.r, mat.baseColorFactor.g, mat.baseColorFactor.b, mat.baseColorFactor.a,
                        mat.metallicFactor, mat.roughnessFactor);
            QL_LOG_INFO("    [DEBUG] emissive=[{:.3f},{:.3f},{:.3f}]",
                        mat.emissiveFactor.r, mat.emissiveFactor.g, mat.emissiveFactor.b);
            QL_LOG_INFO("    [DEBUG] Texture indices: baseColor={} metallicRough={} normal={} emissive={}",
                        mat.baseColorTextureIndex, mat.metallicRoughnessTextureIndex,
                        mat.normalTextureIndex, mat.emissiveTextureIndex);
        }

        GpuBuffer materialBuffer(
            context.GetAllocator(),
            materialData.size() * sizeof(MaterialDataCPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );

        materialBuffer.Upload(materialData.data(), materialData.size() * sizeof(MaterialDataCPU));

        // ====================================================================
        // Generate BRDF Integration LUT for IBL
        // ====================================================================
        // TODO (Optimization): Add disk caching to avoid 6-second regeneration on each run
        // Potential cache format: Binary (.dat) or HDF5 (.h5) with metadata
        // Cache key: resolution + sampleCount + BRDF model (GGX)
        // See docs/DATA_PREPARATION.md for caching strategy
        QL_LOG_INFO("Generating BRDF integration LUT for IBL (this may take 5-10 seconds)...");

        BRDFLutGenerator::Config brdfConfig;
        brdfConfig.resolution = 512;
        brdfConfig.sampleCount = 1024;
        Image brdfLutImage = BRDFLutGenerator::Generate(brdfConfig);

        // Upload BRDF LUT to GPU
        QL_LOG_INFO("  Uploading BRDF LUT to GPU...");
        GpuImage brdfLutTexture(
            context.GetAllocator(),
            context.GetDevice(),
            512,  // width
            512,  // height
            VK_FORMAT_R32G32_SFLOAT,  // RG32F (2 channels, 32-bit float each)
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY
        );

        // Transition image to TRANSFER_DST for upload
        CommandHelper::TransitionImageLayoutImmediate(
            context,
            brdfLutTexture.GetImage(),
            brdfLutTexture.GetFormat(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        );

        // Upload LUT data
        {
            // Convert Image to raw buffer (RG32F format)
            std::vector<f32> lutData(512 * 512 * 2);
            for (u32 y = 0; y < 512; ++y) {
                for (u32 x = 0; x < 512; ++x) {
                    u32 idx = (y * 512 + x) * 2;
                    lutData[idx + 0] = brdfLutImage(x, y, 0);  // R channel (scale)
                    lutData[idx + 1] = brdfLutImage(x, y, 1);  // G channel (bias)
                }
            }

            // Create staging buffer
            VkDeviceSize bufferSize = lutData.size() * sizeof(f32);
            GpuBuffer stagingBuffer(
                context.GetAllocator(),
                bufferSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VMA_MEMORY_USAGE_CPU_TO_GPU
            );
            stagingBuffer.Upload(lutData.data(), bufferSize);

            // Copy buffer to image
            CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
                VkBufferImageCopy region{};
                region.bufferOffset = 0;
                region.bufferRowLength = 0;  // Tightly packed
                region.bufferImageHeight = 0;
                region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.mipLevel = 0;
                region.imageSubresource.baseArrayLayer = 0;
                region.imageSubresource.layerCount = 1;
                region.imageOffset = {0, 0, 0};
                region.imageExtent = {512, 512, 1};

                vkCmdCopyBufferToImage(
                    cmd,
                    stagingBuffer.GetHandle(),
                    brdfLutTexture.GetImage(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    &region
                );
            });
        }

        // Transition image to SHADER_READ_ONLY for sampling
        CommandHelper::TransitionImageLayoutImmediate(
            context,
            brdfLutTexture.GetImage(),
            brdfLutTexture.GetFormat(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );

        // Create sampler for BRDF LUT (linear filtering, clamp to edge)
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;  // No mipmaps
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;

        VkSampler brdfLutSampler = VK_NULL_HANDLE;
        VkResult samplerResult = vkCreateSampler(context.GetDevice(), &samplerInfo, nullptr, &brdfLutSampler);
        if (samplerResult != VK_SUCCESS) {
            throw std::runtime_error("Failed to create BRDF LUT sampler");
        }

        QL_LOG_INFO("  BRDF LUT uploaded successfully");

        // ====================================================================
        // Create Prefiltered Environment Map for IBL Specular
        // ====================================================================
        QL_LOG_INFO("Creating prefiltered environment map for IBL...");

        // Load environment map (equirectangular EXR)
        String envMapPath = config.Get<String>("renderer.environment_map", "");
        std::vector<Image> cubemapFaces;
        u32 envMapSize = 256;  // Default cubemap face size
        constexpr u32 envMapMips = 5;  // Mip chain for roughness levels (roughness 0.0 to 1.0)

        if (!envMapPath.empty() && ImageIO::FileExists(envMapPath)) {
            QL_LOG_INFO("  Loading environment map from: {}", envMapPath);

            auto equirectOpt = ImageIO::ReadEXR(envMapPath);
            if (equirectOpt.has_value()) {
                Image& equirect = equirectOpt.value();
                QL_LOG_INFO("  Environment map loaded: {}x{}, {} channels",
                           equirect.width, equirect.height, equirect.channels);

                // Convert to cubemap (use 512x512 per face for EXR input)
                envMapSize = 512;
                cubemapFaces = EquirectToCubemap(equirect, envMapSize);
            } else {
                QL_LOG_WARN("  Failed to load environment map from {}, using fallback", envMapPath);
                envMapPath = "";  // Trigger fallback
            }
        } else {
            if (!envMapPath.empty()) {
                QL_LOG_WARN("  Environment map not found: {}, using fallback", envMapPath);
            } else {
                QL_LOG_INFO("  No environment map specified in config, using fallback");
            }
        }

        // Fallback: sky-blue cubemap if no EXR loaded
        if (envMapPath.empty()) {
            QL_LOG_INFO("  Creating fallback sky-blue cubemap ({}x{} per face)...", envMapSize, envMapSize);
            cubemapFaces.resize(6);
            for (u32 face = 0; face < 6; ++face) {
                cubemapFaces[face] = Image(envMapSize, envMapSize, 3);
                // Sky-blue color: soft blue gradient
                constexpr f32 skyColor[3] = {0.5f, 0.7f, 1.0f};
                for (u32 y = 0; y < envMapSize; ++y) {
                    for (u32 x = 0; x < envMapSize; ++x) {
                        cubemapFaces[face](x, y, 0) = skyColor[0];
                        cubemapFaces[face](x, y, 1) = skyColor[1];
                        cubemapFaces[face](x, y, 2) = skyColor[2];
                    }
                }
            }
        }

        // Manually create cubemap image (GpuImage doesn't support cubemaps yet)
        VkImage envMapImage = VK_NULL_HANDLE;
        VkImageView envMapView = VK_NULL_HANDLE;
        VmaAllocation envMapAllocation = VK_NULL_HANDLE;

        {
            // Create cubemap image
            VkImageCreateInfo imageInfo{};
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;  // RGBA32F (HDR)
            imageInfo.extent = {envMapSize, envMapSize, 1};
            imageInfo.mipLevels = envMapMips;
            imageInfo.arrayLayers = 6;  // Cubemap faces: +X, -X, +Y, -Y, +Z, -Z
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;  // Enable cubemap view

            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

            VkResult result = vmaCreateImage(context.GetAllocator(), &imageInfo, &allocInfo,
                                              &envMapImage, &envMapAllocation, nullptr);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Failed to create environment cubemap image");
            }

            // Create cubemap view
            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = envMapImage;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;  // Cubemap view
            viewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = envMapMips;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 6;  // All 6 faces

            result = vkCreateImageView(context.GetDevice(), &viewInfo, nullptr, &envMapView);
            if (result != VK_SUCCESS) {
                vmaDestroyImage(context.GetAllocator(), envMapImage, envMapAllocation);
                throw std::runtime_error("Failed to create environment cubemap view");
            }
        }

        // Transition image to TRANSFER_DST for upload
        CommandHelper::TransitionImageLayoutImmediate(
            context,
            envMapImage,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            envMapMips,
            6  // All 6 cubemap faces
        );

        // Upload cubemap faces to GPU (mip level 0 only; TODO: generate mipchain for roughness)
        QL_LOG_INFO("  Uploading cubemap to GPU ({} faces, {}x{} per face)...", cubemapFaces.size(), envMapSize, envMapSize);
        {
            for (u32 face = 0; face < 6; ++face) {
                const Image& faceImage = cubemapFaces[face];

                // Convert RGB to RGBA (add alpha = 1.0)
                std::vector<f32> pixelData(envMapSize * envMapSize * 4);
                for (u32 y = 0; y < envMapSize; ++y) {
                    for (u32 x = 0; x < envMapSize; ++x) {
                        u32 idx = (y * envMapSize + x) * 4;
                        pixelData[idx + 0] = faceImage(x, y, 0);  // R
                        pixelData[idx + 1] = faceImage(x, y, 1);  // G
                        pixelData[idx + 2] = faceImage(x, y, 2);  // B
                        pixelData[idx + 3] = 1.0f;                // A
                    }
                }

                // Create staging buffer for this face
                VkDeviceSize bufferSize = pixelData.size() * sizeof(f32);
                GpuBuffer stagingBuffer(
                    context.GetAllocator(),
                    bufferSize,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VMA_MEMORY_USAGE_CPU_TO_GPU
                );
                stagingBuffer.Upload(pixelData.data(), bufferSize);

                // Upload to mip level 0
                CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
                    VkBufferImageCopy region{};
                    region.bufferOffset = 0;
                    region.bufferRowLength = 0;  // Tightly packed
                    region.bufferImageHeight = 0;
                    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    region.imageSubresource.mipLevel = 0;  // Base mip level
                    region.imageSubresource.baseArrayLayer = face;
                    region.imageSubresource.layerCount = 1;
                    region.imageOffset = {0, 0, 0};
                    region.imageExtent = {envMapSize, envMapSize, 1};

                    vkCmdCopyBufferToImage(
                        cmd,
                        stagingBuffer.GetHandle(),
                        envMapImage,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1,
                        &region
                    );
                });
            }

            // Generate mipmaps for remaining levels (simple box filter)
            // TODO (M2): Replace with proper GGX prefiltering for PBR
            QL_LOG_INFO("  Generating mipmap chain (simple downsampling)...");
            for (u32 mip = 1; mip < envMapMips; ++mip) {
                u32 mipSize = envMapSize >> mip;  // Divide by 2^mip
                if (mipSize == 0) mipSize = 1;

                // For now, just copy the base level (no actual filtering)
                // TODO: Implement proper mipmap generation with GGX kernel
                for (u32 face = 0; face < 6; ++face) {
                    // Create downsampled data (simple box filter)
                    std::vector<f32> mipData(mipSize * mipSize * 4);
                    u32 prevMipSize = envMapSize >> (mip - 1);

                    for (u32 y = 0; y < mipSize; ++y) {
                        for (u32 x = 0; x < mipSize; ++x) {
                            // Sample 2x2 region from previous mip level
                            u32 srcX = x * 2;
                            u32 srcY = y * 2;
                            glm::vec4 sum(0.0f);
                            for (u32 dy = 0; dy < 2 && (srcY + dy) < prevMipSize; ++dy) {
                                for (u32 dx = 0; dx < 2 && (srcX + dx) < prevMipSize; ++dx) {
                                    u32 srcIdx = ((srcY + dy) * prevMipSize + (srcX + dx));
                                    if (srcIdx < cubemapFaces[face].PixelCount()) {
                                        sum.r += cubemapFaces[face].data[srcIdx * 3 + 0];
                                        sum.g += cubemapFaces[face].data[srcIdx * 3 + 1];
                                        sum.b += cubemapFaces[face].data[srcIdx * 3 + 2];
                                        sum.a += 1.0f;
                                    }
                                }
                            }
                            sum /= 4.0f;

                            u32 dstIdx = (y * mipSize + x) * 4;
                            mipData[dstIdx + 0] = sum.r;
                            mipData[dstIdx + 1] = sum.g;
                            mipData[dstIdx + 2] = sum.b;
                            mipData[dstIdx + 3] = 1.0f;
                        }
                    }

                    // Upload this mip level
                    VkDeviceSize bufferSize = mipData.size() * sizeof(f32);
                    GpuBuffer stagingBuffer(
                        context.GetAllocator(),
                        bufferSize,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VMA_MEMORY_USAGE_CPU_TO_GPU
                    );
                    stagingBuffer.Upload(mipData.data(), bufferSize);

                    CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
                        VkBufferImageCopy region{};
                        region.bufferOffset = 0;
                        region.bufferRowLength = 0;
                        region.bufferImageHeight = 0;
                        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                        region.imageSubresource.mipLevel = mip;
                        region.imageSubresource.baseArrayLayer = face;
                        region.imageSubresource.layerCount = 1;
                        region.imageOffset = {0, 0, 0};
                        region.imageExtent = {mipSize, mipSize, 1};

                        vkCmdCopyBufferToImage(
                            cmd,
                            stagingBuffer.GetHandle(),
                            envMapImage,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            1,
                            &region
                        );
                    });
                }
            }
        }

        // Transition image to SHADER_READ_ONLY for sampling
        CommandHelper::TransitionImageLayoutImmediate(
            context,
            envMapImage,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            envMapMips,
            6  // All 6 cubemap faces
        );

        QL_LOG_INFO("  Prefiltered environment map created ({}x{} per face, {} mip levels)",
                    envMapSize, envMapSize, envMapMips);

        // ====================================================================
        // Create Ray Tracing Pipeline
        // ====================================================================
        QL_LOG_INFO("Creating ray tracing pipeline...");

        RayTracingPipeline pipeline(
            context,
            "raygen.spv",
            "closesthit.spv",
            "miss.spv"
        );

        // Bind resources in correct order (bindings 0-7)
        pipeline.BindOutputImage(outputImage);                          // Binding 0
        pipeline.BindAccelerationStructure(tlas.GetHandle());           // Binding 1
        pipeline.BindLUTBuffer(lutBuffer);                              // Binding 2

        // Use first BLAS for geometry buffers (all BLAS share same vertex/index binding)
        if (!blasList.empty()) {
            const GpuBuffer* uvBuffer = blasList[0].HasUVs() ? &blasList[0].GetUVBuffer() : nullptr;
            pipeline.BindGeometryBuffers(blasList[0].GetVertexBuffer(), blasList[0].GetIndexBuffer(), uvBuffer); // Binding 3, 4, 8

            // Bind tangent buffer (always present, uses fallback data if model has no tangents)
            pipeline.BindTangentBuffer(blasList[0].GetTangentBuffer());  // Binding 9
        }

        pipeline.BindMaterialBuffer(materialBuffer);                    // Binding 5

        // Bind textures (bindless arrays)
        pipeline.BindTextures(textureManager.GetImageViews(), textureManager.GetSamplers()); // Binding 6, 7

        // ====================================================================
        // Bind IBL (Image-Based Lighting) resources
        // ====================================================================
        // Binding 10: Prefiltered environment cubemap (with mip chain for roughness)
        // Binding 11: BRDF integration LUT (2D texture)
        // Binding 12: IBL sampler (shared by both textures)
        pipeline.BindPrefilteredEnvMap(envMapView);                      // Binding 10
        pipeline.BindBRDFLut(brdfLutTexture.GetView(), brdfLutSampler);  // Binding 11, 12

        // Set camera parameters (with spectral wavelength and rendering mode)
        CameraData cameraData = camera.GetCameraData();
        cameraData.wavelength_nm = wavelength_nm;  // Override with config wavelength
        cameraData.spectral_mode = static_cast<u32>(spectral_mode);  // Set rendering mode
        pipeline.SetCameraData(cameraData);

        QL_LOG_INFO("  Pipeline created and resources bound");

        // ====================================================================
        // Initialize Performance Logger
        // ====================================================================
        QL_LOG_INFO("Initializing performance logger...");

        PerformanceLogger::Config perfConfig;
        perfConfig.csvFilePath = "quantiloom_performance.csv";
        perfConfig.enableLogging = true;
        PerformanceLogger perfLogger(context, perfConfig);

        // ====================================================================
        // Render Frame with Accumulative Sampling
        // ====================================================================
        if (spectral_mode == SpectralMode::Single ||
            spectral_mode == SpectralMode::MWIR_Fused ||
            spectral_mode == SpectralMode::LWIR_Fused) {
            QL_LOG_INFO("Rendering frame at wavelength {:.1f} nm with {} samples per pixel...", wavelength_nm, spp);
            QL_LOG_WARN("  ⚠️  PREVIEW MODE: Using RGB-averaged spectral albedo.");
            QL_LOG_WARN("  ⚠️  NOT suitable for quantitative analysis.");
            QL_LOG_WARN("  ⚠️  For quantitative results, use mode=\"hs_off\" with measured spectral data.");
        } else {
            QL_LOG_INFO("Rendering frame in RGB mode with {} samples per pixel...", spp);
        }

        try {
            // Frame index for temporal effects (set to 0 for single-frame renders)
            u32 frameIndex = 0;

            // Total accumulated GPU time and rays for all samples
            f32 totalGpuMs = 0.0f;
            f64 totalRays = 0.0;

            // ================================================================
            // SPP Loop: Accumulative Sampling
            // ================================================================
            // Each iteration traces rays with a different random seed and
            // subpixel jitter, accumulating results in the output image.
            // This implements progressive refinement for anti-aliasing and
            // Monte Carlo convergence.
            // ================================================================

            for (u32 sampleIndex = 0; sampleIndex < spp; ++sampleIndex) {
                // Generate unique random seed for this sample
                // Uses system random to ensure different patterns per sample
                u32 randomSeed = static_cast<u32>(std::rand()) ^ (frameIndex * 997 + sampleIndex * 1009);

                // Update sampling parameters in pipeline
                pipeline.SetSamplingParams(frameIndex, sampleIndex, spp, randomSeed);

                QL_LOG_INFO("  [SPP {}/{}] Tracing rays (seed: {})...", sampleIndex + 1, spp, randomSeed);

                // Execute ray tracing for this sample
                CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
                    // Begin performance timing
                    perfLogger.BeginFrame(cmd);

                    // Execute ray tracing
                    pipeline.TraceRays(cmd, width, height);

                    // End performance timing
                    perfLogger.EndFrame(cmd);
                });

                // Accumulate performance metrics
                totalGpuMs += perfLogger.GetLastFrameGpuMs();
                totalRays += perfLogger.GetLastFrameRaysPerSec();

                QL_LOG_INFO("  [SPP {}/{}] Completed - GPU: {:.2f} ms, Progress: {:.1f}%",
                            sampleIndex + 1, spp,
                            perfLogger.GetLastFrameGpuMs(),
                            100.0f * (sampleIndex + 1) / spp);
            }

            // Log aggregated performance metrics
            perfLogger.LogFrame(0, width, height, spp, wavelength_nm, spectralModeStr);
            perfLogger.Flush();

            QL_LOG_INFO("  All samples completed!");
            QL_LOG_INFO("  Total GPU time: {:.2f} ms ({:.2f} ms/sample)",
                        totalGpuMs, totalGpuMs / spp);
            QL_LOG_INFO("  Average throughput: {:.2f} Mrays/s",
                        (totalRays / spp) / 1e6);

        } catch (const std::exception& e) {
            QL_LOG_ERROR("  [DEBUG] GPU execution FAILED: {}", e.what());
            throw;
        }

        QL_LOG_INFO("  Frame rendered ({}x{}) with {} spp - Total GPU: {:.2f} ms",
                    width, height, spp,
                    perfLogger.GetLastFrameGpuMs() * spp);

        // ====================================================================
        // Readback and Save
        // ====================================================================
        QL_LOG_INFO("Reading back and saving image...");
        QL_LOG_INFO("  [DEBUG] Starting image readback...");

        std::vector<f32> pixels = CommandHelper::ReadbackImage(
            context,
            outputImage.GetImage(),
            outputImage.GetFormat(),
            width,
            height
        );

        // Convert to Image object (4 channels: RGBA)
        Image img(width, height, 4);
        img.channelNames = {"R", "G", "B", "A"};
        img.metadata["renderer"] = "Quantiloom Spectral";
        img.metadata["mode"] = spectralModeStr;
        if (spectral_mode == SpectralMode::Single ||
            spectral_mode == SpectralMode::MWIR_Fused ||
            spectral_mode == SpectralMode::LWIR_Fused) {
            img.metadata["wavelength_nm"] = std::to_string(wavelength_nm);
            img.metadata["quality_level"] = "PREVIEW_ONLY";
            img.metadata["warning"] = "RGB-averaged spectral albedo, not quantitative";
            img.metadata["note"] = "For quantitative results use mode=hs_off with measured spectra";
        } else if (spectral_mode == SpectralMode::RGB) {
            img.metadata["quality_level"] = "PREVIEW";
            img.metadata["note"] = "RGB rendering, preview quality";
        }
        img.metadata["resolution"] = std::to_string(width) + "x" + std::to_string(height);
        img.metadata["spp"] = std::to_string(spp);

        // Copy pixel data
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                u32 pixelIndex = (y * width + x) * 4;
                img(x, y, 0) = pixels[pixelIndex + 0];  // R
                img(x, y, 1) = pixels[pixelIndex + 1];  // G
                img(x, y, 2) = pixels[pixelIndex + 2];  // B
                img(x, y, 3) = pixels[pixelIndex + 3];  // A
            }
        }

        // Save as EXR
        if (ImageIO::WriteEXR(outputPath, img)) {
            QL_LOG_INFO("  [OK] Saved spectral image to {}", outputPath);
        } else {
            QL_LOG_ERROR("  [FAIL] Failed to save image to {}", outputPath);
        }

        // ====================================================================
        // Success
        // ====================================================================
        QL_LOG_INFO("========================================");
        QL_LOG_INFO("  Rendering COMPLETED");
        QL_LOG_INFO("========================================");
        QL_LOG_INFO("  Spectral mode: {}", spectralModeStr);
        if (spectral_mode == SpectralMode::Single ||
            spectral_mode == SpectralMode::MWIR_Fused ||
            spectral_mode == SpectralMode::LWIR_Fused) {
            QL_LOG_INFO("  Wavelength: {:.1f} nm", wavelength_nm);
        }
        QL_LOG_INFO("  Output: {}", outputPath);
        QL_LOG_INFO("========================================");

    } catch (const std::exception& e) {
        QL_LOG_ERROR("FATAL ERROR: {}", e.what());
        Log::Shutdown();
        return 1;
    }

    Log::Shutdown();
    return 0;
}
