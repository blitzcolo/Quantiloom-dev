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
#include "scene/Mesh.hpp"
#include "scene/Material.hpp"
#include "scene/Camera.hpp"
#include "SceneBuilder.hpp"

#include <glm/glm.hpp>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <cstddef>  // For offsetof

using namespace quantiloom;

// ============================================================================
// LUT Data Structure (matches shader LUTData structure)
// ============================================================================

struct LUTData {
    glm::vec3 sunDirection;        // FROM surface TO sun (normalized)
    f32 sunRadiance_spectral;       // Spectral radiance at current λ (W·sr⁻¹·m⁻²·nm⁻¹)
    f32 skyRadiance_spectral;       // Spectral radiance at current λ (W·sr⁻¹·m⁻²·nm⁻¹)
    f32 _pad0;
    f32 _pad1;
    f32 _pad2;
};

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
    f32 _pad0;                           // offset 68, size 4
};  // Total: 72 bytes (must match GPU MaterialData in common.hlsli)

// Verify struct layout matches shader expectations
// If this fails, the CPU/GPU struct layouts are mismatched, which WILL cause GPU crashes
static_assert(sizeof(MaterialDataCPU) == 72, "MaterialDataCPU size mismatch! Expected 72 bytes to match GPU MaterialData struct");
static_assert(offsetof(MaterialDataCPU, baseColorTextureIndex) == 16, "baseColorTextureIndex offset mismatch");
static_assert(offsetof(MaterialDataCPU, normalTextureIndex) == 32, "normalTextureIndex offset mismatch");
static_assert(offsetof(MaterialDataCPU, emissiveFactor) == 40, "emissiveFactor offset mismatch");
static_assert(offsetof(MaterialDataCPU, emissiveTextureIndex) == 52, "emissiveTextureIndex offset mismatch");

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
        String spectralMode = config.Get<String>("spectral.mode", "single_wavelength");
        f32 wavelength_nm = config.Get<f32>("spectral.wavelength_nm", 550.0f);

        QL_LOG_INFO("  Spectral mode: {}", spectralMode);
        QL_LOG_INFO("  Wavelength: {:.1f} nm", wavelength_nm);

        if (spectralMode != "single_wavelength") {
            QL_LOG_ERROR("Only 'single_wavelength' mode is supported in this version");
            return 1;
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

        QL_LOG_INFO("  Sun direction: [{:.2f}, {:.2f}, {:.2f}]",
                    sunDirection.x, sunDirection.y, sunDirection.z);
        QL_LOG_INFO("  Sun radiance: [{:.2f}, {:.2f}, {:.2f}]",
                    sunRadiance.x, sunRadiance.y, sunRadiance.z);
        QL_LOG_INFO("  Sky radiance: [{:.2f}, {:.2f}, {:.2f}]",
                    skyRadiance.x, skyRadiance.y, skyRadiance.z);

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
        // Create LUT Buffer (Spectral)
        // ====================================================================
        QL_LOG_INFO("Creating spectral LUT buffer...");

        // Convert RGB radiance to spectral radiance (average of RGB channels)
        // For single-wavelength mode, we approximate spectral radiance from RGB config
        f32 sunRadiance_spectral = (sunRadiance.r + sunRadiance.g + sunRadiance.b) / 3.0f;
        f32 skyRadiance_spectral = (skyRadiance.r + skyRadiance.g + skyRadiance.b) / 3.0f;

        QL_LOG_INFO("  Sun spectral radiance: {:.3f} W·sr⁻¹·m⁻²·nm⁻¹", sunRadiance_spectral);
        QL_LOG_INFO("  Sky spectral radiance: {:.3f} W·sr⁻¹·m⁻²·nm⁻¹", skyRadiance_spectral);

        LUTData lutData;
        lutData.sunDirection = sunDirection;
        lutData.sunRadiance_spectral = sunRadiance_spectral;
        lutData.skyRadiance_spectral = skyRadiance_spectral;

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
            cpuMat._pad0 = 0.0f;

            materialData.push_back(cpuMat);

            QL_LOG_INFO("  Material '{}': base=[{:.2f},{:.2f},{:.2f},{:.2f}] metal={:.2f} rough={:.2f}",
                        mat.name,
                        mat.baseColorFactor.r, mat.baseColorFactor.g, mat.baseColorFactor.b, mat.baseColorFactor.a,
                        mat.metallicFactor, mat.roughnessFactor);
        }

        GpuBuffer materialBuffer(
            context.GetAllocator(),
            materialData.size() * sizeof(MaterialDataCPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );

        materialBuffer.Upload(materialData.data(), materialData.size() * sizeof(MaterialDataCPU));

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
            pipeline.BindGeometryBuffers(blasList[0].GetVertexBuffer(), blasList[0].GetIndexBuffer()); // Binding 3, 4
        }

        pipeline.BindMaterialBuffer(materialBuffer);                    // Binding 5

        // Bind textures (bindless arrays)
        pipeline.BindTextures(textureManager.GetImageViews(), textureManager.GetSamplers()); // Binding 6, 7

        // Set camera parameters (with spectral wavelength)
        CameraData cameraData = camera.GetCameraData();
        cameraData.wavelength_nm = wavelength_nm;  // Override with config wavelength
        pipeline.SetCameraData(cameraData);

        QL_LOG_INFO("  Pipeline created and resources bound");

        // ====================================================================
        // Render Frame
        // ====================================================================
        QL_LOG_INFO("Rendering frame at wavelength {:.1f} nm...", wavelength_nm);
        QL_LOG_INFO("  [DEBUG] Starting TraceRays command submission...");

        try {
            CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
                QL_LOG_INFO("  [DEBUG] Recording TraceRays commands...");
                pipeline.TraceRays(cmd, width, height);
                QL_LOG_INFO("  [DEBUG] TraceRays commands recorded successfully");
            });
            QL_LOG_INFO("  [DEBUG] GPU execution completed successfully");
        } catch (const std::exception& e) {
            QL_LOG_ERROR("  [DEBUG] GPU execution FAILED: {}", e.what());
            throw;
        }

        QL_LOG_INFO("  Frame rendered ({}x{})", width, height);

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
        img.metadata["mode"] = spectralMode;
        img.metadata["wavelength_nm"] = std::to_string(wavelength_nm);
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
        QL_LOG_INFO("  Spectral mode: {}", spectralMode);
        QL_LOG_INFO("  Wavelength: {:.1f} nm", wavelength_nm);
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
