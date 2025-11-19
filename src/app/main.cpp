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
#include "renderer/VulkanContext.hpp"
#include "renderer/RayTracingPipeline.hpp"
#include "renderer/AccelerationStructure.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/CommandHelper.hpp"
#include "scene/Mesh.hpp"
#include "scene/Camera.hpp"
#include "SceneBuilder.hpp"

#include <glm/glm.hpp>
#include <iostream>
#include <filesystem>
#include <stdexcept>

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

struct MaterialDataCPU {
    f32 albedo_spectral;  // Spectral reflectance at current λ [0, 1]
    f32 _pad0;
    f32 _pad1;
    f32 _pad2;
};

// ============================================================================
// Scene Loading Helper
// ============================================================================

Mesh LoadSceneFromConfig(const Config& config) {
    // Check if preset is specified
    if (config.Has("scene.preset")) {
        String preset = config.Get<String>("scene.preset", "cornell_box");

        QL_LOG_INFO("Loading built-in scene preset: {}", preset);

        if (preset == "cornell_box") {
            return TestScenes::CreateCornellBoxScene();
        } else if (preset == "multi_object") {
            return TestScenes::CreateMultiObjectScene();
        } else if (preset == "lighting_test") {
            return TestScenes::CreateLightingTestScene();
        } else {
            QL_LOG_WARN("Unknown scene preset '{}', defaulting to cornell_box", preset);
            return TestScenes::CreateCornellBoxScene();
        }
    }

    // M2+: Load from external OBJ file
    if (config.Has("scene.geometry")) {
        String geometryPath = config.Get<String>("scene.geometry");
        QL_LOG_ERROR("External geometry loading not yet implemented (M2)");
        QL_LOG_INFO("Falling back to cornell_box preset");
        return TestScenes::CreateCornellBoxScene();
    }

    // Default: Cornell box
    QL_LOG_WARN("No scene specified in config, using cornell_box preset");
    return TestScenes::CreateCornellBoxScene();
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
            QL_LOG_ERROR("renderer.resolution must be array of 2 integers");
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
            QL_LOG_ERROR("lighting.sun_direction must be array of 3 floats");
            return 1;
        }
        glm::vec3 sunDirection = glm::normalize(glm::vec3(sunDirArray[0], sunDirArray[1], sunDirArray[2]));

        auto sunRadArray = config.GetArray<f32>("lighting.sun_radiance");
        if (sunRadArray.size() != 3) {
            QL_LOG_ERROR("lighting.sun_radiance must be array of 3 floats");
            return 1;
        }
        glm::vec3 sunRadiance(sunRadArray[0], sunRadArray[1], sunRadArray[2]);

        auto skyRadArray = config.GetArray<f32>("lighting.sky_radiance");
        if (skyRadArray.size() != 3) {
            QL_LOG_ERROR("lighting.sky_radiance must be array of 3 floats");
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
            QL_LOG_ERROR("material.albedo must be array of 3 floats");
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
        QL_LOG_INFO("Loading scene geometry...");
        Mesh sceneMesh = LoadSceneFromConfig(config);
        QL_LOG_INFO("  Mesh: {} vertices, {} triangles",
                    sceneMesh.positions.size(), sceneMesh.indices.size() / 3);

        // ====================================================================
        // Build Acceleration Structures
        // ====================================================================
        QL_LOG_INFO("Building acceleration structures...");

        BLAS blas(context, sceneMesh);
        TLAS tlas(context);

        CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
            blas.Build(cmd);
            tlas.AddInstance(blas);
            tlas.Build(cmd);
        });

        QL_LOG_INFO("  BLAS device address: 0x{:x}", blas.GetDeviceAddress());
        QL_LOG_INFO("  TLAS built with 1 instance");

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
        // Create Material Buffer (Spectral)
        // ====================================================================
        QL_LOG_INFO("Creating spectral material buffer...");

        // Convert RGB albedo to spectral albedo (average of RGB channels)
        // For single-wavelength mode, we approximate spectral reflectance from RGB config
        f32 albedo_spectral = (albedo.r + albedo.g + albedo.b) / 3.0f;

        QL_LOG_INFO("  Material spectral albedo: {:.3f}", albedo_spectral);

        MaterialDataCPU defaultMaterial;
        defaultMaterial.albedo_spectral = albedo_spectral;

        GpuBuffer materialBuffer(
            context.GetAllocator(),
            sizeof(MaterialDataCPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );

        materialBuffer.Upload(&defaultMaterial, sizeof(MaterialDataCPU));

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

        // Bind resources
        pipeline.BindOutputImage(outputImage);
        pipeline.BindAccelerationStructure(tlas.GetHandle());
        pipeline.BindLUTBuffer(lutBuffer);
        pipeline.BindMaterialBuffer(materialBuffer);
        pipeline.BindGeometryBuffers(blas.GetVertexBuffer(), blas.GetIndexBuffer());

        // Set camera parameters (with spectral wavelength)
        CameraData cameraData = camera.GetCameraData();
        cameraData.wavelength_nm = wavelength_nm;  // Override with config wavelength
        pipeline.SetCameraData(cameraData);

        QL_LOG_INFO("  Pipeline created and resources bound");

        // ====================================================================
        // Render Frame
        // ====================================================================
        QL_LOG_INFO("Rendering frame at wavelength {:.1f} nm...", wavelength_nm);

        CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
            pipeline.TraceRays(cmd, width, height);
        });

        QL_LOG_INFO("  Frame rendered ({}x{})", width, height);

        // ====================================================================
        // Readback and Save
        // ====================================================================
        QL_LOG_INFO("Reading back and saving image...");

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
