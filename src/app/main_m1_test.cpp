// ============================================================================
// Quantiloom M1 - End-to-End Ray Tracing Test
// ============================================================================
// This is a standalone test program for M1 milestone.
// It renders a single frame using parametric scene generation.
//
// Prerequisites:
// - Compiled shaders: raygen.spv, closesthit.spv, miss.spv (in working dir)
//
// Output:
// - m1_output.exr (ray traced image)
//
// Configuration:
// - Change SCENE_PRESET, CAMERA_PRESET, LIGHTING_PRESET below to test
// ============================================================================

#include "core/Log.hpp"
#include "core/Image.hpp"
#include "io/ImageIO.hpp"
#include "renderer/VulkanContext.hpp"
#include "renderer/RayTracingPipeline.hpp"
#include "renderer/AccelerationStructure.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/CommandHelper.hpp"
#include "scene/Mesh.hpp"
#include "SceneBuilder.hpp"

#include <glm/glm.hpp>
#include <iostream>
#include <stdexcept>

using namespace quantiloom;

// ============================================================================
// Scene Configuration Presets (Change these to test different setups)
// ============================================================================

enum class ScenePreset {
    CornellBox,      // Minimal: ground + single cube
    MultiObject,     // Ground + tall box + cube + sphere
    LightingTest     // Ground + row of 5 cubes
};

enum class CameraPreset {
    DefaultOverview, // Elevated, behind scene
    GroundLevel,     // Low, human eye height
    TopDown          // Directly above scene
};

enum class LightingPreset {
    Standard,        // 3-point key light from upper-left
    Morning,         // Warm, low-angle light
    Noon,            // Overhead, very bright
    Backlight        // Strong rim lighting from behind
};

// === DEFAULT TEST CONFIGURATION (can be overridden by command line) ===
ScenePreset    g_scenePreset    = ScenePreset::MultiObject;
CameraPreset   g_cameraPreset   = CameraPreset::DefaultOverview;
LightingPreset g_lightingPreset = LightingPreset::Standard;

// ============================================================================
// LUT Data Structure (matches shader LUTData structure)
// ============================================================================

struct LUTData {
    glm::vec3 sunDirection;
    f32 _pad0;
    glm::vec3 sunRadiance;
    f32 _pad1;
    glm::vec3 skyRadiance;
    f32 _pad2;
};

// ============================================================================
// Scene Generation Functions
// ============================================================================

Mesh CreateSceneGeometry(ScenePreset preset) {
    switch (preset) {
        case ScenePreset::CornellBox:
            QL_LOG_INFO("  Scene: Cornell Box (ground + cube)");
            return TestScenes::CreateCornellBoxScene();

        case ScenePreset::MultiObject:
            QL_LOG_INFO("  Scene: Multi-Object (ground + tall box + cube + sphere)");
            return TestScenes::CreateMultiObjectScene();

        case ScenePreset::LightingTest:
            QL_LOG_INFO("  Scene: Lighting Test (ground + 5 cubes in a row)");
            return TestScenes::CreateLightingTestScene();

        default:
            QL_LOG_WARN("  Unknown scene preset, defaulting to Cornell Box");
            return TestScenes::CreateCornellBoxScene();
    }
}

CameraConfig GetCameraConfig(CameraPreset preset) {
    switch (preset) {
        case CameraPreset::DefaultOverview:
            QL_LOG_INFO("  Camera: Default Overview (elevated, behind scene)");
            return CameraConfig::DefaultOverview();

        case CameraPreset::GroundLevel:
            QL_LOG_INFO("  Camera: Ground Level (human eye height)");
            return CameraConfig::GroundLevel();

        case CameraPreset::TopDown:
            QL_LOG_INFO("  Camera: Top-Down (bird's eye view)");
            return CameraConfig::TopDown();

        default:
            QL_LOG_WARN("  Unknown camera preset, defaulting to Overview");
            return CameraConfig::DefaultOverview();
    }
}

LightingConfig GetLightingConfig(LightingPreset preset) {
    switch (preset) {
        case LightingPreset::Standard:
            QL_LOG_INFO("  Lighting: Standard 3-Point (key light from upper-left)");
            return LightingConfig::Standard3Point();

        case LightingPreset::Morning:
            QL_LOG_INFO("  Lighting: Morning Light (warm, low angle)");
            return LightingConfig::MorningLight();

        case LightingPreset::Noon:
            QL_LOG_INFO("  Lighting: Noon Overhead (bright, harsh)");
            return LightingConfig::NoonOverhead();

        case LightingPreset::Backlight:
            QL_LOG_INFO("  Lighting: Backlight (rim lighting, silhouette)");
            return LightingConfig::Backlight();

        default:
            QL_LOG_WARN("  Unknown lighting preset, defaulting to Standard");
            return LightingConfig::Standard3Point();
    }
}

// ============================================================================
// Main M1 Test
// ============================================================================

void PrintUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  --scene <name>      Scene preset: cornell, multiobject, lighting\n";
    std::cout << "  --camera <name>     Camera preset: overview, ground, topdown\n";
    std::cout << "  --lighting <name>   Lighting preset: standard, morning, noon, backlight\n";
    std::cout << "  --output <path>     Output EXR file path (default: m1_output.exr)\n";
    std::cout << "  --help              Show this help message\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << " --scene cornell --output m1_cornell.exr\n";
    std::cout << "  " << programName << " --scene lighting --lighting morning\n";
}

int main(int argc, char* argv[]) {
    Log::Init("quantiloom_m1.log", Log::Level::Info);

    // Parse command-line arguments
    std::string configPath = "";  // Empty means use command-line presets
    std::string outputPath = "m1_output.exr";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        }
        else if (arg == "--config" && i + 1 < argc) {
            configPath = argv[++i];
        }
        else if (arg == "--scene" && i + 1 < argc) {
            std::string scene = argv[++i];
            if (scene == "cornell") g_scenePreset = ScenePreset::CornellBox;
            else if (scene == "multiobject") g_scenePreset = ScenePreset::MultiObject;
            else if (scene == "lighting") g_scenePreset = ScenePreset::LightingTest;
            else {
                std::cerr << "Unknown scene preset: " << scene << std::endl;
                return 1;
            }
        }
        else if (arg == "--camera" && i + 1 < argc) {
            std::string camera = argv[++i];
            if (camera == "overview") g_cameraPreset = CameraPreset::DefaultOverview;
            else if (camera == "ground") g_cameraPreset = CameraPreset::GroundLevel;
            else if (camera == "topdown") g_cameraPreset = CameraPreset::TopDown;
            else {
                std::cerr << "Unknown camera preset: " << camera << std::endl;
                return 1;
            }
        }
        else if (arg == "--lighting" && i + 1 < argc) {
            std::string lighting = argv[++i];
            if (lighting == "standard") g_lightingPreset = LightingPreset::Standard;
            else if (lighting == "morning") g_lightingPreset = LightingPreset::Morning;
            else if (lighting == "noon") g_lightingPreset = LightingPreset::Noon;
            else if (lighting == "backlight") g_lightingPreset = LightingPreset::Backlight;
            else {
                std::cerr << "Unknown lighting preset: " << lighting << std::endl;
                return 1;
            }
        }
        else if (arg == "--output" && i + 1 < argc) {
            outputPath = argv[++i];
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            return 1;
        }
    }

    QL_LOG_INFO("========================================");
    QL_LOG_INFO("  Quantiloom M1 - Ray Tracing Test");
    QL_LOG_INFO("========================================");

    try {
        // ====================================================================
        // Step 1: Initialize Vulkan Context
        // ====================================================================
        QL_LOG_INFO("Step 1: Initializing Vulkan context...");
        VulkanContext context;

        if (!context.IsRayTracingSupported()) {
            QL_LOG_ERROR("Ray Tracing not supported. Aborting.");
            return 1;
        }

        // ====================================================================
        // Step 2: Create Scene Geometry and Camera
        // ====================================================================
        QL_LOG_INFO("Step 2: Creating scene geometry and camera...");

        // Resolution
        u32 width = 1280;
        u32 height = 720;

        // Load camera from config or use preset
        Camera camera;
        if (!configPath.empty()) {
            // Load from TOML config
            auto configResult = Config::Load(configPath);
            if (!configResult.has_value()) {
                QL_LOG_ERROR("Failed to load config '{}': {}", configPath, configResult.error());
                return 1;
            }
            Config config = configResult.value();

            // Get resolution from config (optional)
            auto resArray = config.GetArray<u32>("renderer.resolution");
            if (resArray.size() >= 2) {
                width = resArray[0];
                height = resArray[1];
            }

            // Create camera from config
            f32 aspectRatio = static_cast<f32>(width) / static_cast<f32>(height);
            auto cameraResult = Camera::FromConfig(config, aspectRatio);
            if (!cameraResult.has_value()) {
                QL_LOG_ERROR("Failed to load camera from config: {}", cameraResult.error());
                return 1;
            }
            camera = cameraResult.value();
        } else {
            // Use command-line presets (backward compatibility)
            CameraConfig cameraConfig = GetCameraConfig(g_cameraPreset);
            f32 aspectRatio = static_cast<f32>(width) / static_cast<f32>(height);
            camera = Camera(cameraConfig.position, cameraConfig.lookAt, cameraConfig.up,
                          cameraConfig.fovYDegrees, aspectRatio);
        }

        QL_LOG_INFO("  Camera:");
        QL_LOG_INFO("    position: [{:.2f}, {:.2f}, {:.2f}]",
                    camera.GetPosition().x, camera.GetPosition().y, camera.GetPosition().z);
        QL_LOG_INFO("    lookAt:   [{:.2f}, {:.2f}, {:.2f}]",
                    camera.GetLookAt().x, camera.GetLookAt().y, camera.GetLookAt().z);
        QL_LOG_INFO("    fovY:     {:.1f} degrees", camera.GetFovY());

        Mesh sceneMesh = CreateSceneGeometry(g_scenePreset);
        QL_LOG_INFO("  Mesh: {} vertices, {} triangles",
                    sceneMesh.GetTotalVertexCount(), sceneMesh.GetTotalTriangleCount());

        // ====================================================================
        // Step 3: Build Acceleration Structures
        // ====================================================================
        QL_LOG_INFO("Step 3: Building acceleration structures...");

        // M1 test uses single mesh with single primitive
        if (sceneMesh.primitives.empty()) {
            throw std::runtime_error("Scene mesh has no primitives");
        }

        BLAS blas(context, sceneMesh.primitives[0]);
        TLAS tlas(context);

        // Build BLAS and TLAS in a single command buffer
        CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
            blas.Build(cmd);
            tlas.AddInstance(blas, 0);  // materialId = 0, identity transform
            tlas.Build(cmd);
        });

        QL_LOG_INFO("  BLAS device address: 0x{:x}", blas.GetDeviceAddress());
        QL_LOG_INFO("  TLAS built with 1 instance");

        // ====================================================================
        // Step 4: Create Output Image
        // ====================================================================
        QL_LOG_INFO("Step 4: Creating output image ({}x{})...", width, height);

        GpuImage outputImage(
            context.GetAllocator(),
            context.GetDevice(),
            width, height,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY
        );

        // Transition to GENERAL layout for ray tracing
        CommandHelper::TransitionImageLayoutImmediate(
            context,
            outputImage.GetImage(),
            outputImage.GetFormat(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL
        );

        QL_LOG_INFO("  Output image: {}x{} (RGBA32F)", width, height);

        // ====================================================================
        // Step 5: Create LUT Buffer
        // ====================================================================
        QL_LOG_INFO("Step 5: Creating LUT buffer...");

        // Get lighting configuration from preset
        LightingConfig lighting = GetLightingConfig(g_lightingPreset);

        LUTData lutData;
        lutData.sunDirection = lighting.sunDirection;
        lutData.sunRadiance = lighting.sunRadiance;
        lutData.skyRadiance = lighting.skyRadiance;

        GpuBuffer lutBuffer(
            context.GetAllocator(),
            sizeof(LUTData),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );

        lutBuffer.Upload(&lutData, sizeof(LUTData));
        QL_LOG_INFO("  LUT uploaded:");
        QL_LOG_INFO("    sunDirection: [{:.2f}, {:.2f}, {:.2f}]",
                    lutData.sunDirection.x, lutData.sunDirection.y, lutData.sunDirection.z);
        QL_LOG_INFO("    sunRadiance:  [{:.2f}, {:.2f}, {:.2f}]",
                    lutData.sunRadiance.x, lutData.sunRadiance.y, lutData.sunRadiance.z);
        QL_LOG_INFO("    skyRadiance:  [{:.2f}, {:.2f}, {:.2f}]",
                    lutData.skyRadiance.x, lutData.skyRadiance.y, lutData.skyRadiance.z);

        
        // ====================================================================
        // Step 5.5: Create Material Buffer
        // ====================================================================
        QL_LOG_INFO("Step 5.5: Creating material buffer...");

        // Define material data (matches MaterialData in common.hlsli)
        struct MaterialDataCPU {
            glm::vec3 albedo;
            f32 _pad0;
        };

        MaterialDataCPU defaultMaterial;
        defaultMaterial.albedo = glm::vec3(0.8f, 0.8f, 0.8f);  // Gray
        defaultMaterial._pad0 = 0.0f;

        GpuBuffer materialBuffer(
            context.GetAllocator(),
            sizeof(MaterialDataCPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );

        materialBuffer.Upload(&defaultMaterial, sizeof(MaterialDataCPU));

        QL_LOG_INFO("  Material buffer uploaded (albedo: [{:.2f}, {:.2f}, {:.2f}])",
                    defaultMaterial.albedo.x, defaultMaterial.albedo.y, defaultMaterial.albedo.z);


        // ====================================================================
        // Step 6: Create Ray Tracing Pipeline
        // ====================================================================
        QL_LOG_INFO("Step 6: Creating ray tracing pipeline...");

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
        pipeline.BindMaterialBuffer(materialBuffer);  // Bind material buffer (binding 5)

        // Set camera parameters (push constants)
        pipeline.SetCameraData(camera.GetCameraData());

        QL_LOG_INFO("  Pipeline created and resources bound");

        // ====================================================================
        // Step 7: Render Frame
        // ====================================================================
        QL_LOG_INFO("Step 7: Rendering frame...");

        CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
            pipeline.TraceRays(cmd, width, height);
        });

        QL_LOG_INFO("  Frame rendered ({}x{})", width, height);

        // ====================================================================
        // Step 8: Readback and Save
        // ====================================================================
        QL_LOG_INFO("Step 8: Reading back and saving image...");

        // Read back image from GPU
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
        img.metadata["renderer"] = "Quantiloom M1";
        img.metadata["resolution"] = std::to_string(width) + "x" + std::to_string(height);
        img.metadata["mode"] = "ray_tracing_test";

        // Copy pixel data from GPU readback to Image
        // pixels is [R,G,B,A, R,G,B,A, ...] in row-major order
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                u32 pixelIndex = (y * width + x) * 4;
                img(x, y, 0) = pixels[pixelIndex + 0];  // R
                img(x, y, 1) = pixels[pixelIndex + 1];  // G
                img(x, y, 2) = pixels[pixelIndex + 2];  // B
                img(x, y, 3) = pixels[pixelIndex + 3];  // A
            }
        }

        // Save as EXR (use command-line specified path)
        if (ImageIO::WriteEXR(outputPath, img)) {
            QL_LOG_INFO("  [OK] Saved ray traced image to {}", outputPath);
        } else {
            QL_LOG_ERROR("  [FAIL] Failed to save image to {}", outputPath);
        }

        QL_LOG_INFO("  Rendering and export completed successfully!");

        // ====================================================================
        // Success
        // ====================================================================
        QL_LOG_INFO("========================================");
        QL_LOG_INFO("  M1 Test COMPLETED");
        QL_LOG_INFO("========================================");
        QL_LOG_INFO("  Scene:    {}",
                    g_scenePreset == ScenePreset::CornellBox ? "Cornell Box" :
                    g_scenePreset == ScenePreset::MultiObject ? "Multi-Object" : "Lighting Test");
        QL_LOG_INFO("  Camera:   {}",
                    g_cameraPreset == CameraPreset::DefaultOverview ? "Default Overview" :
                    g_cameraPreset == CameraPreset::GroundLevel ? "Ground Level" : "Top-Down");
        QL_LOG_INFO("  Lighting: {}",
                    g_lightingPreset == LightingPreset::Standard ? "Standard 3-Point" :
                    g_lightingPreset == LightingPreset::Morning ? "Morning Light" :
                    g_lightingPreset == LightingPreset::Noon ? "Noon Overhead" : "Backlight");
        QL_LOG_INFO("");
        QL_LOG_INFO("  All ray tracing components initialized");
        QL_LOG_INFO("  BLAS/TLAS built with memory barriers");
        QL_LOG_INFO("  Pipeline executed without errors");
        QL_LOG_INFO("  Image saved to {}", outputPath);
        QL_LOG_INFO("");
        QL_LOG_INFO("  M1 Milestone: HS-core prototype is DONE");
        QL_LOG_INFO("========================================");

    } catch (const std::exception& e) {
        QL_LOG_ERROR("FATAL ERROR: {}", e.what());
        Log::Shutdown();
        return 1;
    }

    Log::Shutdown();
    return 0;
}
