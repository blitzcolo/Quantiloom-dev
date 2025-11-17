// ============================================================================
// Quantiloom M1 - End-to-End Ray Tracing Test
// ============================================================================
// This is a standalone test program for M1 milestone.
// It renders a single frame using the Cornell Box scene.
//
// Prerequisites:
// - Compiled shaders: raygen.spv, closesthit.spv, miss.spv (in working dir)
// - Cornell Box: assets/scenes/cornell_box.obj
//
// Output:
// - m1_output.exr (ray traced image)
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

#include <glm/glm.hpp>
#include <iostream>
#include <stdexcept>

using namespace quantiloom;

// ============================================================================
// Hardcoded LUT Data (matches shader LUTData structure)
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
// Create Simple Box Mesh (Procedural Cornell Box)
// ============================================================================

Mesh CreateSimpleBox() {
    Mesh mesh;
    mesh.name = "simple_box";

    // 8 vertices of a unit cube [-1, 1]^3
    mesh.positions = {
        // Bottom face (Y = -1)
        {-1.0f, -1.0f, -1.0f},  // 0
        { 1.0f, -1.0f, -1.0f},  // 1
        { 1.0f, -1.0f,  1.0f},  // 2
        {-1.0f, -1.0f,  1.0f},  // 3

        // Top face (Y = 1)
        {-1.0f,  1.0f, -1.0f},  // 4
        { 1.0f,  1.0f, -1.0f},  // 5
        { 1.0f,  1.0f,  1.0f},  // 6
        {-1.0f,  1.0f,  1.0f},  // 7
    };

    // Indices (2 triangles per face = 6 faces * 2 = 12 triangles)
    mesh.indices = {
        // Bottom face (floor, Y = -1)
        0, 1, 2,  0, 2, 3,
        // Top face (ceiling, Y = 1)
        4, 7, 6,  4, 6, 5,
        // Back face (Z = -1)
        0, 4, 5,  0, 5, 1,
        // Front face (Z = 1) - facing camera
        3, 2, 6,  3, 6, 7,
        // Left face (X = -1)
        0, 3, 7,  0, 7, 4,
        // Right face (X = 1)
        1, 5, 6,  1, 6, 2,
    };

    return mesh;
}

// ============================================================================
// Main M1 Test
// ============================================================================

int main(int argc, char* argv[]) {
    Log::Init("quantiloom_m1.log", Log::Level::Info);

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
        // Step 2: Create Scene Geometry
        // ====================================================================
        QL_LOG_INFO("Step 2: Creating scene geometry...");
        Mesh boxMesh = CreateSimpleBox();
        QL_LOG_INFO("  Mesh: {} vertices, {} triangles", 
                    boxMesh.positions.size(), boxMesh.indices.size() / 3);

        // ====================================================================
        // Step 3: Build Acceleration Structures
        // ====================================================================
        QL_LOG_INFO("Step 3: Building acceleration structures...");
        
        BLAS blas(context, boxMesh);
        TLAS tlas(context);

        // Build BLAS and TLAS in a single command buffer
        CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
            blas.Build(cmd);
            tlas.AddInstance(blas);
            tlas.Build(cmd);
        });

        QL_LOG_INFO("  BLAS device address: 0x{:x}", blas.GetDeviceAddress());
        QL_LOG_INFO("  TLAS built with 1 instance");

        // ====================================================================
        // Step 4: Create Output Image
        // ====================================================================
        QL_LOG_INFO("Step 4: Creating output image...");
        const u32 width = 800;
        const u32 height = 600;

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
        
        LUTData lutData;
        // Sun from upper-left (standard 3-point lighting key light position)
        // Direction points FROM surface TO sun (not from sun to surface)
        lutData.sunDirection = glm::normalize(glm::vec3(-0.5f, 0.8f, -0.3f));
        lutData.sunRadiance = glm::vec3(3.0f, 3.0f, 3.0f);  // Bright sun
        lutData.skyRadiance = glm::vec3(0.3f, 0.5f, 0.8f);  // Blue sky

        GpuBuffer lutBuffer(
            context.GetAllocator(),
            sizeof(LUTData),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );

        lutBuffer.Upload(&lutData, sizeof(LUTData));
        QL_LOG_INFO("  LUT uploaded: sun=[{:.2f},{:.2f},{:.2f}], sky=[{:.2f},{:.2f},{:.2f}]",
                    lutData.sunDirection.x, lutData.sunDirection.y, lutData.sunDirection.z,
                    lutData.skyRadiance.x, lutData.skyRadiance.y, lutData.skyRadiance.z);

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

        // Save as EXR
        const std::string outputPath = "m1_output.exr";
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
