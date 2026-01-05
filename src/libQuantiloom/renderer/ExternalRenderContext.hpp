/**
 * @file ExternalRenderContext.hpp
 * @brief External Vulkan context injection API for Qt6 GUI integration
 *
 * Provides ExternalRenderContext class for rendering to externally-managed
 * Vulkan surfaces (e.g., QVulkanWindow). This enables Qt6 GUI applications
 * to use libQuantiloom's ray tracing without creating their own VulkanContext.
 *
 * Key features:
 * - Accepts external VkDevice, VkQueue, VkInstance handles
 * - Uses VK_KHR_dynamic_rendering for RenderPass-free integration
 * - Records commands to external VkCommandBuffer
 * - Supports real-time parameter updates for interactive editing
 *
 * Usage pattern (Qt6 QVulkanWindowRenderer):
 * @code
 * // In initResources():
 * ExternalRenderContext::InitParams params{};
 * params.instance = vulkanInstance->vkInstance();
 * params.physicalDevice = window->physicalDevice();
 * params.device = window->device();
 * params.graphicsQueue = window->graphicsQueue();
 * params.graphicsQueueFamily = window->graphicsQueueFamilyIndex();
 * params.targetColorFormat = window->colorFormat();
 *
 * auto result = ExternalRenderContext::Create(params);
 * m_renderContext = std::move(result.value());
 *
 * // In startNextFrame():
 * m_renderContext->RenderFrame(cmd, targetImageView, width, height);
 * @endcode
 *
 * @note Requires Vulkan 1.3 or VK_KHR_dynamic_rendering extension
 * @note Thread-safe for parameter updates, but not for rendering
 *
 * @author wtflmao
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "core/Config.hpp"
#include "scene/Scene.hpp"
#include "scene/Camera.hpp"
#include "renderer/LightingParams.hpp"

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <memory>
#include <atomic>

// Forward declaration for VMA allocator (avoid including heavy VMA header)
struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;

namespace quantiloom {

// Forward declarations
class GpuBuffer;
class GpuImage;
class RayTracingPipeline;
class TextureManager;
class BLAS;
class TLAS;

/**
 * @class ExternalRenderContext
 * @brief Renders to externally-managed Vulkan surfaces using dynamic rendering
 *
 * This class provides the interface for integrating libQuantiloom's spectral
 * ray tracing into external applications (Qt6, game engines, etc.) that manage
 * their own Vulkan context.
 *
 * Unlike VulkanContext which creates its own VkInstance/VkDevice, this class
 * accepts externally-provided Vulkan handles and adapts the rendering pipeline
 * to work with them.
 */
class QL_API ExternalRenderContext {
public:
    // ========================================================================
    // Initialization Parameters
    // ========================================================================

    /**
     * @struct InitParams
     * @brief Parameters for external context initialization
     *
     * All Vulkan handles must be valid and remain valid for the lifetime
     * of the ExternalRenderContext.
     */
    struct InitParams {
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkQueue graphicsQueue = VK_NULL_HANDLE;
        u32 graphicsQueueFamily = 0;

        // Target surface format (must match Qt's swapchain format)
        VkFormat targetColorFormat = VK_FORMAT_B8G8R8A8_SRGB;

        // Initial viewport size
        u32 width = 1280;
        u32 height = 720;

        // Optional: VMA allocator (if nullptr, creates internal allocator)
        VmaAllocator externalAllocator = VK_NULL_HANDLE;
    };

    // ========================================================================
    // Factory Method
    // ========================================================================

    /**
     * @brief Create ExternalRenderContext from external Vulkan handles
     *
     * Validates parameters and initializes internal resources (VMA, pipeline, etc.)
     * using the provided external Vulkan handles.
     *
     * @param params Initialization parameters
     * @return Result containing context or error message
     *
     * @note The external VkDevice must have been created with:
     *       - VK_KHR_acceleration_structure
     *       - VK_KHR_ray_tracing_pipeline
     *       - VK_KHR_dynamic_rendering (or Vulkan 1.3)
     *       - Buffer device address feature enabled
     */
    static Result<std::unique_ptr<ExternalRenderContext>, String> Create(const InitParams& params);

    ~ExternalRenderContext();

    // Non-copyable, movable
    ExternalRenderContext(const ExternalRenderContext&) = delete;
    ExternalRenderContext& operator=(const ExternalRenderContext&) = delete;
    ExternalRenderContext(ExternalRenderContext&&) noexcept;
    ExternalRenderContext& operator=(ExternalRenderContext&&) noexcept;

    // ========================================================================
    // Scene Loading
    // ========================================================================

    /**
     * @brief Load scene from TOML configuration file
     * @param configPath Path to TOML configuration
     * @return Result indicating success or error
     */
    Result<void, String> LoadSceneFromConfig(const String& configPath);

    /**
     * @brief Load scene from Config object
     * @param config Parsed configuration
     * @return Result indicating success or error
     */
    Result<void, String> LoadScene(const Config& config);

    /**
     * @brief Load scene from glTF file
     * @param gltfPath Path to glTF/GLB file
     * @return Result indicating success or error
     */
    Result<void, String> LoadSceneFromGltf(const String& gltfPath);

    /**
     * @brief Check if scene is loaded
     */
    [[nodiscard]] bool HasScene() const;

    /**
     * @brief Get current scene (read-only)
     */
    [[nodiscard]] const Scene* GetScene() const;

    // ========================================================================
    // Rendering Interface
    // ========================================================================

    /**
     * @brief Render frame to external command buffer
     *
     * Records ray tracing commands to the provided command buffer.
     * The caller is responsible for:
     * - Beginning the command buffer
     * - Calling vkCmdBeginRendering() with appropriate attachments
     * - Calling this method to record ray tracing commands
     * - Calling vkCmdEndRendering()
     * - Submitting the command buffer
     *
     * @param cmd Command buffer (must be in recording state)
     * @param targetImageView Target image view for output
     * @param width Render width
     * @param height Render height
     *
     * @note For real-time preview, use low SPP (1-4)
     * @note Uses progressive accumulation if SPP > 1
     */
    void RenderFrame(
        VkCommandBuffer cmd,
        VkImageView targetImageView,
        u32 width,
        u32 height
    );

    /**
     * @brief Resize render target
     * @param width New width
     * @param height New height
     *
     * Call when viewport size changes (e.g., window resize)
     */
    void Resize(u32 width, u32 height);

    /**
     * @brief Reset accumulation (call when camera/scene changes)
     */
    void ResetAccumulation();

    // ========================================================================
    // Camera Control
    // ========================================================================

    /**
     * @brief Set camera view matrix
     * @param viewMatrix 4x4 view matrix (world -> camera space)
     */
    void SetCameraViewMatrix(const glm::mat4& viewMatrix);

    /**
     * @brief Set camera from position/target/up
     * @param position Camera position (world space)
     * @param target Look-at target (world space)
     * @param up Up vector
     */
    void SetCameraLookAt(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up);

    /**
     * @brief Set camera field of view
     * @param fovYDegrees Vertical FOV in degrees
     */
    void SetCameraFOV(f32 fovYDegrees);

    /**
     * @brief Get current camera
     */
    [[nodiscard]] const Camera& GetCamera() const;

    // ========================================================================
    // Spectral Rendering Parameters
    // ========================================================================

    /**
     * @brief Set spectral rendering mode
     * @param mode Spectral mode (Single, RGB_Fused, MWIR_Fused, etc.)
     */
    void SetSpectralMode(SpectralMode mode);

    /**
     * @brief Set wavelength for single-wavelength mode
     * @param wavelength_nm Wavelength in nanometers
     */
    void SetWavelength(f32 wavelength_nm);

    /**
     * @brief Set samples per pixel
     * @param spp Samples per pixel (1 for real-time, 16-64 for quality)
     */
    void SetSPP(u32 spp);

    /**
     * @brief Get current spectral mode
     */
    [[nodiscard]] SpectralMode GetSpectralMode() const;

    /**
     * @brief Get current wavelength
     */
    [[nodiscard]] f32 GetWavelength() const;

    /**
     * @brief Get current SPP
     */
    [[nodiscard]] u32 GetSPP() const;

    // ========================================================================
    // Lighting Parameters
    // ========================================================================

    /**
     * @brief Update lighting parameters
     * @param params New lighting parameters
     */
    void SetLightingParams(const LightingParams& params);

    /**
     * @brief Set sun direction
     * @param direction Normalized direction vector (from surface to sun)
     */
    void SetSunDirection(const glm::vec3& direction);

    /**
     * @brief Set sun radiance
     * @param radiance RGB radiance (W/m^2/sr)
     */
    void SetSunRadiance(const glm::vec3& radiance);

    /**
     * @brief Set sky radiance
     * @param radiance RGB radiance (W/m^2/sr)
     */
    void SetSkyRadiance(const glm::vec3& radiance);

    /**
     * @brief Get current lighting parameters
     */
    [[nodiscard]] const LightingParams& GetLightingParams() const;

    // ========================================================================
    // Scene Editing (Phase 2)
    // ========================================================================

    /**
     * @brief Add mesh to scene
     * @param mesh Mesh to add
     * @param transform World transform
     * @return Index of added node
     */
    u32 AddMesh(const Mesh& mesh, const glm::mat4& transform);

    /**
     * @brief Remove node from scene
     * @param nodeIndex Index of node to remove
     * @return true if removed successfully
     */
    bool RemoveNode(u32 nodeIndex);

    /**
     * @brief Update node transform
     * @param nodeIndex Index of node
     * @param transform New world transform
     */
    void SetNodeTransform(u32 nodeIndex, const glm::mat4& transform);

    /**
     * @brief Update material properties
     * @param materialIndex Index of material
     * @param material New material properties
     */
    void UpdateMaterial(u32 materialIndex, const Material& material);

    /**
     * @brief Rebuild acceleration structure after scene changes
     *
     * Must be called after AddMesh/RemoveNode/SetNodeTransform
     */
    void RebuildAccelerationStructure();

    // ========================================================================
    // Status and Statistics
    // ========================================================================

    /**
     * @brief Get accumulated sample count
     */
    [[nodiscard]] u32 GetAccumulatedSamples() const;

    /**
     * @brief Get last frame render time in milliseconds
     */
    [[nodiscard]] f32 GetLastFrameTimeMs() const;

    /**
     * @brief Check if context is ready for rendering
     */
    [[nodiscard]] bool IsReady() const;

private:
    // Private constructor - use Create() factory method
    ExternalRenderContext();

    // Initialize internal resources
    Result<void, String> Initialize(const InitParams& params);

    // Build acceleration structures from current scene
    void BuildAccelerationStructures();

    // Update GPU buffers with current state
    void UpdateGpuResources();

    // Create dummy buffers for optional shader bindings
    void CreateDummyBuffers();

    // Create BRDF LUT for IBL
    void CreateBRDFLut();

    // Create fallback environment map
    void CreateFallbackEnvMap();

    // Create ray tracing pipeline and bind resources
    void CreatePipeline();

    // Transition image layout (internal helper)
    void TransitionImageLayoutImmediate(
        VkImage image,
        VkFormat format,
        VkImageLayout oldLayout,
        VkImageLayout newLayout
    );

    // Implementation details (PIMPL)
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace quantiloom
