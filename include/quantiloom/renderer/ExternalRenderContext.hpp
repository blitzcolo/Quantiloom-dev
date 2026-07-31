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
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "core/Config.hpp"
#include "scene/Scene.hpp"
#include "scene/Camera.hpp"
#include "renderer/ConfigApply.hpp"
#include "renderer/LightingParams.hpp"
#include "renderer/Pick.hpp"
#include "atmos/AtmosphereNNConfig.hpp"
#include "core/Image.hpp"
#include "core/SpectralData.hpp"
#include "postprocess/SensorModel.hpp"

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
struct ComplexRefractiveIndex;

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

        // Optional: Pipeline cache directory path
        // If empty, uses platform-specific default:
        //   Windows: %LOCALAPPDATA%/Quantiloom/cache/
        //   Linux:   ~/.cache/Quantiloom/
        //   macOS:   ~/Library/Caches/Quantiloom/
        std::string pipelineCacheDir;
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
     * @brief Load only the scene a configuration names.
     *
     * Reads `scene.gltf` / `scene.usd` and nothing else. For a config that
     * should be honoured -- its illuminant, materials, atmosphere, camera and
     * the rest -- use ApplyConfig().
     *
     * @param config Parsed configuration
     * @return Result indicating success or error
     */
    Result<void, String> LoadScene(const Config& config);

    /**
     * @brief Apply a whole scene configuration: the same reading the CLI does.
     *
     * Loads the scene, then everything else the file says -- spectral mode and
     * wavelength, samples and seed, lighting, the solar spectrum with its
     * normalisation, reflectance curves from CSV and from the NMF basis,
     * complex refractive indices, the default IR surface temperature and
     * [[materials]] overrides, the atmosphere, the environment map, the camera
     * and the sensor model.
     *
     * This exists because reading those keys twice, once here and once in the
     * host, is how the frontend and the CLI came to render the same file
     * differently. One reading, in rendercore::ResolveRenderConfig, serves
     * both.
     *
     * Values that were applied are read back through this class's own getters
     * -- GetLightingParams(), GetAtmosphere(), GetSpectralMode(), GetCamera()
     * and the rest. What the report carries is what no getter can answer: the
     * diagnostics, how much of each kind of data loaded, and the keys a
     * windowed context cannot honour (`renderer.resolution`, which a viewport
     * renders past at its own size; `renderer.output`; `[hyperspectral]`).
     *
     * Accumulation restarts. Safe to call again on the same context: it is
     * also how a host replays a document after a lost device.
     *
     * @param config  Parsed configuration.
     * @param options Strictness, the directory relative paths resolve against,
     *                and the atmosphere model pack to fall back on.
     * @return What was applied and what was not. Check ok() before trusting a
     *         render: under MissingKeyPolicy::Error a required key that is
     *         absent leaves the context partly configured.
     */
    ConfigApplyReport ApplyConfig(const Config& config,
                                  const ConfigApplyOptions& options = {});

    /**
     * @brief Load scene from glTF file
     * @param gltfPath Path to glTF/GLB file
     * @return Result indicating success or error
     */
    Result<void, String> LoadSceneFromGltf(const String& gltfPath);

    /**
     * @brief Load scene from OpenUSD file
     * @param usdPath Path to USD file (.usd, .usda, .usdc, .usdz)
     * @return Result indicating success or error
     *
     * Supports Quantiloom spectral extensions via custom primvars:
     * - quantiloom:materialType - Spectral database type
     * - quantiloom:materialRef - Material name in database
     * - quantiloom:emissivityCurve - Path to emissivity CSV
     * - quantiloom:reflectanceCurve - Path to reflectance CSV
     * - quantiloom:transmittanceCurve - Path to transmittance CSV
     * - quantiloom:temperature_K - Surface temperature (K)
     */
    Result<void, String> LoadSceneFromUsd(const String& usdPath);

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
     * Records ray tracing commands to the provided command buffer, then copies
     * the result to the target swapchain image.
     *
     * The caller is responsible for:
     * - Beginning the command buffer
     * - Ensuring target image is in TRANSFER_DST_OPTIMAL or GENERAL layout
     * - Calling this method to record ray tracing and blit commands
     * - Transitioning target image to PRESENT_SRC_KHR before present
     * - Submitting the command buffer
     *
     * @param cmd Command buffer (must be in recording state)
     * @param targetImage Target swapchain image (from QVulkanWindow::swapChainImage)
     * @param targetLayout Current layout of target image (typically UNDEFINED or PRESENT_SRC_KHR)
     * @param width Render width
     * @param height Render height
     *
     * @note For real-time preview, use low SPP (1-4)
     * @note Uses progressive accumulation if SPP > 1
     * @note There is no tone mapping. The accumulated linear radiance is
     *       blitted to the target, which clamps anything above 1.0 and applies
     *       the target format's transfer function -- so an sRGB target encodes
     *       and a UNORM one displays linear values uncorrected. Ask for an
     *       sRGB format (InitParams::targetColorFormat, and on the host side
     *       QVulkanWindow::setPreferredColorFormats). Bringing a scene into
     *       displayable range is the scene's business:
     *       `lighting.solar_lut_normalise` for an absolute illuminant, or the
     *       CLAHE pass for HDR and thermal content. This note said Reinhard for
     *       a long time and never was.
     */
    void RenderFrame(
        VkCommandBuffer cmd,
        VkImage targetImage,
        VkImageLayout targetLayout,
        u32 width,
        u32 height
    );

    /**
     * @brief Blit the primary-hit depth AOV into a caller-owned image
     *
     * The depth AOV is R32_SFLOAT: the hit distance along the normalized
     * primary camera ray in world units (a Euclidean camera-to-surface
     * distance), or -1.0 where the primary ray missed. It is overwritten
     * every frame -- never accumulated -- so it always reflects the current
     * scene, including mid-drag transform edits.
     *
     * Mirrors RenderFrame's contract: the caller owns the target image and
     * this method only records commands. Record it after RenderFrame in the
     * same command buffer. The target must be R32_SFLOAT with
     * TRANSFER_DST and SAMPLED usage; it is left in
     * VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, ready to be sampled by an
     * overlay pass (the only intended consumer).
     *
     * @param cmd Command buffer (must be in recording state)
     * @param targetImage Caller-owned R32_SFLOAT image
     * @param targetCurrentLayout Current layout of targetImage
     *        (VK_IMAGE_LAYOUT_UNDEFINED on first use)
     * @param width Render width (must match RenderFrame's)
     * @param height Render height (must match RenderFrame's)
     */
    void BlitDepthTo(
        VkCommandBuffer cmd,
        VkImage targetImage,
        VkImageLayout targetCurrentLayout,
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
     * @param mode Spectral mode (RGB, VIS_Fused, Single, MWIR_Fused, etc.)
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
     * @brief Set the seed for the path tracer's per-sample sampling sequence.
     *
     * Same convention as the CLI's `renderer.seed`, and the same default
     * (constants::DEFAULT_SAMPLING_SEED), so an interactive render and a CLI
     * render of the same scene draw the same sequence. Pass 0 to seed from
     * std::random_device instead, giving frame-varying noise.
     *
     * The sequence restarts whenever accumulation restarts, which is what makes
     * a result reproducible: before this existed the context seeded itself from
     * std::random_device once and kept advancing, so the same scene and camera
     * produced a different image depending on how many frames had been drawn
     * beforehand.
     *
     * Resets accumulation.
     *
     * @param seed Nonzero for a reproducible sequence, 0 for a random one
     */
    void SetSamplingSeed(u32 seed);

    /**
     * @brief Get the configured sampling seed (0 means nondeterministic)
     */
    [[nodiscard]] u32 GetSamplingSeed() const;

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
    // Debug Visualization
    // ========================================================================

    /**
     * @brief Set debug visualization mode
     * @param mode Debug mode (None for normal rendering)
     *
     * Enables visualization of intermediate rendering data for debugging:
     * - Geometry: normals, UVs, positions, material IDs
     * - Material: albedo, metallic, roughness, emissive
     * - Lighting: NdotL, NdotV, direct sun, diffuse
     * - BRDF: Fresnel F0, full BRDF evaluation
     * - IBL: prefiltered env, BRDF LUT, specular
     * - Spectral: XYZ tristimulus, pre-correction RGB
     * - IR: temperature, emissivity, emission/reflection
     *
     * @note Resets accumulation when mode changes
     */
    void SetDebugMode(DebugVisualizationMode mode);

    /**
     * @brief Get current debug visualization mode
     */
    [[nodiscard]] DebugVisualizationMode GetDebugMode() const;

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
    // Atmosphere (NN MODTRAN surrogate)
    // ========================================================================

    /**
     * @brief Set the NN atmosphere configuration
     * @param config Weather / geometry configuration; config.modelPackDir
     *               must point at a directory of <band>_<geom>_<net>.safetensors
     *               files when enabled.
     *
     * The spectral LUT is (re)baked lazily before the next frame whenever the
     * bake key (band, weather, quantized h1 / sun geometry) changes. Missing
     * network files are a hard error at bake time -- there is no fallback.
     *
     * Example:
     * @code
     * AtmosphereNNConfig cfg;
     * cfg.modelPackDir = "assets/atmos_models";
     * cfg.ApplyPreset("fog");
     * cfg.enabled = true;
     * context->SetAtmosphere(cfg);
     * @endcode
     *
     * @note Resets accumulation when config changes
     * @throws std::runtime_error if the model pack directory does not exist
     */
    void SetAtmosphere(const AtmosphereNNConfig& config);

    /**
     * @brief Get current NN atmosphere configuration
     */
    [[nodiscard]] const AtmosphereNNConfig& GetAtmosphere() const;

    // ========================================================================
    // Environment Map (IBL)
    // ========================================================================

    /**
     * @brief Load HDR environment map for IBL
     * @param hdrPath Path to equirectangular HDR image (.exr, .hdr)
     * @return Result indicating success or error
     *
     * Loads the HDR image, converts equirectangular to cubemap, and generates
     * prefiltered mip chain for specular IBL.
     *
     * @note Replaces the fallback sky-blue environment map
     * @note Resets accumulation when environment changes
     */
    Result<void, String> LoadEnvironmentMap(const String& hdrPath);

    /**
     * @brief Check if custom environment map is loaded
     * @return true if LoadEnvironmentMap() succeeded, false if using fallback
     */
    [[nodiscard]] bool HasEnvironmentMap() const;

    // ========================================================================
    // Display Enhancement (CLAHE)
    // ========================================================================

    /**
     * @struct CLAHEParams
     * @brief Parameters for GPU CLAHE display enhancement
     *
     * Controls Contrast Limited Adaptive Histogram Equalization for
     * improving visibility of low-contrast images (especially IR bands).
     */
    struct CLAHEParams {
        bool enabled = false;           ///< Enable CLAHE processing
        f32 clipLimit = 2.0f;           ///< Contrast limit (1.0 = no limit, typical 2.0-4.0)
        i32 tileSize = 8;              ///< Tile grid size (4, 8, 16, or 32)
        bool luminanceOnly = true;     ///< Apply only to luminance channel (preserve color)
        bool normalizeOutput = true;   ///< Normalize output to [0,1] range
    };

    /**
     * @brief Set CLAHE display enhancement parameters
     *
     * When enabled, a display image is maintained separately from the
     * raw output image. The display image has CLAHE applied and is used
     * for screen presentation. The raw output image is preserved for
     * export/analysis via CaptureScreenshot().
     *
     * @param params CLAHE parameters
     * @note Takes effect on the next RenderFrame() call
     */
    void SetCLAHEParams(const CLAHEParams& params);

    /**
     * @brief Get current CLAHE parameters
     */
    [[nodiscard]] const CLAHEParams& GetCLAHEParams() const;

    /**
     * @brief Capture display image (with CLAHE applied if enabled)
     *
     * Returns the image as shown on screen. If CLAHE is enabled,
     * the returned image has CLAHE processing applied.
     * If CLAHE is disabled, this is equivalent to CaptureScreenshot().
     *
     * Use this for "what you see is what you get" screenshots.
     * Use CaptureScreenshot() for raw HDR data export.
     *
     * @return Image with display content, or error if not ready
     */
    [[nodiscard]] Result<Image, String> CaptureDisplayImage();

    // ========================================================================
    // GPU Sensor Simulation (Real-time)
    // ========================================================================

    /**
     * @brief Enable/disable GPU-based sensor simulation for real-time preview
     *
     * When enabled, the sensor imaging chain (noise, blur, quantization) is
     * applied in real-time during RenderFrame() using GPU compute shaders.
     * This allows the viewport to display sensor effects at 60 FPS.
     *
     * Performance: ~2-3ms overhead @ 1080p
     *
     * @param enabled true to enable GPU sensor, false to disable
     * @note Takes effect on the next RenderFrame() call
     */
    void SetGPUSensorEnabled(bool enabled);

    /**
     * @brief Set sensor parameters for GPU simulation
     *
     * Updates the sensor parameters used by the GPU sensor chain.
     * Changes take effect immediately on the next frame.
     *
     * @param params Sensor parameters (optics, detector, ADC, noise)
     * @see SensorParams for parameter details
     */
    void SetGPUSensorParams(const SensorParams& params);

    /**
     * @brief Check if GPU sensor is enabled
     * @return true if GPU sensor simulation is active
     */
    [[nodiscard]] bool IsGPUSensorEnabled() const;

    /**
     * @brief Get current GPU sensor parameters
     * @return Current sensor parameters
     */
    [[nodiscard]] const SensorParams& GetGPUSensorParams() const;

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
     * @brief Add complex refractive index data for physical Fresnel
     * @param cri CPU-side complex refractive index (n,k curves)
     * @return Index into CRI buffer (for Material::complexRefractiveIndexIndex)
     *
     * Converts CRI to GPU format (ComplexRefractiveIndexGPU, 64 uniform samples),
     * appends to CRI buffer, and rebinds descriptor. The returned index can be
     * assigned to Material::complexRefractiveIndexIndex before calling UpdateMaterial().
     */
    i32 AddComplexRefractiveIndex(const ComplexRefractiveIndex& cri);

    /**
     * @brief Add a measured spectral reflectance curve for quantitative rendering
     * @param curve CPU-side wavelength/value pairs; resampled to a uniform grid
     * @return Index into the spectral curve buffer, or -1 if the curve is unusable
     *
     * Assign the result to Material::spectralReflectanceCurveIndex and call
     * UpdateMaterial() to make a material use it. A material left at -1 keeps the
     * RGB-upsampled fallback, which is what every material gets by default -- no
     * scene loader sets this index.
     *
     * Same shape as AddComplexRefractiveIndex: appends, reuploads and rebinds.
     */
    i32 AddSpectralCurve(const SpectralCurve& curve);

    /**
     * @brief Set the solar and sky irradiance spectra used for quantitative lighting
     * @param sunIrradiance Direct solar spectral irradiance (W/m^2/nm)
     * @param skyIrradiance Diffuse sky spectral irradiance (W/m^2/nm)
     *
     * One LUT for the scene rather than an indexed set, so this replaces rather than
     * appends.
     *
     * @warning Until this is called the LUT is **zero**, and a quantitative spectral
     *          mode will render black for want of an illuminant. That is deliberate:
     *          silently substituting a standard spectrum would make an unconfigured
     *          scene look plausible while reporting radiance nobody asked for.
     *          assets/luts/astmg173.csv holds AM1.5 if a caller wants that default.
     */
    void SetSolarSpectralLUT(const SpectralCurve& sunIrradiance,
                             const SpectralCurve& skyIrradiance);

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

    /**
     * @brief Absolute path of the pipeline cache file this context reads and writes
     *
     * Resolved once at Create() from InitParams::pipelineCacheDir or the
     * platform default. Hosts that want to know whether shader compilation will
     * be slow should test this path rather than reconstructing the policy --
     * a host-side copy of it silently rots when the default moves.
     */
    [[nodiscard]] const String& GetPipelineCachePath() const;

    // ========================================================================
    // Debug Pixel Reading
    // ========================================================================

    /**
     * @brief Read raw pixel value from render output
     *
     * Reads the raw float4 value from the outputImage at the specified
     * pixel position. This is useful for debug visualization to show
     * the actual rendered values (before swapchain format conversion).
     *
     * @param x X coordinate (pixels, 0 = left)
     * @param y Y coordinate (pixels, 0 = top)
     * @return Raw float4 value from outputImage, or error if out of bounds
     *
     * @note Call this AFTER RenderFrame to get meaningful results
     * @note The returned value is the mapped debug output, use inverse
     *       mapping to recover original values (e.g., for normals:
     *       original = (output - 0.5) * 2)
     */
    [[nodiscard]] Result<glm::vec4, String> ReadPixelValue(u32 x, u32 y);

    /**
     * @brief What is under this pixel: trace the pixel's primary camera ray
     *
     * A 1x1 inline ray-query compute dispatch against the live TLAS, using
     * exactly the raygen shader's ray for the pixel (pixel center, no
     * jitter, no lens offset), so the answer agrees with what is on screen.
     * The instance index is mapped back to a Scene::nodes index host-side.
     *
     * @param x X coordinate (pixels, 0 = left), same space as ReadPixelValue
     * @param y Y coordinate (pixels, 0 = top)
     * @return PickResult (hit == false when the ray reached the sky), or an
     *         error when no scene is loaded or the pixel is out of bounds
     *
     * @note Synchronous: submits and waits like ReadPixelValue. A 1x1
     *       dispatch is sub-millisecond; intended per click, not per frame.
     */
    [[nodiscard]] Result<PickResult, String> Pick(u32 x, u32 y);

    /**
     * @brief Capture current render output to CPU Image
     *
     * Reads the entire outputImage from GPU to CPU memory.
     * The resulting Image contains HDR data (f32 RGBA) before
     * any swapchain format conversion or tone mapping.
     *
     * @return Image with current render output, or error if not ready
     *
     * @note Call this AFTER RenderFrame to capture the frame
     * @note Blocks until GPU transfer completes (synchronous)
     * @note Performs layout transition internally (GENERAL → TRANSFER_SRC → GENERAL)
     */
    [[nodiscard]] Result<Image, String> CaptureScreenshot();

private:
    // Use the Create() factory.
    ExternalRenderContext();

    // Implementation details (PIMPL)
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace quantiloom
