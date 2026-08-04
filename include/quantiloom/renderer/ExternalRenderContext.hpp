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

/// How the camera generates rays. Mirrors Camera::Projection, restated here
/// so a host setting it does not need scene/Camera.hpp.
enum class CameraProjection : u32 {
    Perspective = 0,
    Orthographic = 1,
};

/**
 * @struct SolarLutSpec
 * @brief What a scene declares about its illuminant
 *
 * The `[lighting] solar_lut*` keys, as a struct. Passed to
 * ExternalRenderContext::SetSolarSpectralLUTFromSpec so a host can change the
 * illuminant without re-applying a whole configuration and without deciding
 * for itself what any of these mean -- the core reads them one way, for the
 * file and for the host alike.
 */
struct SolarLutSpec {
    /// A path to a spectrum, or the literal "equal_energy" for CIE illuminant
    /// E -- a flat spectrum at unit luminance, the neutral reference. Not sRGB
    /// white, which is D65.
    String pathOrEqualEnergy;

    /// 1-based columns holding direct sun and diffuse sky. The defaults are
    /// libRadtran uvspec's layout. ASTM G-173 -- assets/luts/astmg173.csv --
    /// wants {4, 3} with diffuseIsGlobal set, because its column 3 is global
    /// irradiance and using it as the sky would count the sun twice.
    u32 directColumn = 2;
    u32 diffuseColumn = 3;
    bool diffuseIsGlobal = false;

    /// Divide both curves by the *sun's* luminance, putting the illuminant at
    /// Y = 1. Published reference spectra are relative, so their absolute level
    /// is arbitrary; this is what makes D65 come out as sRGB (1, 1, 1). Both by
    /// the same divisor, so the sun-to-sky ratio -- the one thing a measured
    /// pair actually tells you -- survives.
    bool normaliseUnitLuminance = false;
};

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
     * @brief Show the accumulated image again without tracing anything
     *
     * RenderFrame traces and presents together, so a host that needs to draw a
     * frame for its own reasons -- an editor compositing a selection outline or
     * a gizmo over the render -- has no way to do it without adding a sample.
     * That makes the sample count of a finished image depend on how many times
     * the user clicked, which for a renderer whose output is a measurement is
     * not a cosmetic problem.
     *
     * This is the second half of RenderFrame on its own: the same source image
     * (sensor chain, then CLAHE, then the raw accumulation), the same
     * format-converting blit, the same PRESENT_SRC transition -- and no trace,
     * no post-processing re-run, and no change to the accumulation. The
     * post-processed images still hold the last frame's result, because
     * nothing has changed the accumulation since.
     *
     * @param cmd           Command buffer to record into
     * @param targetImage   Target swapchain image
     * @param targetLayout  Its current layout
     * @param width         Must equal the current render width
     * @param height        Must equal the current render height
     * @return True when the frame was presented. False when there is nothing
     *         to show -- no context, nothing traced yet, or a size that does
     *         not match the accumulation. **The caller must then call
     *         RenderFrame instead**: nothing has been recorded, so the target
     *         image is left as it was and presenting it would show undefined
     *         contents.
     */
    [[nodiscard]] bool PresentAccumulated(
        VkCommandBuffer cmd,
        VkImage targetImage,
        VkImageLayout targetLayout,
        u32 width,
        u32 height
    );

    /**
     * @brief Re-run post-processing on the accumulation and present, without
     *        tracing anything
     *
     * The gap between its two siblings. RenderFrame re-runs everything and
     * costs a sample; PresentAccumulated costs nothing but re-presents the
     * post-processed images as they were. Neither serves a host whose user
     * just changed a display-stage setting -- sensor simulation on or off,
     * its parameters, CLAHE -- after accumulation has stopped: the first
     * charges the sample budget for a display edit, the second shows the
     * image from before the edit.
     *
     * This is RenderFrame minus the trace: the sensor chain and CLAHE re-run
     * over the accumulation exactly as it stands, then the same blit and
     * PRESENT_SRC transition. No sample is added and the accumulation is not
     * touched, so the result is what RenderFrame would have produced had the
     * new display settings been set all along.
     *
     * @param cmd           Command buffer to record into
     * @param targetImage   Target swapchain image
     * @param targetLayout  Its current layout
     * @param width         Must equal the current render width
     * @param height        Must equal the current render height
     * @return True when the frame was presented. False for the same three
     *         reasons as PresentAccumulated -- no context, nothing traced
     *         yet, or a stale size -- and with the same obligation: nothing
     *         has been recorded, so the caller must call RenderFrame instead.
     */
    [[nodiscard]] bool ReprocessAccumulated(
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
     * @brief Instance an existing node: same mesh, same materials, own transform
     *
     * The shallow copy of a scene editor's copy/paste (Blender's Alt+D,
     * Unreal's asset-by-reference duplicate): the new node references the
     * source's mesh, so geometry and materials are shared and a material
     * edit shows on every copy. The new node starts with the source's
     * transform; move it with SetNodeTransform.
     *
     * Call RebuildAccelerationStructure afterwards -- the instance count
     * changed, so a refit is not enough. Like SetNodeTransform, does not
     * reset accumulation; that is the caller's job.
     *
     * @param sourceNodeIndex Node to instance (a tombstoned source is allowed;
     *                        the copy is created active)
     * @param newName Name for the new node -- what [[nodes]] / [[duplicates]]
     *                config entries match on, so it should be unique
     * @return Index of the new node, or an error for an invalid source
     */
    Result<u32, String> DuplicateNode(u32 sourceNodeIndex, const String& newName);

    /**
     * @brief Remove node from scene (tombstone)
     *
     * Marks the node inactive rather than erasing it: indices held by
     * callers never shift, and RestoreNode can undo the removal. The
     * node contributes no TLAS instances after the next
     * RebuildAccelerationStructure, which the caller must invoke.
     * Does not reset accumulation.
     *
     * @param nodeIndex Index of node to remove
     * @return true if removed; false if out of range or already removed
     */
    bool RemoveNode(u32 nodeIndex);

    /**
     * @brief Reactivate a node removed by RemoveNode
     *
     * The undo of RemoveNode: the node keeps its index, mesh, name and
     * last transform. Call RebuildAccelerationStructure afterwards.
     * Does not reset accumulation.
     *
     * @param nodeIndex Index of node to restore
     * @return true if restored; false if out of range or not removed
     */
    bool RestoreNode(u32 nodeIndex);

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
     * @brief Build an endmember weight map for a material, from its base colour
     * @param materialIndex   Material to unmix
     * @param endmemberColors Linear sRGB colour of each endmember curve, in the
     *                        order their indices will be assigned; at most
     *                        Material::MAX_ENDMEMBERS
     * @return Index into the texture array, or an error explaining what was missing
     *
     * The runtime half of what a scene config gets at load: asks of every texel
     * how much of each endmember would produce that colour, and uploads the
     * answer as a texture. Assign the result to Material::weightTextureIndex,
     * the curve indices from AddSpectralCurve() to
     * Material::spectralReflectanceCurveIndex and endmemberCurveIndex1..3, then
     * call UpdateMaterial().
     *
     * Get the colours from ReflectanceToLinearSrgbD65() on the same curves --
     * they must be what the endmembers look like under D65, or the weights are
     * fitted against something the texture never depicted.
     *
     * Fails when the material has no base-colour texture, or when its pixels
     * were released after upload. The context keeps the base colours of
     * curve-bound materials on the CPU for this; a material that had no curve
     * when the scene loaded will not have them.
     */
    Result<i32, String> BuildEndmemberWeightTexture(
        u32 materialIndex, const Vector<glm::vec3>& endmemberColors);

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
     * @brief Switch between perspective and orthographic ray generation
     *
     * Orthographic rays share a direction and start spread across the film
     * plane, so parallel edges stay parallel and two things the same size
     * measure the same at any depth -- which is what a front, top or side view
     * is for. Picking follows automatically; it reads the same camera.
     *
     * @param projection Which projection to use
     * @param orthoHeight World-space height of the film plane. Ignored in
     *        perspective, and ignored when not positive, which leaves whatever
     *        was set before.
     */
    void SetCameraProjection(CameraProjection projection, f32 orthoHeight = 0.0f);

    /**
     * @brief Set the illuminant the way a scene file would have
     *
     * SetSolarSpectralLUT above takes two curves and asks no questions: the
     * caller has already decided what the file meant, how to normalise it and
     * what colour the non-spectral paths should use. A host that offers the
     * user a choice of illuminant would have to answer all three, and any
     * answer it invented would be a second reading of keys the core already
     * reads -- the class of divergence that made the same TOML render
     * differently in the CLI and in Studio.
     *
     * This takes the declaration instead of the conclusion. It loads the file
     * (or CIE illuminant E for the literal "equal_energy"), normalises it,
     * derives the RGB the non-spectral paths need, uploads the LUT and updates
     * LightingParams -- exactly what [lighting] solar_lut* does, through the
     * same function.
     *
     * @param spec     What to load and how to read it
     * @param baseDir  Directory relative paths resolve against; empty for the
     *                 working directory, as the CLI passes
     * @return Nothing, or why the illuminant could not be loaded. Warnings
     *         that do not prevent loading -- a spectrum too narrow for the
     *         band being rendered -- go to the log.
     */
    Result<void, String> SetSolarSpectralLUTFromSpec(const SolarLutSpec& spec,
                                                     const String& baseDir = "");

    /**
     * @brief Rebuild acceleration structure after scene changes
     *
     * Must be called after AddMesh/RemoveNode/SetNodeTransform
     */
    void RebuildAccelerationStructure();

    /**
     * @brief Refit the acceleration structure after transform-only edits
     *
     * The interactive path for RebuildAccelerationStructure: updates the
     * existing TLAS in place (no allocation, no device idle) so a gizmo
     * drag can apply SetNodeTransform per mouse-move without stalling.
     * Falls back to a full rebuild internally when a refit is not possible
     * (topology changed since the last build).
     *
     * A refit degrades trace quality slightly for large movements; call
     * RebuildAccelerationStructure once when the drag ends.
     *
     * @note Like SetNodeTransform, does not reset accumulation -- that is
     *       the caller's job.
     */
    void RefitAccelerationStructure();

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
