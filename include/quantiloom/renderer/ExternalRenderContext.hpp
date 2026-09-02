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
#include "renderer/DisplayControl.hpp"
#include "renderer/ThermalControl.hpp"
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
     * @param width Target width -- the extent of @p targetImage, not
     *        necessarily what gets traced. See SetRenderScale.
     * @param height Target height, likewise
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
     * @param width         Must equal the current target width
     * @param height        Must equal the current target height
     * @return True when the frame was presented. False when there is nothing
     *         to show -- no context, nothing traced yet, a size that does not
     *         match the accumulation, or a render scale not yet acted on.
     *         **The caller must then call
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
     * @param width         Must equal the current target width
     * @param height        Must equal the current target height
     * @return True when the frame was presented. False for the same reasons
     *         as PresentAccumulated -- no context, nothing traced yet, a
     *         stale size or a pending scale -- and with the same obligation: nothing
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
     * @param width Target width (must match RenderFrame's)
     * @param height Target height (must match RenderFrame's)
     *
     * @note Magnified with VK_FILTER_NEAREST when the render scale is below
     *       1.0, because an interpolated depth across a silhouette would be a
     *       surface that is not in the scene.
     */
    void BlitDepthTo(
        VkCommandBuffer cmd,
        VkImage targetImage,
        VkImageLayout targetCurrentLayout,
        u32 width,
        u32 height
    );

    /**
     * @brief Set the target extent
     * @param width New target width
     * @param height New target height
     *
     * Call when viewport size changes (e.g., window resize). The render extent
     * follows from this and the render scale; RenderFrame does the same thing
     * on its own when the extent it is handed differs from the last one, so
     * calling this is only necessary to resize ahead of a frame.
     *
     * @note Resets the accumulation when the render extent actually changes.
     */
    void Resize(u32 width, u32 height);

    /**
     * @brief Trace at a fraction of the target extent
     *
     * The render extent -- what the trace, the sensor chain, CLAHE and every
     * extent-sized image are sized to -- is the target extent handed to
     * RenderFrame, scaled by this factor and rounded, at least 1x1. The
     * presenting blit magnifies back to the target, linearly where the driver
     * can and NEAREST where it cannot.
     *
     * This buys frames during interaction. Half the linear extent is a quarter
     * of the pixels, so a scene tracing at 20 samples/s traces at about 80
     * while the user is dragging the camera, and both the sample rate and the
     * present rate rise with it. The image is softer for as long as the scale
     * is down, and exactly what it always was once it returns to 1.0 --
     * nothing in the estimator changes, and no frame ever mixes the two.
     *
     * @param scale Clamped to [0.25, 1.0]
     *
     * @note Takes effect on the next RenderFrame(), which is where the extent
     *       change and its vkDeviceWaitIdle can be afforded. A no-op when the
     *       clamped value is the current one. Until then PresentAccumulated
     *       and ReprocessAccumulated refuse, so a host whose loop has stopped
     *       at its target sample count still picks the new scale up.
     * @note Every scale that actually changes the render extent resets the
     *       accumulation. Do not drive this from anything finer-grained than
     *       a user gesture.
     */
    void SetRenderScale(f32 scale);

    /**
     * @brief The factor last given to SetRenderScale, clamped
     */
    [[nodiscard]] f32 GetRenderScale() const;

    /**
     * @brief Width of the image actually being traced
     *
     * Equals the width handed to RenderFrame when the render scale is 1.0.
     * This is the extent of everything CaptureScreenshot, CaptureDisplayImage
     * and the AOV readbacks return.
     */
    [[nodiscard]] u32 GetRenderWidth() const;

    /**
     * @brief Height of the image actually being traced
     */
    [[nodiscard]] u32 GetRenderHeight() const;

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
     * @brief The one number a debug view may need
     *
     * Which number is the view's business. DebugVisualizationMode::
     * SunSensitivity reads it as the sun column to draw: zero is the whole-day
     * response, and 1..sunMemoryLags are the individual hours the solve is
     * remembering. Out of range is clamped rather than refused, because the
     * number of columns is a property of the solve and a panel can hold a
     * stale selection across a re-solve.
     *
     * Every other view ignores it, and it is zero unless something sets it.
     */
    void SetDebugParameter(u32 value);

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
     * @brief Load an HDR environment map and light the scene with it
     * @param hdrPath Path to equirectangular HDR image (.exr, .hdr)
     * @return Result indicating success or error
     *
     * Loads the HDR image, converts equirectangular to cubemap, and generates
     * the prefiltered mip chain for specular IBL.
     *
     * @note On success the map becomes a light source: this raises
     *       LightingParams::enableEnvironmentMap and uploads it. Nothing else
     *       needs to be called.
     * @note On failure the previously bound cubemap stays bound -- the binding
     *       has to hold something -- but image-based lighting is switched off,
     *       so a failed load renders without an environment rather than with a
     *       stale or invented one. GetLightingParams() reports the 0.
     * @note A context with no map loaded binds a black 1x1 placeholder and never
     *       samples it. There is no substitute sky.
     * @note Only meaningful in SpectralMode::RGB. The spectral modes read a
     *       map's RGB as spectral radiance density, which overstates a real sky
     *       by roughly 12x, so they do not sample one; ApplyConfig does not even
     *       load a map for them.
     * @note Resets accumulation, on both the success and failure paths.
     */
    Result<void, String> LoadEnvironmentMap(const String& hdrPath);

    /**
     * @brief Whether a real environment map is loaded
     * @return true if a LoadEnvironmentMap() call succeeded and its map is still
     *         bound; false if the black placeholder is bound instead
     *
     * @note This is load state, not lighting state. A map can be loaded while
     *       image-based lighting is off (a config that sets
     *       renderer.environment_map_enabled = false keeps the path and the
     *       map). Ask GetLightingParams().enableEnvironmentMap for whether it
     *       lights anything.
     */
    [[nodiscard]] bool HasEnvironmentMap() const;

    // ========================================================================
    // Display Enhancement (CLAHE)
    // ========================================================================

    /**
     * @brief Set the tone operator and palette the viewport is displayed with
     *
     * When enabled, a display image is maintained separately from the raw
     * output image. The display image carries the tone mapping and the palette
     * and is what the target is blitted from; the raw output image is
     * untouched and is what CaptureScreenshot() and every export read.
     * CaptureDisplayImage() reads the other one, which is how a false-colour
     * thermogram gets saved.
     *
     * See DisplayControl.hpp for what the modes mean -- in particular that
     * only Linear and Equalize keep "brighter is hotter" true across the whole
     * image.
     *
     * @param params tone mode, palette and their parameters
     * @note Takes effect on the next RenderFrame() call
     */
    void SetDisplayEnhancementParams(const DisplayEnhancementParams& params);

    /**
     * @brief Get the current display enhancement parameters
     */
    [[nodiscard]] const DisplayEnhancementParams& GetDisplayEnhancementParams() const;

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
    // Thermal Solve (interactive surface energy balance)
    // ========================================================================

    /**
     * @brief Set the global parameters for the thermal solve
     * @param params Solve parameters (timestep, layers, forcing, etc.)
     */
    void SetThermalSolveParams(const ThermalSolveParams& params);

    /**
     * @brief Set thermal properties for a material by name
     *
     * A material the config says nothing about keeps whatever temperature it
     * already had rather than being solved. Naming a conductivity of zero
     * explicitly has the same effect.
     */
    void SetThermalMaterial(const String& materialName, const ThermalMaterialParams& params);

    /// Remove all thermal material associations.
    void ClearThermalMaterials();

    /**
     * @brief Write the current thermal solve out, one row per element
     * @param pathOrEmpty  Where to write; empty takes ThermalSolveParams
     *                     ::dumpElementsFile, which is what a config set
     * @return The file written, or why nothing was
     *
     * The interactive twin of `[thermal] dump_elements`, and deliberately a
     * call rather than a parameter. Offline, one run is one dump and writing
     * from inside the solve is right; a viewport re-solves on every scrub of
     * the hour slider, so the same arrangement here would turn dragging a
     * slider into hundreds of writes. The path travels with the parameters and
     * the write happens when a host asks for it.
     *
     * What lands is the instant already on screen and the solver's own numbers
     * -- centroid, normal, the temperature reached, and the (dT/dv, v) pair the
     * shader would apply -- not the image, which carries the per-pixel sun
     * correction and a radiance inversion on top. The header block above the
     * table is the material properties AS THE SOLVE SAW THEM: a material bound
     * to a measured spectrum is solved at the Planck-weighted band average of
     * that curve rather than at the emissivity its config typed.
     */
    Result<String, String> DumpThermalElements(const String& pathOrEmpty = "");

    /**
     * @brief Enable or disable the thermal solve
     *
     * When disabled the viewport reverts to the per-material scalar
     * temperature. Enabling again does not discard the cached exchange or
     * checkpoints — only a geometry change does.
     */
    void SetThermalSolveEnabled(bool enabled);

    /**
     * @brief Set the simulated hour and update the temperature field
     *
     * Steps from the nearest checkpoint, uploads the result, and resets
     * accumulation. The exchange is recomputed only when geometry or rays/topK
     * changed since the last call — scrubbing time alone never reruns it.
     *
     * Must not be called from within a host command buffer recording (it
     * submits its own work via ExecuteImmediate).
     *
     * @return An error string when the solve cannot run (no participating
     *         materials, nodeCount > 32 on a GPU without fallback, etc.)
     */
    Result<void, String> SetThermalTime(f64 time_h);

    /**
     * @brief Status snapshot for the panel
     */
    [[nodiscard]] ThermalSolveStatus GetThermalSolveStatus() const;

    /**
     * @brief What one element did between two hours
     *
     * Replays the trajectory at @p samples evenly spaced instants and reports
     * the temperatures and, where the stepper decomposes its own balance, the
     * six fluxes that produced them. Nothing is re-solved: the checkpoints are
     * already there and this steps between them, which is why a probe is
     * cheap enough to move around a scene.
     *
     * Read-only in every sense that matters -- the field the viewport is
     * showing is not disturbed, and the hour it is showing is restored before
     * this returns.
     *
     * The fluxes are decomposed by the reference CPU balance whichever stepper
     * produced the trajectory, so the six numbers do not change with whether
     * the machine has a GPU. What they describe is the balance AT the state
     * the trajectory reached, which is the trajectory's own either way.
     *
     * @param element  index into the thermal mesh, as a viewport pick reports it
     * @param fromHour, toHour  the stretch to sample; swapped if given backwards
     * @param samples  how many instants, at least two
     *
     * @return An error when there is no solve, the element is out of range, or
     *         fewer than two samples were asked for.
     */
    Result<ThermalElementTrajectory, String> GetElementTrajectory(
        u32 element, f64 fromHour, f64 toHour, u32 samples = 96);

    /**
     * @brief The thermal element a pick landed on
     *
     * A pick reports an instance and a triangle within it; the solve indexes
     * its elements by a flat number, and the map between them is the mesh's.
     * Without this a host holding a PickResult has no way to name the element
     * GetElementTrajectory wants, which is the whole path from a click in the
     * viewport to a chart of that surface's day.
     *
     * @return An error when there is no solve, the pick did not hit, or the
     *         instance it hit is not one the solve carries -- geometry that
     *         does not participate is a real answer, not a zero element.
     */
    [[nodiscard]] Result<u32, String> ThermalElementAt(const PickResult& pick) const;

    /**
     * @brief Show what a material parameter would do, before the re-solve says so
     *
     * The viewport renders T + dT/dp * step instead of T: a first order
     * preview of a slider, exact in the limit of a small step and wrong in the
     * way a linearisation is wrong for a large one. What it is for is the wait
     * -- a re-solve of a day is seconds and a slider is continuous, so the
     * preview is what the user sees while dragging and the solved field is
     * what replaces it when they stop.
     *
     * A step of zero turns it off, which is the state every scene starts in.
     * The tangent is a field of the hour like the temperatures, so scrubbing
     * time moves it too.
     *
     * @return An error when the solve does not carry a derivative with respect
     *         to that parameter. Asking for one is a ThermalSolveParams change
     *         and rebuilds the trajectory, which is why this does not do it
     *         quietly on the caller's behalf.
     */
    Result<void, String> SetThermalWhatIf(ThermalSensitivityParameter parameter, f64 step);

    /**
     * @brief The dT/dp field a what-if preview is built on, one value per element
     *
     * Kelvin per unit of the parameter, at the hour the viewport is showing.
     * The panel driving the preview wants it to say what range a slider is
     * working over; a test wants it because it is the claim the preview makes,
     * and comparing it against a re-solve is the only reading of that claim
     * which is not a restatement of the code.
     *
     * @return An error when the solve does not carry that derivative.
     */
    [[nodiscard]] Result<Vector<f32>, String> GetThermalParameterSensitivity(
        ThermalSensitivityParameter parameter) const;

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
     * @brief Bind (or clear) the spectrum a material emits
     * @param materialIndex  Material to change
     * @param sourceOrEmpty  A built-in token ("d65", "illuminant_a", "halogen",
     *                       "cie_f7", "blackbody_3000k", ...) or a path to a
     *                       table. Empty clears the binding.
     * @param scale          "match_luminance" (the default) or "absolute"
     * @return Warnings worth showing the user, or an error if nothing was bound
     *
     * The interactive twin of `[material_overrides] emissive_curve`, and it goes
     * through the same ResolveEmissionSpectrum the config path does -- a second
     * reading of the levelling rule would put the viewport and the CLI on
     * different lamps.
     *
     * Why this exists rather than the caller doing AddSpectralCurve() and
     * setting an index: binding a lamp is not one assignment. The curve has to
     * be resampled onto the band being rendered, levelled against the material's
     * own emissive triple (published illuminants are relative and carry no
     * absolute level), and then that triple has to be REPLACED by the colour the
     * curve integrates to, or the RGB preview and the emitter-sampling density
     * end up describing a different lamp from the spectral bands. All of that is
     * one decision, and it belongs on one side of the interface.
     *
     * A non-empty result is not a failure. Emission is zero outside a curve's
     * measured span rather than held flat, so a lamp that does not cover the
     * band comes back with a warning saying it will be dark there -- which is
     * the honest answer, and the one a user needs to see.
     *
     * @param baseDir  Directory a relative path resolves against; unused, and
     *                 safely empty, for a built-in token.
     */
    Result<Vector<String>, String> SetMaterialEmissionSpectrum(
        u32 materialIndex, const String& sourceOrEmpty,
        const String& scale = "match_luminance", const String& baseDir = "");

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
     * @brief GPU time of the most recent trace dispatch, in milliseconds
     *
     * Measured with Vulkan timestamp queries around the ray-tracing dispatch
     * alone -- post-processing and the blit are excluded -- so one dispatch
     * being one sample, this is the per-sample cost of the current scene.
     * Results lag by however many frames the host keeps in flight (they are
     * resolved without blocking at the start of the next RenderFrame), and
     * the value is 0 until the first dispatch completes.
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
     * @param x X coordinate in the target extent (pixels, 0 = left) -- the
     *        pixel the host presented into, which below a render scale of 1.0
     *        is mapped into the smaller render grid here
     * @param y Y coordinate in the target extent (pixels, 0 = top)
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
     * @param x X coordinate (pixels, 0 = left), same target-extent space as
     *        ReadPixelValue
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
