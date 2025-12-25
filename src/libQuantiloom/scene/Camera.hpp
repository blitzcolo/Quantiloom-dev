/**
 * @file Camera.hpp
 * @brief Pinhole camera model for ray tracing and spectral rendering
 *
 * Provides Camera class for perspective projection with look-at interface.
 * Used to generate primary camera rays in ray tracing shaders.
 *
 * Features:
 * - Position/look-at/up vector configuration
 * - Field-of-view (FOV) control
 * - Aspect ratio handling
 * - GPU-compatible data export (CameraData struct)
 * - TOML configuration loading
 *
 * Camera rays are generated in raygen.rgen shader using CameraData parameters.
 * The camera uses a standard right-handed coordinate system (OpenGL convention).
 *
 * @author wtflmao
 */

#pragma once

#include "core/Types.hpp"
#include "core/Config.hpp"
#include <glm/glm.hpp>

// ============================================================================
// Camera - Pinhole camera for ray tracing
// ============================================================================

namespace quantiloom {

/**
 * @struct CameraData
 * @brief GPU-compatible camera data structure (matches shader CameraData in common.hlsli)
 *
 * Contains all parameters needed for ray generation in raygen shader:
 * - Camera position and orientation vectors
 * - FOV scale (tan(fovY/2)) for ray direction calculation
 * - Aspect ratio for x/y scaling
 * - Wavelength and spectral mode for spectral rendering
 *
 * @note MUST match shader definition in common.hlsli exactly
 * @note Used in push constants or uniform buffers
 */
struct CameraData {
    glm::vec3 origin;        // Camera position (world space)
    f32 fovScale;            // tan(fovY / 2)
    glm::vec3 forward;       // Forward vector (normalized)
    f32 aspectRatio;         // Width / height
    glm::vec3 right;         // Right vector (normalized)
    f32 wavelength_nm;       // Current wavelength (nanometers) for spectral rendering
    glm::vec3 up;            // Up vector (normalized)
    u32 spectral_mode;       // Spectral rendering mode (see SpectralMode enum)
};

/**
 * @struct PushConstantsRayGen
 * @brief Push constants for ray generation shader (accumulative sampling)
 *
 * Includes camera data plus sampling parameters for progressive refinement:
 * - frameIndex: Temporal counter (for animations, motion blur)
 * - sampleIndex: Current sample within frame (0 to totalSamples-1)
 * - totalSamples: Total SPP (samples per pixel) for this render
 * - randomSeed: Per-frame random seed for stochastic sampling
 *
 * @note MUST match PushConstantsRayGen in raygen.rgen shader
 */
struct PushConstantsRayGen {
    CameraData camera;       // Camera parameters
    u32 frameIndex;          // Frame counter (for temporal effects)
    u32 sampleIndex;         // Current sample index (0 to spp-1)
    u32 totalSamples;        // Total samples per pixel (spp)
    u32 randomSeed;          // Random seed for this frame
};

/**
 * @class Camera
 * @brief Pinhole camera model for ray tracing with perspective projection
 *
 * Implements a standard pinhole camera with:
 * - Look-at interface (position, target, up vector)
 * - Field-of-view control (vertical FOV in degrees)
 * - Automatic aspect ratio handling
 * - Right-handed coordinate system (OpenGL convention)
 *
 * Usage example:
 * @code
 * // Create camera
 * Camera cam(
 *     glm::vec3(0, 2, -8),  // position
 *     glm::vec3(0, 1, 0),   // look-at target
 *     glm::vec3(0, 1, 0),   // up vector
 *     60.0f,                // vertical FOV (degrees)
 *     1920.0f / 1080.0f     // aspect ratio
 * );
 *
 * // Export for GPU
 * CameraData cameraData = cam.GetCameraData();
 * cameraData.wavelength_nm = 550.0f;  // Set wavelength for spectral rendering
 * cameraData.spectral_mode = static_cast<u32>(SpectralMode::Single);
 *
 * // Or load from config
 * auto camResult = Camera::FromConfig(config, aspectRatio);
 * @endcode
 *
 * @note Camera rays use perspective projection (pinhole model, no depth of field)
 * @note Right vector is computed as cross(forward, up) and normalized
 * @note Up vector is recomputed as cross(right, forward) for orthonormality
 *
 * @see CameraData for GPU-compatible data structure
 * @see raygen.rgen for ray generation shader implementation
 */
class QL_API Camera {
public:
    Camera() = default;

    // Construct camera from position and look-at target
    Camera(const glm::vec3& position, const glm::vec3& lookAt,
           const glm::vec3& up = glm::vec3(0, 1, 0),
           f32 fovYDegrees = 60.0f, f32 aspectRatio = 16.0f / 9.0f);

    // Setters
    void SetPosition(const glm::vec3& position);
    void SetLookAt(const glm::vec3& lookAt);
    void SetUp(const glm::vec3& up);
    void SetFovY(f32 fovYDegrees);
    void SetAspectRatio(f32 aspectRatio);

    // Accessors
    [[nodiscard]] glm::vec3 GetPosition() const { return m_position; }
    [[nodiscard]] glm::vec3 GetLookAt() const { return m_lookAt; }
    [[nodiscard]] glm::vec3 GetUp() const { return m_up; }
    [[nodiscard]] glm::vec3 GetForward() const { return m_forward; }
    [[nodiscard]] glm::vec3 GetRight() const { return m_right; }
    [[nodiscard]] f32 GetFovY() const { return m_fovYDegrees; }
    [[nodiscard]] f32 GetAspectRatio() const { return m_aspectRatio; }

    // Get camera data for GPU (push constants)
    [[nodiscard]] CameraData GetCameraData() const;

    // Load camera from TOML config
    static Result<Camera, String> FromConfig(const Config& config, f32 aspectRatio);

private:
    void UpdateVectors();  // Recompute forward/right/up from position/lookAt

    glm::vec3 m_position{0, 2, -8};
    glm::vec3 m_lookAt{0, 1, 0};
    glm::vec3 m_up{0, 1, 0};
    glm::vec3 m_forward{0, 0, 1};
    glm::vec3 m_right{1, 0, 0};

    f32 m_fovYDegrees = 60.0f;
    f32 m_aspectRatio = 16.0f / 9.0f;
};

} // namespace quantiloom
