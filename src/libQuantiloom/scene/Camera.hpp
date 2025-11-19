#pragma once

#include "core/Types.hpp"
#include "core/Config.hpp"
#include <glm/glm.hpp>

// ============================================================================
// Camera - Pinhole camera for ray tracing
// ============================================================================
// Supports perspective projection with look-at interface
// Used to generate camera rays in ray tracing shaders
// ============================================================================

namespace quantiloom {

// Camera data structure (matches shader push constants)
struct CameraData {
    glm::vec3 origin;        // Camera position (world space)
    f32 fovScale;            // tan(fovY / 2)
    glm::vec3 forward;       // Forward vector (normalized)
    f32 aspectRatio;         // Width / height
    glm::vec3 right;         // Right vector (normalized)
    f32 wavelength_nm;       // Current wavelength (nanometers) for spectral rendering
    glm::vec3 up;            // Up vector (normalized)
    f32 _pad1;               // Padding for alignment
};

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
    glm::vec3 GetPosition() const { return m_position; }
    glm::vec3 GetLookAt() const { return m_lookAt; }
    glm::vec3 GetUp() const { return m_up; }
    glm::vec3 GetForward() const { return m_forward; }
    glm::vec3 GetRight() const { return m_right; }
    f32 GetFovY() const { return m_fovYDegrees; }
    f32 GetAspectRatio() const { return m_aspectRatio; }

    // Get camera data for GPU (push constants)
    CameraData GetCameraData() const;

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
