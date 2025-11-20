#include "Camera.hpp"
#include "core/Log.hpp"
#include <glm/gtc/matrix_transform.hpp>

namespace quantiloom {

Camera::Camera(const glm::vec3& position, const glm::vec3& lookAt,
               const glm::vec3& up, f32 fovYDegrees, f32 aspectRatio)
    : m_position(position)
    , m_lookAt(lookAt)
    , m_up(glm::normalize(up))
    , m_fovYDegrees(fovYDegrees)
    , m_aspectRatio(aspectRatio)
{
    UpdateVectors();
}

void Camera::SetPosition(const glm::vec3& position) {
    m_position = position;
    UpdateVectors();
}

void Camera::SetLookAt(const glm::vec3& lookAt) {
    m_lookAt = lookAt;
    UpdateVectors();
}

void Camera::SetUp(const glm::vec3& up) {
    m_up = glm::normalize(up);
    UpdateVectors();
}

void Camera::SetFovY(f32 fovYDegrees) {
    m_fovYDegrees = fovYDegrees;
}

void Camera::SetAspectRatio(f32 aspectRatio) {
    m_aspectRatio = aspectRatio;
}

CameraData Camera::GetCameraData() const {
    CameraData data;
    data.origin = m_position;
    data.forward = m_forward;
    data.right = m_right;
    data.up = m_up;
    data.fovScale = glm::tan(glm::radians(m_fovYDegrees) * 0.5f);
    data.aspectRatio = m_aspectRatio;
    data.wavelength_nm = 550.0f;  // Default wavelength (will be overridden by renderer)
    return data;
}

void Camera::UpdateVectors() {
    // Compute forward (view direction)
    m_forward = glm::normalize(m_lookAt - m_position);

    // Compute right vector (perpendicular to forward and up)
    m_right = glm::normalize(glm::cross(m_forward, m_up));

    // Recompute orthogonal up vector
    m_up = glm::normalize(glm::cross(m_right, m_forward));

    QL_LOG_DEBUG("Camera updated: pos=({:.2f}, {:.2f}, {:.2f}), fwd=({:.2f}, {:.2f}, {:.2f})",
                 m_position.x, m_position.y, m_position.z,
                 m_forward.x, m_forward.y, m_forward.z);
}

Result<Camera, String> Camera::FromConfig(const Config& config, f32 aspectRatio) {
    // Read camera position
    auto posArray = config.GetArray<f32>("camera.position");
    if (posArray.size() != 3) {
        return Result<Camera, String>::Err("camera.position must be array of 3 floats, but got " + std::to_string(posArray.size()));
    }
    glm::vec3 position(posArray[0], posArray[1], posArray[2]);

    // Read look-at target
    auto lookAtArray = config.GetArray<f32>("camera.look_at");
    if (lookAtArray.size() != 3) {
        return Result<Camera, String>::Err("camera.look_at must be array of 3 floats, but got " + std::to_string(lookAtArray.size()));
    }
    glm::vec3 lookAt(lookAtArray[0], lookAtArray[1], lookAtArray[2]);

    // Read up vector (optional, defaults to Y-up)
    auto upArray = config.GetArray<f32>("camera.up");
    glm::vec3 up(0.0f, 1.0f, 0.0f);  // Default: Y-up
    if (!upArray.empty()) {
        if (upArray.size() != 3) {
            return Result<Camera, String>::Err("camera.up must be array of 3 floats, but got " + std::to_string(upArray.size()));
        }
        up = glm::vec3(upArray[0], upArray[1], upArray[2]);
    }

    // Read FOV (optional, defaults to 60 degrees)
    f32 fovY = config.Get<f32>("camera.fov_y", 60.0f);

    // Create camera
    Camera camera(position, lookAt, up, fovY, aspectRatio);

    QL_LOG_INFO("Camera loaded from config:");
    QL_LOG_INFO("  Position: ({:.2f}, {:.2f}, {:.2f})", position.x, position.y, position.z);
    QL_LOG_INFO("  LookAt:   ({:.2f}, {:.2f}, {:.2f})", lookAt.x, lookAt.y, lookAt.z);
    QL_LOG_INFO("  FOV:      {:.1f}°", fovY);

    return camera;  // Direct construction of Result
}

} // namespace quantiloom
