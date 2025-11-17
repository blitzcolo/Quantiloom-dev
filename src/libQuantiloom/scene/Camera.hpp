#pragma once

#include "core/Types.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>  // For lookAt, perspective

// ============================================================================
// Camera - Pinhole camera model
// ============================================================================
// Defines:
// - Position and orientation (position, lookAt, up)
// - Field of view (fov) and aspect ratio
// - Image resolution (width, height)
//
// Usage in rendering:
// - Used to compute ray generation parameters in raygen shader
// - Camera transformations applied on CPU side before rendering
// ============================================================================

namespace quantiloom {

struct Camera {
    // Spatial parameters
    glm::vec3 position{0.0f, 0.0f, 5.0f};  // Camera position in world space
    glm::vec3 lookAt{0.0f, 0.0f, 0.0f};    // Look-at point
    glm::vec3 up{0.0f, 1.0f, 0.0f};        // Up vector (world space)

    // Optical parameters
    f32 fov = 45.0f;  // Vertical field of view (degrees)

    // Image resolution
    u32 width = 1280;
    u32 height = 720;

    // ========================================================================
    // Derived Properties
    // ========================================================================

    // Get aspect ratio
    f32 GetAspectRatio() const {
        return static_cast<f32>(width) / static_cast<f32>(height);
    }

    // Get view matrix (right-handed, looking down -Z)
    glm::mat4 GetViewMatrix() const {
        return glm::lookAt(position, lookAt, up);
    }

    // Get projection matrix (perspective, right-handed)
    glm::mat4 GetProjectionMatrix() const {
        return glm::perspective(glm::radians(fov), GetAspectRatio(), 0.1f, 1000.0f);
    }

    // Get view direction (normalized)
    glm::vec3 GetForward() const {
        return glm::normalize(lookAt - position);
    }
};

} // namespace quantiloom
