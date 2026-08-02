// ============================================================================
// Quantiloom - Unit Tests for scene/Camera.hpp
// ============================================================================
// Tests cover:
// - Camera construction and parameter setting
// - Look-at transformation
// - FOV and aspect ratio handling
// - Camera data generation for GPU
// - Vector orthogonality and normalization
// ============================================================================

#include <gtest/gtest.h>
#include "scene/Camera.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <cmath>

using namespace quantiloom;

// Helper function to check if vectors are approximately equal
bool VecEqual(const glm::vec3& a, const glm::vec3& b, f32 epsilon = 1e-5f) {
    return glm::length(a - b) < epsilon;
}

// Helper function to check if a vector is normalized
bool IsNormalized(const glm::vec3& v, f32 epsilon = 1e-5f) {
    return std::abs(glm::length(v) - 1.0f) < epsilon;
}

// ============================================================================
// Construction Tests
// ============================================================================

TEST(CameraTest, DefaultConstruction) {
    Camera cam;

    EXPECT_EQ(cam.GetPosition(), glm::vec3(0, 2, -8));
    EXPECT_EQ(cam.GetLookAt(), glm::vec3(0, 1, 0));
    EXPECT_EQ(cam.GetUp(), glm::vec3(0, 1, 0));
    EXPECT_EQ(cam.GetFovY(), 60.0f);
    EXPECT_NEAR(cam.GetAspectRatio(), 16.0f / 9.0f, 1e-6f);
}

TEST(CameraTest, ParameterizedConstruction) {
    glm::vec3 position(5, 3, -10);
    glm::vec3 lookAt(0, 0, 0);
    glm::vec3 up(0, 1, 0);
    f32 fovY = 45.0f;
    f32 aspectRatio = 16.0f / 9.0f;

    Camera cam(position, lookAt, up, fovY, aspectRatio);

    EXPECT_EQ(cam.GetPosition(), position);
    EXPECT_EQ(cam.GetLookAt(), lookAt);
    // Note: GetUp() returns orthonormalized up, not necessarily equal to input
    EXPECT_TRUE(IsNormalized(cam.GetUp()));
    EXPECT_EQ(cam.GetFovY(), fovY);
    EXPECT_EQ(cam.GetAspectRatio(), aspectRatio);
}

// ============================================================================
// Look-At Transformation Tests
// ============================================================================

TEST(CameraTest, ForwardVectorComputation) {
    glm::vec3 position(0, 0, 5);
    glm::vec3 lookAt(0, 0, 0);
    glm::vec3 up(0, 1, 0);

    Camera cam(position, lookAt, up);

    glm::vec3 forward = cam.GetForward();

    // Forward should point from position to lookAt (and be normalized)
    glm::vec3 expectedForward = glm::normalize(lookAt - position);

    EXPECT_TRUE(VecEqual(forward, expectedForward));
    EXPECT_TRUE(IsNormalized(forward));
}

TEST(CameraTest, RightVectorComputation) {
    glm::vec3 position(0, 0, 5);
    glm::vec3 lookAt(0, 0, 0);
    glm::vec3 up(0, 1, 0);

    Camera cam(position, lookAt, up);

    glm::vec3 forward = cam.GetForward();
    glm::vec3 right = cam.GetRight();

    // Right should be perpendicular to both forward and up
    EXPECT_NEAR(glm::dot(right, forward), 0.0f, 1e-5f);
    EXPECT_NEAR(glm::dot(right, up), 0.0f, 1e-5f);
    EXPECT_TRUE(IsNormalized(right));
}

TEST(CameraTest, UpVectorComputation) {
    glm::vec3 position(0, 0, 5);
    glm::vec3 lookAt(0, 0, 0);
    glm::vec3 up(0, 1, 0);

    Camera cam(position, lookAt, up);

    glm::vec3 forward = cam.GetForward();
    glm::vec3 right = cam.GetRight();
    glm::vec3 computedUp = cam.GetUp();

    // Up should be perpendicular to both forward and right
    EXPECT_NEAR(glm::dot(computedUp, forward), 0.0f, 1e-5f);
    EXPECT_NEAR(glm::dot(computedUp, right), 0.0f, 1e-5f);
    EXPECT_TRUE(IsNormalized(computedUp));
}

TEST(CameraTest, UpReferenceSurvivesOrthonormalization) {
    // A pitched camera: forward is not horizontal, so the derived up tilts.
    Camera cam(glm::vec3(0, 5, 5), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));

    // The derived up is orthonormal, and therefore not (0,1,0) here...
    EXPECT_TRUE(IsNormalized(cam.GetUp()));
    EXPECT_FALSE(VecEqual(cam.GetUp(), glm::vec3(0, 1, 0)));

    // ...but the reference is exactly what the author said, untouched by the
    // orthonormalization. A host saving camera state reads this one back.
    EXPECT_EQ(cam.GetUpReference(), glm::vec3(0, 1, 0));

    // Moving the camera does not rewrite the reference either.
    cam.SetPosition(glm::vec3(5, 5, 0));
    EXPECT_EQ(cam.GetUpReference(), glm::vec3(0, 1, 0));
}

TEST(CameraTest, NoRollDriftAcrossMoves) {
    // Orbit a pitched camera a quarter turn in small steps. UpdateVectors used
    // to re-derive from the previous step's derived up, so each step leaked a
    // little of the old view plane into the new basis and roll accumulated.
    // The basis after N moves must match a fresh camera built at the final
    // pose -- the path taken to get there must not be visible in it.
    const glm::vec3 target(0, 0, 0);
    const f32 pitchY = 5.0f;
    const f32 radius = 5.0f;

    Camera cam(glm::vec3(0, pitchY, radius), target, glm::vec3(0, 1, 0));

    const int steps = 32;
    for (int i = 1; i <= steps; ++i) {
        const f32 angle = glm::half_pi<f32>() * static_cast<f32>(i) /
                          static_cast<f32>(steps);
        cam.SetPosition(glm::vec3(radius * std::sin(angle), pitchY,
                                  radius * std::cos(angle)));
    }

    Camera fresh(cam.GetPosition(), target, glm::vec3(0, 1, 0));
    EXPECT_TRUE(VecEqual(cam.GetUp(), fresh.GetUp()));
    EXPECT_TRUE(VecEqual(cam.GetRight(), fresh.GetRight()));
    EXPECT_TRUE(VecEqual(cam.GetForward(), fresh.GetForward()));
}

TEST(CameraTest, OrthonormalBasis) {
    glm::vec3 position(5, 3, -10);
    glm::vec3 lookAt(0, 1, 0);
    glm::vec3 up(0, 1, 0);

    Camera cam(position, lookAt, up);

    glm::vec3 forward = cam.GetForward();
    glm::vec3 right = cam.GetRight();
    glm::vec3 computedUp = cam.GetUp();

    // Check orthonormality: all vectors should be normalized and perpendicular
    EXPECT_TRUE(IsNormalized(forward));
    EXPECT_TRUE(IsNormalized(right));
    EXPECT_TRUE(IsNormalized(computedUp));

    EXPECT_NEAR(glm::dot(forward, right), 0.0f, 1e-5f);
    EXPECT_NEAR(glm::dot(forward, computedUp), 0.0f, 1e-5f);
    EXPECT_NEAR(glm::dot(right, computedUp), 0.0f, 1e-5f);
}

// ============================================================================
// Setters Tests
// ============================================================================

TEST(CameraTest, SetPosition) {
    Camera cam;

    glm::vec3 newPosition(10, 5, -20);
    cam.SetPosition(newPosition);

    EXPECT_EQ(cam.GetPosition(), newPosition);
}

TEST(CameraTest, SetLookAt) {
    Camera cam;

    glm::vec3 newLookAt(5, 2, 0);
    cam.SetLookAt(newLookAt);

    EXPECT_EQ(cam.GetLookAt(), newLookAt);
}

TEST(CameraTest, SetUp) {
    // Create a camera looking along +X axis, so Y-up can be changed to Z-up
    Camera cam(glm::vec3(0, 0, 0), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0));

    glm::vec3 newUp(0, 0, 1);  // Z-up instead of Y-up
    cam.SetUp(newUp);

    // Note: GetUp() returns orthonormalized up, not necessarily equal to input
    EXPECT_TRUE(IsNormalized(cam.GetUp()));
    // With forward along +X, setting up to +Z should result in up close to +Z
    EXPECT_GT(glm::abs(glm::dot(cam.GetUp(), newUp)), 0.9f);
}

TEST(CameraTest, SetFovY) {
    Camera cam;

    cam.SetFovY(90.0f);
    EXPECT_EQ(cam.GetFovY(), 90.0f);

    cam.SetFovY(30.0f);
    EXPECT_EQ(cam.GetFovY(), 30.0f);
}

TEST(CameraTest, SetAspectRatio) {
    Camera cam;

    cam.SetAspectRatio(1.0f);  // Square
    EXPECT_EQ(cam.GetAspectRatio(), 1.0f);

    cam.SetAspectRatio(21.0f / 9.0f);  // Ultrawide
    EXPECT_NEAR(cam.GetAspectRatio(), 21.0f / 9.0f, 1e-6f);
}

// ============================================================================
// Camera Data Generation Tests
// ============================================================================

TEST(CameraTest, GetCameraDataOrigin) {
    glm::vec3 position(5, 3, -10);
    Camera cam(position, glm::vec3(0, 0, 0));

    CameraData data = cam.GetCameraData();

    EXPECT_EQ(data.origin, position);
}

TEST(CameraTest, GetCameraDataVectors) {
    glm::vec3 position(0, 0, 5);
    glm::vec3 lookAt(0, 0, 0);
    Camera cam(position, lookAt);

    CameraData data = cam.GetCameraData();

    EXPECT_EQ(data.forward, cam.GetForward());
    EXPECT_EQ(data.right, cam.GetRight());
    EXPECT_EQ(data.up, cam.GetUp());
}

TEST(CameraTest, GetCameraDataFovScale) {
    Camera cam;
    cam.SetFovY(60.0f);

    CameraData data = cam.GetCameraData();

    // fovScale = tan(fovY / 2)
    f32 expectedFovScale = std::tan(glm::radians(60.0f) / 2.0f);
    EXPECT_NEAR(data.fovScale, expectedFovScale, 1e-6f);
}

TEST(CameraTest, GetCameraDataAspectRatio) {
    Camera cam;
    cam.SetAspectRatio(16.0f / 9.0f);

    CameraData data = cam.GetCameraData();

    EXPECT_NEAR(data.aspectRatio, 16.0f / 9.0f, 1e-6f);
}

// ============================================================================
// Special Camera Configurations
// ============================================================================

TEST(CameraTest, LookingDown) {
    glm::vec3 position(0, 10, 0);
    glm::vec3 lookAt(0, 0, 0);    // Looking down
    glm::vec3 up(0, 0, -1);       // Z-axis as up

    Camera cam(position, lookAt, up);

    glm::vec3 forward = cam.GetForward();

    // Forward should point downward (negative Y)
    EXPECT_LT(forward.y, 0.0f);
    EXPECT_TRUE(IsNormalized(forward));
}

TEST(CameraTest, LookingUp) {
    glm::vec3 position(0, 0, 0);
    glm::vec3 lookAt(0, 10, 0);   // Looking up
    glm::vec3 up(0, 0, 1);

    Camera cam(position, lookAt, up);

    glm::vec3 forward = cam.GetForward();

    // Forward should point upward (positive Y)
    EXPECT_GT(forward.y, 0.0f);
    EXPECT_TRUE(IsNormalized(forward));
}

TEST(CameraTest, SidewaysOrientation) {
    glm::vec3 position(10, 0, 0);
    glm::vec3 lookAt(0, 0, 0);    // Looking left along X-axis
    glm::vec3 up(0, 1, 0);

    Camera cam(position, lookAt, up);

    glm::vec3 forward = cam.GetForward();
    glm::vec3 right = cam.GetRight();

    // Forward should point along negative X
    EXPECT_LT(forward.x, 0.0f);
    EXPECT_NEAR(forward.y, 0.0f, 1e-5f);
    EXPECT_NEAR(forward.z, 0.0f, 1e-5f);

    // Right should point along Z-axis
    EXPECT_NEAR(right.x, 0.0f, 1e-5f);
}

// ============================================================================
// FOV and Projection Tests
// ============================================================================

TEST(CameraTest, WideFOV) {
    Camera cam;
    cam.SetFovY(120.0f);  // Wide-angle

    CameraData data = cam.GetCameraData();

    f32 expectedFovScale = std::tan(glm::radians(120.0f) / 2.0f);
    EXPECT_NEAR(data.fovScale, expectedFovScale, 1e-5f);
    EXPECT_GT(data.fovScale, 1.0f);  // Wide FOV has large scale
}

TEST(CameraTest, NarrowFOV) {
    Camera cam;
    cam.SetFovY(20.0f);  // Telephoto

    CameraData data = cam.GetCameraData();

    f32 expectedFovScale = std::tan(glm::radians(20.0f) / 2.0f);
    EXPECT_NEAR(data.fovScale, expectedFovScale, 1e-5f);
    EXPECT_LT(data.fovScale, 0.2f);  // Narrow FOV has small scale
}

// ============================================================================
// Aspect Ratio Tests
// ============================================================================

TEST(CameraTest, SquareAspectRatio) {
    Camera cam;
    cam.SetAspectRatio(1.0f);

    CameraData data = cam.GetCameraData();
    EXPECT_EQ(data.aspectRatio, 1.0f);
}

TEST(CameraTest, WidescreenAspectRatio) {
    Camera cam;
    cam.SetAspectRatio(16.0f / 9.0f);

    CameraData data = cam.GetCameraData();
    EXPECT_NEAR(data.aspectRatio, 1.777777f, 1e-5f);
}

TEST(CameraTest, UltrawideAspectRatio) {
    Camera cam;
    cam.SetAspectRatio(21.0f / 9.0f);

    CameraData data = cam.GetCameraData();
    EXPECT_NEAR(data.aspectRatio, 2.333333f, 1e-5f);
}

TEST(CameraTest, PortraitAspectRatio) {
    Camera cam;
    cam.SetAspectRatio(9.0f / 16.0f);  // Portrait orientation

    CameraData data = cam.GetCameraData();
    EXPECT_NEAR(data.aspectRatio, 0.5625f, 1e-5f);
    EXPECT_LT(data.aspectRatio, 1.0f);
}

// ============================================================================
// Edge Cases
// ============================================================================

// A degenerate basis is not a crash -- glm::normalize returns NaN, which
// propagates into every primary ray and renders garbage with nothing logged.
// UpdateVectors gives both degenerate inputs a defined result instead.

TEST(CameraTest, PositionSameAsLookAt) {
    glm::vec3 position(0, 0, 0);
    glm::vec3 lookAt(0, 0, 0);  // Same as position -> zero view vector

    Camera cam(position, lookAt);

    // Defined fallback direction, and a usable orthonormal basis around it.
    glm::vec3 forward = cam.GetForward();
    EXPECT_TRUE(IsNormalized(forward));
    EXPECT_FALSE(std::isnan(forward.x));
    EXPECT_TRUE(IsNormalized(cam.GetRight()));
    EXPECT_TRUE(IsNormalized(cam.GetUp()));
    EXPECT_NEAR(glm::dot(cam.GetForward(), cam.GetRight()), 0.0f, 1e-5f);
    EXPECT_NEAR(glm::dot(cam.GetForward(), cam.GetUp()), 0.0f, 1e-5f);
}

TEST(CameraTest, LookStraightDownWithDefaultUp) {
    // forward parallel to up: an ordinary top-down camera, not an exotic case.
    Camera cam(glm::vec3(0, 10, 0), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));

    EXPECT_TRUE(IsNormalized(cam.GetForward()));
    EXPECT_TRUE(IsNormalized(cam.GetRight()));
    EXPECT_TRUE(IsNormalized(cam.GetUp()));
    EXPECT_FALSE(std::isnan(cam.GetRight().x));
    EXPECT_NEAR(cam.GetForward().y, -1.0f, 1e-5f);
    EXPECT_NEAR(glm::dot(cam.GetForward(), cam.GetRight()), 0.0f, 1e-5f);
    EXPECT_NEAR(glm::dot(cam.GetRight(), cam.GetUp()), 0.0f, 1e-5f);
}

TEST(CameraTest, LookStraightUpWithDefaultUp) {
    Camera cam(glm::vec3(0, 0, 0), glm::vec3(0, 10, 0), glm::vec3(0, 1, 0));

    EXPECT_TRUE(IsNormalized(cam.GetForward()));
    EXPECT_TRUE(IsNormalized(cam.GetRight()));
    EXPECT_FALSE(std::isnan(cam.GetRight().x));
    EXPECT_NEAR(cam.GetForward().y, 1.0f, 1e-5f);
    EXPECT_NEAR(glm::dot(cam.GetForward(), cam.GetRight()), 0.0f, 1e-5f);
}

TEST(CameraTest, VerySmallFOV) {
    Camera cam;
    cam.SetFovY(1.0f);  // Very narrow

    CameraData data = cam.GetCameraData();
    EXPECT_GT(data.fovScale, 0.0f);
}

TEST(CameraTest, ExtremeWideAngle) {
    Camera cam;
    cam.SetFovY(179.0f);  // Nearly 180 degrees

    CameraData data = cam.GetCameraData();
    EXPECT_GT(data.fovScale, 10.0f);  // Very large scale
}
