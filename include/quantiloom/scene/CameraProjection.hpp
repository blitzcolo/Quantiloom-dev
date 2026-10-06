#pragma once

#include "core/Platform.hpp"
#include "core/Types.hpp"
#include <array>
#include <glm/glm.hpp>

namespace quantiloom::camera {

enum class ProjectionModel : u32 {
    Pinhole,
    BrownConrady,
    Fisheye
};

/// Coordinates are right/down/forward, with centres at (x+0.5,y+0.5).
/// Brown coefficients: k1,k2,p1,p2,k3; fisheye: k1,k2,k3,k4.
/// An unauthored K is resolved from the physical focal length and pixel pitch.
struct CameraProjection {
    ProjectionModel model = ProjectionModel::Pinhole;
    bool explicitIntrinsics = false;
    f64 fx = 0, fy = 0, cx = 0, cy = 0;
    std::array<f64, 5> coefficients{};
    f64 maxThetaRadians = 1.5533430342749532; // 89 degrees, front hemisphere
};

struct ProjectionResult {
    bool valid = false;
    glm::dvec2 pixel{};
};

struct UnprojectionResult {
    bool valid = false;
    glm::dvec3 direction{};
};

enum class LensValidity : u32 {
    Valid = 0,
    OutsideField = 1,
    InverseFailed = 2,
    OutsideImage = 3
};
struct LensResultV2 {
    LensValidity status = LensValidity::InverseFailed;
    glm::dvec3 direction{};
};
QL_API LensResultV2 UnprojectPixelV2(const CameraProjection& projection, const glm::dvec2& pixel,
                                     u32 width, u32 height);

QL_API Result<CameraProjection, String> ResolveProjection(const CameraProjection& authored,
                                                          u32 width, u32 height, f64 focalLengthMm,
                                                          f64 pixelPitchUm);
QL_API Result<void, String> ValidateProjection(const CameraProjection& projection, u32 width,
                                               u32 height);
QL_API ProjectionResult ProjectDirection(const CameraProjection& projection,
                                         const glm::dvec3& direction);
QL_API UnprojectionResult UnprojectPixel(const CameraProjection& projection,
                                         const glm::dvec2& pixel);
QL_API String ProjectionModelName(ProjectionModel model);

} // namespace quantiloom::camera
