#include <gtest/gtest.h>
#include "dataset/ProductGeometry.hpp"
#include <nlohmann/json.hpp>
#include <cmath>

using namespace quantiloom;
TEST(ProductGeometryTest, NativePixelCentresReprojectWithRightDownForwardAxes) {
    Camera camera({2,3,7}, {-1,0,0}, {0,1,0}, 47.0f, 1.7f);
    dataset::ProductGeometry geometry;
    geometry.width = 431;
    geometry.height = 287;
    geometry.referenceTimeSeconds = 12.5;
    geometry.camera = camera.GetCameraData();
    auto description = geometry.ToJson(); ASSERT_TRUE(description);
    const auto j = nlohmann::json::parse(description.value());
    const auto k = j["intrinsics"].get<std::vector<double>>();
    const auto matrix = j["world_to_camera"].get<std::vector<double>>();
    const auto& c = geometry.camera;
    for (double y : {0.5, 143.5, 286.5}) for (double x : {0.5, 215.5, 430.5}) {
        const auto ray = glm::dvec3(c.forward) +
            glm::dvec3(c.right) * ((2*x/geometry.width-1)*c.fovScale*c.aspectRatio) +
            glm::dvec3(c.up) * ((1-2*y/geometry.height)*c.fovScale);
        const auto point = glm::dvec3(c.origin) + 10.0 * ray;
        double local[3]{};
        for (int row=0;row<3;++row)
            local[row] = matrix[row*4]*point.x + matrix[row*4+1]*point.y +
                         matrix[row*4+2]*point.z + matrix[row*4+3];
        EXPECT_GT(local[2], 0.0);
        EXPECT_NEAR(k[0]*local[0]/local[2]+k[2], x, 1e-3);
        EXPECT_NEAR(k[4]*local[1]/local[2]+k[5], y, 1e-3);
    }
}
TEST(ProductGeometryTest, OrthographicHasNoPerspectiveIntrinsics) {
    Camera camera({0,0,3},{0,0,0});
    camera.SetProjection(Camera::Projection::Orthographic);
    camera.SetOrthoHeight(4);
    dataset::ProductGeometry geometry;
    geometry.width=20; geometry.height=10; geometry.camera=camera.GetCameraData();
    const auto description=geometry.ToJson(); ASSERT_TRUE(description);
    const auto j=nlohmann::json::parse(description.value());
    EXPECT_TRUE(j["intrinsics"].is_null());
    EXPECT_EQ(j["projection"], "orthographic");
    EXPECT_DOUBLE_EQ(j["film_height_world_units"].get<double>(), 4.0);
}
TEST(ProductGeometryTest, RejectsNonfiniteAndSingularDescriptions) {
    dataset::ProductGeometry geometry;
    EXPECT_FALSE(geometry.ToJson());
    Camera camera({0,0,3},{0,0,0});
    geometry.width=20; geometry.height=10; geometry.camera=camera.GetCameraData();
    geometry.referenceTimeSeconds=std::numeric_limits<double>::infinity();
    EXPECT_FALSE(geometry.ToJson());
    geometry.referenceTimeSeconds=0;
    geometry.camera.right=geometry.camera.forward;
    EXPECT_FALSE(geometry.ToJson());
}
