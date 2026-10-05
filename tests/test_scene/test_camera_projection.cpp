#include <gtest/gtest.h>
#include "scene/CameraProjection.hpp"
#include <cmath>

using namespace quantiloom;
using namespace quantiloom::camera;

TEST(CameraProjectionTest, PhysicalDefaultsAndHalfPixelConvention) {
    auto p=ResolveProjection({},640,480,8.0,5.0);
    ASSERT_TRUE(p);
    EXPECT_DOUBLE_EQ(p.value().fx,1600);
    EXPECT_DOUBLE_EQ(p.value().cx,320);
    auto ray=UnprojectPixel(*p,{320,240});
    ASSERT_TRUE(ray.valid);
    EXPECT_DOUBLE_EQ(ray.direction.z,1);
    auto pixel=ProjectDirection(*p,{0,0,1});
    ASSERT_TRUE(pixel.valid);
    EXPECT_DOUBLE_EQ(pixel.pixel.x,320);
    EXPECT_FALSE(ProjectDirection(*p,{0,0,-1}).valid);
}

TEST(CameraProjectionTest, BrownMatchesIndependentForwardFormula) {
    CameraProjection p;
    p.explicitIntrinsics=true; p.fx=450; p.fy=460; p.cx=287; p.cy=209;
    p.model=ProjectionModel::BrownConrady;
    p.coefficients={-0.12,0.025,0.002,-0.003,0.001};
    ASSERT_TRUE(ValidateProjection(p,640,480));
    const double x=.3,y=-.2,r2=x*x+y*y;
    const double radial=1-.12*r2+.025*r2*r2+.001*r2*r2*r2;
    const auto q=ProjectDirection(p,{x,y,1});
    ASSERT_TRUE(q.valid);
    EXPECT_NEAR(q.pixel.x,450*(x*radial+2*.002*x*y-.003*(r2+2*x*x))+287,1e-10);
    EXPECT_NEAR(q.pixel.y,460*(y*radial+.002*(r2+2*y*y)-2*.003*x*y)+209,1e-10);
    for(int j=0;j<24;++j) for(int i=0;i<32;++i) {
        const glm::dvec2 pixel(i*20+.5,j*20+.5);
        auto ray=UnprojectPixel(p,pixel);
        ASSERT_TRUE(ray.valid);
        EXPECT_LT(glm::length(ProjectDirection(p,ray.direction).pixel-pixel),1e-4);
    }
}

TEST(CameraProjectionTest, FisheyeEquidistantAndFieldBoundary) {
    CameraProjection p;
    p.explicitIntrinsics=true; p.fx=p.fy=100; p.cx=p.cy=150;
    p.model=ProjectionModel::Fisheye; p.maxThetaRadians=1.2;
    ASSERT_TRUE(ValidateProjection(p,300,300));
    auto q=ProjectDirection(p,{std::sin(.8),0,std::cos(.8)});
    ASSERT_TRUE(q.valid);
    EXPECT_NEAR(q.pixel.x,230,1e-10);
    auto r=UnprojectPixel(p,q.pixel);
    ASSERT_TRUE(r.valid);
    EXPECT_NEAR(r.direction.x,std::sin(.8),1e-12);
    EXPECT_FALSE(UnprojectPixel(p,{290,150}).valid);
    EXPECT_FALSE(ProjectDirection(p,{std::sin(1.3),0,std::cos(1.3)}).valid);
}

TEST(CameraProjectionTest, RejectsInvalidAndFoldingModels) {
    CameraProjection p;
    p.explicitIntrinsics=true; p.fx=p.fy=100; p.cx=p.cy=150;
    p.model=ProjectionModel::BrownConrady; p.coefficients[0]=-1;
    EXPECT_FALSE(ValidateProjection(p,300,300));
    p.model=ProjectionModel::Fisheye; p.maxThetaRadians=1.5707963267948966;
    EXPECT_FALSE(ValidateProjection(p,300,300));
    p.maxThetaRadians=1.2;
    EXPECT_FALSE(ValidateProjection(p,300,300));
}

TEST(CameraProjectionTest, RejectsFisheyeFoldBetweenFormerUniformChecks) {
    CameraProjection p;p.model=ProjectionModel::Fisheye;p.explicitIntrinsics=true;
    p.fx=p.fy=100;p.cx=p.cy=150;p.maxThetaRadians=1.2;
    const double t=1.2*500.5/1024,a=t*t;
    p.coefficients={-(2.000001)/(3*a),1/(5*a*a),0,0,0};
    EXPECT_FALSE(ValidateProjection(p,300,300));
}

TEST(CameraProjectionTest, DetailedLensValidityDoesNotClamp) {
    CameraProjection p;p.model=ProjectionModel::Fisheye;p.explicitIntrinsics=true;
    p.fx=p.fy=10;p.cx=p.cy=4;p.maxThetaRadians=.3;
    EXPECT_EQ(UnprojectPixelV2(p,{4,4},8,8).status,LensValidity::Valid);
    EXPECT_EQ(UnprojectPixelV2(p,{.5,.5},8,8).status,LensValidity::OutsideField);
    EXPECT_EQ(UnprojectPixelV2(p,{-1,4},8,8).status,LensValidity::OutsideImage);
    p.fx=0;
    EXPECT_EQ(UnprojectPixelV2(p,{4,4},8,8).status,LensValidity::InverseFailed);
}
