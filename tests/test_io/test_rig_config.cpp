#include <gtest/gtest.h>
#include "dataset/RigConfig.hpp"
#include "postprocess/CameraConfigIO.hpp"
#include "postprocess/CameraPresets.hpp"
#include <algorithm>

using namespace quantiloom;
using namespace quantiloom::dataset;

namespace {
Config Job(const String& extra="") {
    auto doc=Config::Parse(R"(
[fusion]
seed=17
[rig]
version=1
id="test-rig"
reference_camera="vis"
rig_to_world=[1,0,0,3,0,1,0,4,0,0,1,5,0,0,0,1]
[[rig.cameras]]
id="vis"
[rig.cameras.sensor]
preset="generic_cmos"
[rig.cameras.sensor.optics]
sensor_width_px=16
sensor_height_px=12
[[rig.cameras]]
id="ir"
camera_to_rig=[1,0,0,0.1,0,1,0,0,0,0,1,0,0,0,0,1]
[rig.cameras.sensor]
preset="generic_uncooled_thermal_ir"
[rig.cameras.sensor.optics]
sensor_width_px=8
sensor_height_px=6
)"+extra);
    if(!doc) throw std::runtime_error(doc.error());
    return *doc;
}
}

TEST(RigConfigTest, StableSeedsDefaultPairsAndTransformComposition) {
    auto rig=ParseRigConfig(Job());
    ASSERT_TRUE(rig) << rig.error();
    ASSERT_EQ(rig.value().pairs.size(),1u);
    EXPECT_EQ(rig.value().pairs[0].sourceCamera,"ir");
    EXPECT_EQ(rig.value().pairs[0].targetCamera,"vis");
    auto scene=Config::Parse("[renderer]\nspp=4\n");
    auto composed=RigCameraScene(*scene,*rig,"ir");
    ASSERT_TRUE(composed) << composed.error();
    auto position=composed.value().GetDoubleArray("camera.position");
    ASSERT_EQ(position.size(),3u);
    EXPECT_DOUBLE_EQ(position[0],3.1);
    EXPECT_DOUBLE_EQ(position[1],4);
    EXPECT_DOUBLE_EQ(position[2],5);
    EXPECT_EQ(composed.value().GetUInt("renderer.spp"),4u);
    std::reverse(rig.value().cameras.begin(),rig.value().cameras.end());
    auto reordered=RigCameraScene(*scene,*rig,"ir");
    ASSERT_TRUE(reordered);
    EXPECT_EQ(composed.value().ToToml(),reordered.value().ToToml());
}

TEST(RigConfigTest, RejectsUnknownReferenceUnsafeIdAndNonRigidPose) {
    auto bad=Job().MergedWith(*Config::Parse("[rig]\nreference_camera=\"missing\"\n"));
    EXPECT_FALSE(ParseRigConfig(bad));
    bad=Job().MergedWith(*Config::Parse("[rig]\nid=\"../escape\"\n"));
    EXPECT_FALSE(ParseRigConfig(bad));
    bad=Job().MergedWith(*Config::Parse("[rig]\nrig_to_world=[2,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]\n"));
    EXPECT_FALSE(ParseRigConfig(bad));
}

TEST(RigConfigTest, ProjectionConfigRoundTripAndFastRgbRejection) {
    auto sensor=camera::MakePresetCameraConfig(camera::CameraPresetKind::GenericIngaas);
    ASSERT_TRUE(sensor);
    auto& p=sensor.value().optics.projection;
    p.model=camera::ProjectionModel::BrownConrady;
    p.explicitIntrinsics=true;p.fx=1300;p.fy=1310;p.cx=299;p.cy=240;
    p.coefficients={-.08,.01,.001,-.002,.0001};
    auto doc=Config::Parse(CameraConfigToToml(*sensor));
    ASSERT_TRUE(doc);
    auto restored=ParseCameraConfig(*doc,SpectralMode::Single);
    ASSERT_TRUE(restored) << restored.error();
    EXPECT_EQ(restored.value().optics.projection.coefficients,p.coefficients);
    EXPECT_DOUBLE_EQ(restored.value().optics.projection.cx,299);
    auto invalid=doc.value().MergedWith(*Config::Parse("[sensor]\ninput_kind=\"fast_rgb\"\n"));
    EXPECT_FALSE(ParseCameraConfig(invalid,SpectralMode::Single));
}
