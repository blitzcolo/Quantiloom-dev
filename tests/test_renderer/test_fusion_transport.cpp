#include <gtest/gtest.h>
#include "renderer/FusionTransport.hpp"
#include "dataset/RigConfig.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include "renderer/OfflineRenderer.hpp"
#include "dataset/OpticalCorrespondence.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>

using namespace quantiloom;
namespace {
Scene Box() {
    Scene scene;
    Material m;m.name="Glass.001";m.roughnessFactor=0;scene.materials.push_back(m);
    GeometryPrimitive p;
    p.positions={{-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},{-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}};
    p.indices={0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7};
    Mesh mesh;mesh.primitives.push_back(p);scene.meshes.push_back(mesh);
    SceneNode node;node.name="box";node.meshIndex=0;scene.nodes.push_back(node);return scene;
}
Config Solid() { return *Config::Parse("[material_overrides.\"Glass.001\"]\nfusion_transport=\"solid\"\nfusion_absorption_m_inv=10\n"); }
}

TEST(FusionTransportTest, ClosedSolidLiteralNameAndInsideInitialization) {
    auto scene=Box();auto records=rendercore::ResolveFusionTransport(Solid(),scene);
    ASSERT_TRUE(records) << records.error();
    ASSERT_EQ(records.value().size(),2u);EXPECT_EQ(records.value()[1].mode,2u);
    EXPECT_TRUE(scene.materials[0].doubleSided);
    auto outside=rendercore::InitialFusionMedia(scene,*records,{0,0,3});
    ASSERT_TRUE(outside);EXPECT_TRUE(outside.value().empty());
    auto inside=rendercore::InitialFusionMedia(scene,*records,{0,0,0});
    ASSERT_TRUE(inside);ASSERT_EQ(inside.value().size(),1u);EXPECT_EQ(inside.value()[0],1u);
}
TEST(FusionTransportTest, RejectsOpenRoughAndScatteringSolids) {
    auto scene=Box();scene.meshes[0].primitives[0].indices.resize(33);
    EXPECT_FALSE(rendercore::ResolveFusionTransport(Solid(),scene));
    scene=Box();scene.materials[0].roughnessFactor=.1f;
    EXPECT_FALSE(rendercore::ResolveFusionTransport(Solid(),scene));
    scene=Box();scene.materials[0].scatteringCoeff=1;
    EXPECT_FALSE(rendercore::ResolveFusionTransport(Solid(),scene));
}
TEST(FusionTransportTest, NestedInitialMediaAreOrderedFromOuterToInner) {
    auto scene=Box();scene.nodes[0].transform=glm::scale(glm::mat4(1),glm::vec3(2));
    auto inner=scene.nodes[0];inner.name="inner";inner.transform=glm::mat4(1);scene.nodes.push_back(inner);
    auto records=rendercore::ResolveFusionTransport(Solid(),scene);ASSERT_TRUE(records);
    auto media=rendercore::InitialFusionMedia(scene,*records,{0,0,0});ASSERT_TRUE(media);
    ASSERT_EQ(media.value().size(),2u);EXPECT_EQ(media.value()[0],1u);EXPECT_EQ(media.value()[1],2u);
}
TEST(FusionTransportTest, ThinSheetBudgetAndRigSerialization) {
    auto scene=Box();auto bad=Config::Parse("[material_overrides.\"Glass.001\"]\nfusion_transport=\"thin_sheet\"\nfusion_sheet_reflectance=0.7\nfusion_sheet_transmittance=0.6\n");
    ASSERT_TRUE(bad);EXPECT_FALSE(rendercore::ResolveFusionTransport(*bad,scene));
    auto literal=Solid().GetTable("material_overrides");ASSERT_TRUE(literal);
    ASSERT_TRUE(literal.value().GetNamedTable("Glass.001"));
}

TEST(FusionTransportGpuTest, CurvedMediaTraceRealBackground) {
    const auto base=std::filesystem::path(QUANTILOOM_SOURCE_ROOT)/"docs/dataset/fusion";
    auto job=Config::Load(base/"curved_transmission_job.toml");ASSERT_TRUE(job);
    auto rig=dataset::ParseRigConfig(*job,base.string());ASSERT_TRUE(rig);
    auto scene=Config::Load(base/"curved_transmission_scene.toml");ASSERT_TRUE(scene);
    auto config=dataset::RigCameraScene(*scene,*rig,"reference");ASSERT_TRUE(config);
    OfflineRenderer::InitParams init;init.baseDir=base.string();
    dataset::FusionPathChunk recorded;
    init.onFusionPathChunk=[&](const dataset::FusionPathChunk& chunk) {
        recorded=chunk;
        if(chunk.diagnosticFlags&2) {
            const auto path=std::filesystem::path(QUANTILOOM_SOURCE_ROOT)/"build/curved-diagnostic.bin";
            std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(chunk.bytes.data()),chunk.bytes.size());
        }
    };
    auto renderer=OfflineRenderer::Create(*config,init);
    if(!renderer)GTEST_SKIP()<<renderer.error();
    camera::CaptureState state;
    auto capture=renderer.value()->CaptureCamera(state,0);
    ASSERT_TRUE(capture)<<capture.error();
    auto geometry=renderer.value()->CaptureGeometry(0);ASSERT_TRUE(geometry);
    auto paths=dataset::DecodeOpticalEndpoints(recorded,"paths",geometry.value().geometry);
    ASSERT_FALSE(paths.empty());
    Vector<dataset::OpticalProbe> probes;
    for(size_t i=0;i<std::min<size_t>(8,paths.size());++i)
        probes.push_back({glm::vec2(paths[i].nativePixel),paths[i].wavelengthNm,paths[i].branchMask});
    auto replay=renderer.value()->QueryOpticalPaths(probes,0);ASSERT_TRUE(replay)<<replay.error();
    for(size_t i=0;i<probes.size();++i) {
        EXPECT_EQ(replay.value()[i].surface.instanceId,paths[i].nodeId)<<"ray "<<i<<" flags "<<replay.value()[i].flags;
        EXPECT_LT(glm::length(glm::dvec3(replay.value()[i].surface.worldPosition)-paths[i].position),1e-4)<<"ray "<<i;
    }
}
