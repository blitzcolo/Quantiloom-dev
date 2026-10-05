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
#include <chrono>
#include <nlohmann/json.hpp>
#include "io/ImageIO.hpp"
#include "postprocess/CameraPhysics.hpp"
#include "dataset/FusionExportJob.hpp"

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
    scene=Box();scene.meshes[0].primitives[0].indices.pop_back();
    EXPECT_FALSE(rendercore::ResolveFusionTransport(Solid(),scene));
    scene=Box();scene.materials[0].ior=0;
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

TEST(FusionTransportGpuTest, FisheyeBorderReconstructsEverySampleIndependentOfPathLimit) {
    const auto base=std::filesystem::path(QUANTILOOM_SOURCE_ROOT)/"docs/dataset/fusion";
    auto job=Config::Load(base/"transmission_job.toml");ASSERT_TRUE(job);
    auto rig=dataset::ParseRigConfig(*job,base.string());ASSERT_TRUE(rig);
    auto scene=Config::Load(base/"transmission_scene.toml");ASSERT_TRUE(scene);
    auto config=dataset::RigCameraScene(*scene,*rig,"reference");ASSERT_TRUE(config);
    auto override=Config::Parse(R"(
[material_overrides.Window]
fusion_transport="thin_sheet"
fusion_sheet_reflectance=0.0
fusion_sheet_transmittance=0.0
[sensor.optics.projection]
model="fisheye"
intrinsics=[10.0,10.0,4.0,4.0]
coefficients=[0.0,0.0,0.0,0.0]
max_theta_deg=17.188733853924695
[sensor.products]
traced_radiance=false
)");ASSERT_TRUE(override);
    OfflineRenderer::InitParams init;init.baseDir=base.string();
    auto renderer=OfflineRenderer::Create(config.value().MergedWith(*override),init);
    ASSERT_TRUE(renderer)<<renderer.error();
    dataset::FusionCaptureOptionsV2 options;options.maxRecordedRays=17;
    camera::CaptureState first;
    auto a=renderer.value()->CaptureFusionV2(first,0,options);ASSERT_TRUE(a)<<a.error();
    const auto& fraction=a.value().validSampleFraction;
    ASSERT_GT(fraction(2,1,0),0);ASSERT_LT(fraction(2,1,0),1);
    const auto L=camera::PlanckRadianceWm2SrNm(10000,300);
    const auto omega=camera::ApertureSolidAngleSr(1.4);
    const auto expected=(*L)*(*omega)*std::pow(12e-6,2)*2*fraction(2,1,0);
    EXPECT_NEAR(a.value().linearReference(2,1,0),expected,expected*1e-4);
    EXPECT_EQ(a.value().lensValidity(0,0,0),1);
    camera::CaptureState second;options.maxRecordedRays=0;
    auto b=renderer.value()->CaptureFusionV2(second,0,options);ASSERT_TRUE(b)<<b.error();
    EXPECT_EQ(a.value().linearReference.data,b.value().linearReference.data);
    camera::CaptureState third;options.recordPaths=false;
    auto c=renderer.value()->CaptureFusionV2(third,0,options);ASSERT_TRUE(c)<<c.error();
    EXPECT_TRUE(c.value().paths.empty());
    EXPECT_EQ(a.value().linearReference.data,c.value().linearReference.data);
    for(size_t component=0;component<4;++component)
        EXPECT_EQ(a.value().contributions[component].data,c.value().contributions[component].data);
}

TEST(FusionTransportGpuTest, CancellationDoesNotAdvanceAcquisition) {
    const auto base=std::filesystem::path(QUANTILOOM_SOURCE_ROOT)/"docs/dataset/fusion";
    auto job=Config::Load(base/"transmission_job.toml");ASSERT_TRUE(job);
    auto rig=dataset::ParseRigConfig(*job,base.string());ASSERT_TRUE(rig);
    auto scene=Config::Load(base/"transmission_scene.toml");ASSERT_TRUE(scene);
    auto config=dataset::RigCameraScene(*scene,*rig,"reference");ASSERT_TRUE(config);
    OfflineRenderer::InitParams init;init.baseDir=base.string();
    auto renderer=OfflineRenderer::Create(*config,init);ASSERT_TRUE(renderer)<<renderer.error();
    camera::CaptureState state;dataset::FusionCaptureOptionsV2 options;
    options.cancelled=[]{return true;};
    EXPECT_FALSE(renderer.value()->CaptureFusionV2(state,0,options));
    EXPECT_EQ(state.acquisitionIndex,0u);
    EXPECT_TRUE(state.thermalPixelStateW.empty());
}

TEST(FusionTransportGpuTest, ActualCameraOrderPreservesNoiseAndWarmupProducts) {
    const auto base=std::filesystem::path(QUANTILOOM_SOURCE_ROOT)/"docs/dataset/fusion";
    auto job=Config::Load(base/"transmission_job.toml");ASSERT_TRUE(job);
    auto rig=dataset::ParseRigConfig(*job,base.string());ASSERT_TRUE(rig);
    auto scene=Config::Load(base/"transmission_scene.toml");ASSERT_TRUE(scene);
    for(auto& c:rig.value().cameras){c.sensor.quality.noiseFree=false;c.sensor.warmup.seconds=.04;}
    const auto root=std::filesystem::temp_directory_path()/("quantiloom_fusion_order_"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    struct Cleanup{std::filesystem::path path;~Cleanup(){std::error_code ec;std::filesystem::remove_all(path,ec);}} cleanup{root};
    dataset::FusionExportOptions options;options.sampleId="order";options.renderer.baseDir=base.string();
    options.outputDirectory=(root/"forward").string();
    auto first=dataset::FusionExportJob::Run(*scene,*rig,options);ASSERT_TRUE(first)<<first.error();
    std::reverse(rig.value().cameras.begin(),rig.value().cameras.end());options.outputDirectory=(root/"reverse").string();
    auto second=dataset::FusionExportJob::Run(*scene,*rig,options);ASSERT_TRUE(second)<<second.error();
    for(const auto& c:rig.value().cameras)for(const auto* name:{"raw_dn","measurement","linear_reference","contribution_transmitted"}) {
        const auto role=String(name)=="raw_dn" || String(name)=="measurement" ? "observations" : "ground_truth";
        auto a=ImageIO::ReadImage((root/"forward"/role/c.id/(String(name)+".exr")).string());
        auto b=ImageIO::ReadImage((root/"reverse"/role/c.id/(String(name)+".exr")).string());
        ASSERT_TRUE(a);ASSERT_TRUE(b);EXPECT_EQ(a->data,b->data)<<c.id<<"/"<<name;
    }
    options.outputDirectory=(root/"cancelled").string();
    options.cancelled=[](){return true;};
    EXPECT_FALSE(dataset::FusionExportJob::Run(*scene,*rig,options));
    EXPECT_FALSE(std::filesystem::exists(root/"cancelled"/"order.manifest.json"));
}

TEST(FusionTransportGpuTest, TargetRemovalChangesRealTransmittedSignalAtFixedTemperatures) {
    const auto base=std::filesystem::path(QUANTILOOM_SOURCE_ROOT)/"docs/dataset/fusion";
    auto job=Config::Load(base/"transmission_job.toml");ASSERT_TRUE(job);
    auto rig=dataset::ParseRigConfig(*job,base.string());ASSERT_TRUE(rig);
    auto scene=Config::Load(base/"transmission_scene.toml");ASSERT_TRUE(scene);
    auto config=dataset::RigCameraScene(*scene,*rig,"reference");ASSERT_TRUE(config);
    auto highSamples=Config::Parse("[renderer]\nspp=1024\n");ASSERT_TRUE(highSamples);
    const Config present=config.value().MergedWith(*highSamples);
    auto hidden=Config::Parse("[[nodes]]\nname=\"target\"\ntranslation=[1000000.0,0.0,0.0]\n");ASSERT_TRUE(hidden);
    const Config absent=present.MergedWith(*hidden);
    OfflineRenderer::InitParams init;init.baseDir=base.string();
    std::array<double,2> total{};
    for(size_t variant=0;variant<2;++variant) {
        auto renderer=OfflineRenderer::Create(variant ? absent : present,init);ASSERT_TRUE(renderer)<<renderer.error();
        camera::CaptureState state;dataset::FusionCaptureOptionsV2 options;options.recordPaths=false;
        auto capture=renderer.value()->CaptureFusionV2(state,0,options);ASSERT_TRUE(capture)<<capture.error();
        for(auto value:capture.value().linearReference.data)total[variant]+=value;
        total[variant]/=capture.value().linearReference.data.size();
    }
    const double R=.04,t=std::exp(-10*.2),tau=(1-R)*(1-R)*t/(1-R*R*t*t);
    const auto hot=camera::PlanckRadianceWm2SrNm(10000,350),sky=camera::PlanckRadianceWm2SrNm(10000,260);
    const auto omega=camera::ApertureSolidAngleSr(1.4);
    const double expected=tau*(*hot-*sky)*(*omega)*std::pow(12e-6,2)*2;
    EXPECT_GT(total[0],total[1]);
    EXPECT_NEAR(total[0]-total[1],expected,expected*.01);
}

TEST(FusionTransportGpuTest, InsideSolidAndTotalInternalReflectionReportAnUnknownTail) {
    const auto base=std::filesystem::path(QUANTILOOM_SOURCE_ROOT)/"docs/dataset/fusion";
    auto job=Config::Load(base/"transmission_job.toml");ASSERT_TRUE(job);
    auto rig=dataset::ParseRigConfig(*job,base.string());ASSERT_TRUE(rig);
    // Almost parallel to the slab faces from inside: the z faces are TIR.
    rig.value().cameras[0].cameraToRig={0,0,1,0, 0,-1,0,0, 1,0,0,.6, 0,0,0,1};
    auto scene=Config::Load(base/"transmission_scene.toml");ASSERT_TRUE(scene);
    auto config=dataset::RigCameraScene(*scene,*rig,"reference");ASSERT_TRUE(config);
    auto narrow=Config::Parse("[sensor.optics.projection]\nmodel=\"pinhole\"\nintrinsics=[100.0,100.0,4.0,4.0]\n[[nodes]]\nname=\"window\"\nscale=[100.0,1.0,1.0]\n");ASSERT_TRUE(narrow);
    OfflineRenderer::InitParams init;init.baseDir=base.string();
    init.onFusionPathChunk=[](const dataset::FusionPathChunk& chunk) {
        if(chunk.diagnosticFlags&2u) {
            const auto path=std::filesystem::path(QUANTILOOM_SOURCE_ROOT)/"build/fusion-audit/tir_paths.bin";
            std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(chunk.bytes.data()),chunk.bytes.size());
        }
    };
    auto renderer=OfflineRenderer::Create(config.value().MergedWith(*narrow),init);ASSERT_TRUE(renderer)<<renderer.error();
    Vector<dataset::OpticalProbe> probes={{{4.0f,4.5f},10000.0,255}};
    auto query=renderer.value()->QueryOpticalPaths(probes,0);ASSERT_TRUE(query)<<query.error();
    EXPECT_EQ(query.value()[0].flags&2u,0u);
    EXPECT_EQ(query.value()[0].flags&1u,1u);
    camera::CaptureState state;dataset::FusionCaptureOptionsV2 options;options.maxRecordedRays=0;
    auto capture=renderer.value()->CaptureFusionV2(state,0,options);ASSERT_TRUE(capture)<<capture.error();
    EXPECT_TRUE(std::all_of(capture.value().linearReference.data.begin(),capture.value().linearReference.data.end(),[](float x){return std::isfinite(x); }));
    EXPECT_GT(*std::max_element(capture.value().truncationUnknown.data.begin(),capture.value().truncationUnknown.data.end()),0);
    const auto planck=camera::PlanckRadianceWm2SrNm(10000,300),omega=camera::ApertureSolidAngleSr(1.4);
    const double equilibrium=2.25*(*planck)*(*omega)*std::pow(12e-6,2)*2;
    // The first absorbing segment has optical depth >=28, bounding the
    // untraced tail far below this tolerance even though its mask stays set.
    for(auto value:capture.value().linearReference.data)EXPECT_NEAR(value,equilibrium,equilibrium*1e-4);
}

TEST(FusionTransportGpuTest, DistinctVerifiedBranchesAtSamePixelAreRetained) {
    const auto base=std::filesystem::path(QUANTILOOM_SOURCE_ROOT)/"docs/dataset/fusion";
    auto job=Config::Load(base/"transmission_job.toml");ASSERT_TRUE(job);
    auto rig=dataset::ParseRigConfig(*job,base.string());ASSERT_TRUE(rig);
    auto scene=Config::Load(base/"transmission_scene.toml");ASSERT_TRUE(scene);
    auto config=dataset::RigCameraScene(*scene,*rig,"reference");ASSERT_TRUE(config);
    OfflineRenderer::InitParams init;init.baseDir=base.string();
    auto renderer=OfflineRenderer::Create(*config,init);ASSERT_TRUE(renderer)<<renderer.error();
    Vector<dataset::OpticalProbe> probes={{{4,4},10000,0},{{4,4},10000,6}};
    auto paths=renderer.value()->QueryOpticalPaths(probes,0);ASSERT_TRUE(paths)<<paths.error();
    auto geometry=renderer.value()->CaptureGeometry(0);ASSERT_TRUE(geometry);
    Vector<dataset::OpticalEndpoint> samples;
    for(size_t i=0;i<2;++i) {
        ASSERT_EQ(paths.value()[i].flags,0u);ASSERT_GT(paths.value()[i].throughput,0);
        dataset::OpticalEndpoint e;e.productId="paths";e.row=static_cast<u32>(i);e.wavelengthNm=10000;
        e.nodeId=paths.value()[i].surface.instanceId;e.primitiveId=paths.value()[i].primitiveId;
        e.position=glm::dvec3(paths.value()[i].surface.worldPosition);e.nativePixel={4,4};
        e.branchMask=probes[i].branchMask;e.throughOptics=true;samples.push_back(e);
    }
    auto matched=dataset::MatchOpticalPaths(*renderer.value(),{samples[0]},samples,geometry.value().geometry,1);
    ASSERT_TRUE(matched)<<matched.error();
    const auto mapping=nlohmann::json::parse(*matched);
    ASSERT_EQ(mapping["rows"].size(),1u);
    EXPECT_EQ(mapping["rows"][0]["matches"].size(),2u);
    EXPECT_FALSE(mapping["rows"][0]["unique_solution_proven"].get<bool>());
}
