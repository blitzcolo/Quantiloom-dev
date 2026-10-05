#include <gtest/gtest.h>
#include "dataset/FusionExportJob.hpp"
#include "dataset/ExportSession.hpp"
#include <filesystem>
#include <fstream>
#include <chrono>

using namespace quantiloom;
namespace fs=std::filesystem;

TEST(FusionPublicationTest, CancellationDuringWorkAndBeforeCommitPreservesOldPackage) {
    const auto base=fs::path(QUANTILOOM_SOURCE_ROOT)/"docs/dataset/fusion";
    auto scene=Config::Load(base/"transmission_scene.toml");ASSERT_TRUE(scene);
    auto job=Config::Load(base/"transmission_job.toml");ASSERT_TRUE(job);
    auto rig=dataset::ParseRigConfig(*job,base.string());ASSERT_TRUE(rig);
    for(auto& camera:rig.value().cameras) camera.sensor.quality.wavelengthSamples=3;
    const auto root=fs::temp_directory_path()/("fusion_cancel_"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {fs::path p;~Cleanup(){std::error_code ec;fs::remove_all(p,ec);}} cleanup{root};
    dataset::FusionExportOptions options;options.sampleId="cancel";
    options.outputDirectory=root.string();options.renderer.baseDir=base.string();
    auto first=dataset::FusionExportJob::Run(*scene,*rig,options);ASSERT_TRUE(first)<<first.error();
    const auto read=[](const fs::path& p){std::ifstream in(p,std::ios::binary);return String(
        std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>());};
    const auto previous=read(first.value().recordPath);
    for(const String phase:{"capture","correspondence","ready_to_publish"}) {
        String active;u32 polls=0;bool interrupted=false;
        options.onProgress=[&](const dataset::FusionExportProgress& p){active=p.phase;};
        options.cancelled=[&]{
            // Capture has already integrated two wavelengths; matching has
            // completed its first GPU probe batch before cancellation.
            const u32 completedPolls=phase=="ready_to_publish" ? 0u : phase=="correspondence" ? 3u : 2u;
            const bool cancel=active==phase && ++polls>completedPolls;
            interrupted|=cancel;return cancel;
        };
        auto result=dataset::FusionExportJob::Run(*scene,*rig,options);
        EXPECT_TRUE(interrupted)<<phase;EXPECT_FALSE(result)<<phase;
        EXPECT_EQ(read(first.value().recordPath),previous)<<phase;
        EXPECT_TRUE(dataset::ExportSession::Verify(first.value().recordPath).valid)<<phase;
    }
}
