#pragma once
#include "dataset/RigConfig.hpp"
#include "renderer/OfflineRenderer.hpp"
#include <functional>

namespace quantiloom::dataset {
struct FusionExportProgress {
    u32 completedCameras=0,totalCameras=0;
    String cameraId;
    String phase;
};
struct FusionExportOptions {
    String outputDirectory;
    String sampleId="sample";
    f64 referenceTimeSeconds=0;
    bool dryRun=false;
    bool rectify=false;
    u32 maxRecordedRays=4096; // zero records all camera rays
    u32 maxOpticalSourcePaths=64;
    std::function<void(const FusionExportProgress&)> onProgress;
    std::function<bool()> cancelled;
    OfflineRenderer::InitParams renderer;
};
struct FusionExportResult {
    String recordPath;
    String manifestPath;
    String summaryJson;
};

class QL_API FusionExportJob {
public:
    /// scene and rig are frozen values; callback cancellation aborts publication.
    static Result<FusionExportResult,String> Run(const Config& scene,
        const RigConfig& rig,const FusionExportOptions& options);
    static Result<FusionExportResult,String> RunFile(const String& jobPath,
        bool dryRun=false);
};
} // namespace quantiloom::dataset
