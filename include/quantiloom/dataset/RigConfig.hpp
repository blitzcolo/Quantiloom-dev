#pragma once
#include "core/Config.hpp"
#include "postprocess/CameraPipeline.hpp"
#include <array>
#include <vector>

namespace quantiloom::dataset {

struct RigCamera {
    String id;
    std::array<f64, 16> cameraToRig{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    camera::CameraConfig sensor;
    u32 renderSeed = 0;
};
struct RigPair {
    String sourceCamera, targetCamera;
};
struct RigConfig {
    u32 version = 1;
    u32 seed = 0;
    String id;
    String referenceCamera;
    std::array<f64, 16> rigToWorld{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::vector<RigCamera> cameras;
    std::vector<RigPair> pairs;
};

QL_API Result<RigConfig, String> ParseRigConfig(const Config& document,
                                                const String& baseDirectory = {});
QL_API String RigConfigToToml(const RigConfig& rig);
/// Builds a frozen scene configuration for one camera. Input poses use RDF
/// camera axes and row-major rigid matrices. Seeds depend on identity, not order.
QL_API Result<Config, String> RigCameraScene(const Config& scene, const RigConfig& rig,
                                             const String& cameraId);

} // namespace quantiloom::dataset
