#pragma once
#include "postprocess/CameraPipeline.hpp"
#include <array>
#include <functional>

namespace quantiloom::dataset {
struct FusionCaptureOptionsV2 {
    u32 version = 2;
    u32 maxRecordedRays = 4096;
    bool recordPaths = true;
    std::function<bool()> cancelled;
};
struct FusionPathColumnsV2 {
    Vector<u8> bytes;
    String descriptionJson;
};
struct FusionCaptureResultV2 {
    u32 version = 2;
    camera::CameraOutput products;
    // Same native channels/units as linearReference, before detector state/noise.
    std::array<Image, 4> contributions; // direct, reflected, transmitted, residual
    Image linearReference;
    Image lensValidity;        // LensValidity enum, at the pixel centre
    Image validSampleFraction; // mean over sampled wavelengths and exposure strata
    Image truncationUnknown;   // a nonzero value means an unknown physical tail
    Vector<FusionPathColumnsV2> paths;
};
}
