// ============================================================================
// Quantiloom - push constants and results of the compute kernels
// ============================================================================
// The camera device pipeline (camera_common.hlsli, renderer/GpuCameraPipeline.cpp),
// CLAHE (clahe.comp.hlsl, renderer/GpuClahe.cpp), the display percentile range
// (display_range.comp.hlsl, renderer/GpuDisplayRange.cpp), the hyperspectral
// reconstruction (spectral_reconstruct.comp.hlsl,
// hs_core/GpuSpectralReconstructor.cpp) and the viewport pick (pick.rayq.hlsl,
// renderer/ExternalRenderContext.cpp). Each is defined here once, for both
// languages. ql_shared_prelude.hlsli has the layout rules and how a shader asks
// for one struct (QL_SHARED_WANT_<Name>).
//
// No include guard: a shader includes this once per struct it asks for, and
// every section guards itself (QL_SHARED_HAS_<Name>).
// ============================================================================

#include "ql_shared_prelude.hlsli"

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_CameraPush)) && !defined(QL_SHARED_HAS_CameraPush)
#define QL_SHARED_HAS_CameraPush
QL_SHARED_BEGIN
// One block for every camera pass; the defaults are what a pass that does not
// set a field reads.
struct CameraPush {
    uint width QL_DEFAULT(0);
    uint height QL_DEFAULT(0);
    uint physicalWidth QL_DEFAULT(0);
    uint physicalHeight QL_DEFAULT(0);
    uint channelCount QL_DEFAULT(1);
    uint cfa QL_DEFAULT(0);
    uint detector QL_DEFAULT(0);
    uint flags QL_DEFAULT(0);
    uint seed QL_DEFAULT(0);
    uint acquisitionLo QL_DEFAULT(0);
    uint acquisitionHi QL_DEFAULT(0);
    uint adcBits QL_DEFAULT(0);
    float frameTimeSeconds QL_DEFAULT(0.0f);
    float deltaSeconds QL_DEFAULT(0.0f);
    uint firstThermal QL_DEFAULT(1);
    uint direction QL_DEFAULT(0);
    float sigmaR QL_DEFAULT(0.0f);
    float sigmaG QL_DEFAULT(0.0f);
    float sigmaB QL_DEFAULT(0.0f);
    float sigmaRy QL_DEFAULT(0.0f);
    float sigmaGy QL_DEFAULT(0.0f);
    float sigmaBy QL_DEFAULT(0.0f);
    uint rngNoiseClass QL_DEFAULT(0);
    uint rngCounter QL_DEFAULT(0);
    // Dynamic-exposure compositor (camera_dynamic.comp). The per-stratum
    // camera frames ride in binding 10; these scalars describe the exposure.
    uint timeStratumCount QL_DEFAULT(1);      // T; 0/1 = plain copy of layer 0
    uint rollingShutter QL_DEFAULT(0);        // 0/1: rows integrate at firstRowMid + y*rowDelay
    float exposureSeconds QL_DEFAULT(0.0f);   // E: integration window each row integrates over
    float rowDelaySeconds QL_DEFAULT(0.0f);   // rolling-shutter per-row offset
    float firstRowMidSeconds QL_DEFAULT(0.0f); // t0: midpoint of row 0's exposure. Absolute, so f32-coarse; nothing may subtract with it
    // Multi-phase camera passes (camera_stats) select their stage here; 0 for
    // every other pass.
    uint ispPhase QL_DEFAULT(0);
};
QL_SHARED_LAYOUT(CameraPush, 120)
QL_SHARED_OFFSET(CameraPush, width, 0)
QL_SHARED_OFFSET(CameraPush, height, 4)
QL_SHARED_OFFSET(CameraPush, physicalWidth, 8)
QL_SHARED_OFFSET(CameraPush, physicalHeight, 12)
QL_SHARED_OFFSET(CameraPush, channelCount, 16)
QL_SHARED_OFFSET(CameraPush, cfa, 20)
QL_SHARED_OFFSET(CameraPush, detector, 24)
QL_SHARED_OFFSET(CameraPush, flags, 28)
QL_SHARED_OFFSET(CameraPush, seed, 32)
QL_SHARED_OFFSET(CameraPush, acquisitionLo, 36)
QL_SHARED_OFFSET(CameraPush, acquisitionHi, 40)
QL_SHARED_OFFSET(CameraPush, adcBits, 44)
QL_SHARED_OFFSET(CameraPush, frameTimeSeconds, 48)
QL_SHARED_OFFSET(CameraPush, deltaSeconds, 52)
QL_SHARED_OFFSET(CameraPush, firstThermal, 56)
QL_SHARED_OFFSET(CameraPush, direction, 60)
QL_SHARED_OFFSET(CameraPush, sigmaR, 64)
QL_SHARED_OFFSET(CameraPush, sigmaG, 68)
QL_SHARED_OFFSET(CameraPush, sigmaB, 72)
QL_SHARED_OFFSET(CameraPush, sigmaRy, 76)
QL_SHARED_OFFSET(CameraPush, sigmaGy, 80)
QL_SHARED_OFFSET(CameraPush, sigmaBy, 84)
QL_SHARED_OFFSET(CameraPush, rngNoiseClass, 88)
QL_SHARED_OFFSET(CameraPush, rngCounter, 92)
QL_SHARED_OFFSET(CameraPush, timeStratumCount, 96)
QL_SHARED_OFFSET(CameraPush, rollingShutter, 100)
QL_SHARED_OFFSET(CameraPush, exposureSeconds, 104)
QL_SHARED_OFFSET(CameraPush, rowDelaySeconds, 108)
QL_SHARED_OFFSET(CameraPush, firstRowMidSeconds, 112)
QL_SHARED_OFFSET(CameraPush, ispPhase, 116)
QL_SHARED_END
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_CLAHEPushConstants)) && !defined(QL_SHARED_HAS_CLAHEPushConstants)
#define QL_SHARED_HAS_CLAHEPushConstants
QL_SHARED_BEGIN
// The pipeline-layout range and the record-time payload share this one 48-byte
// block; the words past passIndex are padding in the layout range and the
// tone/palette selectors at record time.
struct CLAHEPushConstants {
    uint imageWidth;
    uint imageHeight;
    uint tileCountX;
    uint tileCountY;
    float clipLimit;           // Typically 2.0-4.0
    uint luminanceOnly;        // 1 = process luminance only, 0 = all channels
    float inputMin;            // Pre-computed min value for normalization
    float inputMax;            // Pre-computed max value for normalization
    uint passIndex;            // Which pass is executing (0, 1, or 2)
    uint toneMode;             // TONE_*
    uint palette;              // PALETTE_*
    uint padding;
};
QL_SHARED_LAYOUT(CLAHEPushConstants, 48)
QL_SHARED_OFFSET(CLAHEPushConstants, imageWidth, 0)
QL_SHARED_OFFSET(CLAHEPushConstants, imageHeight, 4)
QL_SHARED_OFFSET(CLAHEPushConstants, tileCountX, 8)
QL_SHARED_OFFSET(CLAHEPushConstants, tileCountY, 12)
QL_SHARED_OFFSET(CLAHEPushConstants, clipLimit, 16)
QL_SHARED_OFFSET(CLAHEPushConstants, luminanceOnly, 20)
QL_SHARED_OFFSET(CLAHEPushConstants, inputMin, 24)
QL_SHARED_OFFSET(CLAHEPushConstants, inputMax, 28)
QL_SHARED_OFFSET(CLAHEPushConstants, passIndex, 32)
QL_SHARED_OFFSET(CLAHEPushConstants, toneMode, 36)
QL_SHARED_OFFSET(CLAHEPushConstants, palette, 40)
QL_SHARED_OFFSET(CLAHEPushConstants, padding, 44)
QL_SHARED_END
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_RangePush)) && !defined(QL_SHARED_HAS_RangePush)
#define QL_SHARED_HAS_RangePush
QL_SHARED_BEGIN
// The display percentile range's push block; the host calls it DisplayRangePush.
struct RangePush {
    uint width;
    uint height;
    float absMin;
    float scale;
    uint targetLow;
    uint targetHigh;
};
QL_SHARED_LAYOUT(RangePush, 24)
QL_SHARED_OFFSET(RangePush, width, 0)
QL_SHARED_OFFSET(RangePush, height, 4)
QL_SHARED_OFFSET(RangePush, absMin, 8)
QL_SHARED_OFFSET(RangePush, scale, 12)
QL_SHARED_OFFSET(RangePush, targetLow, 16)
QL_SHARED_OFFSET(RangePush, targetHigh, 20)
QL_SHARED_END
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_ReconstructPush)) && !defined(QL_SHARED_HAS_ReconstructPush)
#define QL_SHARED_HAS_ReconstructPush
QL_SHARED_BEGIN
// Push constants for reconstruction parameters; the host calls it
// ReconstructPushConstants.
struct ReconstructPush {
    uint width;             // Image width
    uint height;            // Image height
    uint srcBands;          // Number of source (sparse) bands
    uint dstBands;          // Number of destination (full) bands
    uint interpolationMode; // 0=Linear, 1=CatmullRom, 2=Akima
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
QL_SHARED_LAYOUT(ReconstructPush, 32)
QL_SHARED_OFFSET(ReconstructPush, width, 0)
QL_SHARED_OFFSET(ReconstructPush, height, 4)
QL_SHARED_OFFSET(ReconstructPush, srcBands, 8)
QL_SHARED_OFFSET(ReconstructPush, dstBands, 12)
QL_SHARED_OFFSET(ReconstructPush, interpolationMode, 16)
QL_SHARED_OFFSET(ReconstructPush, _pad0, 20)
QL_SHARED_OFFSET(ReconstructPush, _pad1, 24)
QL_SHARED_OFFSET(ReconstructPush, _pad2, 28)
QL_SHARED_END
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_PickResultGpu)) && !defined(QL_SHARED_HAS_PickResultGpu)
#define QL_SHARED_HAS_PickResultGpu
QL_SHARED_BEGIN
// What the viewport pick writes: one record, read back by the host.
struct PickResultGpu {
    uint  hit;             // 1 = committed triangle hit
    uint  instanceIndex;   // TLAS instance index (matches InstanceIndex())
    uint  primitiveIndex;  // Triangle index within the instance's geometry
    float hitT;            // Distance along the normalized ray
    float3 worldPosition;  // origin + direction * hitT
    uint  _pad;
};
QL_SHARED_LAYOUT(PickResultGpu, 32)
QL_SHARED_OFFSET(PickResultGpu, hit, 0)
QL_SHARED_OFFSET(PickResultGpu, instanceIndex, 4)
QL_SHARED_OFFSET(PickResultGpu, primitiveIndex, 8)
QL_SHARED_OFFSET(PickResultGpu, hitT, 12)
QL_SHARED_OFFSET(PickResultGpu, worldPosition, 16)
QL_SHARED_OFFSET(PickResultGpu, _pad, 28)
QL_SHARED_END
#endif

#if (defined(__cplusplus) || defined(QL_SHARED_WANT_PickPushConstants)) && !defined(QL_SHARED_HAS_PickPushConstants)
#define QL_SHARED_HAS_PickPushConstants
QL_SHARED_BEGIN
// The pick's camera: the basis raygen builds its primary ray from, for one pixel.
struct PickPushConstants {
    float3 origin;   float fovScale;     // tan(fovY / 2), same as CameraData
    float3 forward;  float aspectRatio;
    float3 right;    uint  pixelX;
    float3 up;       uint  pixelY;
    uint   width;
    uint   height;
    uint   projection;      // CAMERA_PROJECTION_* -- must match raygen
    float  orthoHeight;
};
QL_SHARED_LAYOUT(PickPushConstants, 80)
QL_SHARED_OFFSET(PickPushConstants, origin, 0)
QL_SHARED_OFFSET(PickPushConstants, fovScale, 12)
QL_SHARED_OFFSET(PickPushConstants, forward, 16)
QL_SHARED_OFFSET(PickPushConstants, aspectRatio, 28)
QL_SHARED_OFFSET(PickPushConstants, right, 32)
QL_SHARED_OFFSET(PickPushConstants, pixelX, 44)
QL_SHARED_OFFSET(PickPushConstants, up, 48)
QL_SHARED_OFFSET(PickPushConstants, pixelY, 60)
QL_SHARED_OFFSET(PickPushConstants, width, 64)
QL_SHARED_OFFSET(PickPushConstants, height, 68)
QL_SHARED_OFFSET(PickPushConstants, projection, 72)
QL_SHARED_OFFSET(PickPushConstants, orthoHeight, 76)
QL_SHARED_END
#endif
