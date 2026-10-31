#include "renderer/GpuCameraPipeline.hpp"

#include "core/Log.hpp"
#include "core/ResultFail.hpp"
#include "gpu/Binding.hpp"
#include "gpu/Device.hpp"
#include "gpu/Resources.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>


namespace quantiloom::rendercore {
namespace {


struct CameraPush {
    u32 width = 0, height = 0, physicalWidth = 0, physicalHeight = 0;
    u32 channelCount = 1, cfa = 0, detector = 0, flags = 0;
    u32 seed = 0, acquisitionLo = 0, acquisitionHi = 0, adcBits = 0;
    f32 frameTimeSeconds = 0.0f, deltaSeconds = 0.0f;
    u32 firstThermal = 1, direction = 0;
    f32 sigmaR = 0.0f, sigmaG = 0.0f, sigmaB = 0.0f;
    f32 sigmaRy = 0.0f, sigmaGy = 0.0f, sigmaBy = 0.0f;
    u32 rngNoiseClass = 0, rngCounter = 0;
    // Dynamic compositor scalars; per-stratum camera frames ride in the
    // binding-10 buffer. Must match CameraPush in camera_common.hlsli.
    u32 timeStratumCount = 1, rollingShutter = 0;
    f32 exposureSeconds = 0.0f, rowDelaySeconds = 0.0f;
    f32 firstRowMidSeconds = 0.0f;
    // Multi-phase camera passes (camera_stats) select their stage here.
    u32 ispPhase = 0;
};
static_assert(sizeof(CameraPush) == 120);

// DisplayToneMode / DisplayPalette orders must match DisplayControl.hpp and
// the shader constants.
namespace isp {
inline constexpr u32 kFlagDenoise = 1u;
inline constexpr u32 kFlagSharpen = 2u;
inline constexpr u32 kFlagClip = 4u;
inline constexpr u32 kFlagClahePersistent = 8u;
inline constexpr u32 kFlagDefects = 16u;
inline constexpr u32 kFlagEmpiricalNoise = 32u;  // camera_hsv.comp.hlsl bit 5
inline constexpr u32 kFlagTemporalDrift = 64u;   // camera_hsv.comp.hlsl bit 6
inline constexpr u32 kFlagHsv = 128u;  // any HSV parameter off its default

// ispStats buffer layout (uint slots), shared with camera_stats.comp.hlsl /
// camera_demosaic.comp.hlsl. 256 histogram bins, then the named slots; the
// per-channel AWB rows are written by the stats reduce phase.
inline constexpr u32 kStatsSlots = 272;
inline constexpr u32 kStatsMin = 256;
inline constexpr u32 kStatsMax = 257;
inline constexpr u32 kStatsMean = 258;
inline constexpr u32 kStatsSat = 259;
inline constexpr u32 kStatsUnsat = 260;
inline constexpr u32 kStatsWindowLo = 261;
inline constexpr u32 kStatsWindowHi = 262;
inline constexpr u32 kStatsChSumR = 263;
inline constexpr u32 kStatsChCntR = 266;
} // namespace isp

enum CameraFlags : u32 {
    NoiseFree = 1u,
    Shot = 2u,
    DarkCurrent = 4u,
    DarkShot = 8u,
    Read = 16u,
    Fpn = 32u,
    Nuc = 64u,
    NucGainMap = 128u,
    NucOffsetMap = 256u,
    Vignette = 512u
};

using ConfigRows = std::array<std::array<f32, 4>, 9>;

void StorageBarrier(gpu::CommandList& cmd) {
    cmd.Barrier(gpu::GlobalBarrier{gpu::Access::ComputeWrite,
                                   gpu::Access::ComputeRead |
                                       gpu::Access::ComputeWrite});
}

// The access hand-off the destination stage of a checkpoint copy needs:
// the consumer may be the compute passes again or a later transfer copy.
constexpr gpu::Access kCheckpointVisible =
    gpu::Access::ComputeRead | gpu::Access::ComputeWrite |
    gpu::Access::TransferRead;

// Copies one thermal ping-pong pair, bracketing the transfers with
// layout-preserving barriers: the sources move their last writer's access to
// transfer-read, the destinations hand transfer-write back to the consumer
// stage. All four images live in GENERAL throughout the pipeline's life.
void CopyThermalPair(gpu::CommandList& cmd,
                     gpu::TextureRef src0, gpu::TextureRef src1,
                     gpu::TextureRef dst0, gpu::TextureRef dst1,
                     gpu::Access sourceAccess) {
    const gpu::TextureBarrier ready[2] = {
        {src0, sourceAccess, gpu::Access::TransferRead},
        {src1, sourceAccess, gpu::Access::TransferRead}};
    cmd.Barrier(gpu::BarrierBatch{
        {}, {}, std::span<const gpu::TextureBarrier>(ready)});
    const gpu::TextureCopyRegion copy{src0.width, src0.height};
    cmd.CopyTexture(src0, dst0, copy);
    cmd.CopyTexture(src1, dst1, copy);
    const gpu::TextureBarrier visible[2] = {
        {dst0, gpu::Access::TransferWrite, kCheckpointVisible},
        {dst1, gpu::Access::TransferWrite, kCheckpointVisible}};
    cmd.Barrier(gpu::BarrierBatch{
        {}, {}, std::span<const gpu::TextureBarrier>(visible)});
}

// Copies the statistics buffer (previous-frame AGC window + histogram) in
// the direction the checkpoint needs, with the same access hand-off as
// the image pair above.
void CopyStatsBuffer(gpu::CommandList& cmd, gpu::BufferRef source,
                     gpu::BufferRef destination, gpu::Access sourceAccess) {
    cmd.Barrier(gpu::BufferBarrier{source, sourceAccess,
                                   gpu::Access::TransferRead});
    const gpu::BufferCopyRegion copy{0, 0, source.size};
    cmd.CopyBuffer(source, destination,
                   std::span<const gpu::BufferCopyRegion>{&copy, 1});
    cmd.Barrier(gpu::BufferBarrier{destination, gpu::Access::TransferWrite,
                                   kCheckpointVisible});
}

// Builds the ISP configuration rows (binding 12) and the shared ISP flag
// word from a capture config. Row 0 carries the white balance and the flag
// word; rows 1-3 the color correction matrix (row3.y doubles as the defect
// count); row 4 gamma/denoise/sharpen; row 5 percentiles/tone/palette; row 6
// the dynamic exposure/gain/well/ADC ceiling the statistics passes read;
// row 7 the HSV grading (hue offset degrees, saturation scale, value gamma,
// empirical-noise sigma); row 8.x the drift sigma. Configure and every AE/AWB
// effective-config re-upload share this so the two paths cannot drift apart.
void BuildIspRows(const camera::CameraConfig& config,
                  std::array<std::array<f32, 4>, 9>& rows, u32& flags) {
    const auto& isp = config.isp;
    rows = {};
    flags = 0;
    if (isp.denoise && isp.denoiseStrength > 0.0) flags |= isp::kFlagDenoise;
    if (isp.sharpen && isp.sharpenStrength > 0.0) flags |= isp::kFlagSharpen;
    if (isp.clipOutOfGamut) flags |= isp::kFlagClip;
    if (isp.infraredTone == DisplayToneMode::Clahe)
        flags |= isp::kFlagClahePersistent;
    if (!isp.defectPixels.empty()) flags |= isp::kFlagDefects;
    if (isp.hsv.empiricalNoise) flags |= isp::kFlagEmpiricalNoise;
    if (isp.hsv.temporalDrift) flags |= isp::kFlagTemporalDrift;
    // The HSV pass is dispatched at all only when a parameter is off its
    // default -- the CPU caller skips ApplyHsv the same way -- so an
    // untouched chain never reaches the pass and stays bit-identical.
    if (isp.hsv.hueOffsetDegrees != 0.0 || isp.hsv.saturationScale != 1.0 ||
        isp.hsv.valueGamma != 1.0 || isp.hsv.empiricalNoise ||
        isp.hsv.temporalDrift)
        flags |= isp::kFlagHsv;
    rows[0] = {{static_cast<f32>(isp.whiteBalance[0]),
                static_cast<f32>(isp.whiteBalance[1]),
                static_cast<f32>(isp.whiteBalance[2]),
                static_cast<f32>(flags)}};
    for (u32 i = 0; i < 9; ++i)
        rows[1 + i / 4][i % 4] = static_cast<f32>(isp.deviceToLinearSrgb[i]);
    // Row3.y doubles as the defect count (row3.x carries ccm[8]).
    rows[3][1] = static_cast<f32>(isp.defectPixels.size());
    rows[4] = {{static_cast<f32>(isp.toneGamma),
                static_cast<f32>(std::clamp(isp.denoiseStrength, 0.0, 1.0)),
                static_cast<f32>(std::max(isp.sharpenStrength, 0.0)), 0.0f}};
    rows[5] = {{static_cast<f32>(isp.contrastLowPercentile),
                static_cast<f32>(isp.contrastHighPercentile),
                static_cast<f32>(isp.infraredTone),
                static_cast<f32>(isp.infraredPalette)}};
    const f32 adcMax =
        std::exp2(static_cast<f32>(config.readout.adcBits)) - 1.0f;
    rows[6] = {{static_cast<f32>(config.readout.exposureSeconds),
                static_cast<f32>(config.readout.analogGain),
                static_cast<f32>(config.photon.fullWellElectrons), adcMax}};
    rows[7] = {{static_cast<f32>(isp.hsv.hueOffsetDegrees),
                static_cast<f32>(isp.hsv.saturationScale),
                static_cast<f32>(isp.hsv.valueGamma),
                static_cast<f32>(isp.hsv.empiricalNoiseSigma)}};
    rows[8] = {{static_cast<f32>(isp.hsv.temporalDriftSigma), 0.0f, 0.0f,
                0.0f}};
}

f64 EffectiveWavelengthNm(const camera::ResponseStack& response,
                          camera::DetectorKind detector) {
    // Preview-effective wavelength assumes flat incident spectral radiance
    // inside this channel. The physical capture integrates the actual scene
    // spectrum; this one number only sizes the approximate GPU Gaussian PSF.
    // The curve lookup is the shared CameraPhysics one (BaseCurve/CurveAt).
    const camera::ResponseCurve* base =
        camera::BaseCurve(response, detector);
    Vector<f64> knots = base->wavelengthNm;
    for (const auto* extra : {
             response.lensTransmission ? &*response.lensTransmission : nullptr,
             response.filterTransmission ? &*response.filterTransmission : nullptr}) {
        if (!extra) continue;
        for (const f64 lambda : extra->wavelengthNm)
            if (lambda > base->MinNm() && lambda < base->MaxNm())
                knots.push_back(lambda);
    }
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
    const auto density = [&](f64 lambda) {
        f64 value = camera::CurveAt(*base, lambda);
        if (response.lensTransmission)
            value *= camera::CurveAt(*response.lensTransmission, lambda);
        if (response.filterTransmission)
            value *= camera::CurveAt(*response.filterTransmission, lambda);
        // A photon detector's electron response includes photon energy;
        // a thermal absorber is weighted directly by power.
        if (detector == camera::DetectorKind::Photon) value *= lambda;
        return value;
    };
    // Three-point Gauss-Legendre is exact through degree 5 on each knot
    // interval. The stack is at most three linear factors, multiplied by up
    // to lambda^2 for the first moment.
    constexpr f64 node = 0.77459666924148337704;
    const std::array<std::pair<f64, f64>, 3> gauss{{
        {-node, 5.0 / 9.0}, {0.0, 8.0 / 9.0}, {node, 5.0 / 9.0}
    }};
    f64 weight = 0.0, moment = 0.0;
    for (size_t i = 1; i < knots.size(); ++i) {
        const f64 middle = 0.5 * (knots[i - 1] + knots[i]);
        const f64 half = 0.5 * (knots[i] - knots[i - 1]);
        for (const auto& [x, w] : gauss) {
            const f64 lambda = middle + half * x;
            const f64 contribution = half * w * density(lambda);
            weight += contribution;
            moment += lambda * contribution;
        }
    }
    return weight > 0.0 ? moment / weight :
           0.5 * (base->MinNm() + base->MaxNm());
}

Result<void, String> UpdateReadoutRows(
    const camera::CameraConfig& config,
    std::array<std::array<f32, 4>, 9>& rows) {
    const auto& optics = config.optics;
    std::array<f32, 3> thermalNoise{
        static_cast<f32>(config.thermal.readNoiseDnRms),
        static_cast<f32>(config.thermal.readNoiseDnRms),
        static_cast<f32>(config.thermal.readNoiseDnRms)};
    if (config.device.detector == camera::DetectorKind::Thermal &&
        config.thermal.netdKelvin > 0.0) {
        const auto area = camera::PixelCollectionAreaM2(optics);
        if (!area) return Result<void, String>::Err(area.error());
        for (size_t i = 0; i < config.device.channels.size(); ++i) {
            const auto slope = camera::BlackbodyThermalDerivativeWPerK(
                config.thermal.netdReferenceTemperatureK,
                config.device.channels[i].response, optics.fNumber, *area);
            if (!slope) return Result<void, String>::Err(slope.error());
            f64 noise = config.thermal.netdKelvin * *slope *
                        config.thermal.responsivityDnPerWatt;
            if (config.thermal.readoutWindowSeconds > 0.0)
                noise *= std::sqrt(
                    (1.0 / (2.0 * config.thermal.readoutWindowSeconds)) /
                    config.thermal.netdNoiseBandwidthHz);
            thermalNoise[i] = static_cast<f32>(noise);
        }
    }
    const auto& photon = config.photon;
    const auto& readout = config.readout;
    const auto& thermal = config.thermal;
    rows[0] = {{static_cast<f32>(readout.exposureSeconds),
                static_cast<f32>(photon.fullWellElectrons),
                static_cast<f32>(photon.darkCurrentElectronsPerSecond),
                static_cast<f32>(photon.readNoiseElectronsRms)}};
    rows[1] = {{static_cast<f32>(photon.prnuSigma),
                static_cast<f32>(photon.dsnuElectronsRms),
                static_cast<f32>(photon.dsnuReferenceExposureSeconds),
                static_cast<f32>(photon.biasDnRms)}};
    rows[2] = {{static_cast<f32>(readout.analogGain),
                static_cast<f32>(readout.electronsPerDn),
                static_cast<f32>(readout.blackLevelDn),
                static_cast<f32>(photon.nucResidualFraction)}};
    rows[3] = {{static_cast<f32>(thermal.timeConstantSeconds),
                static_cast<f32>(thermal.responsivityDnPerWatt),
                static_cast<f32>(thermal.driftDnPerSecond), thermalNoise[0]}};
    rows[4] = {{thermalNoise[1], thermalNoise[2], 0.0f, 0.0f}};
    return Result<void, String>::Ok();
}

} // namespace

struct GpuCameraPipeline::Impl {
    gpu::Device& device;
    u32 width = 0, height = 0;
    camera::CameraConfig config;
    bool configured = false;
    bool imagesInGeneral = false;
    bool hasCapture = false;
    bool firstCaptureWasFirst = false;
    u64 lastAcquisitionIndex = 0;
    f64 lastFrameTimeSeconds = 0.0;
    f64 beforeFrameTimeSeconds = 0.0;
    f64 currentDeltaSeconds = 0.0;
    u32 stateBefore = 0, stateCurrent = 1;
    u32 ispFlags = 0;
    std::array<f32, 3> sigmaX{}, sigmaY{};
    ConfigRows configRows{};
    // Effective-config rows staged by ApplyEffectiveConfig and written by
    // the next record call through in-command-buffer constant updates (each
    // buffer is far under the 64 KiB limit): a host memcpy at record time
    // could overwrite parameters a previous, still-in-flight submission is
    // reading.
    bool paramsPending = false;
    ConfigRows pendingConfigRows{};
    std::array<std::array<f32, 4>, 9> pendingIspRows{};

    std::unique_ptr<gpu::Texture> psfTemp, blurred, rawDn, corrected, display;
    std::unique_ptr<gpu::Texture> fastRgbRate;
    std::unique_ptr<gpu::Texture> compositedRate;  // dynamic compositor output, PSF input
    std::unique_ptr<gpu::Texture> expectedElectrons, preAdcElectrons;
    std::unique_ptr<gpu::Texture> linearRgb, colorRgb, agcSource;  // ISP stage images
    std::array<std::unique_ptr<gpu::Texture>, 2> thermalState;
    std::array<std::unique_ptr<gpu::Texture>, 2> thermalCheckpoint; // acquisition-history snapshot
    std::unique_ptr<gpu::Texture> randomVectors;
    std::unique_ptr<gpu::Buffer> configBuffer, nucGainBuffer, nucOffsetBuffer;
    std::unique_ptr<gpu::Buffer> layerCameraBuffer;   // binding 10, per-stratum frames
    std::unique_ptr<gpu::Buffer> dynamicCounterBuffer; // binding 11, 4 x u32
    std::unique_ptr<gpu::Buffer> ispConfigBuffer;     // binding 12, 9 x float4
    std::unique_ptr<gpu::Buffer> defectPixelsBuffer;  // binding 13, uint2 array
    std::unique_ptr<gpu::Buffer> ispStatsBuffer;      // binding 14, 272 x u32
    // Readback staging, allocated once with the pipeline instead of per
    // call: both readbacks run between frames and reuse the same copy.
    std::unique_ptr<gpu::Buffer> counterStaging;
    std::unique_ptr<gpu::Buffer> ispStatsStaging;
    std::unique_ptr<gpu::Buffer> cdfBuffer;           // binding 15, 256 x f32
    std::unique_ptr<gpu::Buffer> tileStatsBuffer;     // binding 16, per-tile float4
    std::unique_ptr<gpu::Buffer> ispStatsCheckpoint;  // previous-frame AGC window snapshot (acquisition history)

    // Writes the effective-config rows staged by ApplyEffectiveConfig into
    // the device buffers, inside the recorded command buffer. Queue order
    // then guarantees the write lands after any in-flight frame that read
    // the previous values, which a record-time host memcpy could not.
    void UploadPendingParams(gpu::CommandList& cmd) {
        if (!paramsPending) return;
        cmd.UpdateBuffer(configBuffer->Ref(), 0,
                         std::span<const std::byte>(
                             reinterpret_cast<const std::byte*>(pendingConfigRows.data()),
                             sizeof(pendingConfigRows)));
        cmd.UpdateBuffer(ispConfigBuffer->Ref(), 0,
                         std::span<const std::byte>(
                             reinterpret_cast<const std::byte*>(pendingIspRows.data()),
                             sizeof(pendingIspRows)));
        const gpu::BufferBarrier uploaded[2] = {
            {configBuffer->Ref(), gpu::Access::TransferWrite,
             gpu::Access::ComputeRead},
            {ispConfigBuffer->Ref(), gpu::Access::TransferWrite,
             gpu::Access::ComputeRead}};
        cmd.Barrier(gpu::BarrierBatch{
            {}, std::span<const gpu::BufferBarrier>(uploaded), {}});
        paramsPending = false;
    }

    std::unique_ptr<gpu::BindingLayout> setLayout;
    std::unique_ptr<gpu::KernelLayout> pipelineLayout;
    std::unique_ptr<gpu::ComputeKernel> psfKernel;
    std::unique_ptr<gpu::ComputeKernel> readoutKernel;
    std::unique_ptr<gpu::ComputeKernel> statsTilesKernel;
    std::unique_ptr<gpu::ComputeKernel> statsReduceKernel;
    std::unique_ptr<gpu::ComputeKernel> statsHistKernel;
    std::unique_ptr<gpu::ComputeKernel> cdfKernel;
    std::unique_ptr<gpu::ComputeKernel> demosaicKernel;
    std::unique_ptr<gpu::ComputeKernel> hsvKernel;
    std::unique_ptr<gpu::ComputeKernel> rngKernel;
    std::unique_ptr<gpu::ComputeKernel> fastRgbKernel;
    std::unique_ptr<gpu::ComputeKernel> dynamicKernel;
    std::unique_ptr<gpu::BindGroupPool> descriptorPool;
    // 0/1: PSF H/V reading the caller's measured rate directly (unstratified
    // path). 2/3: readout ping/pong. 4: display. 5: RNG audit. 6: fast RGB.
    // 7: dynamic compositor (strata array inputs, composited output).
    // 8/9: PSF H/V reading the compositor output (stratified path).
    // 10: acquisition statistics. 11: demosaic. 12: unused. 13: CDF.
    // 14: HSV (display image bound as both input and in-place output).
    // Immutable while frames may be in flight; Configure waits idle first.
    std::array<gpu::BindGroup, 15> groups{};
    // What the "the caller's image changed" guards compare: the identity a
    // descriptor names. Zero means the lazy binding has not happened yet.
    gpu::NativeHandle measurementIdentity = 0;
    gpu::NativeHandle fastRgbInputIdentity = 0;
    gpu::NativeHandle strataRateIdentity = 0;
    gpu::NativeHandle strataDepthIdentity = 0;

    Impl(gpu::Device& dev, u32 w, u32 h) : device(dev), width(w), height(h) {}

    bool MakeImage(std::unique_ptr<gpu::Texture>& target, gpu::Format format) {
        auto texture = device.CreateTexture(gpu::TextureDesc{
            width, height, 1, 1, format,
            gpu::TextureUsage::Storage | gpu::TextureUsage::Sampled |
                gpu::TextureUsage::TransferSrc | gpu::TextureUsage::TransferDst,
            gpu::TextureDimension::Tex2D, gpu::MemoryClass::DeviceLocal});
        if (!texture.has_value()) return false;
        target = std::move(texture.value());
        return true;
    }

    Result<void, String> Initialize();
    void TransitionImages(gpu::CommandList& cmd);
    void UpdateSet(gpu::BindGroup set, const std::array<gpu::TextureRef, 7>& images);
    void BindAndDispatch(gpu::CommandList& cmd, const gpu::ComputeKernel& kernel,
                         gpu::BindGroup set, const CameraPush& push);
    void BindAndDispatchGrid(gpu::CommandList& cmd, const gpu::ComputeKernel& kernel,
                             gpu::BindGroup set, const CameraPush& push,
                             u32 groupsX, u32 groupsY);
    CameraPush MakePush(u64 acquisitionIndex, f64 frameTimeSeconds) const;
};

Result<void, String> GpuCameraPipeline::Impl::Initialize() {
    if (width == 0 || height == 0)
        return Result<void, String>::Err("GPU camera extent must be positive");
    if (!MakeImage(psfTemp, gpu::Format::RGBA32Float) ||
        !MakeImage(blurred, gpu::Format::RGBA32Float) ||
        !MakeImage(rawDn, gpu::Format::RGBA32Float) ||
        !MakeImage(corrected, gpu::Format::RGBA32Float) ||
        !MakeImage(display, gpu::Format::RGBA32Float) ||
        !MakeImage(fastRgbRate, gpu::Format::RGBA32Float) ||
        !MakeImage(compositedRate, gpu::Format::RGBA32Float) ||
        !MakeImage(expectedElectrons, gpu::Format::RGBA32Float) ||
        !MakeImage(preAdcElectrons, gpu::Format::RGBA32Float) ||
        !MakeImage(linearRgb, gpu::Format::RGBA32Float) ||
        !MakeImage(colorRgb, gpu::Format::RGBA32Float) ||
        !MakeImage(agcSource, gpu::Format::RGBA32Float) ||
        !MakeImage(thermalState[0], gpu::Format::RGBA32Float) ||
        !MakeImage(thermalState[1], gpu::Format::RGBA32Float) ||
        !MakeImage(thermalCheckpoint[0], gpu::Format::RGBA32Float) ||
        !MakeImage(thermalCheckpoint[1], gpu::Format::RGBA32Float) ||
        !MakeImage(randomVectors, gpu::Format::RGBA32Uint))
        return Result<void, String>::Err("cannot allocate GPU camera images");

    // Bindings 0-6 are storage images, 7-16 storage buffers, mirroring the
    // register classes in camera_*.comp.hlsl (t7-t10, t12, t13 read-only;
    // u11, u14-u16 read-write). Bindings 0/1 may hold 2D-array views (the
    // dynamic compositor's strata) that other sets of this shared layout
    // never use; the compositor is the only consumer and only touches them
    // when a stratified measurement is actually recorded.
    constexpr gpu::BindingSlot kSlots[] = {
        {0, gpu::BindingKind::StorageTexture, 1, true},
        {1, gpu::BindingKind::StorageTexture, 1, true},
        {2, gpu::BindingKind::StorageTexture, 1, false},
        {3, gpu::BindingKind::StorageTexture, 1, false},
        {4, gpu::BindingKind::StorageTexture, 1, false},
        {5, gpu::BindingKind::StorageTexture, 1, false},
        {6, gpu::BindingKind::StorageTexture, 1, false},
        {7, gpu::BindingKind::ReadOnlyBuffer, 1, false},
        {8, gpu::BindingKind::ReadOnlyBuffer, 1, false},
        {9, gpu::BindingKind::ReadOnlyBuffer, 1, false},
        {10, gpu::BindingKind::ReadOnlyBuffer, 1, false},
        {11, gpu::BindingKind::ReadWriteBuffer, 1, false},
        {12, gpu::BindingKind::ReadOnlyBuffer, 1, false},
        {13, gpu::BindingKind::ReadOnlyBuffer, 1, false},
        {14, gpu::BindingKind::ReadWriteBuffer, 1, false},
        {15, gpu::BindingKind::ReadWriteBuffer, 1, false},
        {16, gpu::BindingKind::ReadWriteBuffer, 1, false},
    };
    auto layoutResult = device.CreateBindingLayout(
        gpu::BindingLayoutDesc{0, std::span<const gpu::BindingSlot>(kSlots)});
    if (!layoutResult.has_value())
        return Result<void, String>::Err("cannot create GPU camera descriptor layout");

    const gpu::BindingLayout* spaces[] = {layoutResult.value().get()};
    auto kernelLayoutResult = device.CreateKernelLayout(gpu::KernelLayoutDesc{
        std::span<const gpu::BindingLayout* const>(spaces), sizeof(CameraPush)});
    if (!kernelLayoutResult.has_value())
        return Result<void, String>::Err("cannot create GPU camera pipeline layout");

    const StringView kShaderNames[] = {
        "camera_psf.comp",
        "camera_readout.comp",
        "camera_stats_tiles.comp",
        "camera_stats_reduce.comp",
        "camera_stats_hist.comp",
        "camera_cdf.comp",
        "camera_demosaic.comp",
        "camera_hsv.comp",
        "camera_rng_vectors.comp",
        "camera_fast_rgb.comp",
        "camera_dynamic.comp",
    };
    auto shaders = device.Shaders().Load(kShaderNames);
    if (!shaders.has_value())
        return Result<void, String>::Err(
            "cannot load camera compute kernels: " + shaders.error());

    const gpu::KernelLayout* kernelLayout = kernelLayoutResult.value().get();
    auto psfResult = device.CreateComputeKernel(
        gpu::ComputeKernelDesc{&shaders.value()[0], kernelLayout});
    if (!psfResult.has_value())
        return Result<void, String>::Err("cannot create camera compute pipeline");
    auto readoutResult = device.CreateComputeKernel(
        gpu::ComputeKernelDesc{&shaders.value()[1], kernelLayout});
    if (!readoutResult.has_value())
        return Result<void, String>::Err("cannot create camera compute pipeline");
    auto statsTilesResult = device.CreateComputeKernel(
        gpu::ComputeKernelDesc{&shaders.value()[2], kernelLayout});
    if (!statsTilesResult.has_value())
        return Result<void, String>::Err("cannot create camera compute pipeline");
    auto statsReduceResult = device.CreateComputeKernel(
        gpu::ComputeKernelDesc{&shaders.value()[3], kernelLayout});
    if (!statsReduceResult.has_value())
        return Result<void, String>::Err("cannot create camera compute pipeline");
    auto statsHistResult = device.CreateComputeKernel(
        gpu::ComputeKernelDesc{&shaders.value()[4], kernelLayout});
    if (!statsHistResult.has_value())
        return Result<void, String>::Err("cannot create camera compute pipeline");
    auto cdfResult = device.CreateComputeKernel(
        gpu::ComputeKernelDesc{&shaders.value()[5], kernelLayout});
    if (!cdfResult.has_value())
        return Result<void, String>::Err("cannot create camera compute pipeline");
    auto demosaicResult = device.CreateComputeKernel(
        gpu::ComputeKernelDesc{&shaders.value()[6], kernelLayout});
    if (!demosaicResult.has_value())
        return Result<void, String>::Err("cannot create camera compute pipeline");
    auto hsvResult = device.CreateComputeKernel(
        gpu::ComputeKernelDesc{&shaders.value()[7], kernelLayout});
    if (!hsvResult.has_value())
        return Result<void, String>::Err("cannot create camera compute pipeline");
    auto rngResult = device.CreateComputeKernel(
        gpu::ComputeKernelDesc{&shaders.value()[8], kernelLayout});
    if (!rngResult.has_value())
        return Result<void, String>::Err("cannot create camera compute pipeline");
    auto fastRgbResult = device.CreateComputeKernel(
        gpu::ComputeKernelDesc{&shaders.value()[9], kernelLayout});
    if (!fastRgbResult.has_value())
        return Result<void, String>::Err("cannot create camera compute pipeline");
    auto dynamicResult = device.CreateComputeKernel(
        gpu::ComputeKernelDesc{&shaders.value()[10], kernelLayout});
    if (!dynamicResult.has_value())
        return Result<void, String>::Err("cannot create camera compute pipeline");

    auto poolResult = device.CreateBindGroupPool(
        *layoutResult.value(), static_cast<u32>(groups.size()));
    if (!poolResult.has_value())
        return Result<void, String>::Err("cannot create GPU camera descriptor pool");
    auto allocated = poolResult.value()->Allocate(static_cast<u32>(groups.size()));
    if (!allocated.has_value())
        return Result<void, String>::Err("cannot allocate GPU camera descriptor sets");

    // Per-stratum camera frames (binding 10) and the compositor's disocclusion
    // counters (binding 11). Both are bound into every set, used or not. The
    // frame buffer lives in device memory: each measurement writes it inside
    // the recorded command buffer, so an in-flight frame from a previous
    // submission can no longer read a half-overwritten copy.
    auto makeDeviceBuffer = [&](u64 size, gpu::BufferUsage usage)
        -> Result<std::unique_ptr<gpu::Buffer>, String> {
        auto buffer = device.CreateBuffer(
            gpu::BufferDesc{size, usage, gpu::MemoryClass::DeviceLocal, 0});
        if (!buffer.has_value())
            return Fail<std::unique_ptr<gpu::Buffer>>(buffer.error());
        return std::move(buffer.value());
    };
    auto layerCameraResult = makeDeviceBuffer(
        8u * sizeof(DynamicLayerCamera),
        gpu::BufferUsage::Storage | gpu::BufferUsage::TransferDst);
    auto dynamicCounterResult = makeDeviceBuffer(
        4u * sizeof(u32),
        gpu::BufferUsage::Storage | gpu::BufferUsage::TransferDst |
            gpu::BufferUsage::TransferSrc);
    // ISP statistics (binding 14), Equalize CDF (15) and per-tile partials
    // (16). The statistics buffer is zeroed so the first acquisition's
    // previous-frame window is degenerate and the display pass falls back to
    // the current frame's range.
    auto ispStatsResult = makeDeviceBuffer(
        isp::kStatsSlots * sizeof(u32),
        gpu::BufferUsage::Storage | gpu::BufferUsage::TransferDst |
            gpu::BufferUsage::TransferSrc);
    auto cdfResultBuffer = makeDeviceBuffer(
        256u * sizeof(f32), gpu::BufferUsage::Storage);
    const u64 tileCount =
        static_cast<u64>((width + 15u) / 16u) * ((height + 15u) / 16u);
    // Four float4 rows per tile (all-element min/max/sum/sat, per-channel
    // unsaturated sums and counts, unsaturated channel-0 display range --
    // camera_stats phases 0/1), then the per-tile 256-bin histogram block
    // the phase-2 flush writes and the CDF pass merges.
    auto tileStatsResult = makeDeviceBuffer(
        tileCount * (16u + 256u) * sizeof(u32), gpu::BufferUsage::Storage);
    if (!layerCameraResult.has_value() || !dynamicCounterResult.has_value() ||
        !ispStatsResult.has_value() || !cdfResultBuffer.has_value() ||
        !tileStatsResult.has_value())
        return Result<void, String>::Err(
            "cannot allocate GPU camera dynamic buffers");
    // Acquisition history: the checkpoint twin of the statistics buffer. The previous
    // frame's Linear-AGC window rides in the stats slots; without a snapshot
    // a replayed tick's display tone would read the discarded run's window
    // and the replay would not be bit-identical.
    auto ispStatsCheckpointResult = makeDeviceBuffer(
        isp::kStatsSlots * sizeof(u32),
        gpu::BufferUsage::Storage | gpu::BufferUsage::TransferDst |
            gpu::BufferUsage::TransferSrc);
    if (!ispStatsCheckpointResult.has_value())
        return Result<void, String>::Err(
            "cannot allocate GPU camera checkpoint buffer");
    auto counterStagingResult = device.CreateBuffer(gpu::BufferDesc{
        4u * sizeof(u32), gpu::BufferUsage::TransferDst,
        gpu::MemoryClass::Readback, 0});
    auto ispStatsStagingResult = device.CreateBuffer(gpu::BufferDesc{
        isp::kStatsSlots * sizeof(u32), gpu::BufferUsage::TransferDst,
        gpu::MemoryClass::Readback, 0});
    if (!counterStagingResult.has_value() || !ispStatsStagingResult.has_value())
        return Result<void, String>::Err(
            "cannot allocate GPU camera statistics staging");

    setLayout = std::move(layoutResult.value());
    pipelineLayout = std::move(kernelLayoutResult.value());
    psfKernel = std::move(psfResult.value());
    readoutKernel = std::move(readoutResult.value());
    statsTilesKernel = std::move(statsTilesResult.value());
    statsReduceKernel = std::move(statsReduceResult.value());
    statsHistKernel = std::move(statsHistResult.value());
    cdfKernel = std::move(cdfResult.value());
    demosaicKernel = std::move(demosaicResult.value());
    hsvKernel = std::move(hsvResult.value());
    rngKernel = std::move(rngResult.value());
    fastRgbKernel = std::move(fastRgbResult.value());
    dynamicKernel = std::move(dynamicResult.value());
    descriptorPool = std::move(poolResult.value());
    groups = std::move(allocated.value());
    layerCameraBuffer = std::move(layerCameraResult.value());
    dynamicCounterBuffer = std::move(dynamicCounterResult.value());
    ispStatsBuffer = std::move(ispStatsResult.value());
    cdfBuffer = std::move(cdfResultBuffer.value());
    tileStatsBuffer = std::move(tileStatsResult.value());
    ispStatsCheckpoint = std::move(ispStatsCheckpointResult.value());
    counterStaging = std::move(counterStagingResult.value());
    ispStatsStaging = std::move(ispStatsStagingResult.value());

    device.ExecuteImmediate([&](gpu::CommandList& cmd) {
        cmd.FillBuffer(layerCameraBuffer->Ref(), 0, gpu::kWholeSize, 0u);
        cmd.FillBuffer(dynamicCounterBuffer->Ref(), 0, gpu::kWholeSize, 0u);
        cmd.FillBuffer(ispStatsBuffer->Ref(), 0, gpu::kWholeSize, 0u);
    });
    return Result<void, String>::Ok();
}

void GpuCameraPipeline::Impl::TransitionImages(gpu::CommandList& cmd) {
    if (imagesInGeneral) return;
    const gpu::Texture* images[] = {
        psfTemp.get(), blurred.get(), rawDn.get(), corrected.get(),
        display.get(), fastRgbRate.get(), expectedElectrons.get(), preAdcElectrons.get(),
        thermalState[0].get(), thermalState[1].get(), randomVectors.get(),
        compositedRate.get(), linearRgb.get(), colorRgb.get(), agcSource.get(),
        thermalCheckpoint[0].get(), thermalCheckpoint[1].get()};
    gpu::TextureBarrier barriers[17];
    for (size_t i = 0; i < std::size(images); ++i) {
        barriers[i] = gpu::TextureBarrier{
            images[i]->Ref(), gpu::Access::None,
            gpu::Access::ComputeRead | gpu::Access::ComputeWrite, true};
    }
    cmd.Barrier(gpu::BarrierBatch{
        {}, {}, std::span<const gpu::TextureBarrier>(barriers)});
    imagesInGeneral = true;
}

void GpuCameraPipeline::Impl::UpdateSet(
    gpu::BindGroup set, const std::array<gpu::TextureRef, 7>& images) {
    gpu::BindingWrite writes[17];
    for (u32 i = 0; i < images.size(); ++i)
        writes[i] = gpu::BindingWrite::StorageTexture(i, images[i]);
    // The buffer slot's read-only/read-write class follows the HLSL
    // declaration; the Vulkan backend sees STORAGE_BUFFER either way.
    constexpr gpu::BindingKind kBufferKinds[10] = {
        gpu::BindingKind::ReadOnlyBuffer,   // 7 NUC gain
        gpu::BindingKind::ReadOnlyBuffer,   // 8 NUC offset
        gpu::BindingKind::ReadOnlyBuffer,   // 9 camera config
        gpu::BindingKind::ReadOnlyBuffer,   // 10 per-stratum frames
        gpu::BindingKind::ReadWriteBuffer,  // 11 dynamic counters
        gpu::BindingKind::ReadOnlyBuffer,   // 12 ISP config
        gpu::BindingKind::ReadOnlyBuffer,   // 13 defect list
        gpu::BindingKind::ReadWriteBuffer,  // 14 ISP statistics
        gpu::BindingKind::ReadWriteBuffer,  // 15 CDF
        gpu::BindingKind::ReadWriteBuffer,  // 16 tile statistics
    };
    const gpu::Buffer* buffers[] = {
        nucGainBuffer.get(), nucOffsetBuffer.get(), configBuffer.get(),
        layerCameraBuffer.get(), dynamicCounterBuffer.get(),
        ispConfigBuffer.get(), defectPixelsBuffer.get(),
        ispStatsBuffer.get(), cdfBuffer.get(), tileStatsBuffer.get()};
    for (u32 i = 0; i < 10; ++i) {
        writes[7 + i] = gpu::BindingWrite::Buffer(
            7 + i, kBufferKinds[i], buffers[i]->Ref(), 0, buffers[i]->Ref().size);
    }
    device.WriteBindGroup(set, std::span<const gpu::BindingWrite>(writes));
}

void GpuCameraPipeline::Impl::BindAndDispatch(
    gpu::CommandList& cmd, const gpu::ComputeKernel& kernel, gpu::BindGroup set,
    const CameraPush& push) {
    BindAndDispatchGrid(cmd, kernel, set, push,
                        (width + 15) / 16, (height + 15) / 16);
}

void GpuCameraPipeline::Impl::BindAndDispatchGrid(
    gpu::CommandList& cmd, const gpu::ComputeKernel& kernel, gpu::BindGroup set,
    const CameraPush& push, u32 groupsX, u32 groupsY) {
    cmd.SetKernel(kernel);
    cmd.SetBindGroup(*pipelineLayout, 0, set);
    cmd.PushConstants(*pipelineLayout, push);
    cmd.Dispatch(groupsX, groupsY, 1);
}

CameraPush GpuCameraPipeline::Impl::MakePush(
    u64 acquisitionIndex, f64 frameTimeSeconds) const {
    CameraPush push;
    push.width = width;
    push.height = height;
    push.physicalWidth = config.optics.sensorWidthPx;
    push.physicalHeight = config.optics.sensorHeightPx;
    push.channelCount = static_cast<u32>(config.device.channels.size());
    push.cfa = static_cast<u32>(config.device.cfa);
    push.detector = static_cast<u32>(config.device.detector);
    push.seed = camera::DeviceRandomSeed(config);
    push.acquisitionLo = static_cast<u32>(acquisitionIndex);
    push.acquisitionHi = static_cast<u32>(acquisitionIndex >> 32);
    push.adcBits = config.readout.adcBits;
    push.frameTimeSeconds = static_cast<f32>(frameTimeSeconds);
    push.deltaSeconds = static_cast<f32>(currentDeltaSeconds);
    push.firstThermal = firstCaptureWasFirst ? 1u : 0u;
    push.sigmaR = sigmaX[0];
    push.sigmaG = sigmaX[1];
    push.sigmaB = sigmaX[2];
    push.sigmaRy = sigmaY[0];
    push.sigmaGy = sigmaY[1];
    push.sigmaBy = sigmaY[2];
    if (config.optics.cosFourthVignetting) push.flags |= Vignette;
    if (config.quality.noiseFree) push.flags |= NoiseFree;
    if (config.device.detector == camera::DetectorKind::Photon) {
        if (config.photon.enableShotNoise) push.flags |= Shot;
        if (config.photon.enableDarkCurrent) push.flags |= DarkCurrent;
        if (config.photon.enableDarkShotNoise) push.flags |= DarkShot;
        if (config.photon.enableReadNoise) push.flags |= Read;
        if (config.photon.enableFpn) push.flags |= Fpn;
        if (config.photon.applyNuc) push.flags |= Nuc;
        if (!config.photon.nucGainMap.empty()) push.flags |= NucGainMap;
        if (!config.photon.nucOffsetElectronsMap.empty()) push.flags |= NucOffsetMap;
    } else {
        if (!config.thermal.nucGainMap.empty()) push.flags |= NucGainMap;
        if (!config.thermal.nucOffsetDnMap.empty()) push.flags |= NucOffsetMap;
    }
    return push;
}

GpuCameraPipeline::GpuCameraPipeline(std::unique_ptr<Impl> impl)
    : m_impl(std::move(impl)) {}

GpuCameraPipeline::~GpuCameraPipeline() = default;

Result<std::unique_ptr<GpuCameraPipeline>, String>
GpuCameraPipeline::Create(gpu::Device& device, u32 width, u32 height) {
    auto impl = std::make_unique<Impl>(device, width, height);
    auto initialized = impl->Initialize();
    if (!initialized)
        return Fail<std::unique_ptr<GpuCameraPipeline>>(initialized.error());
    return std::unique_ptr<GpuCameraPipeline>(
        new GpuCameraPipeline(std::move(impl)));
}

Result<void, String>
GpuCameraPipeline::Configure(const camera::CameraConfig& config) {
    const auto valid = camera::ValidateCameraConfig(config);
    if (!valid) return Result<void, String>::Err(valid.error());
    if (config.device.channels.size() > 3)
        return Result<void, String>::Err(
            "GPU camera supports at most three response channels");
    if ((m_impl->width != config.optics.sensorWidthPx ||
         m_impl->height != config.optics.sensorHeightPx) &&
        config.quality.backend != camera::ProcessingBackend::GpuPreview)
        return Result<void, String>::Err(
            "physical CPU/reference camera requires full sensor extent; "
            "set gpu_preview for an explicitly scaled viewport");
    // Descriptors and static parameter buffers are immutable during a run.
    // A Qt window may have older command buffers in flight when a panel
    // changes the camera; wait before replacing anything they still read.
    if (!m_impl->device.WaitIdle().has_value())
        return Result<void, String>::Err(
            "device lost while reconfiguring GPU camera");
    m_impl->config = config;
    const auto& optics = config.optics;
    for (size_t i = 0; i < config.device.channels.size(); ++i) {
        f64 sigma = optics.psfSigmaPixelsOverride;
        if (sigma < 0.0) {
            const f64 wavelength = EffectiveWavelengthNm(
                config.device.channels[i].response, config.device.detector);
            sigma = 0.437 * wavelength * 1e-3 * optics.fNumber /
                    optics.pixelPitchUm;
        }
        const f64 sigmaX = sigma * m_impl->width / optics.sensorWidthPx;
        const f64 sigmaY = sigma * m_impl->height / optics.sensorHeightPx;
        if (std::max(sigmaX, sigmaY) > 4.0)
            QL_LOG_WARN("GPU camera effective PSF sigma ({}, {}) preview px is "
                        "capped at 4 px", sigmaX, sigmaY);
        m_impl->sigmaX[i] = static_cast<f32>(std::clamp(sigmaX, 0.0, 4.0));
        m_impl->sigmaY[i] = static_cast<f32>(std::clamp(sigmaY, 0.0, 4.0));
    }
    if (config.device.channels.size() == 1) {
        m_impl->sigmaX[1] = m_impl->sigmaX[2] = m_impl->sigmaX[0];
        m_impl->sigmaY[1] = m_impl->sigmaY[2] = m_impl->sigmaY[0];
    }

    m_impl->configRows = {};
    if (auto rows = UpdateReadoutRows(config, m_impl->configRows); !rows)
        return rows;
    // The RGB approximation starts with linear scene RGB. Each row maps it
    // to one latent device channel, then applies the same absolute radiance,
    // aperture, pixel-area and response integral as the CPU adapter.
    const auto aperture = camera::ApertureSolidAngleSr(optics.fNumber);
    const auto pixelArea = camera::PixelCollectionAreaM2(optics);
    if (!aperture || !pixelArea)
        return Result<void, String>::Err(
            aperture ? pixelArea.error() : aperture.error());
    const f64 radianceScale = config.fastRgbRadianceScale > 0.0
        ? config.fastRgbRadianceScale : 1.0;
    for (size_t channel = 0; channel < config.device.channels.size(); ++channel) {
        const auto& response = config.device.channels[channel].response;
        const auto& base = response.systemResponse ? *response.systemResponse :
            config.device.detector == camera::DetectorKind::Photon
                ? *response.quantumEfficiency : *response.thermalAbsorptance;
        Vector<camera::SpectralIrradianceSample> flat;
        flat.reserve(base.wavelengthNm.size());
        for (const f64 wavelength : base.wavelengthNm)
            flat.push_back({wavelength, 1.0});
        f64 responseIntegral = 0.0;
        if (config.device.detector == camera::DetectorKind::Photon) {
            const auto measured = camera::IntegratePhoton(
                flat, response, pixelArea.value());
            if (!measured) return Result<void, String>::Err(measured.error());
            responseIntegral = measured.value().electronRatePerSecond;
        } else {
            const auto measured = camera::IntegrateThermal(
                flat, response, pixelArea.value());
            if (!measured) return Result<void, String>::Err(measured.error());
            responseIntegral = measured.value().absorbedPowerW;
        }
        const f64 factor = radianceScale * aperture.value() * responseIntegral;
        for (size_t rgb = 0; rgb < 3; ++rgb)
            m_impl->configRows[5 + channel][rgb] = static_cast<f32>(
                config.fastRgbToDevice[channel * 3 + rgb] * factor);
    }
    m_impl->configRows[8][0] = static_cast<f32>(optics.focalLengthMm * 1e-3);
    m_impl->configRows[8][1] = static_cast<f32>(optics.pixelPitchUm * 1e-6);
    const auto makeBuffer = [&](const void* data, u64 size,
                                gpu::BufferUsage usage)
        -> Result<std::unique_ptr<gpu::Buffer>, String> {
        auto buffer = m_impl->device.CreateBuffer(
            gpu::BufferDesc{size, usage, gpu::MemoryClass::Upload, 0});
        if (!buffer.has_value())
            return Fail<std::unique_ptr<gpu::Buffer>>(buffer.error());
        buffer.value()->Upload(data, size);
        return std::move(buffer.value());
    };
    auto configBufferResult = makeBuffer(
        m_impl->configRows.data(), sizeof(m_impl->configRows),
        gpu::BufferUsage::Storage | gpu::BufferUsage::TransferDst);
    if (!configBufferResult.has_value())
        return Result<void, String>::Err(configBufferResult.error());
    const auto& gainMap = config.device.detector == camera::DetectorKind::Photon
        ? config.photon.nucGainMap : config.thermal.nucGainMap;
    const auto& offsetMap = config.device.detector == camera::DetectorKind::Photon
        ? config.photon.nucOffsetElectronsMap : config.thermal.nucOffsetDnMap;
    const auto uploadMap = [&](const std::vector<f64>& values, f32 fallback)
        -> Result<std::unique_ptr<gpu::Buffer>, String> {
        std::vector<f32> upload;
        upload.reserve(std::max<size_t>(values.size(), 1));
        if (values.empty()) upload.push_back(fallback);
        else for (const f64 value : values) upload.push_back(static_cast<f32>(value));
        return makeBuffer(upload.data(), upload.size() * sizeof(f32),
                          gpu::BufferUsage::Storage);
    };
    auto nucGainResult = uploadMap(gainMap, 1.0f);
    if (!nucGainResult.has_value())
        return Result<void, String>::Err(nucGainResult.error());
    auto nucOffsetResult = uploadMap(offsetMap, 0.0f);
    if (!nucOffsetResult.has_value())
        return Result<void, String>::Err(nucOffsetResult.error());

    // The ISP configuration (binding 12) is one row of float4 per parameter
    // group; see BuildIspRows for the layout. The buffer is CPU_TO_GPU so the
    // AE/AWB loop can re-upload the dynamic rows (white balance, row 6
    // exposure/gain, HSV) per acquisition through ApplyEffectiveConfig. The
    // defect list (binding 13) is a uint2 array with a one-element
    // placeholder when empty, matching the NUC map convention.
    std::array<std::array<f32, 4>, 9> ispRows{};
    BuildIspRows(config, ispRows, m_impl->ispFlags);
    auto ispConfigResult = makeBuffer(
        ispRows.data(), sizeof(ispRows),
        gpu::BufferUsage::Storage | gpu::BufferUsage::TransferDst);
    std::vector<std::array<u32, 2>> defects = config.isp.defectPixels;
    if (defects.empty()) defects.push_back({0u, 0u});
    if (!ispConfigResult.has_value())
        return Result<void, String>::Err(ispConfigResult.error());
    auto defectResult = m_impl->device.CreateBuffer(gpu::BufferDesc{
        std::max<size_t>(defects.size(), 1u) * 2u * sizeof(u32),
        gpu::BufferUsage::Storage, gpu::MemoryClass::Upload, 0});
    if (!defectResult.has_value())
        return Result<void, String>::Err("cannot allocate GPU camera defect list");
    defectResult.value()->Upload(defects.data(), defects.size() * 2u * sizeof(u32));
    m_impl->configBuffer = std::move(configBufferResult.value());
    m_impl->nucGainBuffer = std::move(nucGainResult.value());
    m_impl->nucOffsetBuffer = std::move(nucOffsetResult.value());
    m_impl->ispConfigBuffer = std::move(ispConfigResult.value());
    m_impl->defectPixelsBuffer = std::move(defectResult.value());

    const gpu::TextureRef dummy = m_impl->psfTemp->Ref();
    m_impl->UpdateSet(m_impl->groups[0], {
        dummy, m_impl->psfTemp->Ref(),
        dummy, dummy, dummy, dummy, dummy});
    m_impl->UpdateSet(m_impl->groups[1], {
        m_impl->psfTemp->Ref(), m_impl->blurred->Ref(),
        dummy, dummy, dummy, dummy, dummy});
    m_impl->UpdateSet(m_impl->groups[2], {
        m_impl->blurred->Ref(), m_impl->thermalState[0]->Ref(),
        m_impl->rawDn->Ref(), m_impl->corrected->Ref(),
        m_impl->expectedElectrons->Ref(), m_impl->preAdcElectrons->Ref(),
        m_impl->thermalState[1]->Ref()});
    m_impl->UpdateSet(m_impl->groups[3], {
        m_impl->blurred->Ref(), m_impl->thermalState[1]->Ref(),
        m_impl->rawDn->Ref(), m_impl->corrected->Ref(),
        m_impl->expectedElectrons->Ref(), m_impl->preAdcElectrons->Ref(),
        m_impl->thermalState[0]->Ref()});
    // Display: the color-pass product feeds the visible branch, the corrected
    // scalar feeds the infrared branch; both write the display and agcSource.
    m_impl->UpdateSet(m_impl->groups[4], {
        m_impl->corrected->Ref(), m_impl->linearRgb->Ref(),
        m_impl->colorRgb->Ref(), m_impl->display->Ref(),
        m_impl->agcSource->Ref(), dummy, dummy});
    m_impl->UpdateSet(m_impl->groups[5], {
        dummy, dummy, m_impl->randomVectors->Ref(),
        dummy, dummy, dummy, dummy});
    m_impl->UpdateSet(m_impl->groups[6], {
        dummy, m_impl->fastRgbRate->Ref(),
        dummy, dummy, dummy, dummy, dummy});
    // Dynamic compositor set: bindings 0/1 re-pointed at the caller's strata
    // array views on the first stratified measurement (their PARTIALLY_BOUND
    // flag covers every other state of this set).
    m_impl->UpdateSet(m_impl->groups[7], {
        m_impl->compositedRate->Ref(), m_impl->compositedRate->Ref(),
        m_impl->compositedRate->Ref(), dummy, dummy, dummy, dummy});
    // PSF pair reading the compositor output (stratified path).
    m_impl->UpdateSet(m_impl->groups[8], {
        m_impl->compositedRate->Ref(), m_impl->psfTemp->Ref(),
        dummy, dummy, dummy, dummy, dummy});
    m_impl->UpdateSet(m_impl->groups[9], {
        m_impl->psfTemp->Ref(), m_impl->blurred->Ref(),
        dummy, dummy, dummy, dummy, dummy});
    // Acquisition statistics read the corrected product; demosaic, color and
    // the CDF pass chain the ISP stage images.
    m_impl->UpdateSet(m_impl->groups[10], {
        m_impl->corrected->Ref(), dummy, dummy, dummy, dummy, dummy, dummy});
    // Fused ISP pass: demosaic reads the corrected product and writes the
    // linearRgb/colorRgb analysis products plus the display and agcSource
    // images in one dispatch.
    m_impl->UpdateSet(m_impl->groups[11], {
        m_impl->corrected->Ref(), m_impl->linearRgb->Ref(),
        m_impl->colorRgb->Ref(), m_impl->display->Ref(),
        m_impl->agcSource->Ref(), dummy, dummy});
    m_impl->UpdateSet(m_impl->groups[12], {
        dummy, m_impl->linearRgb->Ref(), m_impl->colorRgb->Ref(),
        dummy, dummy, dummy, dummy});
    m_impl->UpdateSet(m_impl->groups[13], {
        dummy, dummy, dummy, dummy, dummy, dummy, dummy});
    // HSV: the display image is both input (binding 3) and in-place output
    // (binding 4); the pass runs after the display pass has finished and
    // every thread touches only its own texel.
    m_impl->UpdateSet(m_impl->groups[14], {
        dummy, dummy, dummy,
        m_impl->display->Ref(), m_impl->display->Ref(),
        dummy, dummy});
    m_impl->measurementIdentity = 0;
    m_impl->fastRgbInputIdentity = 0;
    m_impl->strataRateIdentity = 0;
    m_impl->strataDepthIdentity = 0;
    m_impl->configured = true;
    ResetState();
    return Result<void, String>::Ok();
}

Result<void, String>
GpuCameraPipeline::ApplyEffectiveConfig(const camera::CameraConfig& config) {
    if (!m_impl->configured)
        return Result<void, String>::Err("GPU camera must be configured first");
    // The effective capture config differs from the authored one only where
    // the AE/AWB loop moved exposure, gain or white balance; adopt it as the
    // pipeline's config so MakePush and the display/stats shading see one
    // consistent state. Refresh detector/readout rows 0-4 and the ISP buffer;
    // PSF sigmas and response integrals in rows 5-8 remain tied to the
    // measurement config and are never changed by this path. The rows are
    // staged here and written by the next record call inside its command
    // buffer (Impl::UploadPendingParams): a host memcpy at record time could
    // overwrite parameters a still-in-flight frame is reading.
    m_impl->config = config;
    if (auto rows = UpdateReadoutRows(config, m_impl->configRows); !rows)
        return rows;
    if (!m_impl->configBuffer || !m_impl->ispConfigBuffer)
        return Result<void, String>::Err(
            "GPU camera parameter buffers are not allocated");
    m_impl->pendingConfigRows = m_impl->configRows;
    BuildIspRows(config, m_impl->pendingIspRows, m_impl->ispFlags);
    m_impl->paramsPending = true;
    return Result<void, String>::Ok();
}

Result<void, String>
GpuCameraPipeline::RecordDisplayReprocess(
    gpu::CommandList& cmd, const camera::CameraConfig& config) {
    if (!m_impl->configured)
        return Result<void, String>::Err(
            "GPU camera display reprocess needs a configured pipeline");
    auto applied = ApplyEffectiveConfig(config);
    if (!applied) return applied;
    m_impl->TransitionImages(cmd);
    m_impl->UploadPendingParams(cmd);
    // The same acquisition index keys the HSV noise streams, so a reprocess
    // of unchanged parameters is bit-identical to the original display.
    auto push = m_impl->MakePush(m_impl->lastAcquisitionIndex,
                                 m_impl->lastFrameTimeSeconds);
    // Fused demosaic/color/display -> HSV. No statistics passes: the
    // Linear-AGC window keeps riding in the statistics buffer, which is what
    // makes a reprocess stateless. corrected still holds the last
    // acquisition and serves as the input.
    m_impl->BindAndDispatch(cmd, *m_impl->demosaicKernel, m_impl->groups[11],
                            push);
    StorageBarrier(cmd);
    if (m_impl->ispFlags & isp::kFlagHsv) {
        m_impl->BindAndDispatch(cmd, *m_impl->hsvKernel, m_impl->groups[14],
                                push);
        StorageBarrier(cmd);
    }
    return Result<void, String>::Ok();
}

Result<void, String> GpuCameraPipeline::RecordReadoutReprocess(
    gpu::CommandList& cmd, const camera::CameraConfig& config) {
    if (!m_impl->configured || !m_impl->hasCapture)
        return Result<void, String>::Err(
            "GPU camera readout reprocess needs a completed acquisition");
    if (auto applied = ApplyEffectiveConfig(config); !applied) return applied;
    m_impl->TransitionImages(cmd);
    m_impl->UploadPendingParams(cmd);
    auto push = m_impl->MakePush(m_impl->lastAcquisitionIndex,
                                 m_impl->lastFrameTimeSeconds);
    const gpu::BindGroup readoutGroup =
        m_impl->stateBefore == 0 ? m_impl->groups[2] : m_impl->groups[3];
    // The standing blurred rate belongs to the last measurement. Reading
    // stateBefore and overwriting stateCurrent replays that same device tick;
    // it does not step the thermal state to a new acquisition.
    m_impl->BindAndDispatch(cmd, *m_impl->readoutKernel, readoutGroup, push);
    StorageBarrier(cmd);
    m_impl->BindAndDispatch(cmd, *m_impl->statsTilesKernel, m_impl->groups[10], push);
    StorageBarrier(cmd);
    m_impl->BindAndDispatchGrid(cmd, *m_impl->statsReduceKernel,
                                m_impl->groups[10], push, 1, 1);
    StorageBarrier(cmd);
    m_impl->BindAndDispatch(cmd, *m_impl->statsHistKernel, m_impl->groups[10], push);
    StorageBarrier(cmd);
    m_impl->BindAndDispatchGrid(cmd, *m_impl->cdfKernel,
                                m_impl->groups[13], push, 1, 1);
    StorageBarrier(cmd);
    m_impl->BindAndDispatch(cmd, *m_impl->demosaicKernel, m_impl->groups[11], push);
    StorageBarrier(cmd);
    if (m_impl->ispFlags & isp::kFlagHsv) {
        m_impl->BindAndDispatch(cmd, *m_impl->hsvKernel, m_impl->groups[14], push);
        StorageBarrier(cmd);
    }
    return Result<void, String>::Ok();
}

void GpuCameraPipeline::ResetState() {
    m_impl->hasCapture = false;
    m_impl->firstCaptureWasFirst = false;
    m_impl->lastAcquisitionIndex = 0;
    m_impl->lastFrameTimeSeconds = 0.0;
    m_impl->beforeFrameTimeSeconds = 0.0;
    m_impl->currentDeltaSeconds = 0.0;
    m_impl->stateBefore = 0;
    m_impl->stateCurrent = 1;
    // The statistics buffer persists the previous frame's window and
    // histogram; resetting history makes the next acquisition the first one
    // again, whose window must fall back to its own range.
    if (m_impl->ispStatsBuffer) {
        m_impl->device.ExecuteImmediate([&](gpu::CommandList& cmd) {
            cmd.FillBuffer(m_impl->ispStatsBuffer->Ref(), 0, gpu::kWholeSize, 0u);
        });
    }
}

Result<GpuCameraPipeline::GpuCheckpoint, String>
GpuCameraPipeline::RecordStateCheckpoint(gpu::CommandList& cmd) {
    if (!m_impl->configured)
        return Fail<GpuCheckpoint>(
            "GPU camera checkpoint needs a configured pipeline");
    m_impl->TransitionImages(cmd);
    CopyThermalPair(cmd, m_impl->thermalState[0]->Ref(),
                    m_impl->thermalState[1]->Ref(),
                    m_impl->thermalCheckpoint[0]->Ref(),
                    m_impl->thermalCheckpoint[1]->Ref(),
                    gpu::Access::ComputeRead | gpu::Access::ComputeWrite);
    CopyStatsBuffer(cmd, m_impl->ispStatsBuffer->Ref(),
                    m_impl->ispStatsCheckpoint->Ref(),
                    gpu::Access::ComputeRead | gpu::Access::ComputeWrite);
    GpuCheckpoint checkpoint;
    checkpoint.acquisitionIndex = m_impl->lastAcquisitionIndex;
    checkpoint.frameTimeSeconds = m_impl->lastFrameTimeSeconds;
    checkpoint.hasCapture = m_impl->hasCapture;
    checkpoint.firstCaptureWasFirst = m_impl->firstCaptureWasFirst;
    checkpoint.beforeFrameTimeSeconds = m_impl->beforeFrameTimeSeconds;
    checkpoint.currentDeltaSeconds = m_impl->currentDeltaSeconds;
    checkpoint.stateBefore = m_impl->stateBefore;
    checkpoint.stateCurrent = m_impl->stateCurrent;
    return checkpoint;
}

Result<void, String>
GpuCameraPipeline::RestoreStateCheckpoint(const GpuCheckpoint& checkpoint,
                                          gpu::CommandList& cmd) {
    if (!m_impl->configured)
        return Result<void, String>::Err(
            "GPU camera checkpoint restore needs a configured pipeline");
    m_impl->TransitionImages(cmd);
    CopyThermalPair(cmd, m_impl->thermalCheckpoint[0]->Ref(),
                    m_impl->thermalCheckpoint[1]->Ref(),
                    m_impl->thermalState[0]->Ref(),
                    m_impl->thermalState[1]->Ref(),
                    gpu::Access::ComputeRead | gpu::Access::ComputeWrite |
                        gpu::Access::TransferWrite);
    CopyStatsBuffer(cmd, m_impl->ispStatsCheckpoint->Ref(),
                    m_impl->ispStatsBuffer->Ref(),
                    gpu::Access::ComputeRead | gpu::Access::ComputeWrite |
                        gpu::Access::TransferWrite);
    m_impl->hasCapture = checkpoint.hasCapture;
    m_impl->firstCaptureWasFirst = checkpoint.firstCaptureWasFirst;
    m_impl->lastAcquisitionIndex = checkpoint.acquisitionIndex;
    m_impl->lastFrameTimeSeconds = checkpoint.frameTimeSeconds;
    m_impl->beforeFrameTimeSeconds = checkpoint.beforeFrameTimeSeconds;
    m_impl->currentDeltaSeconds = checkpoint.currentDeltaSeconds;
    m_impl->stateBefore = checkpoint.stateBefore;
    m_impl->stateCurrent = checkpoint.stateCurrent;
    return Result<void, String>::Ok();
}

GpuCameraPipeline::Images GpuCameraPipeline::GetOutputs() const {
    if (!m_impl->configured) return {};
    return {m_impl->rawDn->Ref(),
            m_impl->corrected->Ref(),
            m_impl->display->Ref(),
            m_impl->expectedElectrons->Ref(),
            m_impl->preAdcElectrons->Ref(),
            m_impl->strataRateIdentity != 0 ? m_impl->compositedRate->Ref()
                                            : gpu::TextureRef{},
            m_impl->linearRgb->Ref(),
            m_impl->colorRgb->Ref(),
            m_impl->agcSource->Ref()};
}

gpu::TextureRef GpuCameraPipeline::GetRandomVectors() const {
    return m_impl->randomVectors->Ref();
}

Result<std::array<u32, 4>, String>
GpuCameraPipeline::ReadDynamicCounters() const {
    if (!m_impl->dynamicCounterBuffer)
        return Result<std::array<u32, 4>, String>::Err(
            "GPU camera dynamic counters are not allocated");
    if (!m_impl->counterStaging)
        return Result<std::array<u32, 4>, String>::Err(
            "GPU camera counter staging is not allocated");
    std::array<u32, 4> counts{};
    gpu::Buffer& staging = *m_impl->counterStaging;
    m_impl->device.ExecuteImmediate([&](gpu::CommandList& cmd) {
        cmd.Barrier(gpu::BufferBarrier{m_impl->dynamicCounterBuffer->Ref(),
                                       gpu::Access::ComputeWrite,
                                       gpu::Access::TransferRead});
        const gpu::BufferCopyRegion copy{0, 0, sizeof(counts)};
        cmd.CopyBuffer(m_impl->dynamicCounterBuffer->Ref(), staging.Ref(),
                       std::span<const gpu::BufferCopyRegion>{&copy, 1});
    });
    const void* data = staging.MapRead();
    if (!data)
        return Result<std::array<u32, 4>, String>::Err(
            "cannot map GPU camera counter staging");
    std::memcpy(counts.data(), data, sizeof(counts));
    staging.Unmap();
    return counts;
}

void GpuCameraPipeline::RecordDynamicCounterCopy(gpu::CommandList& cmd,
                                                 gpu::BufferRef dst,
                                                 u64 dstOffset) const {
    if (!m_impl->dynamicCounterBuffer) return;
    cmd.Barrier(gpu::BufferBarrier{m_impl->dynamicCounterBuffer->Ref(),
                                   gpu::Access::ComputeWrite,
                                   gpu::Access::TransferRead});
    const gpu::BufferCopyRegion copy{0, dstOffset, 4u * sizeof(u32)};
    cmd.CopyBuffer(m_impl->dynamicCounterBuffer->Ref(), dst,
                   std::span<const gpu::BufferCopyRegion>{&copy, 1});
}

Result<GpuCameraPipeline::IspStats, String>
GpuCameraPipeline::ReadIspStats() const {
    if (!m_impl->ispStatsBuffer)
        return Result<IspStats, String>::Err(
            "GPU camera statistics are not allocated");
    if (!m_impl->ispStatsStaging)
        return Result<IspStats, String>::Err(
            "GPU camera statistics staging is not allocated");
    std::array<u32, isp::kStatsSlots> slots{};
    gpu::Buffer& staging = *m_impl->ispStatsStaging;
    m_impl->device.ExecuteImmediate([&](gpu::CommandList& cmd) {
        cmd.Barrier(gpu::BufferBarrier{m_impl->ispStatsBuffer->Ref(),
                                       gpu::Access::ComputeWrite,
                                       gpu::Access::TransferRead});
        const gpu::BufferCopyRegion copy{0, 0, sizeof(slots)};
        cmd.CopyBuffer(m_impl->ispStatsBuffer->Ref(), staging.Ref(),
                       std::span<const gpu::BufferCopyRegion>{&copy, 1});
    });
    const void* data = staging.MapRead();
    if (!data)
        return Result<IspStats, String>::Err(
            "cannot map GPU camera statistics staging");
    std::memcpy(slots.data(), data, sizeof(slots));
    staging.Unmap();
    IspStats stats;
    std::memcpy(&stats.minValue, &slots[isp::kStatsMin], sizeof(f32));
    std::memcpy(&stats.maxValue, &slots[isp::kStatsMax], sizeof(f32));
    std::memcpy(&stats.meanValue, &slots[isp::kStatsMean], sizeof(f32));
    stats.saturatedCount = slots[isp::kStatsSat];
    stats.unsaturatedCount = slots[isp::kStatsUnsat];
    std::copy_n(slots.data(), stats.histogram.size(), stats.histogram.data());
    // Per-channel unsaturated sums (float bits) and counts (exact float bits
    // converted back), written by the stats reduce phase for the AWB loop.
    std::memcpy(stats.channelSums.data(), &slots[isp::kStatsChSumR],
                3u * sizeof(f32));
    f32 channelCounts[3]{};
    std::memcpy(channelCounts, &slots[isp::kStatsChCntR], 3u * sizeof(f32));
    for (u32 c = 0; c < 3u; ++c)
        stats.channelCounts[c] = static_cast<u32>(channelCounts[c] + 0.5f);
    return stats;
}

Result<void, String> GpuCameraPipeline::RecordMeasurement(
    gpu::CommandList& cmd, gpu::TextureRef measuredRate, u64 acquisitionIndex,
    f64 frameTimeSeconds, TimingQueries timing,
    const DynamicExposureInput& dynamic) {
    if (!m_impl->configured)
        return Result<void, String>::Err("GPU camera must be configured first");
    if (!measuredRate.IsValid() ||
        measuredRate.format != gpu::Format::RGBA32Float ||
        measuredRate.width != m_impl->width ||
        measuredRate.height != m_impl->height)
        return Result<void, String>::Err(
            "GPU camera needs a same-extent RGBA32F measured-rate image");
    if (!std::isfinite(frameTimeSeconds))
        return Result<void, String>::Err("GPU camera frame time must be finite");

    // The dynamic pass runs whenever the caller supplies strata images, even
    // for a single stratum: the PSF pair it feeds expects the compositor's
    // output image, and the caller's strata may be array views a 2D-typed PSF
    // set cannot bind. Without strata the unstratified path reads the caller's
    // measured rate directly, exactly as before temporal stratification.
    const bool stratified = dynamic.strataRate.IsValid() &&
                            dynamic.strataDepth.IsValid() &&
                            dynamic.layerCameras != nullptr;
    if (stratified &&
        (dynamic.strataRate.format != gpu::Format::RGBA32Float ||
         dynamic.strataDepth.format != gpu::Format::R32Float ||
         dynamic.strataRate.width != m_impl->width ||
         dynamic.strataRate.height != m_impl->height ||
         dynamic.strataDepth.width != m_impl->width ||
         dynamic.strataDepth.height != m_impl->height))
        return Result<void, String>::Err(
            "GPU camera dynamic inputs must match the sensor extent "
            "(strata rate RGBA32F array, strata depth R32F array)");
    if (stratified &&
        ((m_impl->strataRateIdentity != 0 &&
          m_impl->strataRateIdentity != dynamic.strataRate.Identity()) ||
         (m_impl->strataDepthIdentity != 0 &&
          m_impl->strataDepthIdentity != dynamic.strataDepth.Identity())))
        return Result<void, String>::Err(
            "GPU camera strata images changed; reconfigure after "
            "synchronizing in-flight frames");

    if (m_impl->measurementIdentity != 0 &&
        m_impl->measurementIdentity != measuredRate.Identity() && !stratified)
        return Result<void, String>::Err(
            "GPU camera measured-rate image changed; reconfigure after "
            "synchronizing in-flight frames");

    if (!m_impl->hasCapture) {
        m_impl->firstCaptureWasFirst = true;
        m_impl->currentDeltaSeconds = 0.0;
    } else if (acquisitionIndex == m_impl->lastAcquisitionIndex) {
        if (std::abs(frameTimeSeconds - m_impl->lastFrameTimeSeconds) > 1e-9)
            return Result<void, String>::Err(
                "same acquisition index must keep the same frame time");
        // Keep the pre-capture state and elapsed interval, then overwrite
        // this tick's current state. SPP convergence changes the measurement,
        // never the detector's count of captures or its elapsed history.
    } else if (acquisitionIndex == m_impl->lastAcquisitionIndex + 1) {
        if (frameTimeSeconds < m_impl->lastFrameTimeSeconds)
            return Result<void, String>::Err(
                "GPU camera acquisitions must advance scene time");
        m_impl->stateBefore = m_impl->stateCurrent;
        m_impl->stateCurrent = 1u - m_impl->stateCurrent;
        m_impl->beforeFrameTimeSeconds = m_impl->lastFrameTimeSeconds;
        m_impl->currentDeltaSeconds =
            frameTimeSeconds - m_impl->beforeFrameTimeSeconds;
        m_impl->firstCaptureWasFirst = false;
    } else {
        return Result<void, String>::Err(
            "GPU camera acquisition index skipped or moved backward; process "
            "intermediate captures or explicitly ResetState after a scrub");
    }
    m_impl->lastAcquisitionIndex = acquisitionIndex;
    m_impl->lastFrameTimeSeconds = frameTimeSeconds;
    m_impl->hasCapture = true;

    auto push = m_impl->MakePush(acquisitionIndex, frameTimeSeconds);
    m_impl->TransitionImages(cmd);
    m_impl->UploadPendingParams(cmd);
    if (!stratified && m_impl->measurementIdentity == 0) {
        const gpu::TextureRef dummy = m_impl->psfTemp->Ref();
        m_impl->UpdateSet(m_impl->groups[0], {
            measuredRate, m_impl->psfTemp->Ref(),
            dummy, dummy, dummy, dummy, dummy});
        m_impl->measurementIdentity = measuredRate.Identity();
    }

    if (stratified) {
        if (dynamic.strataCount > kCameraTimeStrataMax)
            return Result<void, String>::Err(
                "GPU camera strata count exceeds the supported maximum");
        if (dynamic.strataLayers > 0u && dynamic.strataCount > dynamic.strataLayers)
            return Result<void, String>::Err(
                "GPU camera strata count exceeds the strata images' layer count");
        // The per-stratum frames change per acquisition. Write them inside
        // the command buffer (a 640-byte constant update) so the update is
        // ordered with the compositor's read and cannot overwrite frames a
        // previous, still-in-flight submission is reading.
        std::array<DynamicLayerCamera, 8> frames{};
        const u32 count = std::min<u32>(dynamic.strataCount, 8u);
        std::copy_n(dynamic.layerCameras, count, frames.data());
        cmd.UpdateBuffer(m_impl->layerCameraBuffer->Ref(), 0,
                         std::span<const std::byte>(
                             reinterpret_cast<const std::byte*>(frames.data()),
                             sizeof(frames)));
        cmd.Barrier(gpu::BufferBarrier{m_impl->layerCameraBuffer->Ref(),
                                       gpu::Access::TransferWrite,
                                       gpu::Access::ComputeRead});
        if (m_impl->strataRateIdentity == 0) {
            const gpu::TextureRef dummy = m_impl->psfTemp->Ref();
            m_impl->UpdateSet(m_impl->groups[7], {
                dynamic.strataRate, dynamic.strataDepth,
                m_impl->compositedRate->Ref(), dummy, dummy, dummy, dummy});
            m_impl->strataRateIdentity = dynamic.strataRate.Identity();
            m_impl->strataDepthIdentity = dynamic.strataDepth.Identity();
        }
        push.timeStratumCount = count;
        push.rollingShutter = dynamic.rowDelaySeconds > 0.0 ? 1u : 0u;
        push.exposureSeconds = static_cast<f32>(dynamic.exposureSeconds);
        push.rowDelaySeconds = static_cast<f32>(dynamic.rowDelaySeconds);
        push.firstRowMidSeconds = static_cast<f32>(dynamic.firstRowMidSeconds);
        cmd.FillBuffer(m_impl->dynamicCounterBuffer->Ref(), 0, gpu::kWholeSize, 0u);
        cmd.Barrier(gpu::BufferBarrier{m_impl->dynamicCounterBuffer->Ref(),
                                       gpu::Access::TransferWrite,
                                       gpu::Access::ComputeRead |
                                           gpu::Access::ComputeWrite});
    }

    const auto stamp = [&](u32 offset) {
        if (timing.pool.handle != 0)
            cmd.WriteTimestamp(timing.pool, timing.firstQuery + offset,
                               gpu::Stage::Compute);
    };
    if (timing.pool.handle != 0)
        cmd.ResetTimestamps(timing.pool, timing.firstQuery, kTimingQueryCount);
    stamp(0);
    if (stratified) {
        m_impl->BindAndDispatch(cmd, *m_impl->dynamicKernel,
                                m_impl->groups[7], push);
        StorageBarrier(cmd);
    }
    stamp(1);
    push.direction = 0;
    const gpu::BindGroup psfHorizontal =
        stratified ? m_impl->groups[8] : m_impl->groups[0];
    const gpu::BindGroup psfVertical =
        stratified ? m_impl->groups[9] : m_impl->groups[1];
    m_impl->BindAndDispatch(cmd, *m_impl->psfKernel, psfHorizontal, push);
    StorageBarrier(cmd);
    push.direction = 1;
    m_impl->BindAndDispatch(cmd, *m_impl->psfKernel, psfVertical, push);
    StorageBarrier(cmd);
    stamp(2);
    const gpu::BindGroup readoutGroup =
        m_impl->stateBefore == 0 ? m_impl->groups[2] : m_impl->groups[3];
    m_impl->BindAndDispatch(cmd, *m_impl->readoutKernel, readoutGroup, push);
    StorageBarrier(cmd);
    stamp(3);
    // M5: the classic ISP, restructured. Statistics first: per-tile
    // partials, a global reduce that also derives the next Linear-AGC
    // window from the previous frame's histogram (preloaded into shared
    // memory there), and per-tile groupshared histograms flushed to the
    // tile histogram block -- no whole-image global atomics. The CDF pass
    // merges the tile histograms into the persistent global histogram and
    // prefix-sums it in the same single workgroup. The demosaic pass then
    // runs the fused demosaic + color + display product chain (former
    // camera_demosaic) in one dispatch.
    m_impl->BindAndDispatch(cmd, *m_impl->statsTilesKernel, m_impl->groups[10],
                            push);
    StorageBarrier(cmd);
    m_impl->BindAndDispatchGrid(cmd, *m_impl->statsReduceKernel,
                                m_impl->groups[10], push, 1, 1);
    StorageBarrier(cmd);
    m_impl->BindAndDispatch(cmd, *m_impl->statsHistKernel, m_impl->groups[10],
                            push);
    StorageBarrier(cmd);
    m_impl->BindAndDispatchGrid(cmd, *m_impl->cdfKernel, m_impl->groups[13],
                                push, 1, 1);
    StorageBarrier(cmd);
    m_impl->BindAndDispatch(cmd, *m_impl->demosaicKernel, m_impl->groups[11],
                            push);
    StorageBarrier(cmd);
    stamp(4);
    // Display HSV grading and the empirical display effects, in place
    // on the display image. Skipped entirely when every HSV parameter sits at
    // its default (the kFlagHsv bit), so an untouched chain is bit-identical
    // to the pre-HSV pipeline.
    if (m_impl->ispFlags & isp::kFlagHsv) {
        m_impl->BindAndDispatch(cmd, *m_impl->hsvKernel, m_impl->groups[14],
                                push);
        StorageBarrier(cmd);
    }
    stamp(5);
    stamp(6);
    return Result<void, String>::Ok();
}

Result<void, String> GpuCameraPipeline::RecordFastRgbMeasurement(
    gpu::CommandList& cmd, gpu::TextureRef linearRgb, u64 acquisitionIndex,
    f64 frameTimeSeconds, TimingQueries timing) {
    if (!m_impl->configured)
        return Result<void, String>::Err("GPU camera must be configured first");
    if (m_impl->config.inputKind != camera::CameraInputKind::FastRgbApproximation)
        return Result<void, String>::Err(
            "fast RGB measurement requires input_kind=fast_rgb");
    if (!linearRgb.IsValid() || linearRgb.format != gpu::Format::RGBA32Float)
        return Result<void, String>::Err(
            "GPU fast RGB camera needs an RGBA32F linear-RGB image");
    if (m_impl->fastRgbInputIdentity != 0 &&
        m_impl->fastRgbInputIdentity != linearRgb.Identity())
        return Result<void, String>::Err(
            "GPU fast RGB input image changed; reconfigure after synchronizing "
            "in-flight frames");

    m_impl->TransitionImages(cmd);
    if (m_impl->fastRgbInputIdentity == 0) {
        const gpu::TextureRef dummy = m_impl->psfTemp->Ref();
        m_impl->UpdateSet(m_impl->groups[6], {
            linearRgb, m_impl->fastRgbRate->Ref(),
            dummy, dummy, dummy, dummy, dummy});
        m_impl->fastRgbInputIdentity = linearRgb.Identity();
    }
    auto push = m_impl->MakePush(acquisitionIndex, frameTimeSeconds);
    m_impl->BindAndDispatch(
        cmd, *m_impl->fastRgbKernel, m_impl->groups[6], push);
    StorageBarrier(cmd);
    return RecordMeasurement(
        cmd, m_impl->fastRgbRate->Ref(), acquisitionIndex, frameTimeSeconds,
        timing);
}

Result<void, String> GpuCameraPipeline::RecordRandomVectors(
    gpu::CommandList& cmd, u32 seed, u64 acquisitionIndex,
    camera::NoiseClass noiseClass, u32 counter) {
    if (!m_impl->configured)
        return Result<void, String>::Err(
            "GPU camera RNG vectors need a configured pipeline");
    m_impl->TransitionImages(cmd);
    auto push = m_impl->MakePush(acquisitionIndex, 0.0);
    push.seed = seed;
    push.rngNoiseClass = static_cast<u32>(noiseClass);
    push.rngCounter = counter;
    m_impl->BindAndDispatch(cmd, *m_impl->rngKernel, m_impl->groups[5], push);
    return Result<void, String>::Ok();
}

} // namespace quantiloom::rendercore
